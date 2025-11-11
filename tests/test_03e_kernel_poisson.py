import math

import numpy as np
from numba import cuda

"""
Debug exactly what happens when calling Poisson inside the kernel
"""

@cuda.jit(device=True)
def hash_seed(seed):
	seed = (seed ^ 61) ^ (seed >> 16)
	seed = seed + (seed << 3)
	seed = seed ^ (seed >> 4)
	seed = seed * 0x27d4eb2d
	seed = seed ^ (seed >> 15)
	return seed & 0xFFFFFFFF

@cuda.jit(device=True)
def pcg_random(state):
	oldstate = state[0]
	state[0] = (oldstate * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
	xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
	rot = (oldstate >> 59) & 0xFFFFFFFF
	return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

@cuda.jit(device=True)
def random_uniform(state):
	r = pcg_random(state)
	return r / 4294967296.0

@cuda.jit(device=True)
def random_poisson_debug(state, lam, debug):
	"""Poisson with debug output"""
	if lam < 10.0:
		L = math.exp(-lam)
		debug[0] = L  # Store L for debugging
		k = 0
		p = 1.0
		for iter_num in range(1000):
			k += 1
			u = random_uniform(state)
			p *= u
			if iter_num < 5:  # Store first few iterations
				debug[1 + iter_num] = u
			if p <= L:
				return k - 1
		return int(lam)
	else:
		return 0  # Simplified for debugging

@cuda.jit(device=True)
def get_cell_seed(x, y, offset):
	return (y * 65536 + x + offset) & 0xFFFFFFFF

@cuda.jit
def test_kernel_poisson_call(output, debug_array, lam_value, seed_offset):
	"""
	Test calling Poisson exactly as the render kernel does.

	output[thread_id] = number of grains generated
	debug_array[thread_id, :] = debug info from Poisson call
	"""
	thread_id = cuda.grid(1)
	if thread_id >= output.shape[0]:
		return
	
	# Simulate cell coordinates
	ncx = thread_id % 8
	ncy = thread_id // 8
	
	# Seed RNG exactly as in render kernel
	seed = get_cell_seed(ncx, ncy, seed_offset)
	seed = hash_seed(seed)
	rng_state = cuda.local.array(1, dtype=np.uint64)
	# Properly initialize 64-bit state from 32-bit seed
	seed64 = (seed << 32) | hash_seed(seed + 1)
	rng_state[0] = seed64
	
	# Debug array for this thread
	thread_debug = cuda.local.array(10, dtype=np.float32)
	for i in range(10):
		thread_debug[i] = 0.0
	
	# Call Poisson
	n_grains = random_poisson_debug(rng_state, lam_value, thread_debug)
	
	# Write output
	output[thread_id] = n_grains
	for i in range(10):
		debug_array[thread_id, i] = thread_debug[i]

def test_kernel_call():
	"""Test Poisson being called from kernel context"""
	print("=" * 60)
	print("Testing Poisson called from kernel (as render does)")
	print("=" * 60)
	
	n_threads = 64
	lam_value = 0.188
	seed_offset = 2016
	
	output = np.zeros(n_threads, dtype=np.float32)
	debug_array = np.zeros((n_threads, 10), dtype=np.float32)
	
	threads_per_block = 64
	blocks = (n_threads + threads_per_block - 1) // threads_per_block
	
	test_kernel_poisson_call[blocks, threads_per_block](
		output, debug_array, lam_value, seed_offset
	)
	cuda.synchronize()
	
	print(f"\nLambda value: {lam_value}")
	print(f"Expected L = exp(-lambda) = {np.exp(-lam_value):.6f}")
	
	# Check a few threads
	for tid in [0, 1, 10, 20]:
		print(f"\nThread {tid}:")
		print(f"  Grains generated: {int(output[tid])}")
		print(f"  L value: {debug_array[tid, 0]:.6f}")
		print(f"  First uniform samples: {debug_array[tid, 1:4]}")
	
	# Statistics
	mean_grains = np.mean(output)
	print("\nOverall statistics:")
	print(f"  Mean grains: {mean_grains:.4f} (expected {lam_value:.4f})")
	
	unique, counts = np.unique(output, return_counts=True)
	print("  Distribution:")
	for val, count in zip(unique, counts):
		pct = count / len(output) * 100
		print(f"    {int(val)}: {count:3d} ({pct:5.1f}%)")
	
	if mean_grains < 0.01:
		print("\n✗ FAIL: Still getting all zeros!")
		print("  Checking debug values...")
		if np.all(debug_array[:, 0] == 0):
			print("  ERROR: L value is 0 (exp(-lambda) calculation failed)")
		elif np.all(debug_array[:, 1] == 0):
			print("  ERROR: Uniform samples are 0 (RNG not working in this context)")
		else:
			print("  ERROR: Unknown issue")
		return False
	else:
		print("\n✓ PASS: Poisson works when called from kernel")
		return True

if __name__ == '__main__':
	try:
		test_kernel_call()
	except Exception as e:
		print(f"\n✗ Error: {e}")
		import traceback
		
		traceback.print_exc()
