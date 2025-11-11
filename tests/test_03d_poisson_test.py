import math

import numpy as np
from numba import cuda

"""
Test Poisson generation specifically
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
def random_poisson(state, lam):
	"""Generate Poisson random variable"""
	if lam < 10.0:
		L = math.exp(-lam)
		k = 0
		p = 1.0
		for _ in range(1000):
			k += 1
			u = random_uniform(state)
			p *= u
			if p <= L:
				return k - 1
		return int(lam)
	else:
		u1 = random_uniform(state)
		u2 = random_uniform(state)
		u1 = max(u1, 1e-10)
		r = math.sqrt(-2.0 * math.log(u1))
		theta = 2.0 * math.pi * u2
		g = r * math.cos(theta)
		value = lam + g * math.sqrt(lam)
		return max(0, int(round(value)))

@cuda.jit
def test_poisson_kernel(output, lam, n_samples):
	"""Each thread generates n_samples Poisson values"""
	thread_id = cuda.grid(1)
	if thread_id >= output.shape[0]:
		return
	
	seed = thread_id + 12345
	seed = hash_seed(seed)
	rng_state = cuda.local.array(1, dtype=np.uint64)
	rng_state[0] = seed
	
	for i in range(n_samples):
		val = random_poisson(rng_state, lam)
		output[thread_id, i] = val

def test_poisson(lam=0.188):
	"""Test Poisson generation with specific lambda"""
	print("=" * 60)
	print(f"Testing Poisson RNG with λ={lam}")
	print("=" * 60)
	
	n_threads = 64
	n_samples = 100
	output = np.zeros((n_threads, n_samples), dtype=np.float32)
	
	threads_per_block = 64
	blocks = (n_threads + threads_per_block - 1) // threads_per_block
	
	test_poisson_kernel[blocks, threads_per_block](output, lam, n_samples)
	cuda.synchronize()
	
	flat = output.flatten()
	mean = np.mean(flat)
	std = np.std(flat)
	
	# Count distribution
	unique, counts = np.unique(flat, return_counts=True)
	
	print(f"\nGenerated {n_threads * n_samples} samples")
	print("Statistics:")
	print(f"  Mean: {mean:.4f} (expected {lam:.4f})")
	print(f"  Std:  {std:.4f} (expected {np.sqrt(lam):.4f})")
	
	print("\nValue distribution:")
	for val, count in zip(unique, counts):
		pct = count / len(flat) * 100
		print(f"  {int(val)}: {count:5d} ({pct:5.1f}%)")
	
	# Check if reasonable
	if len(unique) == 1 and unique[0] == 0:
		print("\n✗ FAIL: All values are 0!")
		return False
	
	if abs(mean - lam) / lam > 0.2:
		print(f"\n✗ FAIL: Mean {mean:.4f} is far from expected {lam:.4f}")
		return False
	
	print("\n✓ PASS: Poisson distribution looks correct")
	return True

if __name__ == '__main__':
	try:
		# Test with lambda=0.188 (what the film grain uses)
		success = test_poisson(lam=0.188)
		
		print("\n" + "=" * 60)
		
		if not success:
			# Try with higher lambda to see if it's a numerical issue
			print("\nTrying with higher lambda...")
			test_poisson(lam=5.0)
	
	except Exception as e:
		print(f"\n✗ Error: {e}")
		import traceback
		
		traceback.print_exc()
