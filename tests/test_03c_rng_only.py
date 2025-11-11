"""
Test just the RNG functions in isolation
"""
import numpy as np
from numba import cuda


@cuda.jit(device=True)
def hash_seed(seed):
    """Hash function for seeding"""
    seed = (seed ^ 61) ^ (seed >> 16)
    seed = seed + (seed << 3)
    seed = seed ^ (seed >> 4)
    seed = seed * 0x27d4eb2d
    seed = seed ^ (seed >> 15)
    return seed & 0xFFFFFFFF


@cuda.jit(device=True)
def pcg_random(state):
    """PCG random number generator"""
    oldstate = state[0]
    state[0] = (oldstate * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
    rot = (oldstate >> 59) & 0xFFFFFFFF
    return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF


@cuda.jit(device=True)
def random_uniform(state):
    """Generate uniform random float in [0, 1)"""
    r = pcg_random(state)
    return r / 4294967296.0


@cuda.jit
def test_rng_kernel(output, n_samples):
    """Test RNG: each thread generates n_samples random numbers"""
    thread_id = cuda.grid(1)
    if thread_id >= output.shape[0]:
        return

    # Initialize RNG state
    seed = thread_id + 12345
    seed = hash_seed(seed)
    rng_state = cuda.local.array(1, dtype=np.uint64)
    rng_state[0] = seed

    # Generate samples
    for i in range(n_samples):
        u = random_uniform(rng_state)
        output[thread_id, i] = u


def test_uniform_rng():
    """Test that uniform RNG produces varied output"""
    print("="*60)
    print("Testing PCG Uniform RNG")
    print("="*60)

    n_threads = 64
    n_samples = 10
    output = np.zeros((n_threads, n_samples), dtype=np.float32)

    threads_per_block = 64
    blocks = (n_threads + threads_per_block - 1) // threads_per_block

    test_rng_kernel[blocks, threads_per_block](output, n_samples)
    cuda.synchronize()

    print(f"\nGenerated {n_threads} threads × {n_samples} samples")
    print(f"First thread samples: {output[0, :]}")
    print(f"Second thread samples: {output[1, :]}")

    # Statistics
    mean = np.mean(output)
    std = np.std(output)
    min_val = np.min(output)
    max_val = np.max(output)

    print(f"\nStatistics:")
    print(f"  Mean: {mean:.4f} (expected ~0.5)")
    print(f"  Std:  {std:.4f} (expected ~0.289)")
    print(f"  Min:  {min_val:.4f}")
    print(f"  Max:  {max_val:.4f}")

    # Check for obvious failures
    if min_val == max_val:
        print("\n✗ FAIL: All values are identical!")
        return False

    if mean < 0.3 or mean > 0.7:
        print(f"\n✗ FAIL: Mean {mean:.4f} is far from 0.5")
        return False

    if std < 0.1:
        print(f"\n✗ FAIL: Std {std:.4f} is too low (values not varied)")
        return False

    print("\n✓ PASS: RNG is producing varied uniform values")
    return True


if __name__ == '__main__':
    try:
        test_uniform_rng()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
