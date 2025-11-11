"""
Test CUDA-compatible RNG functions against NumPy equivalents
"""
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import math

# CUDA device function for Poisson sampling
# Using Knuth's algorithm for small lambda, Gaussian approximation for large lambda
@cuda.jit(device=True)
def cuda_poisson(rng_states, thread_id, lam):
    """Generate Poisson random variable with parameter lambda"""
    if lam < 10.0:
        # Knuth's algorithm for small lambda
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = xoroshiro128p_uniform_float32(rng_states, thread_id)
            p *= u
            if p <= L:
                return k - 1
    else:
        # Gaussian approximation for large lambda: N(lambda, sqrt(lambda))
        g = xoroshiro128p_normal_float32(rng_states, thread_id)
        value = lam + g * math.sqrt(lam)
        return max(0, int(round(value)))


@cuda.jit
def test_uniform_kernel(rng_states, output):
    """Test uniform random generation"""
    thread_id = cuda.grid(1)
    if thread_id < output.size:
        output[thread_id] = xoroshiro128p_uniform_float32(rng_states, thread_id)


@cuda.jit
def test_normal_kernel(rng_states, output):
    """Test normal random generation"""
    thread_id = cuda.grid(1)
    if thread_id < output.size:
        output[thread_id] = xoroshiro128p_normal_float32(rng_states, thread_id)


@cuda.jit
def test_poisson_kernel(rng_states, lam, output):
    """Test Poisson random generation"""
    thread_id = cuda.grid(1)
    if thread_id < output.size:
        output[thread_id] = cuda_poisson(rng_states, thread_id, lam)


def test_cuda_uniform(n_samples=10000, seed=42):
    """Test CUDA uniform distribution matches NumPy"""
    print(f"\n{'='*60}")
    print("Testing CUDA Uniform Distribution")
    print('='*60)

    # NumPy baseline
    np.random.seed(seed)
    np_samples = np.random.random(n_samples)
    np_mean = np.mean(np_samples)
    np_std = np.std(np_samples)

    # CUDA samples
    rng_states = create_xoroshiro128p_states(n_samples, seed=seed)
    cuda_samples = np.zeros(n_samples, dtype=np.float32)

    threads_per_block = 256
    blocks = (n_samples + threads_per_block - 1) // threads_per_block

    test_uniform_kernel[blocks, threads_per_block](rng_states, cuda_samples)
    cuda.synchronize()

    cuda_mean = np.mean(cuda_samples)
    cuda_std = np.std(cuda_samples)

    print(f"NumPy:  mean={np_mean:.4f}, std={np_std:.4f}")
    print(f"CUDA:   mean={cuda_mean:.4f}, std={cuda_std:.4f}")
    print(f"Expected: mean≈0.5, std≈0.2887")

    # Check if within reasonable bounds
    mean_ok = abs(cuda_mean - 0.5) < 0.01
    std_ok = abs(cuda_std - 0.2887) < 0.01

    if mean_ok and std_ok:
        print("✓ CUDA uniform distribution is correct")
        return True
    else:
        print("✗ CUDA uniform distribution is incorrect")
        return False


def test_cuda_normal(n_samples=10000, seed=42):
    """Test CUDA normal distribution matches NumPy"""
    print(f"\n{'='*60}")
    print("Testing CUDA Normal Distribution")
    print('='*60)

    # NumPy baseline
    np.random.seed(seed)
    np_samples = np.random.randn(n_samples)
    np_mean = np.mean(np_samples)
    np_std = np.std(np_samples)

    # CUDA samples
    rng_states = create_xoroshiro128p_states(n_samples, seed=seed)
    cuda_samples = np.zeros(n_samples, dtype=np.float32)

    threads_per_block = 256
    blocks = (n_samples + threads_per_block - 1) // threads_per_block

    test_normal_kernel[blocks, threads_per_block](rng_states, cuda_samples)
    cuda.synchronize()

    cuda_mean = np.mean(cuda_samples)
    cuda_std = np.std(cuda_samples)

    print(f"NumPy:  mean={np_mean:.4f}, std={np_std:.4f}")
    print(f"CUDA:   mean={cuda_mean:.4f}, std={cuda_std:.4f}")
    print(f"Expected: mean≈0.0, std≈1.0")

    # Check if within reasonable bounds
    mean_ok = abs(cuda_mean) < 0.05
    std_ok = abs(cuda_std - 1.0) < 0.05

    if mean_ok and std_ok:
        print("✓ CUDA normal distribution is correct")
        return True
    else:
        print("✗ CUDA normal distribution is incorrect")
        return False


def test_cuda_poisson(lam=5.0, n_samples=10000, seed=42):
    """Test CUDA Poisson distribution matches NumPy"""
    print(f"\n{'='*60}")
    print(f"Testing CUDA Poisson Distribution (λ={lam})")
    print('='*60)

    # NumPy baseline
    np.random.seed(seed)
    np_samples = np.random.poisson(lam, n_samples)
    np_mean = np.mean(np_samples)
    np_std = np.std(np_samples)

    # CUDA samples
    rng_states = create_xoroshiro128p_states(n_samples, seed=seed)
    cuda_samples = np.zeros(n_samples, dtype=np.float32)

    threads_per_block = 256
    blocks = (n_samples + threads_per_block - 1) // threads_per_block

    test_poisson_kernel[blocks, threads_per_block](rng_states, lam, cuda_samples)
    cuda.synchronize()

    cuda_mean = np.mean(cuda_samples)
    cuda_std = np.std(cuda_samples)

    print(f"NumPy:  mean={np_mean:.4f}, std={np_std:.4f}")
    print(f"CUDA:   mean={cuda_mean:.4f}, std={cuda_std:.4f}")
    print(f"Expected: mean≈{lam}, std≈{np.sqrt(lam):.4f}")

    # Check if within reasonable bounds (10% tolerance for Poisson)
    mean_ok = abs(cuda_mean - lam) / lam < 0.10
    std_ok = abs(cuda_std - np.sqrt(lam)) / np.sqrt(lam) < 0.10

    if mean_ok and std_ok:
        print("✓ CUDA Poisson distribution is correct")
        return True
    else:
        print("✗ CUDA Poisson distribution is incorrect")
        return False


if __name__ == '__main__':
    print("="*60)
    print("CUDA RNG Verification Tests")
    print("="*60)

    try:
        uniform_ok = test_cuda_uniform()
        normal_ok = test_cuda_normal()
        poisson_small_ok = test_cuda_poisson(lam=5.0)
        poisson_large_ok = test_cuda_poisson(lam=20.0)

        print(f"\n{'='*60}")
        print("Summary")
        print('='*60)
        print(f"Uniform: {'✓' if uniform_ok else '✗'}")
        print(f"Normal:  {'✓' if normal_ok else '✗'}")
        print(f"Poisson (λ=5):  {'✓' if poisson_small_ok else '✗'}")
        print(f"Poisson (λ=20): {'✓' if poisson_large_ok else '✗'}")

        if uniform_ok and normal_ok and poisson_small_ok and poisson_large_ok:
            print("\n✓ All RNG tests passed!")
        else:
            print("\n✗ Some RNG tests failed")

    except Exception as e:
        print(f"\n✗ Error running tests: {e}")
        import traceback
        traceback.print_exc()
