"""
Test GPU capability and Numba CUDA installation
"""
import sys

def check_gpu_capability():
    """Check if CUDA GPU is available and working"""
    print("="*60)
    print("GPU Capability Check")
    print("="*60)

    # Check if numba.cuda is available
    try:
        from numba import cuda
        print("✓ numba.cuda module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numba.cuda: {e}")
        print("  Install with: pip install numba")
        return False

    # Check if CUDA is available
    if not cuda.is_available():
        print("✗ CUDA is not available")
        print("  Possible issues:")
        print("    - No NVIDIA GPU detected")
        print("    - CUDA toolkit not installed")
        print("    - GPU drivers not installed")
        return False

    print("✓ CUDA is available")

    # Get GPU info
    try:
        gpu = cuda.get_current_device()
        print(f"\nGPU Information:")
        print(f"  Name: {gpu.name.decode('utf-8')}")
        print(f"  Compute Capability: {gpu.compute_capability}")
        print(f"  Total Memory: {gpu.MAX_THREADS_PER_BLOCK} threads/block max")
        print(f"  Multiprocessors: {gpu.MULTIPROCESSOR_COUNT}")
    except Exception as e:
        print(f"✗ Failed to get GPU info: {e}")
        return False

    # Test simple kernel launch
    try:
        @cuda.jit
        def simple_kernel(output):
            idx = cuda.grid(1)
            if idx < output.size:
                output[idx] = idx * 2

        import numpy as np
        test_array = np.zeros(10, dtype=np.int32)

        threads_per_block = 32
        blocks = (test_array.size + threads_per_block - 1) // threads_per_block

        simple_kernel[blocks, threads_per_block](test_array)
        cuda.synchronize()

        expected = np.arange(10) * 2
        if np.allclose(test_array, expected):
            print("\n✓ Successfully launched and executed test kernel")
        else:
            print("\n✗ Kernel execution produced incorrect results")
            print(f"  Expected: {expected}")
            print(f"  Got: {test_array}")
            return False

    except Exception as e:
        print(f"\n✗ Failed to launch test kernel: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ GPU is ready for film grain rendering")
    print("="*60)
    return True


if __name__ == '__main__':
    success = check_gpu_capability()
    sys.exit(0 if success else 1)
