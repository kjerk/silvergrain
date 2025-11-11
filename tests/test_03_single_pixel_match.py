"""
Test that GPU and CPU produce similar results for single pixels.

Note: Results won't be identical due to different RNG implementations,
but should be statistically similar.
"""
import numpy as np
from silvergrain.renderer import FilmGrainRenderer
from silvergrain.renderer_gpu import FilmGrainRendererGPU


def test_single_pixel(intensity=0.5, grain_radius=0.12, n_monte_carlo=200):
    """
    Test single pixel rendering on both CPU and GPU.

    Since RNG implementations differ, we test multiple pixels and compare statistics.
    """
    print("="*60)
    print("Single Pixel Rendering Test (CPU vs GPU)")
    print("="*60)

    # Create test image (single constant intensity)
    img = np.full((8, 8), intensity, dtype=np.float32)

    print(f"Test parameters:")
    print(f"  Input intensity: {intensity}")
    print(f"  Grain radius: {grain_radius}")
    print(f"  MC samples: {n_monte_carlo}")

    # Create renderers
    cpu_renderer = FilmGrainRenderer(
        grain_radius=grain_radius,
        n_monte_carlo=n_monte_carlo,
        seed=2016
    )

    gpu_renderer = FilmGrainRendererGPU(
        grain_radius=grain_radius,
        n_monte_carlo=n_monte_carlo,
        seed=2016
    )

    # Render on CPU
    print("\nRendering on CPU...")
    cpu_result = cpu_renderer._render_single_channel(img, zoom=1.0, output_size=None)
    cpu_mean = np.mean(cpu_result)
    cpu_std = np.std(cpu_result)

    # Render on GPU
    print("Rendering on GPU...")
    gpu_result = gpu_renderer._render_single_channel(img, zoom=1.0, output_size=None)
    gpu_mean = np.mean(gpu_result)
    gpu_std = np.std(gpu_result)

    # Compare statistics
    print(f"\n{'='*60}")
    print("Results:")
    print('='*60)
    print(f"CPU: mean={cpu_mean:.4f}, std={cpu_std:.4f}")
    print(f"GPU: mean={gpu_mean:.4f}, std={gpu_std:.4f}")
    print(f"Expected: mean≈{intensity:.4f}")

    # Check if means are close (within 20% - loose tolerance for small sample)
    mean_diff_pct = abs(cpu_mean - gpu_mean) / cpu_mean * 100 if cpu_mean > 0 else 0
    print(f"\nMean difference: {mean_diff_pct:.1f}%")

    # Both should be reasonably close to input intensity
    cpu_ok = abs(cpu_mean - intensity) / intensity < 0.3  # 30% tolerance
    gpu_ok = abs(gpu_mean - intensity) / intensity < 0.3
    similar_ok = mean_diff_pct < 30  # CPU and GPU within 30% of each other

    if cpu_ok and gpu_ok and similar_ok:
        print("✓ Results are statistically similar")
        return True
    else:
        print("✗ Results differ significantly")
        if not cpu_ok:
            print(f"  CPU mean {cpu_mean:.4f} is far from input {intensity:.4f}")
        if not gpu_ok:
            print(f"  GPU mean {gpu_mean:.4f} is far from input {intensity:.4f}")
        if not similar_ok:
            print(f"  CPU and GPU differ by {mean_diff_pct:.1f}%")
        return False


def test_multiple_intensities():
    """Test across different intensity levels"""
    print("\n" + "="*60)
    print("Testing Multiple Intensities")
    print("="*60)

    intensities = [0.2, 0.5, 0.8]
    all_ok = True

    for intensity in intensities:
        print(f"\n--- Testing intensity {intensity} ---")
        ok = test_single_pixel(intensity=intensity, grain_radius=0.12, n_monte_carlo=100)
        if not ok:
            all_ok = False

    return all_ok


if __name__ == '__main__':
    try:
        # Single test with detailed output
        ok1 = test_single_pixel(intensity=0.5, grain_radius=0.12, n_monte_carlo=200)

        # Multiple intensities
        ok2 = test_multiple_intensities()

        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        if ok1 and ok2:
            print("✓ All tests passed - GPU and CPU are statistically similar")
        else:
            print("✗ Some tests failed - results differ significantly")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
