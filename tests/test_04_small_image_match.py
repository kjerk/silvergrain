"""
Test GPU and CPU on small image with visual comparison
"""
import numpy as np
from PIL import Image
from silvergrain.renderer import FilmGrainRenderer
from silvergrain.renderer_gpu import FilmGrainRendererGPU


def create_test_image(size=64):
    """Create a test image with gradient"""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    # Horizontal gradient
    img = X.astype(np.float32)
    return img


def test_small_image():
    """Render same image on CPU and GPU, compare visually"""
    print("="*60)
    print("Small Image Test (64x64)")
    print("="*60)

    # Create test image
    img = create_test_image(size=64)
    print(f"Created {img.shape} test image")

    # Rendering parameters
    grain_radius = 0.12
    n_monte_carlo = 100  # Low for speed
    seed = 2016

    print(f"\nRendering parameters:")
    print(f"  Grain radius: {grain_radius}")
    print(f"  MC samples: {n_monte_carlo}")
    print(f"  Seed: {seed}")

    # CPU rendering
    print("\nRendering on CPU...")
    cpu_renderer = FilmGrainRenderer(
        grain_radius=grain_radius,
        n_monte_carlo=n_monte_carlo,
        seed=seed
    )
    cpu_result = cpu_renderer._render_single_channel(img, zoom=1.0, output_size=None)
    cpu_mean = np.mean(cpu_result)
    cpu_std = np.std(cpu_result)

    # GPU rendering
    print("Rendering on GPU...")
    gpu_renderer = FilmGrainRendererGPU(
        grain_radius=grain_radius,
        n_monte_carlo=n_monte_carlo,
        seed=seed
    )
    gpu_result = gpu_renderer._render_single_channel(img, zoom=1.0, output_size=None)
    gpu_mean = np.mean(gpu_result)
    gpu_std = np.std(gpu_result)

    # Statistics
    print(f"\n{'='*60}")
    print("Statistics:")
    print('='*60)
    print(f"CPU: mean={cpu_mean:.4f}, std={cpu_std:.4f}, range=[{cpu_result.min():.3f}, {cpu_result.max():.3f}]")
    print(f"GPU: mean={gpu_mean:.4f}, std={gpu_std:.4f}, range=[{gpu_result.min():.3f}, {gpu_result.max():.3f}]")

    mean_diff_pct = abs(cpu_mean - gpu_mean) / cpu_mean * 100 if cpu_mean > 0 else 0
    std_diff_pct = abs(cpu_std - gpu_std) / cpu_std * 100 if cpu_std > 0 else 0

    print(f"\nDifferences:")
    print(f"  Mean: {mean_diff_pct:.1f}%")
    print(f"  Std:  {std_diff_pct:.1f}%")

    # Save images for visual comparison
    print("\nSaving images...")

    # Input
    img_uint8 = (np.clip(img * 255, 0, 255)).astype(np.uint8)
    Image.fromarray(img_uint8, mode='L').save('test_04_input.png')

    # CPU output
    cpu_uint8 = (np.clip(cpu_result * 255, 0, 255)).astype(np.uint8)
    Image.fromarray(cpu_uint8, mode='L').save('test_04_cpu.png')

    # GPU output
    gpu_uint8 = (np.clip(gpu_result * 255, 0, 255)).astype(np.uint8)
    Image.fromarray(gpu_uint8, mode='L').save('test_04_gpu.png')

    # Difference map (amplified)
    diff = np.abs(cpu_result - gpu_result)
    diff_amplified = np.clip(diff * 10.0, 0, 1.0)  # 10x amplification
    diff_uint8 = (diff_amplified * 255).astype(np.uint8)
    Image.fromarray(diff_uint8, mode='L').save('test_04_diff.png')

    print("✓ Saved:")
    print("  - test_04_input.png (original)")
    print("  - test_04_cpu.png (CPU result)")
    print("  - test_04_gpu.png (GPU result)")
    print("  - test_04_diff.png (difference, 10x amplified)")

    # Check if results are reasonable
    stats_ok = mean_diff_pct < 20 and std_diff_pct < 30

    if stats_ok:
        print("\n✓ CPU and GPU results are statistically similar")
        return True
    else:
        print("\n⚠ CPU and GPU results differ more than expected")
        print("  This may be due to different RNG implementations")
        print("  Check visual outputs to verify both look reasonable")
        return False


if __name__ == '__main__':
    try:
        test_small_image()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
