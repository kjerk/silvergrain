import time

import numpy as np

from silvergrain.renderer import FilmGrainRenderer
from silvergrain.renderer_gpu import FilmGrainRendererGPU

"""
Benchmark GPU vs CPU performance
"""

def benchmark(renderer, img, name, warmup=True):
    """Benchmark a renderer on an image"""
    if warmup:
        # Warmup run (for JIT compilation)
        _ = renderer._render_single_channel(img, zoom=1.0, output_size=None)

    # Timed run
    start = time.time()
    result = renderer._render_single_channel(img, zoom=1.0, output_size=None)
    elapsed = time.time() - start

    return result, elapsed


def run_benchmark(size, n_monte_carlo, grain_radius=0.12):
    """Run benchmark for a given image size"""
    print("="*60)
    print(f"Benchmark: {size}x{size} @ {n_monte_carlo} MC samples")
    print("="*60)

    # Create test image (gradient)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    img = X.astype(np.float32)

    print(f"Image size: {size}x{size} = {size*size:,} pixels")
    print(f"MC samples: {n_monte_carlo}")
    print(f"Total evaluations: {size*size*n_monte_carlo:,}")

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

    # Benchmark CPU
    print("\nBenchmarking CPU...")
    cpu_result, cpu_time = benchmark(cpu_renderer, img, "CPU", warmup=True)
    print(f"  Time: {cpu_time:.2f} seconds")
    print(f"  Throughput: {size*size/cpu_time:.0f} pixels/sec")

    # Benchmark GPU
    print("\nBenchmarking GPU...")
    gpu_result, gpu_time = benchmark(gpu_renderer, img, "GPU", warmup=True)
    print(f"  Time: {gpu_time:.2f} seconds")
    print(f"  Throughput: {size*size/gpu_time:.0f} pixels/sec")

    # Speedup
    speedup = cpu_time / gpu_time
    print(f"\n{'='*60}")
    print(f"Speedup: {speedup:.1f}x")
    print('='*60)

    # Verify results are similar
    cpu_mean = np.mean(cpu_result)
    gpu_mean = np.mean(gpu_result)
    mean_diff_pct = abs(cpu_mean - gpu_mean) / cpu_mean * 100 if cpu_mean > 0 else 0
    print("\nResult verification:")
    print(f"  CPU mean: {cpu_mean:.4f}")
    print(f"  GPU mean: {gpu_mean:.4f}")
    print(f"  Difference: {mean_diff_pct:.1f}%")

    return speedup, cpu_time, gpu_time


def estimate_large_image(base_size, base_time, target_size):
    """Estimate time for larger image based on smaller benchmark"""
    # Time scales roughly linearly with number of pixels
    scale_factor = (target_size / base_size) ** 2
    estimated_time = base_time * scale_factor
    return estimated_time


if __name__ == '__main__':
    print("GPU vs CPU Performance Benchmark")
    print("="*60)

    try:
        # Small image (quick test)
        print("\n\n")
        speedup_small, cpu_small, gpu_small = run_benchmark(
            size=128,
            n_monte_carlo=100,
            grain_radius=0.12
        )

        # Medium image
        print("\n\n")
        speedup_med, cpu_med, gpu_med = run_benchmark(
            size=256,
            n_monte_carlo=100,
            grain_radius=0.12
        )

        # Estimate for 1408x2058 (original target size)
        target_h, target_w = 1408, 2058
        target_pixels = target_h * target_w

        # Use medium benchmark to estimate
        med_pixels = 256 * 256
        scale = target_pixels / med_pixels

        est_cpu_time = cpu_med * scale
        est_gpu_time = gpu_med * scale

        print("\n\n")
        print("="*60)
        print(f"Estimated time for {target_w}x{target_h} @ 100 MC samples:")
        print("="*60)
        print(f"  CPU: {est_cpu_time:.1f} seconds ({est_cpu_time/60:.1f} minutes)")
        print(f"  GPU: {est_gpu_time:.1f} seconds")
        print(f"  Speedup: ~{speedup_med:.1f}x")

        # Summary
        print("\n\n")
        print("="*60)
        print("Summary")
        print("="*60)
        print(f"128x128:  {speedup_small:.1f}x speedup")
        print(f"256x256:  {speedup_med:.1f}x speedup")
        print(f"Estimated 1408x2058: {est_gpu_time:.1f}s (vs {est_cpu_time/60:.1f}min on CPU)")

        if speedup_med > 20:
            print("\n✓ Excellent speedup! GPU acceleration is working well.")
        elif speedup_med > 10:
            print("\n✓ Good speedup. GPU acceleration is effective.")
        elif speedup_med > 5:
            print("\n⚠ Moderate speedup. May need optimization.")
        else:
            print("\n✗ Low speedup. GPU may not be properly utilized.")

    except Exception as e:
        print(f"\n✗ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
