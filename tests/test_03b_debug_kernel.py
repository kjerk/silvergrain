import math

import numpy as np
from numba import cuda

"""
Debug version to diagnose why GPU kernel returns zeros
"""

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


@cuda.jit(device=True)
def random_poisson(state, lam):
    """Generate Poisson random variable"""
    if lam < 10.0:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        for _ in range(1000):  # Safety limit
            k += 1
            u = random_uniform(state)
            p *= u
            if p <= L:
                return k - 1
        return int(lam)
    else:
        # Gaussian approximation
        u1 = random_uniform(state)
        u2 = random_uniform(state)
        u1 = max(u1, 1e-10)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        g = r * math.cos(theta)
        value = lam + g * math.sqrt(lam)
        return max(0, int(round(value)))


@cuda.jit(device=True)
def get_cell_seed(x, y, offset):
    """Generate unique seed for a cell"""
    return (y * 65536 + x + offset) & 0xFFFFFFFF


@cuda.jit
def debug_kernel(
    img_in, debug_output,
    m_in, n_in, m_out, n_out,
    grain_radius, seed_offset,
    lambda_lut
):
    """
    Debug kernel that outputs diagnostic info
    debug_output has 5 values per pixel:
    [0] = n_monte_carlo iterations run
    [1] = number of cells checked
    [2] = total grains generated
    [3] = coverage count
    [4] = final pixel value
    """
    x_out = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y_out = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x_out >= n_out or y_out >= m_out:
        return

    MAX_GREY_LEVEL = 255
    EPSILON_GREY_LEVEL = 0.1
    grain_radius_sq = grain_radius * grain_radius
    max_radius = grain_radius
    ag = 1.0 / math.ceil(1.0 / grain_radius)

    # Simplified: no zoom, no gaussian shift, just 10 MC samples
    n_monte_carlo = 10
    x_in = float(x_out) + 0.5
    y_in = float(y_out) + 0.5

    pix_out = 0.0
    total_cells = 0
    total_grains = 0
    total_coverage = 0

    for i in range(n_monte_carlo):
        x_gaussian = x_in
        y_gaussian = y_in

        # Bounding box
        min_x = int(math.floor((x_gaussian - max_radius) / ag))
        max_x = int(math.floor((x_gaussian + max_radius) / ag))
        min_y = int(math.floor((y_gaussian - max_radius) / ag))
        max_y = int(math.floor((y_gaussian + max_radius) / ag))

        pt_covered = False

        for ncx in range(min_x, max_x + 1):
            if pt_covered:
                break
            for ncy in range(min_y, max_y + 1):
                if pt_covered:
                    break

                total_cells += 1

                cell_corner_x = ag * ncx
                cell_corner_y = ag * ncy

                # Get intensity
                ix = max(0, min(int(math.floor(cell_corner_x)), n_in - 1))
                iy = max(0, min(int(math.floor(cell_corner_y)), m_in - 1))
                u = img_in[iy, ix]

                # Lookup lambda
                u_ind = int(math.floor(u * (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)))
                u_ind = max(0, min(u_ind, MAX_GREY_LEVEL))
                curr_lambda = lambda_lut[u_ind]

                # Seed RNG
                seed = get_cell_seed(ncx, ncy, seed_offset)
                seed = hash_seed(seed)
                rng_state = cuda.local.array(1, dtype=np.uint64)
                rng_state[0] = seed

                # Draw grains
                n_cell = random_poisson(rng_state, curr_lambda)
                total_grains += n_cell

                # Check each grain
                for k in range(n_cell):
                    x_centre_grain = cell_corner_x + ag * random_uniform(rng_state)
                    y_centre_grain = cell_corner_y + ag * random_uniform(rng_state)

                    dx = x_centre_grain - x_gaussian
                    dy = y_centre_grain - y_gaussian
                    dist_sq = dx * dx + dy * dy

                    if dist_sq < grain_radius_sq:
                        pix_out += 1.0
                        pt_covered = True
                        total_coverage += 1
                        break

    # Write debug info
    idx = y_out * n_out + x_out
    debug_output[idx, 0] = n_monte_carlo
    debug_output[idx, 1] = total_cells
    debug_output[idx, 2] = total_grains
    debug_output[idx, 3] = total_coverage
    debug_output[idx, 4] = pix_out / n_monte_carlo


def test_debug():
    """Run debug kernel and print diagnostics"""
    print("="*60)
    print("Debug Kernel Test")
    print("="*60)

    # Small test image
    img = np.full((8, 8), 0.5, dtype=np.float32)
    m_in, n_in = img.shape
    m_out, n_out = img.shape

    grain_radius = 0.12
    seed_offset = 2016

    # Compute lambda LUT
    MAX_GREY_LEVEL = 255
    EPSILON_GREY_LEVEL = 0.1
    lambda_lut = np.zeros(MAX_GREY_LEVEL + 1, dtype=np.float32)
    ag = 1.0 / np.ceil(1.0 / grain_radius)

    for i in range(MAX_GREY_LEVEL + 1):
        u = i / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        lam = -(ag * ag) / (np.pi * grain_radius * grain_radius) * np.log(1.0 - u)
        lambda_lut[i] = lam

    print(f"Lambda for u=0.5: {lambda_lut[127]:.4f}")

    # Allocate debug output
    debug_output = np.zeros((m_out * n_out, 5), dtype=np.float32)

    # Copy to GPU
    d_img_in = cuda.to_device(img)
    d_debug_out = cuda.to_device(debug_output)
    d_lambda_lut = cuda.to_device(lambda_lut)

    # Launch kernel
    threads_per_block = (4, 4)
    blocks_x = (n_out + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (m_out + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    print(f"\nLaunching kernel with grid={blocks_per_grid}, block={threads_per_block}")

    debug_kernel[blocks_per_grid, threads_per_block](
        d_img_in, d_debug_out,
        m_in, n_in, m_out, n_out,
        grain_radius, seed_offset,
        d_lambda_lut
    )

    # Copy back
    d_debug_out.copy_to_host(debug_output)

    # Print results for center pixel
    center_idx = (m_out // 2) * n_out + (n_out // 2)
    print("\nCenter pixel (4, 4) diagnostics:")
    print(f"  MC iterations: {debug_output[center_idx, 0]:.0f}")
    print(f"  Cells checked: {debug_output[center_idx, 1]:.0f}")
    print(f"  Grains generated: {debug_output[center_idx, 2]:.0f}")
    print(f"  Coverage hits: {debug_output[center_idx, 3]:.0f}")
    print(f"  Final value: {debug_output[center_idx, 4]:.4f}")

    # Statistics across all pixels
    print(f"\nStatistics across all {m_out}x{n_out} pixels:")
    print(f"  Avg cells checked: {np.mean(debug_output[:, 1]):.1f}")
    print(f"  Avg grains generated: {np.mean(debug_output[:, 2]):.1f}")
    print(f"  Avg coverage hits: {np.mean(debug_output[:, 3]):.1f}")
    print(f"  Avg final value: {np.mean(debug_output[:, 4]):.4f}")
    print("  Expected value: ~0.5")

    if np.mean(debug_output[:, 4]) > 0.01:
        print("\n✓ Kernel is producing non-zero output")
    else:
        print("\n✗ Kernel is still producing zeros")
        print("\nPossible issues:")
        if np.mean(debug_output[:, 2]) == 0:
            print("  - No grains being generated (check Poisson RNG or lambda)")
        elif np.mean(debug_output[:, 3]) == 0:
            print("  - Grains generated but no coverage (check distance calculation)")
        else:
            print("  - Unknown issue")


if __name__ == '__main__':
    try:
        test_debug()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
