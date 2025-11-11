import numpy as np
from numba import cuda
import math
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings

"""
GPU-Accelerated Film Grain Renderer using Numba CUDA

Ports the CPU rendering algorithm to CUDA for massive speedup.
Each thread renders one output pixel independently.
"""

# ============================================================================
# CUDA DEVICE FUNCTIONS - RNG
# ============================================================================

@cuda.jit(device=True)
def hash_seed(seed):
    """Hash function for seeding (similar to wang_hash)"""
    seed = (seed ^ 61) ^ (seed >> 16)
    seed = seed + (seed << 3)
    seed = seed ^ (seed >> 4)
    seed = seed * 0x27d4eb2d
    seed = seed ^ (seed >> 15)
    return seed & 0xFFFFFFFF


@cuda.jit(device=True)
def pcg_random(state):
    """PCG random number generator - returns (new_state, random_uint)"""
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
def random_normal(state):
    """Generate standard normal random variable using Box-Muller"""
    u1 = random_uniform(state)
    u2 = random_uniform(state)
    # Clamp u1 to avoid log(0)
    u1 = max(u1, 1e-10)
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta)


@cuda.jit(device=True)
def random_poisson(state, lam):
    """Generate Poisson random variable"""
    if lam < 10.0:
        # Knuth's algorithm for small lambda
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random_uniform(state)
            p *= u
            if p <= L:
                return k - 1
            if k > 1000:  # Safety limit
                return int(lam)
    else:
        # Gaussian approximation for large lambda
        g = random_normal(state)
        value = lam + g * math.sqrt(lam)
        return max(0, int(round(value)))


@cuda.jit(device=True)
def get_cell_seed(x, y, offset):
    """Generate unique seed for a cell"""
    return (y * 65536 + x + offset) & 0xFFFFFFFF


# ============================================================================
# CUDA DEVICE FUNCTIONS - GEOMETRY
# ============================================================================

@cuda.jit(device=True)
def sq_distance(x1, y1, x2, y2):
    """Squared Euclidean distance"""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


# ============================================================================
# CUDA KERNEL
# ============================================================================

@cuda.jit
def render_grain_kernel(
    img_in, img_out,
    m_in, n_in, m_out, n_out,
    grain_radius, grain_sigma, sigma_filter,
    n_monte_carlo, seed_offset,
    x_a, y_a, x_b, y_b,
    lambda_lut,
    x_gaussian_list, y_gaussian_list
):
    """
    CUDA kernel: each thread renders one output pixel.

    Grid/Block organization:
    - 2D grid of threads, one per output pixel
    - Each thread runs full Monte Carlo loop independently
    """
    # Get pixel coordinates
    x_out = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y_out = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Bounds check
    if x_out >= n_out or y_out >= m_out:
        return

    # Constants
    NORMAL_QUANTILE = 3.0902
    MAX_GREY_LEVEL = 255
    EPSILON_GREY_LEVEL = 0.1

    grain_radius_sq = grain_radius * grain_radius
    max_radius = grain_radius

    # Log-normal parameters
    if grain_sigma > 0.0:
        sigma = math.sqrt(math.log((grain_sigma / grain_radius) ** 2 + 1.0))
        sigma_sq = sigma * sigma
        mu = math.log(grain_radius) - sigma_sq / 2.0
        log_normal_quantile = math.exp(mu + sigma * NORMAL_QUANTILE)
        max_radius = log_normal_quantile
    else:
        mu = 0.0
        sigma = 0.0

    # Cell size
    ag = 1.0 / math.ceil(1.0 / grain_radius)

    # Scale factors
    s_x = (n_out - 1) / (x_b - x_a)
    s_y = (m_out - 1) / (y_b - y_a)

    # Convert output pixel to input coordinates
    x_in = x_a + (x_out + 0.5) * ((x_b - x_a) / n_out)
    y_in = y_a + (y_out + 0.5) * ((y_b - y_a) / m_out)

    pix_out = 0.0

    # Monte Carlo loop
    for i in range(n_monte_carlo):
        # Apply Gaussian shift
        x_gaussian = x_in + x_gaussian_list[i] / s_x
        y_gaussian = y_in + y_gaussian_list[i] / s_y

        # Bounding box of cells
        min_x = int(math.floor((x_gaussian - max_radius) / ag))
        max_x = int(math.floor((x_gaussian + max_radius) / ag))
        min_y = int(math.floor((y_gaussian - max_radius) / ag))
        max_y = int(math.floor((y_gaussian + max_radius) / ag))

        pt_covered = False

        # Check all cells in bounding box
        for ncx in range(min_x, max_x + 1):
            if pt_covered:
                break
            for ncy in range(min_y, max_y + 1):
                if pt_covered:
                    break

                # Cell corner
                cell_corner_x = ag * ncx
                cell_corner_y = ag * ncy

                # Get intensity at cell corner
                ix = max(0, min(int(math.floor(cell_corner_x)), n_in - 1))
                iy = max(0, min(int(math.floor(cell_corner_y)), m_in - 1))
                u = img_in[iy, ix]

                # Lookup lambda
                u_ind = int(math.floor(u * (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)))
                u_ind = max(0, min(u_ind, MAX_GREY_LEVEL))
                curr_lambda = lambda_lut[u_ind]

                # Seed RNG for this cell
                seed = get_cell_seed(ncx, ncy, seed_offset)
                seed = hash_seed(seed)
                rng_state = cuda.local.array(1, dtype=np.uint64)
                # Properly initialize 64-bit state from 32-bit seed
                seed64 = (seed << 32) | hash_seed(seed + 1)
                rng_state[0] = seed64

                # Draw number of grains
                n_cell = random_poisson(rng_state, curr_lambda)

                # Check each grain
                for k in range(n_cell):
                    # Draw grain center
                    x_centre_grain = cell_corner_x + ag * random_uniform(rng_state)
                    y_centre_grain = cell_corner_y + ag * random_uniform(rng_state)

                    # Draw grain radius
                    if grain_sigma > 0.0:
                        gauss_val = random_normal(rng_state)
                        curr_radius = min(math.exp(mu + sigma * gauss_val), max_radius)
                        curr_grain_radius_sq = curr_radius * curr_radius
                    else:
                        curr_grain_radius_sq = grain_radius_sq

                    # Test if point is covered
                    if sq_distance(x_centre_grain, y_centre_grain,
                                 x_gaussian, y_gaussian) < curr_grain_radius_sq:
                        pix_out += 1.0
                        pt_covered = True
                        break

    # Write output
    img_out[y_out, x_out] = pix_out / n_monte_carlo


# ============================================================================
# HOST FUNCTION
# ============================================================================

def film_grain_rendering_gpu(
    img_in, grain_radius, grain_sigma, sigma_filter,
    n_monte_carlo, seed_offset,
    x_a, y_a, x_b, y_b, m_out, n_out
):
    """
    GPU-accelerated film grain rendering.

    Uses CUDA to parallelize rendering across all output pixels.
    Expected speedup: 50-100x over CPU version.
    """
    m_in, n_in = img_in.shape

    # Precompute lambda lookup table
    MAX_GREY_LEVEL = 255
    EPSILON_GREY_LEVEL = 0.1

    lambda_lut = np.zeros(MAX_GREY_LEVEL + 1, dtype=np.float32)
    ag = 1.0 / np.ceil(1.0 / grain_radius)

    for i in range(MAX_GREY_LEVEL + 1):
        u = i / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        lam = -(ag * ag) / (np.pi * (grain_radius * grain_radius +
                                     grain_sigma * grain_sigma)) * np.log(1.0 - u)
        lambda_lut[i] = lam

    # Generate Monte Carlo translation vectors
    np.random.seed(2016)
    x_gaussian_list = np.random.normal(0, sigma_filter, n_monte_carlo).astype(np.float32)
    y_gaussian_list = np.random.normal(0, sigma_filter, n_monte_carlo).astype(np.float32)

    # Allocate output
    img_out = np.zeros((m_out, n_out), dtype=np.float32)

    # Copy data to GPU
    d_img_in = cuda.to_device(img_in)
    d_img_out = cuda.to_device(img_out)
    d_lambda_lut = cuda.to_device(lambda_lut)
    d_x_gaussian = cuda.to_device(x_gaussian_list)
    d_y_gaussian = cuda.to_device(y_gaussian_list)

    # Configure CUDA grid
    threads_per_block = (16, 16)
    blocks_x = (n_out + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (m_out + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    # Launch kernel
    render_grain_kernel[blocks_per_grid, threads_per_block](
        d_img_in, d_img_out,
        m_in, n_in, m_out, n_out,
        grain_radius, grain_sigma, sigma_filter,
        n_monte_carlo, seed_offset,
        x_a, y_a, x_b, y_b,
        d_lambda_lut,
        d_x_gaussian, d_y_gaussian
    )

    # Copy result back
    d_img_out.copy_to_host(img_out)

    return img_out


# ============================================================================
# USER-FRIENDLY API
# ============================================================================

class FilmGrainRendererGPU:
    """
    GPU-accelerated film grain renderer.

    Same API as FilmGrainRenderer but uses CUDA for 50-100x speedup.
    """

    def __init__(
        self,
        grain_radius: float = 0.1,
        grain_sigma: float = 0.0,
        sigma_filter: float = 0.8,
        n_monte_carlo: int = 800,
        seed: int = 2016
    ):
        self.grain_radius = grain_radius
        self.grain_sigma = grain_sigma
        self.sigma_filter = sigma_filter
        self.n_monte_carlo = n_monte_carlo
        self.seed = seed

        # Validate parameters
        if grain_radius <= 0:
            raise ValueError("grain_radius must be positive")
        if grain_sigma < 0:
            raise ValueError("grain_sigma must be non-negative")
        if sigma_filter <= 0:
            raise ValueError("sigma_filter must be positive")
        if n_monte_carlo < 1:
            raise ValueError("n_monte_carlo must be at least 1")

    def _render_single_channel(
        self,
        image: np.ndarray,
        zoom: float,
        output_size: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """Render a single channel using GPU"""
        m_in, n_in = image.shape

        # Determine output size
        if output_size is not None:
            n_out, m_out = output_size
        else:
            m_out = int(np.floor(zoom * m_in))
            n_out = int(np.floor(zoom * n_in))

        # Image bounds
        x_a, y_a = 0.0, 0.0
        x_b, y_b = float(n_in), float(m_in)

        # Ensure float32
        image = image.astype(np.float32)

        # Call GPU rendering
        output = film_grain_rendering_gpu(
            image,
            self.grain_radius,
            self.grain_sigma,
            self.sigma_filter,
            self.n_monte_carlo,
            self.seed,
            x_a, y_a, x_b, y_b,
            m_out, n_out
        )

        return output

    def render(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        zoom: float = 1.0,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Render film grain on an image using GPU"""
        # Load image if path provided
        if isinstance(image, (Path, str)):
            image = Image.open(image)

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            pil_image = image
            if pil_image.mode not in ['L', 'RGB']:
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image, dtype=np.float32)

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            warnings.warn("Image values > 1.0 detected, normalizing to [0, 1]")
            image = image / 255.0

        # Handle color vs grayscale
        is_color = len(image.shape) == 3 and image.shape[2] == 3

        if is_color:
            # Process each channel independently
            channels = []
            for c in range(3):
                rendered_channel = self._render_single_channel(
                    image[:, :, c], zoom, output_size
                )
                channels.append(rendered_channel)
            output = np.stack(channels, axis=2)
        else:
            # Single channel
            if len(image.shape) == 3:
                image = image[:, :, 0]
            output = self._render_single_channel(image, zoom, output_size)
            output = np.stack([output] * 3, axis=2)

        # Convert back to uint8 and PIL
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output)


if __name__ == "__main__":
    print("GPU Film Grain Renderer Test")
    print("=" * 60)

    # Check CUDA availability
    if not cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("Please ensure you have:")
        print("  - NVIDIA GPU")
        print("  - CUDA toolkit installed")
        print("  - numba with CUDA support")
        exit(1)

    print(f"✓ CUDA available")
    print(f"  GPU: {cuda.get_current_device().name.decode('utf-8')}")

    # Quick test
    print("\nRunning quick test on 64x64 image...")
    renderer = FilmGrainRendererGPU(
        grain_radius=0.1,
        n_monte_carlo=100
    )

    test_img = np.random.rand(64, 64).astype(np.float32)
    result = renderer._render_single_channel(test_img, zoom=1.0, output_size=None)

    print(f"✓ Success! Output shape: {result.shape}")
    print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
    print("\nGPU renderer is ready!")
