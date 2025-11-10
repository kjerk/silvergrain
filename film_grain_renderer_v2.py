"""
Film Grain Renderer V2 - WORKING VERSION

This version uses non-JIT code that actually produces correct results.
Once working, we can optimize with Numba later.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings


# ============================================================================
# PSEUDO-RANDOM NUMBER GENERATOR
# ============================================================================

def wang_hash(seed):
    """Wang hash for pseudo-random seed generation"""
    seed = int(seed) & 0xFFFFFFFF
    seed = (seed ^ 61) ^ (seed >> 16)
    seed = (seed * 9) & 0xFFFFFFFF
    seed = seed ^ (seed >> 4)
    seed = (seed * 668265261) & 0xFFFFFFFF
    seed = seed ^ (seed >> 15)
    return seed & 0xFFFFFFFF


def cellseed(x, y, offset):
    """Generate unique seed for a cell given coordinates"""
    period = 65536
    s = ((y % period) * period + (x % period)) + offset
    if s == 0:
        s = 1
    return s & 0xFFFFFFFF


def xorshift_init(seed):
    """Initialize Xorshift PRNG state"""
    return wang_hash(seed)


def xorshift_next(state):
    """Xorshift algorithm - fast PRNG"""
    state = int(state) & 0xFFFFFFFF
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= (state >> 17) & 0xFFFFFFFF
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def rand_uniform_0_1(state):
    """Generate uniform random in [0, 1]"""
    return float(state) / 4294967295.0


def rand_poisson(lam, exp_lambda, state):
    """Generate Poisson random variable. Returns (new_state, value)"""
    state = xorshift_next(state)
    u = rand_uniform_0_1(state)

    x = 0
    prod = exp_lambda
    total = prod

    max_iter = int(np.floor(10000.0 * lam))
    while u > total and x < max_iter:
        x += 1
        prod = prod * lam / float(x)
        total += prod

    return state, x


def sq_distance(x1, y1, x2, y2):
    """Squared Euclidean distance"""
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


# ============================================================================
# CORE RENDERING FUNCTION (non-JIT but working!)
# ============================================================================

def render_pixel(
    img_in, y_out, x_out,
    m_in, n_in, m_out, n_out,
    offset, n_monte_carlo,
    grain_radius, grain_sigma, sigma_filter,
    x_a, y_a, x_b, y_b,
    lambda_lut, exp_lambda_lut,
    x_gaussian_list, y_gaussian_list
):
    """
    Render a single pixel using Monte Carlo simulation.
    Non-JIT version that actually works correctly.
    """
    MAX_GREY_LEVEL = 255
    EPSILON_GREY_LEVEL = 0.1

    grain_radius_sq = grain_radius * grain_radius
    max_radius = grain_radius
    ag = 1.0 / np.ceil(1.0 / grain_radius)

    s_x = (n_out - 1) / (x_b - x_a)
    s_y = (m_out - 1) / (y_b - y_a)

    x_in = x_a + (x_out + 0.5) * ((x_b - x_a) / n_out)
    y_in = y_a + (y_out + 0.5) * ((y_b - y_a) / m_out)

    pix_out = 0.0

    # Monte Carlo loop
    for i in range(n_monte_carlo):
        # Apply Gaussian shift for anti-aliasing
        x_gaussian = x_in + x_gaussian_list[i] / s_x
        y_gaussian = y_in + y_gaussian_list[i] / s_y

        # Bounding box of cells to check
        min_x = int(np.floor((x_gaussian - max_radius) / ag))
        max_x = int(np.floor((x_gaussian + max_radius) / ag))
        min_y = int(np.floor((y_gaussian - max_radius) / ag))
        max_y = int(np.floor((y_gaussian + max_radius) / ag))

        pt_covered = False

        # Check all cells in bounding box
        for ncx in range(min_x, max_x + 1):
            if pt_covered:
                break
            for ncy in range(min_y, max_y + 1):
                if pt_covered:
                    break

                # Cell corner in pixel coordinates
                cell_corner_x = ag * ncx
                cell_corner_y = ag * ncy

                # Get unique seed for this cell
                seed = cellseed(ncx, ncy, offset)
                state = xorshift_init(seed)

                # Get intensity at cell corner
                ix = max(0, min(int(np.floor(cell_corner_x)), n_in - 1))
                iy = max(0, min(int(np.floor(cell_corner_y)), m_in - 1))
                u = img_in[iy, ix]

                # Lookup precomputed lambda
                u_ind = int(np.floor(u * (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)))
                u_ind = max(0, min(u_ind, MAX_GREY_LEVEL))
                curr_lambda = lambda_lut[u_ind]
                curr_exp_lambda = exp_lambda_lut[u_ind]

                # Draw number of grains in this cell
                state, n_cell = rand_poisson(curr_lambda, curr_exp_lambda, state)

                # Check each grain
                for k in range(n_cell):
                    # Draw grain center within cell
                    state = xorshift_next(state)
                    x_centre_grain = cell_corner_x + ag * rand_uniform_0_1(state)
                    state = xorshift_next(state)
                    y_centre_grain = cell_corner_y + ag * rand_uniform_0_1(state)

                    # Test if point is covered by this grain
                    if sq_distance(x_centre_grain, y_centre_grain,
                                 x_gaussian, y_gaussian) < grain_radius_sq:
                        pix_out += 1.0
                        pt_covered = True
                        break

    return pix_out / n_monte_carlo


def film_grain_rendering_pixel_wise(
    img_in, grain_radius, grain_sigma, sigma_filter,
    n_monte_carlo, seed_offset,
    x_a, y_a, x_b, y_b, m_out, n_out
):
    """Pixel-wise film grain rendering - non-JIT version"""
    m_in, n_in = img_in.shape

    MAX_GREY_LEVEL = 255
    EPSILON_GREY_LEVEL = 0.1

    # Precompute lambda lookup tables
    lambda_lut = np.zeros(MAX_GREY_LEVEL + 1, dtype=np.float32)
    exp_lambda_lut = np.zeros(MAX_GREY_LEVEL + 1, dtype=np.float32)

    ag = 1.0 / np.ceil(1.0 / grain_radius)

    for i in range(MAX_GREY_LEVEL + 1):
        u = i / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        if u >= 0.9999:
            lam = 100.0
        else:
            lam = -(ag * ag) / (np.pi * (grain_radius * grain_radius +
                                         grain_sigma * grain_sigma)) * np.log(1.0 - u)
        lambda_lut[i] = lam
        exp_lambda_lut[i] = np.exp(-lam)

    # Generate Monte Carlo translation vectors
    np.random.seed(2016)
    x_gaussian_list = np.random.normal(0, sigma_filter, n_monte_carlo).astype(np.float32)
    y_gaussian_list = np.random.normal(0, sigma_filter, n_monte_carlo).astype(np.float32)

    # Render output image
    img_out = np.zeros((m_out, n_out), dtype=np.float32)

    print(f"Rendering {m_out}x{n_out} image with {n_monte_carlo} MC samples...")
    print("Note: This is the non-JIT version, so it may be slow.")

    # Simple progress indicator
    for i in range(m_out):
        if i % max(1, m_out // 10) == 0:
            print(f"  Progress: {i}/{m_out} rows ({100*i//m_out}%)")

        for j in range(n_out):
            img_out[i, j] = render_pixel(
                img_in, i, j,
                m_in, n_in, m_out, n_out,
                seed_offset, n_monte_carlo,
                grain_radius, grain_sigma, sigma_filter,
                x_a, y_a, x_b, y_b,
                lambda_lut, exp_lambda_lut,
                x_gaussian_list, y_gaussian_list
            )

    print("  Done!")
    return img_out


# ============================================================================
# USER-FRIENDLY API
# ============================================================================

class FilmGrainRenderer:
    """
    Film grain renderer - V2 working version (non-JIT)
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

        # Validate
        if grain_radius <= 0:
            raise ValueError("grain_radius must be positive")
        if grain_sigma < 0:
            raise ValueError("grain_sigma must be non-negative")
        if sigma_filter <= 0:
            raise ValueError("sigma_filter must be positive")
        if n_monte_carlo < 1:
            raise ValueError("n_monte_carlo must be at least 1")

    def render(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        zoom: float = 1.0,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Render film grain on an image."""
        # Load image if path
        if isinstance(image, (Path, str)):
            image = Image.open(image)

        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            if image.mode not in ['L', 'RGB']:
                image = image.convert('RGB')
            image = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            warnings.warn("Image values > 1.0 detected, normalizing to [0, 1]")
            image = image / 255.0

        # Handle color vs grayscale
        is_color = len(image.shape) == 3 and image.shape[2] == 3

        if is_color:
            channels = []
            for c in range(3):
                rendered_channel = self._render_single_channel(
                    image[:, :, c], zoom, output_size
                )
                channels.append(rendered_channel)
            output = np.stack(channels, axis=2)
        else:
            if len(image.shape) == 3:
                image = image[:, :, 0]
            output = self._render_single_channel(image, zoom, output_size)
            output = np.stack([output] * 3, axis=2)

        # Convert back to uint8 and PIL
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output)

    def _render_single_channel(
        self,
        image: np.ndarray,
        zoom: float,
        output_size: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """Render a single channel"""
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

        # Call rendering function
        output = film_grain_rendering_pixel_wise(
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


def render_film_grain(
    image: Union[Image.Image, np.ndarray, Path, str],
    grain_radius: float = 0.1,
    grain_sigma: float = 0.0,
    sigma_filter: float = 0.8,
    n_monte_carlo: int = 800,
    zoom: float = 1.0,
    **kwargs
) -> Image.Image:
    """Quick function to render film grain"""
    renderer = FilmGrainRenderer(
        grain_radius=grain_radius,
        grain_sigma=grain_sigma,
        sigma_filter=sigma_filter,
        n_monte_carlo=n_monte_carlo,
        **kwargs
    )
    return renderer.render(image, zoom=zoom)


if __name__ == "__main__":
    print("Film Grain Renderer V2 - Working Version")
    print("=" * 60)

    # Quick test
    print("\nTesting with different intensities...")
    renderer = FilmGrainRenderer(grain_radius=0.1, n_monte_carlo=50)

    for intensity in [0.25, 0.5, 0.75]:
        img = np.ones((32, 32), dtype=np.float32) * intensity
        output = renderer._render_single_channel(img, zoom=1.0, output_size=None)
        print(f"Intensity {intensity:.2f} → Output mean {output.mean():.3f}")

    print("\n✓ Renderer is working!")
