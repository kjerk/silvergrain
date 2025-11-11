import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from numba import njit, prange

@njit
def get_cell_seed(x, y, offset):
	"""Generate unique seed for a cell"""
	# Simple deterministic hash from coordinates
	return (y * 65536 + x + offset) & 0xFFFFFFFF

@njit
def sq_distance(x1, y1, x2, y2):
	"""Squared Euclidean distance"""
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

@njit
def render_pixel(
	img_in, y_out, x_out,
	m_in, n_in, m_out, n_out,
	offset, n_monte_carlo,
	grain_radius, grain_sigma, x_a, y_a, x_b, y_b,
	lambda_lut,
	x_gaussian_list, y_gaussian_list
):
	"""
	Render a single pixel using Monte Carlo simulation.

	This is the core of the pixel-wise algorithm - translated directly
	from the C++ reference implementation.
	"""
	# Constants
	NORMAL_QUANTILE = 3.0902  # For alpha=0.999
	MAX_GREY_LEVEL = 255
	EPSILON_GREY_LEVEL = 0.1
	
	grain_radius_sq = grain_radius * grain_radius
	max_radius = grain_radius
	
	# Log-normal parameters if we have grain size variation
	if grain_sigma > 0.0:
		sigma = np.sqrt(np.log((grain_sigma / grain_radius) ** 2 + 1.0))
		sigma_sq = sigma * sigma
		mu = np.log(grain_radius) - sigma_sq / 2.0
		log_normal_quantile = np.exp(mu + sigma * NORMAL_QUANTILE)
		max_radius = log_normal_quantile
	else:
		mu = 0.0
		sigma = 0.0
	
	# Cell size for spatial hashing
	ag = 1.0 / np.ceil(1.0 / grain_radius)
	
	# Scale factors for coordinate conversion
	s_x = (n_out - 1) / (x_b - x_a)
	s_y = (m_out - 1) / (y_b - y_a)
	
	# Convert output pixel to input coordinates (sample pixel center)
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
				
				# Get intensity at cell corner to determine Poisson lambda
				ix = max(0, min(int(np.floor(cell_corner_x)), n_in - 1))
				iy = max(0, min(int(np.floor(cell_corner_y)), m_in - 1))
				u = img_in[iy, ix]
				
				# Lookup precomputed lambda
				u_ind = int(np.floor(u * (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)))
				u_ind = max(0, min(u_ind, MAX_GREY_LEVEL))
				curr_lambda = lambda_lut[u_ind]
				
				# Seed RNG for this cell
				seed = get_cell_seed(ncx, ncy, offset)
				np.random.seed(seed)
				
				# Draw number of grains in this cell
				n_cell = np.random.poisson(curr_lambda)
				
				# Check each grain
				for k in range(n_cell):
					# Draw grain center within cell
					x_centre_grain = cell_corner_x + ag * np.random.random()
					y_centre_grain = cell_corner_y + ag * np.random.random()
					
					# Draw grain radius
					if grain_sigma > 0.0:
						# Log-normal distribution
						gauss_val = np.random.randn()
						curr_radius = min(np.exp(mu + sigma * gauss_val), max_radius)
						curr_grain_radius_sq = curr_radius * curr_radius
					else:
						curr_grain_radius_sq = grain_radius_sq
					
					# Test if point is covered by this grain
					if sq_distance(x_centre_grain, y_centre_grain,
								   x_gaussian, y_gaussian) < curr_grain_radius_sq:
						pix_out += 1.0
						pt_covered = True
						break
	
	return pix_out / n_monte_carlo

@njit(parallel=True)
def film_grain_rendering_pixel_wise(
	img_in, grain_radius, grain_sigma, sigma_filter,
	n_monte_carlo, seed_offset,
	x_a, y_a, x_b, y_b, m_out, n_out
):
	"""
	Pixel-wise film grain rendering algorithm with parallel execution.

	Uses Monte Carlo simulation to render each pixel independently,
	enabling efficient parallelization.
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
	
	# Generate Monte Carlo translation vectors (done once, shared across pixels)
	# Note: Using deterministic seed for reproducibility
	np.random.seed(2016)
	x_gaussian_list = np.random.normal(0, sigma_filter, n_monte_carlo).astype(np.float32)
	y_gaussian_list = np.random.normal(0, sigma_filter, n_monte_carlo).astype(np.float32)
	
	# Render output image
	img_out = np.zeros((m_out, n_out), dtype=np.float32)
	
	# Parallel loop over output pixels (like OpenMP in C++)
	for i in prange(m_out):
		for j in range(n_out):
			img_out[i, j] = render_pixel(
				img_in, i, j,
				m_in, n_in, m_out, n_out,
				seed_offset, n_monte_carlo,
				grain_radius, grain_sigma,
				x_a, y_a, x_b, y_b,
				lambda_lut,
				x_gaussian_list, y_gaussian_list
			)
	
	return img_out

class FilmGrainRendererCPU:
	"""
	CPU implementation of film grain renderer using pixel-wise algorithm.

	Uses Numba JIT compilation for performance with parallel execution.
	"""
	
	def __init__(
		self,
		grain_radius: float = 0.1,
		grain_sigma: float = 0.0,
		sigma_filter: float = 0.8,
		n_monte_carlo: int = 800,
		algorithm: str = 'pixel_wise',
		seed: int = 2016
	):
		self.grain_radius = grain_radius
		self.grain_sigma = grain_sigma
		self.sigma_filter = sigma_filter
		self.n_monte_carlo = n_monte_carlo
		self.algorithm = algorithm
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
		if algorithm not in ['pixel_wise']:
			raise ValueError(f"Unknown algorithm: {algorithm}")
	
	def render(
		self,
		image: Union[Image.Image, np.ndarray, Path, str],
		zoom: float = 1.0,
		output_size: Optional[Tuple[int, int]] = None
	) -> Image.Image:
		"""
		Render film grain on an image.

		Parameters
		----------
		image : PIL.Image, np.ndarray, Path, or str
			Input image. Can be grayscale or RGB.
			If path, will load automatically.

		zoom : float, default=1.0
			Output resolution multiplier.
			2.0 = double resolution, 0.5 = half resolution

		output_size : tuple of (width, height), optional
			Explicit output size. If provided, overrides zoom.

		Returns
		-------
		PIL.Image
			Rendered image with film grain applied
		"""
		# Load image if path provided
		if isinstance(image, (Path, str)):
			image = Image.open(image)
		
		# Convert PIL to numpy if needed
		if isinstance(image, Image.Image):
			pil_image = image
			# Convert to RGB if needed
			if pil_image.mode not in ['L', 'RGB']:
				pil_image = pil_image.convert('RGB')
			image = np.array(pil_image, dtype=np.float32)
		else:
			pil_image = None
		
		# Ensure float32 and normalized to [0, 1]
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
				rendered_channel = self.render_single_channel(
					image[:, :, c], zoom, output_size
				)
				channels.append(rendered_channel)
			
			# Stack channels
			output = np.stack(channels, axis=2)
		else:
			# Single channel
			if len(image.shape) == 3:
				image = image[:, :, 0]  # Extract first channel
			output = self.render_single_channel(image, zoom, output_size)
			output = np.stack([output] * 3, axis=2)  # Convert to RGB
		
		# Convert back to uint8 and PIL
		output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
		return Image.fromarray(output)
	
	def render_single_channel(
		self,
		image: np.ndarray,
		zoom: float,
		output_size: Optional[Tuple[int, int]]
	) -> np.ndarray:
		"""Render a single channel (grayscale image)"""
		m_in, n_in = image.shape
		
		# Determine output size
		if output_size is not None:
			n_out, m_out = output_size
		else:
			m_out = int(np.floor(zoom * m_in))
			n_out = int(np.floor(zoom * n_in))
		
		# Image bounds (full image by default)
		x_a, y_a = 0.0, 0.0
		x_b, y_b = float(n_in), float(m_in)
		
		# Ensure float32
		image = image.astype(np.float32)
		
		# Call core rendering function
		if self.algorithm == 'pixel_wise':
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
		else:
			raise NotImplementedError(f"Algorithm {self.algorithm} not yet implemented")
		
		return output
	
	def render_from_file(
		self,
		input_path: Union[Path, str],
		output_path: Union[Path, str],
		zoom: float = 1.0,
		output_size: Optional[Tuple[int, int]] = None
	):
		"""
		Convenience method to render from file to file.

		Parameters
		----------
		input_path : Path or str
			Input image file path
		output_path : Path or str
			Output image file path
		zoom : float, default=1.0
			Output resolution multiplier
		output_size : tuple of (width, height), optional
			Explicit output size
		"""
		input_path = Path(input_path)
		output_path = Path(output_path)
		
		print(f"Loading image from {input_path}")
		image = Image.open(input_path)
		
		print(f"Rendering with grain_radius={self.grain_radius}, "
			  f"n_monte_carlo={self.n_monte_carlo}")
		output = self.render(image, zoom=zoom, output_size=output_size)
		
		print(f"Saving to {output_path}")
		output.save(output_path)
		print("Done!")

def check_grain_renderer():
	print("Film Grain Renderer CPU - basic test")
	print("=" * 60)
	
	# Test basic functionality
	print("Creating test renderer...")
	renderer = FilmGrainRendererCPU(grain_radius=0.1, n_monte_carlo=100)
	
	print("Generating test image (64x64)...")
	test_img = np.random.rand(64, 64).astype(np.float32)
	
	print("Rendering (first run will trigger Numba JIT compilation)...")
	result = renderer.render_single_channel(test_img, zoom=1.0, output_size=None)
	
	print(f"Success! Output shape: {result.shape}")
	print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
	print("\nRenderer is ready to use!")

if __name__ == "__main__":
	check_grain_renderer()
