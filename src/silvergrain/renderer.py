from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

"""
Film Grain Renderer - Public API

Facade that selects appropriate backend (CPU or GPU) based on device parameter.
"""

class FilmGrainRenderer:
	"""
	User-friendly interface for physically-based film grain rendering.

	This renderer uses a stochastic geometry approach (Boolean model) to
	simulate realistic photographic grain. It models grain as random circles
	placed according to a Poisson point process.

	Automatically selects CPU or GPU backend based on device parameter and availability.

	Parameters
	----------
	grain_radius : float, default=0.1
		Average grain radius in pixels. Smaller = finer grain.
		Typical range: 0.05 to 0.5

	grain_sigma : float, default=0.0
		Standard deviation of grain radii (log-normal distribution).
		0.0 = constant grain size, higher = more variation.
		Typical range: 0.0 to 0.3 * grain_radius

	sigma_filter : float, default=0.8
		Standard deviation of Gaussian filter for anti-aliasing.
		Higher = smoother grain appearance.
		Typical range: 0.5 to 1.5

	n_monte_carlo : int, default=800
		Number of Monte Carlo samples per pixel.
		Higher = better quality but slower.
		Typical range: 100 to 2000

	device : str, default='auto'
		Computing device to use: 'cpu', 'gpu', or 'auto'.
		'cpu' - Use CPU rendering (always available)
		'gpu' - Use GPU acceleration (requires CUDA, raises error if unavailable)
		'auto' - Use GPU if available, otherwise fall back to CPU

	algorithm : str, default='pixel_wise'
		Rendering algorithm to use.
		Options: 'pixel_wise' (currently only option implemented)

	seed : int, default=2016
		Random seed for reproducibility

	Examples
	--------
	>>> from PIL import Image
	>>> renderer = FilmGrainRenderer(grain_radius=0.1)
	>>> img = Image.open("input.png")
	>>> output = renderer.render(img)
	>>> output.save("output.png")

	>>> # Adjust grain parameters
	>>> renderer = FilmGrainRenderer(
	...     grain_radius=0.15,
	...     grain_sigma=0.03,
	...     n_monte_carlo=400
	... )

	>>> # Use GPU acceleration
	>>> renderer = FilmGrainRenderer(device='gpu', grain_radius=0.12)
	>>> output = renderer.render(img)
	"""
	
	def __init__(
		self,
		grain_radius: float = 0.1,
		grain_sigma: float = 0.0,
		sigma_filter: float = 0.8,
		n_monte_carlo: int = 800,
		device: str = 'auto',
		algorithm: str = 'pixel_wise',
		seed: int = 2016
	):
		# Validate device parameter
		if device not in ['auto', 'cpu', 'gpu']:
			raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cpu', or 'gpu'")
		
		# Select and instantiate backend
		self._impl = self._create_backend(
			device=device,
			grain_radius=grain_radius,
			grain_sigma=grain_sigma,
			sigma_filter=sigma_filter,
			n_monte_carlo=n_monte_carlo,
			seed=seed
		)
	
	def _create_backend(self, device: str, **params):
		"""Create appropriate backend based on device parameter"""
		if device == 'cpu':
			from silvergrain.renderer_cpu import FilmGrainRendererCPU
			self._device = 'cpu'
			return FilmGrainRendererCPU(**params)
		elif device in ('gpu', 'auto'):
			try:
				# Try GPU, fall back to CPU if needed
				from numba import cuda
				if not cuda.is_available():
					if device == 'gpu':
						raise RuntimeError(
							"GPU device requested but CUDA is not available. "
							"Ensure you have an NVIDIA GPU, CUDA toolkit installed, "
							"and numba with CUDA support (pip install silvergrain[gpu])"
						)
					# Fall back to CPU for 'auto' mode
					from silvergrain.renderer_cpu import FilmGrainRendererCPU
					self._device = 'cpu'
					return FilmGrainRendererCPU(**params)
				else:
					# GPU is available
					from silvergrain.renderer_gpu import FilmGrainRendererGPU
					self._device = 'gpu'
					return FilmGrainRendererGPU(**params)
			except ImportError:
				if device == 'gpu':
					raise RuntimeError(
						"GPU device requested but numba-cuda is not installed. "
						"Install with: pip install silvergrain[gpu]"
					)
				# Fall back to CPU for 'auto' mode
				from silvergrain.renderer_cpu import FilmGrainRendererCPU
				self._device = 'cpu'
				return FilmGrainRendererCPU(**params)
	
	@property
	def device(self) -> str:
		"""Return the actual device being used ('cpu' or 'gpu')"""
		return self._device

	def render(self, image: Union[Image.Image, np.ndarray, Path, str], zoom: float = 1.0, output_size: Optional[Tuple[int, int]] = None) -> Image.Image:
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
		return self._impl.render(image, zoom, output_size)
	
	def render_single_channel(self, image: np.ndarray, zoom: float, output_size: Optional[Tuple[int, int]]) -> np.ndarray:
		"""
		Render film grain on a single channel (grayscale array).

		Parameters
		----------
		image : np.ndarray
			Input grayscale image as 2D numpy array, normalized to [0, 1]

		zoom : float
			Output resolution multiplier

		output_size : tuple of (width, height), optional
			Explicit output size. If provided, overrides zoom.

		Returns
		-------
		np.ndarray
			Rendered grayscale image as 2D float32 array in [0, 1]
		"""
		return self._impl.render_single_channel(image, zoom, output_size)
	
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
		return self._impl.render_from_file(input_path, output_path, zoom, output_size)

def render_film_grain(
	image: Union[Image.Image, np.ndarray, Path, str],
	grain_radius: float = 0.1,
	grain_sigma: float = 0.0,
	sigma_filter: float = 0.8,
	n_monte_carlo: int = 800,
	zoom: float = 1.0,
	**kwargs
) -> Image.Image:
	"""
	Quick function to render film grain with default settings.

	This is a convenience wrapper around FilmGrainRenderer for one-off usage.

	Parameters
	----------
	image : PIL.Image, np.ndarray, Path, or str
		Input image
	grain_radius : float, default=0.1
		Average grain radius in pixels
	grain_sigma : float, default=0.0
		Standard deviation of grain radii
	sigma_filter : float, default=0.8
		Anti-aliasing filter strength
	n_monte_carlo : int, default=800
		Number of Monte Carlo samples
	zoom : float, default=1.0
		Output resolution multiplier
	**kwargs
		Additional arguments passed to FilmGrainRenderer

	Returns
	-------
	PIL.Image
		Rendered image with film grain

	Examples
	--------
	>>> from PIL import Image
	>>> img = Image.open("input.png")
	>>> output = render_film_grain(img, grain_radius=0.15)
	>>> output.save("output.png")
	"""
	renderer = FilmGrainRenderer(
		grain_radius=grain_radius,
		grain_sigma=grain_sigma,
		sigma_filter=sigma_filter,
		n_monte_carlo=n_monte_carlo,
		**kwargs
	)
	return renderer.render(image, zoom=zoom)

def check_grain_renderer():
	print("Film Grain Renderer - basic test")
	print("=" * 60)
	
	# Test basic functionality
	print("Creating test renderer...")
	renderer = FilmGrainRenderer(grain_radius=0.1, n_monte_carlo=100)
	
	print("Generating test image (64x64)...")
	test_img = np.random.rand(64, 64).astype(np.float32)
	
	print("Rendering (first run will trigger Numba JIT compilation)...")
	result = renderer.render_single_channel(test_img, zoom=1.0, output_size=None)
	
	print(f"Success! Output shape: {result.shape}")
	print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
	print("\nRenderer is ready to use!")

if __name__ == "__main__":
	check_grain_renderer()
