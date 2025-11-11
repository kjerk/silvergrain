from pathlib import Path

import numpy as np
from PIL import Image

from silvergrain.renderer import FilmGrainRenderer, render_film_grain

"""
Example usage of the Film Grain Renderer

This script demonstrates different ways to use the renderer.
"""

def example_1_basic():
	"""Basic usage with default settings"""
	print("\n" + "=" * 60)
	print("Example 1: Basic usage")
	print("=" * 60)
	
	# Create a simple test image
	test_image = Image.new('L', (256, 256), color=128)
	
	# Add some gradients for visual interest
	img_array = np.array(test_image, dtype=np.float32)
	for i in range(256):
		img_array[i, :] = 50 + i * 0.8  # Gradient
	test_image = Image.fromarray(img_array.astype(np.uint8), mode='L')
	
	# Render with default settings
	renderer = FilmGrainRenderer()
	output = renderer.render(test_image)
	
	# Save
	output_path = Path("output_basic.png")
	output.save(output_path)
	print(f"Saved to {output_path}")

def example_2_fine_grain():
	"""Fine grain with variation"""
	print("\n" + "=" * 60)
	print("Example 2: Fine grain with size variation")
	print("=" * 60)
	
	# Create test image
	test_image = Image.new('RGB', (256, 256), color=(100, 120, 140))
	
	# Fine grain with variation
	renderer = FilmGrainRenderer(
		grain_radius=0.08,  # Fine grain
		grain_sigma=0.02,  # Some size variation
		sigma_filter=0.6,  # Less smoothing
		n_monte_carlo=400  # Lower samples for speed
	)
	
	output = renderer.render(test_image)
	output_path = Path("output_fine_grain.png")
	output.save(output_path)
	print(f"Saved to {output_path}")

def example_3_heavy_grain():
	"""Heavy, coarse grain"""
	print("\n" + "=" * 60)
	print("Example 3: Heavy coarse grain")
	print("=" * 60)
	
	# Create test image
	test_image = Image.new('RGB', (256, 256), color=(150, 150, 150))
	
	# Coarse grain
	renderer = FilmGrainRenderer(
		grain_radius=0.25,  # Coarse grain
		grain_sigma=0.05,  # Moderate variation
		sigma_filter=1.0,  # Smoother
		n_monte_carlo=200  # Even lower for speed
	)
	
	output = renderer.render(test_image)
	output_path = Path("output_heavy_grain.png")
	output.save(output_path)
	print(f"Saved to {output_path}")

def example_4_file_to_file():
	"""Process from file to file"""
	print("\n" + "=" * 60)
	print("Example 4: File to file processing")
	print("=" * 60)
	
	# First create an input file
	test_image = Image.new('RGB', (256, 256))
	pixels = np.array(test_image)
	# Create a simple pattern
	for i in range(256):
		for j in range(256):
			pixels[i, j] = [i // 2, j // 2, (i + j) // 4]
	test_image = Image.fromarray(pixels, 'RGB')
	test_image.save("input_test.png")
	
	# Now process it
	renderer = FilmGrainRenderer(grain_radius=0.12, n_monte_carlo=300)
	renderer.render_from_file("input_test.png", "output_from_file.png")

def example_5_convenience_function():
	"""Using the convenience function for one-offs"""
	print("\n" + "=" * 60)
	print("Example 5: Convenience function")
	print("=" * 60)
	
	# Create test image
	test_image = Image.new('L', (256, 256), color=100)
	
	# One-liner rendering
	output = render_film_grain(
		test_image,
		grain_radius=0.15,
		n_monte_carlo=300
	)
	
	output.save("output_convenience.png")
	print("Saved to output_convenience.png")

def example_6_zoom():
	"""Upscaling with grain"""
	print("\n" + "=" * 60)
	print("Example 6: Zoom/upscaling")
	print("=" * 60)
	
	# Create small test image
	test_image = Image.new('L', (128, 128), color=120)
	
	# Render at 2x resolution
	renderer = FilmGrainRenderer(grain_radius=0.1, n_monte_carlo=200)
	output = renderer.render(test_image, zoom=2.0)
	
	print(f"Input size: {test_image.size}")
	print(f"Output size: {output.size}")
	
	output.save("output_zoom.png")
	print("Saved to output_zoom.png")

if __name__ == "__main__":
	print("Film Grain Renderer - Usage Examples")
	print("=" * 60)
	print("\nThis will create several example outputs to demonstrate the renderer.")
	print("Running on small images (256x256) for speed.")
	print("\nNote: First run will be slower due to Numba JIT compilation.\n")
	
	# Run examples
	try:
		example_1_basic()
		example_2_fine_grain()
		example_3_heavy_grain()
		example_4_file_to_file()
		example_5_convenience_function()
		example_6_zoom()
		
		print("\n" + "=" * 60)
		print("All examples completed!")
		print("=" * 60)
		print("\nOutput files created:")
		print("  - output_basic.png")
		print("  - output_fine_grain.png")
		print("  - output_heavy_grain.png")
		print("  - output_from_file.png")
		print("  - output_convenience.png")
		print("  - output_zoom.png")
	
	except Exception as e:
		print(f"\nError occurred: {e}")
		import traceback
		
		traceback.print_exc()
