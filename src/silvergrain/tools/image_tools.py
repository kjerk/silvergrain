from pathlib import Path

import numpy

"""
Shared image processing utilities
"""

DEFAULT_JPEG_QUALITY = 98
DEFAULT_PNG_COMPRESS_LEVEL = 3

def get_pil_save_kwargs(output_path: Path) -> dict:
	"""
	Get appropriate save kwargs based on output file format.

	Ensures high quality output by using proper settings for each format:
	- JPEG: quality=98 (instead of PIL's default 75)
	- PNG: compress_level=3 for balanced speed/compression

	Args:
		output_path: Path to the output file

	Returns:
		Dictionary of kwargs to pass to PIL Image.save()
	"""
	ext = output_path.suffix.lower()
	
	if ext in ['.jpg', '.jpeg']:
		return {'quality': DEFAULT_JPEG_QUALITY}
	elif ext == '.png':
		return {'compress_level': DEFAULT_PNG_COMPRESS_LEVEL}
	else:
		# For other formats, use reasonable defaults
		return {'optimize': True}

def blend_images(base_image: numpy.ndarray, target_image: numpy.ndarray, strength: float) -> numpy.ndarray:
	"""
	Blend two images with a weighted average.

	Args:
		base_image: Base image array (strength=0.0 returns 100% of this)
		target_image: Target image array (strength=1.0 returns 100% of this)
		strength: Blend strength [0.0, 1.0]

	Returns:
		(1.0 - strength) * base_image + strength * target_image
	"""
	stacked = numpy.stack([base_image, target_image])
	weights = (1.0 - strength, strength)
	return numpy.average(stacked, axis=0, weights=weights)

if __name__ == '__main__':
	print('__main__ not supported in modules.')
