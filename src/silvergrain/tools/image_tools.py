import numpy

"""
Shared image processing utilities
"""

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
