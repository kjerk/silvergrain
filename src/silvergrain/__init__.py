"""
SilverGrain - Physically-Based Film Grain Renderer

A high-performance film grain renderer using stochastic geometry (Boolean model)
to simulate realistic photographic grain. Supports both CPU and GPU acceleration.
"""

__version__ = "0.2"


from .renderer import FilmGrainRenderer, render_film_grain


__all__ = [
	"FilmGrainRenderer",
	"render_film_grain",
]
