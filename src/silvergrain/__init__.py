from .renderer import FilmGrainRenderer, render_film_grain

"""
SilverGrain - Physically-Based Film Grain Renderer

A high-performance film grain renderer using stochastic geometry (Boolean model)
to simulate realistic photographic grain. Supports both CPU and GPU acceleration.
"""

__version__ = "0.1.0"

__all__ = [
    "FilmGrainRenderer",
    "render_film_grain",
]

# Conditionally export GPU renderer if available
try:
    from .renderer_gpu import FilmGrainRendererGPU
    __all__.append("FilmGrainRendererGPU")
except ImportError:
    pass
