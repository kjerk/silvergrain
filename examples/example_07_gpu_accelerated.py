from pathlib import Path

from PIL import Image

try:
	from numba import cuda
	
	from silvergrain.renderer_gpu import FilmGrainRendererGPU
	
	GPU_AVAILABLE = cuda.is_available()
except ImportError:
	GPU_AVAILABLE = False

def apply_gpu_grain():
	"""Apply grain using GPU acceleration if available"""
	input_path = Path(__file__).parent / "example_input.png"
	output_path = Path(__file__).parent / f"{Path(__file__).stem}.png"
	
	if not GPU_AVAILABLE:
		print("GPU not available, skipping this example")
		print("Install with: pip install silvergrain[gpu]")
		return
	
	print(f"Loading {input_path.name}...")
	image = Image.open(input_path)
	
	print("Applying grain with GPU acceleration...")
	renderer = FilmGrainRendererGPU(grain_radius=0.12, n_monte_carlo=200)
	
	output = renderer.render(image)
	
	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_gpu_grain()
