from pathlib import Path

from PIL import Image
from silvergrain import FilmGrainRenderer

def apply_gpu_grain():
	"""Apply grain using GPU acceleration if available"""
	input_path = Path(__file__).parent / "example_input.png"
	output_path = Path(__file__).parent / f"{Path(__file__).stem}.png"
	
	print(f"Loading {input_path.name}...")
	image = Image.open(input_path)
	
	print("Applying grain with GPU acceleration...")
	try:
		renderer = FilmGrainRenderer(device='gpu', grain_radius=0.12, n_monte_carlo=200)
	except RuntimeError as e:
		print(f"GPU not available: {e}")
		print("Install with: pip install silvergrain[gpu]")
		return
	
	output = renderer.render(image)
	
	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_gpu_grain()
