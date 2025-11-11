from pathlib import Path

from PIL import Image
from silvergrain import FilmGrainRenderer

def apply_fine_grain():
	"""Apply fine, subtle grain to example image"""
	input_path = Path(__file__).parent / "example_input.png"
	output_path = Path(__file__).parent / f"{Path(__file__).stem}.png"
	
	print(f"Loading {input_path.name}...")
	image = Image.open(input_path)
	
	print("Applying fine grain...")
	renderer = FilmGrainRenderer(
		grain_radius=0.08, grain_sigma=0.02, sigma_filter=0.6, n_monte_carlo=200
	)
	
	output = renderer.render(image)
	
	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_fine_grain()
