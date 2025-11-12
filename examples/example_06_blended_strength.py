from pathlib import Path

from PIL import Image
from silvergrain import FilmGrainRenderer

def apply_blended_grain():
	"""Apply grain at 50% strength, blending with original"""
	input_path = Path(__file__).parent / "example_input.png"
	output_path = Path(__file__).parent / f"{Path(__file__).stem}.png"

	print(f"Loading {input_path.name}...")
	image = Image.open(input_path)

	print("Applying grain at 50% strength...")
	renderer = FilmGrainRenderer(grain_radius=0.12, n_monte_carlo=200)

	output = renderer.process_image(image, strength=0.5)

	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_blended_grain()
