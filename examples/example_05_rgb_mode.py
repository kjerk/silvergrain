from pathlib import Path

from PIL import Image
from silvergrain import FilmGrainRenderer

def apply_rgb_grain():
	"""Apply grain independently to each RGB channel"""
	input_path = Path(__file__).parent / "example_input.png"
	output_path = Path(__file__).parent / f"{Path(__file__).stem}.png"

	print(f"Loading {input_path.name}...")
	image = Image.open(input_path)

	print("Applying grain to each RGB channel independently...")
	renderer = FilmGrainRenderer(grain_radius=0.12, n_monte_carlo=200)

	output = renderer.process_image(image, mode='rgb')

	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_rgb_grain()
