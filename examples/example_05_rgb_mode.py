from pathlib import Path

import numpy as np
from PIL import Image
from silvergrain import FilmGrainRenderer

def apply_rgb_grain():
	"""Apply grain independently to each RGB channel"""
	input_path = Path(__file__).parent / "example_input.png"
	output_path = Path(__file__).parent / f"{Path(__file__).stem}.png"
	
	print(f"Loading {input_path.name}...")
	image = Image.open(input_path)
	if image.mode != "RGB":
		image = image.convert("RGB")
	
	print("Applying grain to each RGB channel independently...")
	renderer = FilmGrainRenderer(grain_radius=0.12, n_monte_carlo=200)
	
	img_array = np.array(image, dtype=np.float32) / 255.0
	
	# Process each channel independently
	channels = []
	for c in range(3):
		print(f"  Processing channel {c + 1}/3...")
		rendered = renderer._render_single_channel(img_array[:, :, c], zoom=1.0, output_size=None)
		channels.append(rendered)
	
	output_array = np.stack(channels, axis=2)
	output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
	output = Image.fromarray(output_array)
	
	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_rgb_grain()
