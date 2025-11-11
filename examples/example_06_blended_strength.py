from pathlib import Path

import numpy as np
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
	
	grained_output = renderer.render(image)
	
	# Blend 50% with original
	strength = 0.5
	original_array = np.array(image, dtype=np.float32)
	grained_array = np.array(grained_output, dtype=np.float32)
	
	stacked = np.stack([original_array, grained_array])
	weights = (1.0 - strength, strength)
	blended = np.average(stacked, axis=0, weights=weights)
	blended = np.clip(blended, 0, 255).astype(np.uint8)
	
	output = Image.fromarray(blended)
	
	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_blended_grain()
