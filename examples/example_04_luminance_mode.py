from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from silvergrain import FilmGrainRenderer

def apply_luminance_grain():
	"""Apply grain only to luminance channel, preserving color"""
	input_path = Path(__file__).parent / "example_input.png"
	output_path = Path(__file__).parent / f"{Path(__file__).stem}.png"
	
	print(f"Loading {input_path.name}...")
	image = Image.open(input_path)
	if image.mode != 'RGB':
		image = image.convert('RGB')
	
	print("Applying grain to luminance only (preserves color)...")
	renderer = FilmGrainRenderer(
		grain_radius=0.12,
		n_monte_carlo=200
	)
	
	img_array = np.array(image, dtype=np.float32) / 255.0
	
	# Convert RGB to YUV
	img_uint8 = (img_array * 255).astype(np.uint8)
	yuv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
	
	# Render grain on Y (luminance) channel only
	y_rendered = renderer._render_single_channel(yuv[:, :, 0], zoom=1.0, output_size=None)
	yuv[:, :, 0] = y_rendered
	
	# Convert back to RGB
	yuv_uint8 = (np.clip(yuv * 255.0, 0, 255)).astype(np.uint8)
	output_array = cv2.cvtColor(yuv_uint8, cv2.COLOR_YUV2RGB)
	output = Image.fromarray(output_array)
	
	print(f"Saving to {output_path.name}...")
	output.save(output_path)
	print("Done!")

if __name__ == "__main__":
	apply_luminance_grain()
