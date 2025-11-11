import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from silvergrain import FilmGrainRenderer, file_tools

"""
SilverGrain Batch CLI - Batch process directories of images
"""

def process_image(input_path: Path, output_path: Path, renderer, mode: str, strength: float, verbose: bool = False) -> bool:
	"""Process a single image, return success status"""
	try:
		# Load image
		if verbose:
			print(f"  Loading {input_path.name}...")
		image = Image.open(input_path)
		
		# Convert to RGB if needed
		if image.mode not in ['L', 'RGB']:
			image = image.convert('RGB')
		
		# Render
		img_array = np.array(image, dtype=np.float32) / 255.0
		
		if mode == 'luminance':
			# Luminance mode
			if len(img_array.shape) == 2:
				output_array = renderer.render_single_channel(img_array, zoom=1.0, output_size=None)
				output_array = np.stack([output_array] * 3, axis=2)
			else:
				img_uint8 = (img_array * 255).astype(np.uint8)
				yuv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
				y_rendered = renderer.render_single_channel(yuv[:, :, 0], zoom=1.0, output_size=None)
				yuv[:, :, 0] = y_rendered
				yuv_uint8 = (np.clip(yuv * 255.0, 0, 255)).astype(np.uint8)
				output_array = cv2.cvtColor(yuv_uint8, cv2.COLOR_YUV2RGB).astype(np.float32)
		else:
			# RGB mode
			if len(img_array.shape) == 2:
				output_array = renderer.render_single_channel(img_array, zoom=1.0, output_size=None)
				output_array = np.stack([output_array] * 3, axis=2)
			else:
				channels = []
				for c in range(3):
					rendered = renderer.render_single_channel(img_array[:, :, c], zoom=1.0, output_size=None)
					channels.append(rendered)
				output_array = np.stack(channels, axis=2)
			output_array = output_array * 255.0
		
		# Blend if needed
		if strength < 1.0:
			original_array = np.array(image, dtype=np.float32)
			if output_array.max() <= 1.0:
				output_array = output_array * 255.0
			
			stacked = np.stack([original_array, output_array])
			weights = (1.0 - strength, strength)
			output_array = np.average(stacked, axis=0, weights=weights)
		
		# Convert to image
		output_array = np.clip(output_array, 0, 255).astype(np.uint8)
		output = Image.fromarray(output_array)
		
		# Save
		if verbose:
			print(f"  Saving {output_path.name}...")
		output.save(output_path)
		
		return True
	
	except Exception as e:
		print(f"  ERROR processing {input_path.name}: {e}", file=sys.stderr)
		return False

def main() -> int:
	"""Main CLI entry point for batch processing"""
	parser = argparse.ArgumentParser(
		description='Batch apply film grain to directory of images',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Process all images in directory
  silvergrain-batch input_dir/ output_dir/

  # Heavy grain with fast quality
  silvergrain-batch input_dir/ output_dir/ --intensity heavy --quality fast

  # Subtle grain effect
  silvergrain-batch input_dir/ output_dir/ --strength 0.3

  # Randomize grain per image (different seed for each)
  silvergrain-batch input_dir/ output_dir/ --random-seed

Presets:
  Intensity: fine (subtle) | medium (default) | heavy (strong)
  Quality:   fast | balanced (default) | high
  Mode:      luminance (default) | rgb
  Device:    auto (default, uses GPU if available) | cpu | gpu
        """
	)
	
	parser.add_argument('input_dir', type=str, help='Input directory containing images')
	parser.add_argument('output_dir', type=str, help='Output directory for processed images')
	
	# User-facing options
	parser.add_argument('--intensity', type=str, choices=['fine', 'medium', 'heavy'], default='medium', help='Grain intensity (default: medium)')
	parser.add_argument('--quality', type=str, choices=['fast', 'balanced', 'high'], default='balanced', help='Quality/speed tradeoff (default: balanced)')
	parser.add_argument('--mode', type=str, choices=['rgb', 'luminance'], default='luminance', help='Grain mode (default: luminance)')
	parser.add_argument('--strength', type=float, default=1.0, help='Grain strength 0.0-1.0 (default: 1.0)')
	parser.add_argument('--random-seed', action='store_true', help='Use different random seed for each image (for data augmentation)')
	
	# Advanced options
	parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'gpu'], default='auto', help='Device to use (default: auto)')
	parser.add_argument('--grain-radius', type=float, help='Override grain radius (0.05-0.25)')
	parser.add_argument('--samples', type=int, help='Override Monte Carlo samples (100-800)')
	parser.add_argument('--grain-sigma', type=float, default=0.0, help='Grain size variation')
	parser.add_argument('--sigma-filter', type=float, default=0.8, help='Anti-aliasing filter')
	parser.add_argument('--seed', type=int, default=2016, help='Base random seed (default: 2016)')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	
	args = parser.parse_args()
	
	# Validate directories
	input_dir = Path(args.input_dir)
	if not input_dir.exists():
		print(f"Error: Input directory '{args.input_dir}' not found", file=sys.stderr)
		return 1
	if not input_dir.is_dir():
		print(f"Error: '{args.input_dir}' is not a directory", file=sys.stderr)
		return 1
	
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# Validate strength
	if args.strength < 0.0 or args.strength > 1.0:
		print(f"Error: --strength must be between 0.0 and 1.0, got {args.strength}", file=sys.stderr)
		return 1
	
	# Find images
	print(f"Scanning {input_dir}...")
	images = file_tools.list_images(input_dir)
	if not images:
		print(f"Error: No images found in {input_dir}", file=sys.stderr)
		print(f"Supported formats: {', '.join(file_tools.IMAGE_EXTENSIONS)}", file=sys.stderr)
		return 1
	
	print(f"Found {len(images)} images")
	
	# Presets for easier use
	intensity_map = {'fine': 0.08, 'medium': 0.12, 'heavy': 0.20}
	quality_map = {'fast': 100, 'balanced': 200, 'high': 400}
	
	grain_radius = args.grain_radius if args.grain_radius else intensity_map[args.intensity]
	n_samples = args.samples if args.samples else quality_map[args.quality]
	
	# Create a test renderer to determine actual device (for display)
	try:
		test_renderer = FilmGrainRenderer(
			grain_radius=grain_radius,
			grain_sigma=args.grain_sigma,
			sigma_filter=args.sigma_filter,
			n_monte_carlo=n_samples,
			device=args.device,
			seed=args.seed
		)
	except RuntimeError as e:
		print(f"Error: {e}", file=sys.stderr)
		return 1
	
	device_str = args.device
	if args.device == 'auto':
		device_str = 'GPU' if test_renderer.device == 'gpu' else 'CPU'
	elif args.device == 'gpu':
		device_str = 'GPU'
	else:
		device_str = 'CPU'
	
	# Print configuration
	print("\nProcessing with:")
	print(f"  Device: {device_str}")
	print(f"  Intensity: {args.intensity}")
	print(f"  Quality: {args.quality}")
	print(f"  Mode: {args.mode}")
	
	if args.strength < 1.0:
		print(f"  Strength: {args.strength:.2f}")
	if args.random_seed:
		print("  Random seed per image: enabled")
	print()
	
	# Process images
	start_time = time.time()
	success_count = 0
	fail_count = 0
	
	for idx, input_path in enumerate(images, 1):
		print(f"[{idx}/{len(images)}] {input_path.name}")
		
		# Create renderer (new seed if random_seed enabled)
		seed = args.seed
		if args.random_seed:
			seed = args.seed + idx
		
		renderer = FilmGrainRenderer(
			grain_radius=grain_radius,
			grain_sigma=args.grain_sigma,
			sigma_filter=args.sigma_filter,
			n_monte_carlo=n_samples,
			device=args.device,
			seed=seed
		)
		
		# Process
		output_path = output_dir / input_path.name
		if process_image(input_path, output_path, renderer, args.mode, args.strength, args.verbose):
			success_count += 1
		else:
			fail_count += 1
	
	# Summary
	elapsed = time.time() - start_time
	print(f"\n{'=' * 60}")
	print(f"Completed in {elapsed:.1f}s ({elapsed / len(images):.2f}s per image)")
	print(f"Success: {success_count}/{len(images)}")
	if fail_count > 0:
		print(f"Failed: {fail_count}")
	print('=' * 60)
	
	return 0 if fail_count == 0 else 1

if __name__ == '__main__':
	sys.exit(main())
