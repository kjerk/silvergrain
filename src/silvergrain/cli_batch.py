import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from silvergrain import FilmGrainRenderer, file_tools

"""
SilverGrain Batch CLI - Batch process directories of images
"""

console = Console()

def process_image(input_path: Path, output_path: Path, renderer, mode: str, strength: float) -> Tuple[bool, str]:
	"""Process a single image, return (success, error_message)"""
	try:
		# Load image
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
		output.save(output_path)
		
		return True, ""
	
	except Exception as e:
		return False, str(e)

def main() -> int:
	"""Main CLI entry point for batch processing"""
	parser = argparse.ArgumentParser(
		description='Batch apply film grain to directory of images',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Process all images in directory (in-place with -grainy suffix)
  silvergrain-batch input_dir/

  # Process to separate output directory
  silvergrain-batch input_dir/ output_dir/

  # Heavy grain with fast quality
  silvergrain-batch input_dir/ --intensity heavy --quality fast

  # Search subdirectories recursively
  silvergrain-batch input_dir/ --recursive

  # Randomize grain per image (different seed for each)
  silvergrain-batch input_dir/ --random-seed

Presets:
  Intensity: fine (subtle) | medium (default) | heavy (strong)
  Quality:   fast | balanced (default) | high
  Mode:      luminance (default) | rgb
  Device:    auto (default, uses GPU if available) | cpu | gpu
        """
	)
	
	parser.add_argument('input_dir', type=str, help='Input directory containing images')
	parser.add_argument('output_dir', type=str, nargs='?', default=None, help='Output directory for processed images (optional, defaults to in-place with -grainy suffix)')
	
	# User-facing options
	parser.add_argument('--intensity', type=str, choices=['fine', 'medium', 'heavy'], default='medium', help='Grain intensity (default: medium)')
	parser.add_argument('--quality', type=str, choices=['fast', 'balanced', 'high'], default='balanced', help='Quality/speed tradeoff (default: balanced)')
	parser.add_argument('--mode', type=str, choices=['rgb', 'luminance'], default='luminance', help='Grain mode (default: luminance)')
	parser.add_argument('--strength', type=float, default=1.0, help='Grain strength 0.0-1.0 (default: 1.0)')
	parser.add_argument('--random-seed', action='store_true', help='Use different random seed for each image (for data augmentation)')
	
	# File handling options
	parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
	
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
		console.print(f"[red]Error:[/red] Input directory [bright_yellow]{escape(str(input_dir))}[/bright_yellow] not found", file=sys.stderr)
		return 1
	if not input_dir.is_dir():
		console.print(f"[red]Error:[/red] [bright_yellow]{escape(str(input_dir))}[/bright_yellow] is not a directory", file=sys.stderr)
		return 1
	
	# Handle output directory
	inplace_mode = args.output_dir is None
	if inplace_mode:
		output_dir = None
	else:
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
	
	# Validate strength
	if args.strength < 0.0 or args.strength > 1.0:
		console.print(f"[red]Error:[/red] --strength must be between 0.0 and 1.0, got {args.strength}", file=sys.stderr)
		return 1
	
	# Find images
	recursive = args.recursive
	search_mode = "recursively" if recursive else "in current directory only"
	console.print(f"[cyan]Scanning[/cyan] [bright_yellow]{escape(str(input_dir))}[/bright_yellow] {search_mode}...")
	images = file_tools.list_images(input_dir, recursive=recursive)
	if not images:
		console.print(f"[red]Error:[/red] No images found in [bright_yellow]{escape(str(input_dir))}[/bright_yellow]", file=sys.stderr)
		console.print(f"Supported formats: {', '.join(file_tools.IMAGE_EXTENSIONS)}", file=sys.stderr)
		return 1
	
	mode_str = "in-place (adding -grainy suffix)" if inplace_mode else f"to [bright_yellow]{escape(str(output_dir))}[/bright_yellow]"
	console.print(f"[green]Found {len(images)} images[/green] - processing {mode_str}")
	
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
		console.print(f"[red]Error:[/red] {e}", file=sys.stderr)
		return 1
	
	device_str = 'GPU' if test_renderer.device == 'gpu' else 'CPU'
	
	# Display configuration panel
	config_table = Table.grid(padding=(0, 2))
	config_table.add_column(style="cyan", justify="right")
	config_table.add_column(style="white")
	
	config_table.add_row("Images:", f"{len(images)}")
	if inplace_mode:
		config_table.add_row("Output:", "[yellow]In-place with -grainy suffix[/yellow]")
	else:
		config_table.add_row("Output:", f"[bright_yellow]{escape(str(output_dir))}[/bright_yellow]")
	if recursive:
		config_table.add_row("Recursive:", "[green]Yes[/green]")
	config_table.add_row("Device:", f"[bold]{device_str}[/bold]")
	config_table.add_row("Intensity:", args.intensity)
	config_table.add_row("Quality:", args.quality)
	config_table.add_row("Mode:", args.mode)
	
	if args.strength < 1.0:
		config_table.add_row("Strength:", f"{args.strength:.2f}")
	if args.random_seed:
		config_table.add_row("Random seeds:", "[yellow]enabled[/yellow]")
	
	console.print()
	console.print(Panel(config_table, title="[bold]Film Grain Batch Processing[/bold]", border_style="blue"))
	console.print()
	
	# Process images with progress bar
	start_time = time.time()
	success_count = 0
	fail_count = 0
	failed_images = []
	
	progress_columns = [
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		BarColumn(),
		TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
		TextColumn("|"),
		TextColumn("[cyan]{task.completed}/{task.total}"),
		TextColumn("|"),
		TimeElapsedColumn(),
		TextColumn("|"),
		TimeRemainingColumn()
	]
	
	with Progress(*progress_columns, console=console) as progress:
		task = progress.add_task("[green]Processing images...", total=len(images))
		
		for idx, input_path in enumerate(images, 1):
			# Update progress with current file
			progress.update(task, description=f"[green]Processing[/green] [bright_yellow]{escape(input_path.name)}[/bright_yellow]")
			
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
			
			# Determine output path
			if inplace_mode:
				# In-place mode: save with -grainy suffix in same directory
				stem = input_path.stem
				output_path = input_path.parent / f"{stem}-grainy.png"
			else:
				# Output directory mode: preserve filename
				output_path = output_dir / input_path.name
			
			# Process
			success, error = process_image(input_path, output_path, renderer, args.mode, args.strength)
			
			if success:
				success_count += 1
			else:
				fail_count += 1
				failed_images.append((input_path.name, error))
			
			progress.advance(task)
	
	# Display summary
	elapsed = time.time() - start_time
	avg_time = elapsed / len(images)
	
	console.print()
	
	# Create summary table
	summary = Table(show_header=True, header_style="bold cyan", border_style="blue")
	summary.add_column("Metric", style="cyan", justify="right")
	summary.add_column("Value", style="white")
	
	summary.add_row("Total images", str(len(images)))
	summary.add_row("Successful", f"[green]{success_count}[/green]")
	
	if fail_count > 0:
		summary.add_row("Failed", f"[red]{fail_count}[/red]")
	
	summary.add_row("Total time", f"{elapsed:.1f}s")
	summary.add_row("Average time", f"{avg_time:.2f}s per image")
	
	if len(images) > 0:
		throughput = len(images) / elapsed
		summary.add_row("Throughput", f"{throughput:.2f} images/sec")
	
	console.print(Panel(summary, title="[bold]Batch Processing Summary[/bold]", border_style="green" if fail_count == 0 else "yellow"))
	
	# Report failed images if any
	if failed_images:
		console.print()
		error_table = Table(show_header=True, header_style="bold red", border_style="red")
		error_table.add_column("File", style="bright_yellow")
		error_table.add_column("Error", style="white")
		
		for filename, error in failed_images:
			error_table.add_row(escape(filename), error)
		
		console.print(Panel(error_table, title="[bold red]Failed Images[/bold red]", border_style="red"))
	
	console.print()
	
	return 0 if fail_count == 0 else 1

if __name__ == '__main__':
	sys.exit(main())
