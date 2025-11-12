import argparse
import random
import sys
import time
from pathlib import Path
from typing import Tuple, Union

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
SilverGrain Augment CLI - Generate augmented image datasets
"""

console = Console()

def parse_param(value_str: str, param_type=float) -> Union[Tuple[str, float], Tuple[str, float, float]]:
	"""Parse a parameter that can be fixed or a range.

	Returns:
		('fixed', value) for fixed values
		('range', low, high) for ranges
	"""
	if ':' in value_str:
		low, high = value_str.split(':')
		return ('range', param_type(low), param_type(high))
	else:
		return ('fixed', param_type(value_str))

def sample_param(param_spec: Union[Tuple[str, float], Tuple[str, float, float]]) -> float:
	"""Sample a value from a parameter specification."""
	if param_spec[0] == 'fixed':
		return param_spec[1]
	else:  # range
		low, high = param_spec[1], param_spec[2]
		return random.uniform(low, high)

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
	"""Main CLI entry point for dataset augmentation"""
	parser = argparse.ArgumentParser(
		description='Generate augmented image datasets with film grain variations',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Generate 5 augmentation variants with randomized grain radius
  silvergrain-augment input_dir/ output_dir/ --count 5 --grain-radius 0.08:0.20

  # Fixed strength, ranged grain, random mode per variant
  silvergrain-augment input_dir/ output_dir/ --count 10 \\
      --grain-radius 0.05:0.25 \\
      --strength 0.9 \\
      --mode rand

  # High quality with parameter variation
  silvergrain-augment input_dir/ output_dir/ --count 20 \\
      --quality high \\
      --grain-radius 0.08:0.15 \\
      --grain-sigma 0.0:0.1 \\
      --strength 0.7:1.0

Output Structure:
  output_dir/
  ├── aug_0/
  │   ├── image_001.png
  │   ├── image_002.png
  ├── aug_1/
  │   ├── image_001.png
  │   └── ...
  └── ...

Parameters:
  Fixed value:  --grain-radius 0.12
  Range:        --grain-radius 0.08:0.20 (uniform sampling per variant)
  Mode options: luminance | rgb | rand (randomly pick per variant)
        """
	)
	
	parser.add_argument('input_dir', type=str, help='Input directory containing clean images')
	parser.add_argument('output_dir', type=str, help='Output directory for augmented datasets')
	parser.add_argument('--count', type=int, default=1, help='Number of augmentation variants to generate (default: 1)')
	
	# Parameter options (can be fixed or ranges)
	parser.add_argument('--grain-radius', type=str, default='0.12', help='Grain radius: fixed (0.12) or range (0.08:0.20) (default: 0.12)')
	parser.add_argument('--grain-sigma', type=str, default='0.0', help='Grain size variation: fixed or range (default: 0.0)')
	parser.add_argument('--sigma-filter', type=str, default='0.8', help='Anti-aliasing filter: fixed or range (default: 0.8)')
	parser.add_argument('--strength', type=str, default='1.0', help='Grain strength: fixed or range (default: 1.0)')
	parser.add_argument('--samples', type=str, help='Monte Carlo samples: fixed or range (overrides --quality)')
	
	# Preset and mode options
	parser.add_argument('--quality', type=str, choices=['fast', 'balanced', 'high'], default='balanced', help='Quality preset (default: balanced)')
	parser.add_argument('--mode', type=str, choices=['luminance', 'rgb', 'rand'], default='luminance', help='Grain mode: luminance, rgb, or rand (random per variant) (default: luminance)')
	
	# File handling
	parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
	
	# Device
	parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'gpu'], default='auto', help='Device to use (default: auto)')
	
	args = parser.parse_args()
	
	# Validate directories
	input_dir = Path(args.input_dir)
	if not input_dir.exists():
		console.print(f"[red]Error:[/red] Input directory [bright_yellow]{escape(str(input_dir))}[/bright_yellow] not found", file=sys.stderr)
		return 1
	if not input_dir.is_dir():
		console.print(f"[red]Error:[/red] [bright_yellow]{escape(str(input_dir))}[/bright_yellow] is not a directory", file=sys.stderr)
		return 1
	
	output_dir = Path(args.output_dir)
	if output_dir.exists() and any(output_dir.iterdir()):
		response = console.input(f"[yellow]Output directory [bright_yellow]{escape(str(output_dir))}[/bright_yellow] exists and is not empty. Continue? [y/N][/yellow] ")
		if response.lower() != 'y':
			console.print("[yellow]Cancelled.[/yellow]")
			return 0
	
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# Validate count
	if args.count < 1:
		console.print(f"[red]Error:[/red] --count must be at least 1, got {args.count}", file=sys.stderr)
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
	
	console.print(f"[green]Found {len(images)} images[/green]")
	
	# Parse parameter specifications
	grain_radius_spec = parse_param(args.grain_radius, float)
	grain_sigma_spec = parse_param(args.grain_sigma, float)
	sigma_filter_spec = parse_param(args.sigma_filter, float)
	strength_spec = parse_param(args.strength, float)
	
	quality_map = {'fast': 100, 'balanced': 200, 'high': 400}
	
	if args.samples:
		samples_spec = parse_param(args.samples, int)
	else:
		samples_spec = ('fixed', quality_map[args.quality])
	
	# Validate strength ranges
	if strength_spec[0] == 'range':
		low, high = strength_spec[1], strength_spec[2]
		if low < 0.0 or high > 1.0:
			console.print(f"[red]Error:[/red] --strength range must be within 0.0-1.0, got {low}:{high}", file=sys.stderr)
			return 1
	elif strength_spec[0] == 'fixed':
		if strength_spec[1] < 0.0 or strength_spec[1] > 1.0:
			console.print(f"[red]Error:[/red] --strength must be between 0.0 and 1.0, got {strength_spec[1]}", file=sys.stderr)
			return 1
	
	# Create a test renderer to determine actual device (for display)
	try:
		test_renderer = FilmGrainRenderer(
			grain_radius=0.12,
			grain_sigma=0.0,
			sigma_filter=0.8,
			n_monte_carlo=200,
			device=args.device,
			seed=2016
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
	config_table.add_row("Augmentation variants:", f"[bold]{args.count}[/bold]")
	config_table.add_row("Output:", f"[bright_yellow]{escape(str(output_dir))}[/bright_yellow]")
	if recursive:
		config_table.add_row("Recursive:", "[green]Yes[/green]")
	config_table.add_row("Device:", f"[bold]{device_str}[/bold]")
	config_table.add_row("Mode:", args.mode)
	
	# Show parameter specs
	config_table.add_row("", "")
	config_table.add_row("[dim]Parameters:[/dim]", "")
	
	def format_param_spec(spec):
		if spec[0] == 'fixed':
			return f"{spec[1]}"
		else:
			return f"[yellow]{spec[1]}:{spec[2]}[/yellow]"
	
	config_table.add_row("Grain radius:", format_param_spec(grain_radius_spec))
	config_table.add_row("Grain sigma:", format_param_spec(grain_sigma_spec))
	config_table.add_row("Sigma filter:", format_param_spec(sigma_filter_spec))
	config_table.add_row("Strength:", format_param_spec(strength_spec))
	config_table.add_row("Samples:", format_param_spec(samples_spec))
	
	console.print()
	console.print(Panel(config_table, title="[bold]Film Grain Dataset Augmentation[/bold]", border_style="blue"))
	console.print()
	
	# Process images with progress bar
	start_time = time.time()
	total_operations = len(images) * args.count
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
		task = progress.add_task("[green]Generating augmentations...", total=total_operations)
		
		for variant_idx in range(args.count):
			# Sample parameters for this variant
			grain_radius = sample_param(grain_radius_spec)
			grain_sigma = sample_param(grain_sigma_spec)
			sigma_filter = sample_param(sigma_filter_spec)
			strength = sample_param(strength_spec)
			n_samples = int(sample_param(samples_spec))
			
			# Select mode for this variant
			if args.mode == 'rand':
				mode = random.choice(['luminance', 'rgb'])
			else:
				mode = args.mode
			
			# Create output directory for this variant
			variant_dir = output_dir / f"aug_{variant_idx}"
			variant_dir.mkdir(parents=True, exist_ok=True)
			
			for img_idx, input_path in enumerate(images):
				# Update progress
				progress.update(task, description=f"[green]Variant {variant_idx + 1}/{args.count}[/green] - [bright_yellow]{escape(input_path.name)}[/bright_yellow]")
				
				# Create renderer with random seed per image
				seed = random.randint(0, 2 ** 31 - 1)
				
				renderer = FilmGrainRenderer(
					grain_radius=grain_radius,
					grain_sigma=grain_sigma,
					sigma_filter=sigma_filter,
					n_monte_carlo=n_samples,
					device=args.device,
					seed=seed
				)
				
				# Determine output path (preserve filename)
				output_path = variant_dir / input_path.name
				
				# Process
				success, error = process_image(input_path, output_path, renderer, mode, strength)
				
				if success:
					success_count += 1
				else:
					fail_count += 1
					failed_images.append((f"aug_{variant_idx}/{input_path.name}", error))
				
				progress.advance(task)
	
	# Display summary
	elapsed = time.time() - start_time
	avg_time = elapsed / total_operations if total_operations > 0 else 0
	
	console.print()
	
	# Create summary table
	summary = Table(show_header=True, header_style="bold cyan", border_style="blue")
	summary.add_column("Metric", style="cyan", justify="right")
	summary.add_column("Value", style="white")
	
	summary.add_row("Total operations", str(total_operations))
	summary.add_row("Successful", f"[green]{success_count}[/green]")
	
	if fail_count > 0:
		summary.add_row("Failed", f"[red]{fail_count}[/red]")
	
	summary.add_row("Total time", f"{elapsed:.1f}s")
	summary.add_row("Average time", f"{avg_time:.2f}s per image")
	
	if total_operations > 0:
		throughput = total_operations / elapsed
		summary.add_row("Throughput", f"{throughput:.2f} images/sec")
	
	console.print(Panel(summary, title="[bold]Augmentation Summary[/bold]", border_style="green" if fail_count == 0 else "yellow"))
	
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
