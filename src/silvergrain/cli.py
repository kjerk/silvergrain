import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from silvergrain.renderer import FilmGrainRenderer

"""
SilverGrain CLI - Single image film grain rendering
"""

console = Console()

def render_luminance_mode(pil_image: Image.Image, renderer: FilmGrainRenderer) -> Image.Image:
	"""
	Render grain only on luminance channel, preserving color information.
	"""
	img_array = np.array(pil_image, dtype=np.float32) / 255.0
	
	if len(img_array.shape) == 2:
		# Already grayscale
		output = renderer.render_single_channel(img_array, zoom=1.0, output_size=None)
		output = np.stack([output] * 3, axis=2)
	else:
		# Convert RGB to YUV
		img_uint8 = (img_array * 255).astype(np.uint8)
		yuv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
		
		# Render grain on Y (luminance) channel only
		y_rendered = renderer.render_single_channel(yuv[:, :, 0], zoom=1.0, output_size=None)
		yuv[:, :, 0] = y_rendered
		
		# Convert back to RGB
		yuv_uint8 = (np.clip(yuv * 255.0, 0, 255)).astype(np.uint8)
		output = cv2.cvtColor(yuv_uint8, cv2.COLOR_YUV2RGB)
		return Image.fromarray(output)
	
	# Clip and convert to uint8
	output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
	return Image.fromarray(output)

def render_rgb_mode(pil_image: Image.Image, renderer: FilmGrainRenderer, show_progress: bool = True) -> Image.Image:
	"""
	Render grain independently on each RGB channel.
	"""
	img_array = np.array(pil_image, dtype=np.float32) / 255.0

	if len(img_array.shape) == 2:
		# Grayscale - process once, copy to RGB
		output = renderer.render_single_channel(img_array, zoom=1.0, output_size=None)
		output = np.stack([output] * 3, axis=2)
	else:
		# Process each channel independently
		channels = []
		channel_names = ['Red', 'Green', 'Blue']
		for c in range(3):
			if show_progress:
				console.print(f"  [cyan]Processing {channel_names[c]} channel ({c + 1}/3)...[/cyan]")
			rendered = renderer.render_single_channel(img_array[:, :, c], zoom=1.0, output_size=None)
			channels.append(rendered)
		output = np.stack(channels, axis=2)

	# Clip and convert to uint8
	output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
	return Image.fromarray(output)

def main() -> int:
	"""Main CLI entry point for single image processing"""
	parser = argparse.ArgumentParser(
		description='Apply physically-based film grain to images',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Basic usage (medium intensity, balanced quality, luminance mode)
  silvergrain input.jpg output.jpg

  # Fine grain with high quality
  silvergrain input.jpg output.jpg --intensity fine --quality high

  # Heavy grain, fast rendering
  silvergrain input.jpg output.jpg --intensity heavy --quality fast

  # RGB mode (adds grain to each color channel independently)
  silvergrain input.jpg output.jpg --mode rgb

  # Subtle grain effect (50%% blend with original)
  silvergrain input.jpg output.jpg --strength 0.5

Presets:
  Intensity: fine (subtle) | medium (default) | heavy (strong)
  Quality:   fast (~1s GPU/1min CPU) | balanced (~2s GPU/3min CPU) | high (~5s GPU/8min CPU, 1080p)
  Mode:      luminance (default, preserves color) | rgb (per-channel grain)
  Device:    auto (default, uses GPU if available) | cpu | gpu
        """
	)
	
	parser.add_argument('input', type=str, help='Input image file')
	parser.add_argument('output', type=str, help='Output image file')
	
	# Simple user-facing options
	parser.add_argument('--intensity', type=str, choices=['fine', 'medium', 'heavy'], default='medium', help='Grain intensity: fine (subtle), medium (noticeable), heavy (strong) (default: medium)')
	parser.add_argument('--quality', type=str, choices=['fast', 'balanced', 'high'], default='balanced', help='Quality/speed tradeoff: fast (~1 min), balanced (~2-3 min), high (~5-8 min for 1080p) (default: balanced)')
	parser.add_argument('--mode', type=str, choices=['rgb', 'luminance'], default='luminance', help='Grain mode: "luminance" preserves color, "rgb" adds grain to each channel (default: luminance)')
	parser.add_argument('--strength', type=float, default=1.0, help='Grain strength: blend between original (0.0) and full grain (1.0) (default: 1.0, range: 0.0-1.0)')
	
	# Advanced options (most users won't need these)
	parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'gpu'], default='auto', help='Device to use: auto (GPU if available), cpu, gpu (default: auto)')
	parser.add_argument('--grain-radius', type=float, help='Override grain radius (advanced, 0.05-0.25)')
	parser.add_argument('--samples', type=int, help='Override Monte Carlo samples (advanced, 100-800)')
	parser.add_argument('--grain-sigma', type=float, default=0.0, help='Grain size variation (advanced, default: 0.0)')
	parser.add_argument('--sigma-filter', type=float, default=0.8, help='Anti-aliasing (advanced, default: 0.8)')
	parser.add_argument('--seed', type=int, default=2016, help='Random seed (advanced, default: 2016)')
	
	args = parser.parse_args()
	
	# Presets for easier use
	intensity_map = {'fine': 0.08, 'medium': 0.12, 'heavy': 0.20}
	quality_map = {'fast': 100, 'balanced': 200, 'high': 400}
	
	grain_radius = args.grain_radius if args.grain_radius else intensity_map[args.intensity]
	n_samples = args.samples if args.samples else quality_map[args.quality]
	
	# Validate inputs
	if args.strength < 0.0 or args.strength > 1.0:
		console.print(f"[red]Error:[/red] --strength must be between 0.0 and 1.0, got {args.strength}", file=sys.stderr)
		return 1

	input_path = Path(args.input)
	if not input_path.exists():
		console.print(f"[red]Error:[/red] Input file [bright_yellow]{escape(str(input_path))}[/bright_yellow] not found", file=sys.stderr)
		return 1

	output_path = Path(args.output)
	if output_path.exists():
		response = console.input(f"[yellow]Output file [bright_yellow]{escape(str(output_path))}[/bright_yellow] exists. Overwrite? [y/N][/yellow] ")
		if response.lower() != 'y':
			console.print("[yellow]Cancelled.[/yellow]")
			return 0

	# Load image
	console.print(f"[cyan]Loading[/cyan] [bright_yellow]{escape(input_path.name)}[/bright_yellow]...")
	try:
		image = Image.open(input_path)
	except Exception as e:
		console.print(f"[red]Error loading image:[/red] {e}", file=sys.stderr)
		return 1

	# Convert to RGB if needed
	if image.mode not in ['L', 'RGB']:
		console.print(f"[cyan]Converting from {image.mode} to RGB...[/cyan]")
		image = image.convert('RGB')

	width, height = image.size
	megapixels = (width * height) / 1_000_000
	
	# Create renderer with device parameter
	try:
		renderer = FilmGrainRenderer(
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

	device_str = 'GPU' if renderer.device == 'gpu' else 'CPU'

	# Display configuration panel
	config_table = Table.grid(padding=(0, 2))
	config_table.add_column(style="cyan", justify="right")
	config_table.add_column(style="white")

	config_table.add_row("Image:", f"[bright_yellow]{escape(input_path.name)}[/bright_yellow]")
	config_table.add_row("Size:", f"{width}x{height} ({megapixels:.1f} MP)")
	config_table.add_row("Device:", f"[bold]{device_str}[/bold]")
	config_table.add_row("Intensity:", args.intensity)
	config_table.add_row("Quality:", args.quality)
	config_table.add_row("Mode:", args.mode)

	if args.strength < 1.0:
		config_table.add_row("Strength:", f"{args.strength:.2f}")

	if args.grain_radius or args.samples:
		config_table.add_row("", "")
		config_table.add_row("[dim]Advanced:[/dim]", "")
		if args.grain_radius:
			config_table.add_row("Grain radius:", f"{grain_radius:.3f}")
		if args.samples:
			config_table.add_row("Samples:", f"{n_samples}")

	console.print()
	console.print(Panel(config_table, title="[bold]Film Grain Rendering[/bold]", border_style="blue"))
	console.print()
	
	# Render based on mode
	start_time = time.time()

	try:
		if args.mode == 'luminance':
			with Progress(SpinnerColumn(), TextColumn("[cyan]Rendering grain on luminance channel...[/cyan]"), console=console) as progress:
				progress.add_task("render", total=None)
				output = render_luminance_mode(image, renderer)
		else:  # rgb
			console.print("[cyan]Rendering grain on RGB channels:[/cyan]")
			output = render_rgb_mode(image, renderer, show_progress=True)

		render_time = time.time() - start_time

	except Exception as e:
		console.print(f"[red]Error during rendering:[/red] {e}", file=sys.stderr)
		import traceback
		traceback.print_exc()
		return 1

	# Blend with original if strength < 1.0
	if args.strength < 1.0:
		with Progress(SpinnerColumn(), TextColumn(f"[cyan]Blending at strength {args.strength:.2f}...[/cyan]"), console=console) as progress:
			progress.add_task("blend", total=None)
			original_array = np.array(image, dtype=np.float32)
			output_array = np.array(output, dtype=np.float32)

			stacked = np.stack([original_array, output_array])
			weights = (1.0 - args.strength, args.strength)
			blended = np.average(stacked, axis=0, weights=weights)
			blended = np.clip(blended, 0, 255).astype(np.uint8)
			output = Image.fromarray(blended)

	# Save output
	console.print(f"[cyan]Saving to[/cyan] [bright_yellow]{escape(output_path.name)}[/bright_yellow]...")

	try:
		output.save(output_path)
	except Exception as e:
		console.print(f"[red]Error saving image:[/red] {e}", file=sys.stderr)
		return 1

	# Display completion summary
	console.print()
	console.print(f"[green]✓ Done![/green] Rendered in [bold]{render_time:.2f}s[/bold]")
	console.print(f"  Output: [bright_yellow]{escape(str(output_path))}[/bright_yellow]")
	console.print()

	return 0

if __name__ == '__main__':
	sys.exit(main())
