import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from silvergrain import FilmGrainRenderer
from silvergrain.tools.image_tools import get_pil_save_kwargs
from silvergrain.tools.print_tools import console, help_console

"""
SilverGrain CLI - Single image film grain rendering
"""

# Default suffix for auto-generated output filenames
DEFAULT_AUTONAME_SUFFIX = "-grainy"

def render_help():
	"""Render beautiful custom help output using Rich"""
	
	# Header
	help_console.print()
	help_console.print(Panel.fit(
		"[bold cyan]SilverGrain[/bold cyan]\n"
		"Physically-based film grain for your images",
		border_style="cyan"
	))
	help_console.print()
	
	# Quick Start
	quick_start = Table.grid(padding=(0, 2))
	quick_start.add_column(style="dim")
	quick_start.add_row("silvergrain input.jpg")
	quick_start.add_row("silvergrain input.jpg output.jpg")
	quick_start.add_row("silvergrain input.jpg --intensity heavy --grain-variation 0.3")
	
	help_console.print(Panel(quick_start, title="[bold]Quick Start[/bold]", border_style="green"))
	help_console.print()
	
	# Basic Options
	basic = Table.grid(padding=(0, 1))
	basic.add_column(style="cyan", justify="left")
	basic.add_column(style="white")
	
	basic.add_row("[bold]Intensity Presets[/bold]", "")
	basic.add_row("  --intensity", "fine | medium | heavy")
	basic.add_row("", "[dim]Default: medium[/dim]")
	basic.add_row("", "")
	
	basic.add_row("[bold]Quality Presets[/bold]", "")
	basic.add_row("  --quality", "fast | balanced | high")
	basic.add_row("", "[dim]Default: balanced[/dim]")
	basic.add_row("", "")
	
	basic.add_row("[bold]Grain Mode[/bold]", "")
	basic.add_row("  --mode", "luminance | rgb")
	basic.add_row("", "[dim]luminance: preserves color (recommended)[/dim]")
	basic.add_row("", "[dim]rgb: per-channel grain (more intense)[/dim]")
	basic.add_row("", "")
	
	basic.add_row("[bold]Blend Strength[/bold]", "")
	basic.add_row("  --strength", "0.0-1.0")
	basic.add_row("", "[dim]Blend between original (0.0) and full grain (1.0)[/dim]")
	
	help_console.print(Panel(basic, title="[bold]Basic Options[/bold]", border_style="blue"))
	help_console.print()
	
	# Preset Reference Tables
	intensity_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
	intensity_table.add_column("Intensity", style="cyan")
	intensity_table.add_column("Grain Size", justify="center")
	intensity_table.add_column("Use Case")
	intensity_table.add_row("fine", "0.08", "Subtle texture, modern film")
	intensity_table.add_row("medium", "0.12", "Classic film look [dim](default)[/dim]")
	intensity_table.add_row("heavy", "0.20", "Vintage, high-ISO aesthetic")
	
	quality_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
	quality_table.add_column("Quality", style="cyan")
	quality_table.add_column("Samples", justify="center")
	quality_table.add_row("fast", "100")
	quality_table.add_row("balanced", "200 [dim](default)[/dim]")
	quality_table.add_row("high", "400")
	
	preset_grid = Table.grid(padding=(0, 2))
	preset_grid.add_column()
	preset_grid.add_column()
	preset_grid.add_row(intensity_table, quality_table)
	
	help_console.print(Panel(preset_grid, title="[bold]Preset Reference[/bold]", border_style="magenta"))
	help_console.print()
	
	# Advanced Options
	advanced = Table.grid(padding=(0, 1))
	advanced.add_column(style="yellow", width=20)
	advanced.add_column(style="dim", width=10)
	advanced.add_column(style="white")
	
	advanced.add_row("--grain-variation", "0.0-1.0", "Randomize grain sizes")
	advanced.add_row("", "", "[dim]0.0 = uniform (fast)[/dim]")
	advanced.add_row("", "", "[dim]0.3 = subtle variation[/dim]")
	advanced.add_row("", "", "[dim]1.0 = maximum variation (slow!)[/dim]")
	advanced.add_row("", "", "")
	advanced.add_row("--device", "", "auto | cpu | gpu")
	advanced.add_row("", "", "[dim]Force CPU or GPU (default: auto)[/dim]")
	advanced.add_row("", "", "")
	advanced.add_row("--grain-radius", "float", "Manual grain size (0.05-0.25)")
	advanced.add_row("--samples", "int", "Manual sample count (100-800)")
	advanced.add_row("--sigma-filter", "float", "Anti-aliasing filter (default: 0.8)")
	advanced.add_row("--seed", "int", "Random seed (default: 2016)")
	
	help_console.print(Panel(advanced, title="[bold]Advanced Options[/bold]", border_style="yellow"))
	help_console.print()
	
	# Examples
	examples = Table.grid(padding=(0, 0))
	examples.add_column(style="white")
	
	examples.add_row("[dim]# Basic usage (auto-named output: photo-grainy.png)[/dim]")
	examples.add_row("[green]silvergrain[/green] photo.jpg")
	examples.add_row("")
	examples.add_row("[dim]# Specify output filename[/dim]")
	examples.add_row("[green]silvergrain[/green] photo.jpg output.jpg")
	examples.add_row("")
	examples.add_row("[dim]# Heavy grain with size variation[/dim]")
	examples.add_row("[green]silvergrain[/green] photo.jpg --intensity heavy --grain-variation 0.3")
	examples.add_row("")
	examples.add_row("[dim]# Blend 50% with original[/dim]")
	examples.add_row("[green]silvergrain[/green] photo.jpg --strength 0.5")
	
	help_console.print(Panel(examples, title="[bold]Examples[/bold]", border_style="green"))
	help_console.print()
	
	# Footer
	help_console.print("[dim]For batch processing, see: [cyan]silvergrain-batch --help[/cyan][/dim]")
	help_console.print("[dim]For dataset augmentation, see: [cyan]silvergrain-augment --help[/cyan][/dim]")
	help_console.print()

def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Apply physically-based film grain to images',
		add_help=False  # Disable default help to use our custom one
	)
	
	parser.add_argument('input', type=str, help='Input image file')
	parser.add_argument('output', type=str, nargs='?', default=None, help='Output image file (optional, defaults to {input}-grainy.png)')
	
	# Simple user-facing options
	parser.add_argument('--intensity', type=str, choices=['fine', 'medium', 'heavy'], default='medium', help='Grain intensity: fine (subtle), medium (noticeable), heavy (strong) (default: medium)')
	parser.add_argument('--quality', type=str, choices=['fast', 'balanced', 'high'], default='balanced', help='Quality/speed tradeoff: fast (~1 min), balanced (~2-3 min), high (~5-8 min for 1080p) (default: balanced)')
	parser.add_argument('--mode', type=str, choices=['rgb', 'luminance'], default='luminance', help='Grain mode: "luminance" preserves color, "rgb" adds grain to each channel (default: luminance)')
	parser.add_argument('--strength', type=float, default=1.0, help='Grain strength: blend between original (0.0) and full grain (1.0) (default: 1.0, range: 0.0-1.0)')
	
	# Advanced options (most users won't need these)
	parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'gpu'], default='auto', help='Device to use: auto (GPU if available), cpu, gpu (default: auto)')
	parser.add_argument('--grain-radius', type=float, help='Override grain radius (advanced, 0.05-0.25)')
	parser.add_argument('--samples', type=int, help='Override Monte Carlo samples (advanced, 100-800)')
	parser.add_argument('--grain-variation', type=float, default=0.0, help='Grain size variation: 0.0 (uniform) to 1.0 (maximum variation) (advanced, default: 0.0)')
	parser.add_argument('--grain-sigma', type=float, default=None, help=argparse.SUPPRESS)  # Hidden advanced override
	parser.add_argument('--sigma-filter', type=float, default=0.8, help='Anti-aliasing (advanced, default: 0.8)')
	parser.add_argument('--seed', type=int, default=2016, help='Random seed (advanced, default: 2016)')
	
	args = parser.parse_args()
	
	return args

def main() -> int:
	"""Main CLI entry point for single image processing"""
	# Check for help flag before parsing
	if '--help' in sys.argv or '-h' in sys.argv:
		render_help()
		return 0
	
	args = parse_arguments()
	
	# Presets for easier use
	intensity_map = {'fine': 0.08, 'medium': 0.12, 'heavy': 0.20}
	quality_map = {'fast': 100, 'balanced': 200, 'high': 400}
	
	grain_radius = args.grain_radius if args.grain_radius else intensity_map[args.intensity]
	n_samples = args.samples if args.samples else quality_map[args.quality]
	
	# Calculate grain_sigma from grain_variation or use explicit override
	if args.grain_sigma is not None:
		# Advanced user explicitly set grain-sigma
		grain_sigma = args.grain_sigma
		# Warn if grain_sigma is high relative to grain_radius
		sigma_ratio = grain_sigma / grain_radius
		if sigma_ratio > 0.5:
			console.print(f"[yellow]Warning:[/yellow] --grain-sigma ({grain_sigma:.3f}) exceeds recommended limit (0.5 × grain_radius = {0.5 * grain_radius:.3f})", file=sys.stderr)
			console.print(f"[yellow]         This may cause significant slowdown. Recommended: --grain-sigma ≤ {0.5 * grain_radius:.3f}[/yellow]", file=sys.stderr)
		elif sigma_ratio > 0.2:
			console.print(f"[yellow]Note:[/yellow] --grain-sigma ({grain_sigma:.3f}) may significantly increase render time (~{int(sigma_ratio / 0.2 * 4)}× slower)", file=sys.stderr)
	else:
		# Calculate from grain-variation (user-friendly parameter)
		grain_sigma = args.grain_variation * 0.5 * grain_radius
		# Warn if grain_variation is high
		if args.grain_variation > 1.0:
			console.print(f"[yellow]Warning:[/yellow] --grain-variation ({args.grain_variation:.2f}) exceeds recommended limit (1.0)", file=sys.stderr)
			console.print(f"[yellow]         This may cause significant slowdown and unrealistic results.[/yellow]", file=sys.stderr)
		elif args.grain_variation > 0.4:
			console.print(f"[yellow]Note:[/yellow] --grain-variation ({args.grain_variation:.2f}) may significantly increase render time", file=sys.stderr)
	
	# Validate inputs
	if args.strength < 0.0 or args.strength > 1.0:
		console.print(f"[red]Error:[/red] --strength must be between 0.0 and 1.0, got {args.strength}", file=sys.stderr)
		return 1
	
	if args.grain_variation < 0.0:
		console.print(f"[red]Error:[/red] --grain-variation must be non-negative, got {args.grain_variation}", file=sys.stderr)
		return 1
	
	input_path = Path(args.input)
	if not input_path.exists():
		console.print(f"[red]Error:[/red] Input file [bright_yellow]{escape(str(input_path))}[/bright_yellow] not found", file=sys.stderr)
		return 1
	
	# Determine output path
	if args.output is None:
		# Auto-generate output filename
		output_path = input_path.parent / f"{input_path.stem}{DEFAULT_AUTONAME_SUFFIX}.png"
	else:
		output_path = Path(args.output)
	
	# Check for overwrite
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
			grain_sigma=grain_sigma,
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
	
	if args.grain_variation > 0.0:
		config_table.add_row("Grain variation:", f"{args.grain_variation:.2f}")
	
	if args.grain_radius or args.samples or args.grain_sigma is not None:
		config_table.add_row("", "")
		config_table.add_row("[dim]Advanced:[/dim]", "")
		if args.grain_radius:
			config_table.add_row("Grain radius:", f"{grain_radius:.3f}")
		if args.samples:
			config_table.add_row("Samples:", f"{n_samples}")
		if args.grain_sigma is not None:
			config_table.add_row("Grain sigma:", f"{grain_sigma:.3f}")
	
	console.print()
	console.print(Panel(config_table, title="[bold]Film Grain Rendering[/bold]", border_style="blue"))
	console.print()
	
	# Render with film grain
	start_time = time.time()
	
	try:
		mode_text = "luminance channel" if args.mode == 'luminance' else "RGB channels"
		with Progress(SpinnerColumn(), TextColumn(f"[cyan]Rendering grain on {mode_text}...[/cyan]"), console=console) as progress:
			progress.add_task("render", total=None)
			output = renderer.process_image(image, mode=args.mode, strength=args.strength)
		
		render_time = time.time() - start_time
	
	except Exception as e:
		console.print(f"[red]Error during rendering:[/red] {e}", file=sys.stderr)
		import traceback
		traceback.print_exc()
		return 1
	
	# Save output
	console.print(f"[cyan]Saving to[/cyan] [bright_yellow]{escape(output_path.name)}[/bright_yellow]...")
	
	try:
		save_kwargs = get_pil_save_kwargs(output_path)
		output.save(output_path, **save_kwargs)
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
