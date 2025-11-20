import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from silvergrain import FilmGrainRenderer
from silvergrain.tools import file_tools

"""
SilverGrain Batch CLI - Batch process directories of images
"""

console = Console()

def get_save_kwargs(output_path: Path) -> dict:
	"""Get appropriate save kwargs based on output file format"""
	ext = output_path.suffix.lower()

	if ext in ['.jpg', '.jpeg']:
		return {'quality': 98, 'optimize': True}
	elif ext == '.png':
		return {'compress_level': 3, 'optimize': True}
	else:
		# For other formats, use reasonable defaults
		return {'optimize': True}

def render_help():
	"""Render beautiful custom help output using Rich"""
	# Use 80 chars or console width, whichever is smaller
	help_width = min(80, console.width)
	help_console = Console(width=help_width)

	# Header
	help_console.print()
	help_console.print(Panel.fit(
		"[bold cyan]SilverGrain Batch[/bold cyan]\n"
		"Batch apply film grain to directories of images",
		border_style="cyan"
	))
	help_console.print()

	# Quick Start
	quick_start = Table.grid(padding=(0, 2))
	quick_start.add_column(style="dim")
	quick_start.add_row("silvergrain-batch input_dir/")
	quick_start.add_row("silvergrain-batch input_dir/ output_dir/")
	quick_start.add_row("silvergrain-batch input_dir/ --intensity heavy --quality fast")

	help_console.print(Panel(quick_start, title="[bold]Quick Start[/bold]", border_style="green"))
	help_console.print()

	# Usage Pattern
	usage = Table.grid(padding=(0, 1))
	usage.add_column(style="cyan")
	usage.add_column(style="white")
	usage.add_row("[bold]Input only:[/bold]", "Saves in-place with -grainy suffix")
	usage.add_row("", "[dim]silvergrain-batch photos/[/dim]")
	usage.add_row("", "")
	usage.add_row("[bold]Input + Output:[/bold]", "Saves to output directory")
	usage.add_row("", "[dim]silvergrain-batch photos/ processed/[/dim]")

	help_console.print(Panel(usage, title="[bold]Usage Patterns[/bold]", border_style="blue"))
	help_console.print()

	# Options
	options = Table.grid(padding=(0, 1))
	options.add_column(style="cyan", width=20)
	options.add_column(style="white")

	options.add_row("[bold]Basic Options[/bold]", "")
	options.add_row("  --intensity", "fine | medium | heavy  [dim](default: medium)[/dim]")
	options.add_row("  --quality", "fast | balanced | high  [dim](default: balanced)[/dim]")
	options.add_row("  --mode", "luminance | rgb  [dim](default: luminance)[/dim]")
	options.add_row("  --strength", "0.0-1.0  [dim](default: 1.0)[/dim]")
	options.add_row("  --recursive", "Search subdirectories")
	options.add_row("  --random-seed", "Different seed per image (for augmentation)")
	options.add_row("", "")
	options.add_row("[bold]Advanced Options[/bold]", "")
	options.add_row("  --grain-variation", "0.0-1.0  [dim]Grain size randomness[/dim]")
	options.add_row("  --device", "auto | cpu | gpu  [dim](default: auto)[/dim]")
	options.add_row("  --grain-radius", "float  [dim]Manual grain size (0.05-0.25)[/dim]")
	options.add_row("  --samples", "int  [dim]Manual sample count (100-800)[/dim]")

	help_console.print(Panel(options, title="[bold]Options[/bold]", border_style="blue"))
	help_console.print()

	# Examples
	examples = Table.grid(padding=(0, 0))
	examples.add_column(style="white")

	examples.add_row("[dim]# Process all images in directory (in-place)[/dim]")
	examples.add_row("[green]silvergrain-batch[/green] photos/")
	examples.add_row("")
	examples.add_row("[dim]# Process to separate output directory[/dim]")
	examples.add_row("[green]silvergrain-batch[/green] photos/ processed/")
	examples.add_row("")
	examples.add_row("[dim]# Heavy grain, fast, search recursively[/dim]")
	examples.add_row("[green]silvergrain-batch[/green] photos/ --intensity heavy --quality fast --recursive")
	examples.add_row("")
	examples.add_row("[dim]# Different grain per image (for augmentation)[/dim]")
	examples.add_row("[green]silvergrain-batch[/green] dataset/ output/ --random-seed")

	help_console.print(Panel(examples, title="[bold]Examples[/bold]", border_style="green"))
	help_console.print()

	# Footer
	help_console.print("[dim]For single images, see: [cyan]silvergrain --help[/cyan][/dim]")
	help_console.print("[dim]For dataset augmentation, see: [cyan]silvergrain-augment --help[/cyan][/dim]")
	help_console.print()

def main() -> int:
	"""Main CLI entry point for batch processing"""
	# Check for help flag before parsing
	if '--help' in sys.argv or '-h' in sys.argv:
		render_help()
		return 0

	parser = argparse.ArgumentParser(
		description='Batch apply film grain to directory of images',
		add_help=False  # Disable default help to use our custom one
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
	parser.add_argument('--grain-variation', type=float, default=0.0, help='Grain size variation: 0.0 (uniform) to 1.0 (maximum variation) (default: 0.0)')
	parser.add_argument('--grain-sigma', type=float, default=None, help=argparse.SUPPRESS)  # Hidden advanced override
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

	if args.grain_variation < 0.0:
		console.print(f"[red]Error:[/red] --grain-variation must be non-negative, got {args.grain_variation}", file=sys.stderr)
		return 1

	# Create a test renderer to determine actual device (for display)
	try:
		test_renderer = FilmGrainRenderer(
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
	if args.grain_variation > 0.0:
		config_table.add_row("Grain variation:", f"{args.grain_variation:.2f}")
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
				grain_sigma=grain_sigma,
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
			
			# Process image
			try:
				image = Image.open(input_path)
				output = renderer.process_image(image, mode=args.mode, strength=args.strength)
				save_kwargs = get_save_kwargs(output_path)
				output.save(output_path, **save_kwargs)
				success_count += 1
			except Exception as e:
				fail_count += 1
				failed_images.append((input_path.name, str(e)))
			
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
