from rich.console import Console

console = Console()

help_width = min(80, console.width)
help_console = Console(width=help_width)

if __name__ == '__main__':
	print('__main__ not supported in modules.')
