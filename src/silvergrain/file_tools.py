from pathlib import Path
from typing import List

import natsort

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def list_images(input_dir: Path, recursive: bool = True) -> List[Path]:
	search_glob = '**/*' if recursive else '*'
	image_paths = []
	
	for file_path in input_dir.glob(search_glob):
		if file_path.suffix.lower() in IMAGE_EXTENSIONS:
			image_paths.append(file_path)
	
	image_paths = list(set(image_paths))
	
	return natsort.os_sorted(image_paths)

if __name__ == '__main__':
	print('__main__ not supported in modules.')
