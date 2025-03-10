#!/bin/bash

# This script demonstrates how to run the YOLO model with different input sizes
# to avoid unnecessary padding and improve efficiency.

# Example 1: Run with default square input size (640x640)
# This will use the original square resizing with padding
echo "Running with default square input size (640x640)..."
python main.py --input-size 640 --test --yaml_file utils/args_bionano.yaml

# Example 2: Run with rectangular input size (640x320)
# This will resize images to 640x320 without padding, which is more efficient
# for images that are naturally wider than they are tall
echo "Running with rectangular input size (640x320)..."
python main.py --input-size 640,320 --test --yaml_file utils/args_bionano.yaml

# Example 3: Run with rectangular input size (320x640)
# This will resize images to 320x640 without padding, which is more efficient
# for images that are naturally taller than they are wide
echo "Running with rectangular input size (320x640)..."
python main.py --input-size 320,640 --test --yaml_file utils/args_bionano.yaml

# Note: The optimal input size depends on your dataset's typical image aspect ratio.
# For example:
# - For landscape images (wider than tall), try sizes like 640,320 or 800,450
# - For portrait images (taller than wide), try sizes like 320,640 or 450,800
# - For square images, the original 640,640 is still appropriate
