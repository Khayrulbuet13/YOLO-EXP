import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import random

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_image_and_label(image_path, label_path, output_image_path, output_label_path, crop_width=128):
    """
    Process an image by cropping it from 1280x128 to 256x128 and update its label.
    
    Args:
        image_path: Path to the original image
        label_path: Path to the original label
        output_image_path: Path to save the processed image
        output_label_path: Path to save the processed label
        crop_width: Width of the cropped image (default: 256)
    
    Returns:
        True if processing was successful, False otherwise
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return False
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Read the label
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label_lines = f.read().strip().splitlines()
        
        labels = []
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])
                
                # Convert normalized coordinates to absolute
                abs_x_center = x_center * width
                abs_y_center = y_center * height
                abs_width = bbox_width * width
                abs_height = bbox_height * height
                
                # Calculate bbox boundaries
                x1 = abs_x_center - abs_width / 2
                x2 = abs_x_center + abs_width / 2
                
                labels.append({
                    'class_id': class_id,
                    'x_center': abs_x_center,
                    'y_center': abs_y_center,
                    'width': abs_width,
                    'height': abs_height,
                    'x1': x1,
                    'x2': x2
                })
    else:
        labels = []
    
    # Determine crop position
    if labels:
        # Strategy: Try to keep the bounding box in the crop
        # Find the leftmost and rightmost points of all bounding boxes
        min_x = min([label['x1'] for label in labels])
        max_x = max([label['x2'] for label in labels])
        
        # Calculate the center of all bounding boxes
        bbox_center_x = (min_x + max_x) / 2
        
        # Calculate the leftmost possible crop position that would include all bboxes
        leftmost_crop = max(0, max_x - crop_width)
        
        # Calculate the rightmost possible crop position that would include all bboxes
        rightmost_crop = min(width - crop_width, min_x)
        
        # If we can't fit all bboxes in the crop, center the crop on the bbox center
        if leftmost_crop > rightmost_crop:
            # Center the crop on the bbox center
            crop_x = max(0, min(width - crop_width, int(bbox_center_x - crop_width / 2)))
        else:
            # Randomly choose a crop position that includes all bboxes
            crop_x = int(random.uniform(rightmost_crop, leftmost_crop))
    else:
        # If no bounding boxes, choose a random crop position
        crop_x = random.randint(0, width - crop_width)
    
    # Ensure crop_x is within valid range
    crop_x = max(0, min(width - crop_width, crop_x))
    
    # Crop the image
    cropped_image = image[:, crop_x:crop_x + crop_width]
    
    # Update the labels
    new_labels = []
    for label in labels:
        # Check if the bbox is still in the cropped image
        if (label['x1'] < crop_x + crop_width) and (label['x2'] > crop_x):
            # Calculate the new coordinates relative to the cropped image
            new_x_center = label['x_center'] - crop_x
            new_width = label['width']
            
            # If the bbox is partially outside the crop, adjust it
            if label['x1'] < crop_x:
                # Left side of bbox is outside the crop
                new_width = label['x2'] - crop_x
                new_x_center = new_width / 2
            elif label['x2'] > crop_x + crop_width:
                # Right side of bbox is outside the crop
                new_width = crop_x + crop_width - label['x1']
                new_x_center = new_width / 2 + (label['x1'] - crop_x)
            
            # Normalize the coordinates for the new image dimensions
            norm_x_center = new_x_center / crop_width
            norm_y_center = label['y_center'] / height
            norm_width = new_width / crop_width
            norm_height = label['height'] / height
            
            # Ensure the normalized coordinates are within [0, 1]
            norm_x_center = max(0, min(1, norm_x_center))
            norm_y_center = max(0, min(1, norm_y_center))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            new_labels.append(f"{label['class_id']} {norm_x_center} {norm_y_center} {norm_width} {norm_height}")
    
    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_image)
    
    # Save the updated label
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(new_labels))
    
    return True

def process_dataset(input_dir, output_dir):
    """
    Process the entire dataset.
    
    Args:
        input_dir: Path to the original dataset
        output_dir: Path to save the processed dataset
    """
    # Create output directories
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, 'images'))
    ensure_dir(os.path.join(output_dir, 'images', 'train'))
    ensure_dir(os.path.join(output_dir, 'images', 'val'))
    ensure_dir(os.path.join(output_dir, 'images', 'test'))
    ensure_dir(os.path.join(output_dir, 'labels'))
    ensure_dir(os.path.join(output_dir, 'labels', 'train'))
    ensure_dir(os.path.join(output_dir, 'labels', 'val'))
    ensure_dir(os.path.join(output_dir, 'labels', 'test'))
    
    # Copy train.txt and val.txt
    for file in ['train.txt', 'val.txt']:
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)
        if os.path.exists(input_file):
            # Read the content and update the paths
            with open(input_file, 'r') as f:
                content = f.read()
            
            # Replace the old directory with the new one in the paths
            content = content.replace(input_dir, output_dir)
            
            with open(output_file, 'w') as f:
                f.write(content)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(input_dir, 'images', split)
        labels_dir = os.path.join(input_dir, 'labels', split)
        
        if not os.path.exists(images_dir):
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {split} split: {len(image_files)} images")
        
        for image_file in tqdm(image_files):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            output_image_path = os.path.join(output_dir, 'images', split, image_file)
            output_label_path = os.path.join(output_dir, 'labels', split, label_file)
            
            process_image_and_label(image_path, label_path, output_image_path, output_label_path)

def main():
    input_dir = 'Dataset/bionano_cell'
    output_dir = 'Dataset/bionano_cellv3'
    
    process_dataset(input_dir, output_dir)
    
    print(f"Dataset processing complete. Modified dataset saved to {output_dir}")

if __name__ == "__main__":
    main()
