"""
Combined script that first cleans images by removing ICC profiles and converting to proper RGB,
then places legends on all CAD images using universal aspect ratios and proportional sizing.

Usage:
    python combined_image_processing.py

Legend aspect ratios (width/height):
  surface_mount_led_motionsensor.png: 0.354
  surface_mount_led_batten_luminer.png: 3.278
  surface_mount_led_downlight.png: 0.972
  exit_sign.png: 1.646
  lighting_motionsensor.png: 0.941
"""

import os
import random
import argparse
import json
import warnings
from PIL import Image
import numpy as np
import pandas as pd
import cv2

# Configuration
LEGENDS_DIR = 'data_agumentation/sample_legends_finalized'
IMAGES_DIR = 'data_agumentation/images'
OUTPUT_DIR = 'data_agumentation/drawn_images/images'
ANNOTATION_FILE = 'data_agumentation/drawn_images/legend_annotations_signle_class.json'
REFERENCE_IMAGE_PATH = "data_agumentation/without_takeoff_images/Diagram.png"
REFERENCE_PIL = Image.open(REFERENCE_IMAGE_PATH).convert('RGBA')
REFERENCE_IMG_HEIGHT = 3840
REFERENCE_IMG_WIDTH = 3400 # Best suited

def fix_image_profiles(input_dir, output_dir):
    """Remove ICC profiles and convert to proper RGB without warnings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Suppress PIL warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    
    processed_count = 0
    error_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            
            try:
                # Method 1: OpenCV load (ICC profile ignore)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"⚠️  Could not load: {filename}")
                    error_count += 1
                    continue
                
                # RGB convert 
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # PIL save 
                pil_img = Image.fromarray(img_rgb)
                
                # RGB mode ensure 
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                output_path = os.path.join(output_dir, filename)
                
                # Save without ICC profile
                pil_img.save(output_path, quality=95, optimize=True)
                processed_count += 1
                
                print(f"✅ Processed: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                error_count += 1
                
                # Alternative method: Try with PIL directly
                try:
                    with Image.open(img_path) as img:
                        # Remove ICC profile
                        if 'icc_profile' in img.info:
                            del img.info['icc_profile']
                        
                        # Convert to RGB
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        output_path = os.path.join(output_dir, filename)
                        img.save(output_path, quality=95, optimize=True)
                        processed_count += 1
                        print(f"✅ Processed (PIL method): {filename}")
                        
                except Exception as e2:
                    print(f"❌ Failed both methods for {filename}: {e2}")
    
    print(f"\n🎉 Image cleaning complete!")
    print(f"📊 Processed: {processed_count} images")
    print(f"❌ Errors: {error_count} images")
    print(f"📁 Clean images saved to: {output_dir}")

def get_diagram_bbox(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, img.shape[1], img.shape[0])
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)

def is_overlapping(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def get_scaling_factor(reference_img, target_img_data):
    tgt_w, _ = target_img_data
    scale = tgt_w / REFERENCE_IMG_WIDTH

    print("scale", scale,"tgt_w:",tgt_w,"REFERENCE_IMG_WIDTH",REFERENCE_IMG_WIDTH )
    return scale

def resize_patch_keep_aspect(patch, scale):
    orig_w, orig_h = patch.size
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    return patch.resize((new_w, new_h))

def process_image(image_path, output_path, vgg_annotations, symbol_usage):
    cad_img = Image.open(image_path).convert('RGBA')
    DIAGRAM_BOX = get_diagram_bbox(image_path)
    diag_w = DIAGRAM_BOX[2] - DIAGRAM_BOX[0]
    diag_h = DIAGRAM_BOX[3] - DIAGRAM_BOX[1]

    #LEGENDS DATA PREPARATION
    legends = []
    for f in os.listdir(LEGENDS_DIR):
        if f.endswith('.png'):
            reference_image = Image.open(REFERENCE_IMAGE_PATH).convert('RGBA')
            patch_img = Image.open(os.path.join(LEGENDS_DIR, f)).convert('RGBA')
            scalefactor =  get_scaling_factor(reference_img=reference_image, target_img_data=(diag_w, diag_h) )
            resized_patch = resize_patch_keep_aspect(patch_img, scale=scalefactor)
            legends.append((f, resized_patch))

    # Sort symbols by usage count (least used first) for balanced distribution
    sorted_legends = sorted(legends, key=lambda x: symbol_usage.get(x[0], 0))
    
    # Select symbols to place 30+ instances per image
    target_instances = random.randint(30, 35)
    selected_symbols = []
    current_instances = 0
    
    for legend_name, legend_img in sorted_legends:
        if current_instances >= target_instances:
            break
        # Add all 4 rotations (0°, 90°, 180°, 270°) for each symbol
        selected_symbols.extend([(legend_name, legend_img)] * 4)
        current_instances += 4
    
    placed_boxes = []
    regions = []

    for legend_name, legend_img in selected_symbols[:target_instances]:
        tries = 0
        while tries < 100:
            # Each symbol gets one of the 4 rotations (0°, 90°, 180°, 270°)
            rotation_angle = random.choice([0, 90, 180, 270])
            img_to_place = legend_img.rotate(rotation_angle, expand=True)
            
            lw, lh = img_to_place.size  
            x = random.randint(DIAGRAM_BOX[0], DIAGRAM_BOX[2] - lw)
            y = random.randint(DIAGRAM_BOX[1], DIAGRAM_BOX[3] - lh)
            new_box = (x, y, lw, lh)
            if all(not is_overlapping(new_box, b) for b in placed_boxes):
                cad_img.alpha_composite(img_to_place, (x, y))  
                placed_boxes.append(new_box)
                regions.append({
                    'shape_attributes': {
                        'name': 'rect',
                        'x': x,
                        'y': y,
                        'width': lw,
                        'height': lh
                    },
                    'region_attributes': {
                        'Layout_detect': 'legend',  # Single class for all symbols
                        'Text': ''
                    }
                })
                # Update usage tracking
                symbol_usage[legend_name] = symbol_usage.get(legend_name, 0) + 1
                break
            tries += 1
        else:
            print(f"Warning: Could not place {legend_name} without overlap in {os.path.basename(image_path)}.")

    cad_img.convert('RGB').save(output_path)
    img_size_bytes = os.path.getsize(output_path)
    key = os.path.basename(output_path) + str(img_size_bytes)
    vgg_annotations[key] = {
        'filename': os.path.basename(output_path),
        'size': img_size_bytes,
        'regions': regions,
        'file_attributes': {}
    }
    print(f"Augmented image saved to {output_path}")

def main():
    print("🚀 Starting combined image processing pipeline...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vgg_annotations = {}
    
    # Initialize symbol usage tracking
    symbol_usage = {}
    
    # Suppress PIL warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    
    for fname in os.listdir(IMAGES_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(IMAGES_DIR, fname)
            
            try:
                # Step 1: Clean image (in memory)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️  Could not load: {fname}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # Save temporarily for diagram bbox detection
                temp_path = f"temp_{fname}"
                pil_img.save(temp_path, quality=95, optimize=True)
                
                # Step 2: Apply legend augmentation
                output_path = os.path.join(OUTPUT_DIR, fname)
                process_image(temp_path, output_path, vgg_annotations, symbol_usage)
                
                # Clean up temp file
                os.remove(temp_path)
                
                print(f"✅ Processed: {fname}")
                
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")
    
    with open(ANNOTATION_FILE, 'w') as f:
        json.dump(vgg_annotations, f, indent=2)
    
    # Print symbol usage statistics
    print(f"\n📊 Symbol Distribution Report:")
    for symbol, count in sorted(symbol_usage.items()):
        print(f"{symbol}: {count} instances")
    
    total_instances = sum(symbol_usage.values())
    print(f"\n🎯 Total Instances: {total_instances}")
    print(f"📈 Average per Symbol: {total_instances/len(symbol_usage):.1f}")
    
    print(f"\n🎉 Pipeline complete!")
    print(f"📁 Final output: {OUTPUT_DIR}")
    print(f"📄 Annotations: {ANNOTATION_FILE}")

if __name__ == '__main__':
    main()