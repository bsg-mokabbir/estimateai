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
IMAGES_DIR = 'iteration4'
OUTPUT_DIR = 'data_agumentation/drawn_images/images_with_red_legends2'
ANNOTATION_FILE = 'data_agumentation/drawn_images/legend_annotations_with_red_legends2.json'
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

# Add this global dictionary to track detailed statistics
detailed_symbol_stats = {}

def generate_excel_report(symbol_usage, detailed_stats, output_file='symbol_distribution_report.xlsx'):
    """Generate detailed Excel report with rotation and color statistics"""
    
    report_data = []
    sl_no = 1
    
    for symbol_name in sorted(symbol_usage.keys()):
        stats = detailed_stats.get(symbol_name, {
            'normal_0': 0, 'normal_90': 0, 'normal_180': 0, 'normal_270': 0,
            'red_0': 0, 'red_90': 0, 'red_180': 0, 'red_270': 0
        })
        
        total_count = symbol_usage[symbol_name]
        
        report_data.append({
            'SL': sl_no,
            'Symbol Name': symbol_name,
            'Rotate - 0': stats['normal_0'],
            'Rotate 90': stats['normal_90'], 
            'Rotate 180': stats['normal_180'],
            'Rotate 270': stats['normal_270'],
            'Color Augment_red 0': stats['red_0'],
            'Color Augment_red Rotate 90': stats['red_90'],
            'Color Augment_red Rotate 180': stats['red_180'], 
            'Color Augment_red Rotate 270': stats['red_270'],
            'Total Count': total_count
        })
        sl_no += 1
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(report_data)
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"📊 Excel report generated: {output_file}")
    print(f"📈 Total symbols: {len(report_data)}")
    print(f"🎯 Total instances: {sum(symbol_usage.values())}")
    
    return output_file

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

def process_image(image_path, output_path, vgg_annotations, symbol_usage, preloaded_legends=None, detailed_stats=None):
    cad_img = Image.open(image_path).convert('RGBA')
    DIAGRAM_BOX = get_diagram_bbox(image_path)
    diag_w = DIAGRAM_BOX[2] - DIAGRAM_BOX[0]
    diag_h = DIAGRAM_BOX[3] - DIAGRAM_BOX[1]

    # Use preloaded legends if available, otherwise load them (for backward compatibility)
    if preloaded_legends is not None:
        legends = []
        scalefactor = get_scaling_factor(reference_img=REFERENCE_PIL, target_img_data=(diag_w, diag_h))
        for legend_name, legend_img in preloaded_legends:
            resized_patch = resize_patch_keep_aspect(legend_img, scale=scalefactor)
            legends.append((legend_name, resized_patch))
    else:
        #LEGENDS DATA PREPARATION (fallback)
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
    target_instances = random.randint(35, 42)
    selected_symbols = []
    current_instances = 0
    
    for legend_name, legend_img in sorted_legends:
        if current_instances >= target_instances:
            break
        # Random instances per symbol (5-8 times)
        instances_per_symbol = random.randint(5, 8)
        
        # 50% red, 50% normal distribution
        red_count = instances_per_symbol // 2
        normal_count = instances_per_symbol - red_count
        
        # Add red symbols with rotations
        for _ in range(red_count):
            selected_symbols.append((legend_name, legend_img, 'red'))
            current_instances += 1
            if current_instances >= target_instances:
                break
        
        # Add normal symbols with rotations  
        for _ in range(normal_count):
            selected_symbols.append((legend_name, legend_img, 'normal'))
            current_instances += 1
            if current_instances >= target_instances:
                break
    
    placed_boxes = []
    regions = []

    for legend_name, legend_img, color_type in selected_symbols[:target_instances]:
        tries = 0
        max_attempts = 100  # Increased attempts to reduce overlap warnings
        placed = False
        
        while tries < max_attempts:
            # Both red and normal symbols get random rotation
            rotation_angle = random.choice([0, 90, 180, 270])
            img_to_place = legend_img.rotate(rotation_angle, expand=True)
            
            if color_type == 'red':
                # Convert to red color for transparent background symbols
                img_array = np.array(img_to_place)
                if img_array.shape[2] == 4:  # RGBA
                    alpha = img_array[:, :, 3]
                    # Only color non-transparent pixels
                    visible_pixels = alpha > 0
                    img_array[visible_pixels, 0] = 255  # Red
                    img_array[visible_pixels, 1] = 0    # Green  
                    img_array[visible_pixels, 2] = 0    # Blue
                    # Keep alpha channel unchanged
                img_to_place = Image.fromarray(img_array, 'RGBA')
            
            lw, lh = img_to_place.size  
            
            # Better placement strategy with margin
            margin = 5
            if DIAGRAM_BOX[2] - DIAGRAM_BOX[0] > lw + margin and DIAGRAM_BOX[3] - DIAGRAM_BOX[1] > lh + margin:
                x = random.randint(DIAGRAM_BOX[0] + margin, DIAGRAM_BOX[2] - lw - margin)
                y = random.randint(DIAGRAM_BOX[1] + margin, DIAGRAM_BOX[3] - lh - margin)
                new_box = (x, y, lw, lh)
                
                # Check overlap with increased tolerance
                overlap_found = False
                for existing_box in placed_boxes:
                    if is_overlapping(new_box, existing_box):
                        overlap_found = True
                        break
                
                if not overlap_found:
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
                    
                    # Update detailed statistics
                    if detailed_stats is not None:
                        if legend_name not in detailed_stats:
                            detailed_stats[legend_name] = {
                                'normal_0': 0, 'normal_90': 0, 'normal_180': 0, 'normal_270': 0,
                                'red_0': 0, 'red_90': 0, 'red_180': 0, 'red_270': 0
                            }
                        
                        stat_key = f"{color_type}_{rotation_angle}"
                        detailed_stats[legend_name][stat_key] += 1
                    
                    placed = True
                    break
            tries += 1
        
        if not placed:
            # Try smaller size if placement failed
            smaller_img = img_to_place.resize((int(lw*0.8), int(lh*0.8)))
            for _ in range(20):  # Quick attempts with smaller size
                x = random.randint(DIAGRAM_BOX[0], DIAGRAM_BOX[2] - smaller_img.width)
                y = random.randint(DIAGRAM_BOX[1], DIAGRAM_BOX[3] - smaller_img.height)
                new_box = (x, y, smaller_img.width, smaller_img.height)
                
                overlap_found = False
                for existing_box in placed_boxes:
                    if is_overlapping(new_box, existing_box):
                        overlap_found = True
                        break
                
                if not overlap_found:
                    cad_img.alpha_composite(smaller_img, (x, y))
                    placed_boxes.append(new_box)
                    regions.append({
                        'shape_attributes': {
                            'name': 'rect',
                            'x': x,
                            'y': y,
                            'width': smaller_img.width,
                            'height': smaller_img.height
                        },
                        'region_attributes': {
                            'Layout_detect': 'legend',
                            'Text': ''
                        }
                    })
                    symbol_usage[legend_name] = symbol_usage.get(legend_name, 0) + 1
                    
                    # Update detailed statistics  
                    if detailed_stats is not None:
                        if legend_name not in detailed_stats:
                            detailed_stats[legend_name] = {
                                'normal_0': 0, 'normal_90': 0, 'normal_180': 0, 'normal_270': 0,
                                'red_0': 0, 'red_90': 0, 'red_180': 0, 'red_270': 0
                            }
                        
                        stat_key = f"{color_type}_{rotation_angle}"
                        detailed_stats[legend_name][stat_key] += 1
                    
                    placed = True
                    break

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
    detailed_symbol_stats = {}  # Add detailed statistics tracking
    
    # Suppress PIL warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    
    # 🚀 PERFORMANCE FIX: Preload all legends once
    print("📦 Preloading legend images...")
    preloaded_legends = []
    for f in os.listdir(LEGENDS_DIR):
        if f.endswith('.png'):
            patch_img = Image.open(os.path.join(LEGENDS_DIR, f)).convert('RGBA')
            preloaded_legends.append((f, patch_img))
    print(f"✅ Loaded {len(preloaded_legends)} legend templates")
    
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
                
                # Step 2: Apply legend augmentation with detailed stats
                output_path = os.path.join(OUTPUT_DIR, fname)
                process_image(temp_path, output_path, vgg_annotations, symbol_usage, preloaded_legends, detailed_symbol_stats)
                
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
    
    # Generate Excel report
    excel_file = generate_excel_report(symbol_usage, detailed_symbol_stats)
    
    print(f"\n🎉 Pipeline complete!")
    print(f"📁 Final output: {OUTPUT_DIR}")
    print(f"📄 Annotations: {ANNOTATION_FILE}")
    print(f"📊 Excel Report: {excel_file}")

if __name__ == '__main__':
    main()