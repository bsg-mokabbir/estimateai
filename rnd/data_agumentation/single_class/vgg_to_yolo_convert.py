import os
import json
from PIL import Image

# Paths
input_json_path = "data_agumentation/drawn_images_with_red_legends/legend_annotations_with_red_legends2.json"
image_folder = "data_agumentation/drawn_images_with_red_legends/images"         # Folder where images are stored
output_dir = "data_agumentation/drawn_images_with_red_legends/labels"      # Where YOLO .txt files will be saved

# Class name to ID mapping
class_mapping = {
    "legend": 0
}

os.makedirs(output_dir, exist_ok=True)

with open(input_json_path, 'r') as f:
    data = json.load(f)

for key, value in data.items():
    filename = value['filename']
    image_path = os.path.join(image_folder, filename)

    # Load image size using PIL
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        continue

    yolo_lines = []

    for region in value['regions']:
        shape = region['shape_attributes']
        attr = region['region_attributes']

        if shape['name'] != 'rect':
            continue

        x = shape['x']
        y = shape['y']
        w = shape['width']
        h = shape['height']

        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        norm_w = w / width
        norm_h = h / height

        class_name = attr.get('Layout_detect')
        class_id = class_mapping.get(class_name, -1)

        if class_id == -1:
            print(f"Unknown class label: {class_name}")
            continue

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    # Save YOLO label file
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, base_name + ".txt")
    with open(output_path, "w") as out_f:
        out_f.write("\n".join(yolo_lines))

print("Vgg to yolo annotation converion completed")
