import os
import json
import re

# === Config ===
input_folder = r"Output\Project 1\annotation"
output_json_path = r"Output\Project 1\annotation\project_confidance_filter.json"

# === Init ===
final_data = {}

def parse_bbox(bbox_str):
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", bbox_str)))
    x1, y1, x2, y2 = nums
    x = int(x1)
    y = int(y1)
    width = int(x2 - x1)
    height = int(y2 - y1)
    return x, y, width, height

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()

        image_name = None
        actual_size = 0
        regions = []

        current_class = None
        current_bbox = None
        current_confidence = None

        for line in lines:
            line = line.strip()
            if line.startswith("Image:"):
                image_name = line.split("Image:")[1].strip()
                image_path = os.path.join(input_folder, image_name)
                if os.path.exists(image_path):
                    actual_size = os.path.getsize(image_path)
                else:
                    print(f"⚠️ Image not found: {image_path}")
            elif line.startswith("Detection"):
                # Reset per detection
                current_class = None
                current_bbox = None
                current_confidence = None
            elif line.startswith("Class:"):
                current_class = line.split("Class:")[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    current_confidence = float(line.split("Confidence:")[1].strip())
                except Exception as e:
                    current_confidence = None
            elif line.startswith("Bbox:"):
                current_bbox = line.split("Bbox:")[1].strip()
                # Only add if confidence is >= 0.5
                if current_confidence is not None and current_confidence >= 0.5:
                    x, y, w, h = parse_bbox(current_bbox)

                    region = {
                        "shape_attributes": {
                            "name": "rect",
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h
                        },
                        "region_attributes": {
                            "Layout_detect": current_class
                        }
                    }
                    regions.append(region)

        if image_name and actual_size > 0:
            key = f"{image_name}{actual_size}"
            final_data[key] = {
                "filename": image_name,
                "size": actual_size,
                "regions": regions
            }

# === Save to JSON ===
with open(output_json_path, "w", encoding="utf-8") as out_file:
    json.dump(final_data, out_file, indent=4)

print(f"✅ Done! Output saved to {output_json_path}")
