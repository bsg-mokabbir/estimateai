import cv2
import numpy as np
import os

# === Root folder where subfolders exist ===
root_dir = "legend_symbols"

# === Loop over each subfolder inside root_dir ===
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    # Skip if not a folder
    if not os.path.isdir(subfolder_path):
        continue

    # Output folder name
    output_folder = f"{subfolder}_bg_remove"
    output_path = os.path.join(root_dir, output_folder)

    # Create output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process each image in the subfolder
    for file in os.listdir(subfolder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_img_path = os.path.join(subfolder_path, file)
            output_img_path = os.path.join(output_path, file)

            # Load image
            image = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)

            # Skip if failed to read
            if image is None:
                print(f"❌ Could not read {input_img_path}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Threshold for mask
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

            # Split channels
            channels = cv2.split(image)
            if len(channels) == 3:
                b, g, r = channels
            elif len(channels) == 4:
                b, g, r, _ = channels  # discard original alpha

            # Use mask as alpha
            alpha = thresh
            rgba = cv2.merge((b, g, r, alpha))

            # Save transparent image
            cv2.imwrite(output_img_path, rgba)
            print(f"✅ Saved: {output_img_path}")
