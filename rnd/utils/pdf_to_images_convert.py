from pdf2image import convert_from_path
import os

# === Config ===
input_folder = 'input_folder_path'         # Folder where multiple PDFs are located
output_folder = 'Output_path'      # Where to save the output images
dpi_setting = 300             # Set DPI for image quality
poppler_path = None           # Set this if you're on Windows

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === Process each PDF file ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]  # Filename without .pdf

        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi_setting,
                poppler_path=poppler_path
            )

            for i, img in enumerate(images):
                img_filename = os.path.join(
                    output_folder,
                    f'{base_name}_page_{i+1:03d}.png'
                )
                img.save(img_filename, 'PNG')
            print(f"Saved {len(images)} pages from {filename}")
        
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
