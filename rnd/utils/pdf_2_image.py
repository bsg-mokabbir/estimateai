from pdf2image import convert_from_path
import os

# === Config ===
pdf_path = '/Users/nuhash/Documents/Blockstak/cad/legent-count-detection/data_agumentation/pdfs/133 VICTORIA PARADE, FITZROY, VIC 3065 Rev. 1 - without takeoffs.pdf'
output_folder = 'output_images'
dpi_setting = 300  # Lower DPI for faster conversion
poppler_path = None  # Set only if you're on Windows

# Optional: Set page range to process (e.g., pages 1-3)
first_page = 6
last_page = 10

os.makedirs(output_folder, exist_ok=True)

images = convert_from_path(
    pdf_path,
    dpi=dpi_setting,
    poppler_path=poppler_path,
    first_page=first_page,
    last_page=last_page
)

for i, img in enumerate(images):
    filename = os.path.join(output_folder, f'page_{i+first_page:03d}.png')
    img.save(filename, 'PNG')

print(f"Saved {len(images)} pages to {output_folder}")
