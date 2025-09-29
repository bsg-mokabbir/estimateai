# Electrical Legend Count Detection

YOLOv8 based electrical symbol detection and counting system for architectural drawings with PDF processing capabilities.

## Quick Start

### 1. Download Pre-trained Models
Download the required models from [Google Drive](https://drive.google.com/drive/folders/1b-XSuxp9F1i46TKnU-Dn_ak9EwZcquRo?usp=sharing):

- **PDF Classifier Model**: From "pdf classifier" folder 
- **YOLO Detection Model**: From "detection model" folder 
- **Legend Samples**: From "Row_Legend" folder

### 2. Update Model Paths
Configure the model paths in `development\streamlit_app\main_app.py` according to your downloaded model locations.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run development\streamlit_app\main_app.py
```

## Project Structure

```
📂 LEGEND-COUNT-DETECTION
├── 📂 rnd/
│   ├── 📂 data_agumentation/
│   │   ├── 📂 images/
│   │   ├── 📂 drawn_images/
│   │   ├── 📂 sample_legends_finalized/
│   │   └── 📂 without_takeoff_images/
│   │       ├── 📂 multiple_class/
│   │       └── 📂 single_class/
│   │
│   ├── 📂 layout_detection/
│   │   ├── 📂 config/
│   │   ├── 📂 data/
│   │   ├── 📂 inference/
│   │   │   └── 📂 streamlit/
│   │   ├── train.py
│   │   └── data_utils.py
│   │
│   ├── 📂 document_classification/
│   │   ├── drawing_index_extractor.py
│   │   └── inference_pdf_classifier.py
│   ├── 📂 cable_length/
│   │   ├── cable_size.py
│   │
│   └── 📂 streamlit_app/
│       ├── main_app.py (main entry point)
│       ├── tab1_pdf_processing.py
│       ├── tab2_legend_detection.py
│       └── tab3_symbol_counting.py
│       └──tab4_cable_length_with_cable_size.py.py
│
├── 📂 development/
│   ├── 📂 layout_detection/
│   │   ├── inference_layout_detector.py
│   │   └── matching_symbol.py
│   │
│   ├── 📂 document_classification/
│   │   ├── drawing_index_extractor.py
│   │   └── inference_pdf_classifier.py
│   ├── 📂 cable_length/
│   │   ├── cable_size.py
│   │
│   └── 📂 streamlit_app/
│       ├── main_app.py (main entry point)
│       ├── tab1_pdf_processing.py
│       ├── tab2_legend_detection.py
│       └── tab3_symbol_counting.py
│       └──tab4_cable_length_with_cable_size.py.py
│
│
└── 📂 deployment/
```

## Key Features

- **PDF Processing**: Extract and classify architectural drawing pages
- **Legend Detection**: Identify electrical symbols using YOLOv8
- **Symbol Counting**: Automated counting of detected electrical symbols
- **Interactive UI**: Streamlit-based web interface with multiple tabs
- **Data Augmentation**: Tools for expanding training datasets
- **Format Conversion**: VGG to YOLO annotation format conversion

## Development Workflow

### Data Preparation
```bash
# For multiple classes
python rnd\data_agumentation\multiple_class\place_legends_on_cad.py

# For single class
python rnd\data_agumentation\single_class\place_legends_on_cad_black_red_ratio50.py

# Convert annotations
python rnd\data_agumentation\single_class\vgg_to_yolo_convert.py
```

### Model Training
```bash
# Split dataset
python rnd\layout_detection\train.py --raw_data_path data_agumentation/drawn_images --data_path data --split_only

# Train model
python rnd\layout_detection\train.py --data_path data --epochs 150
```

### Inference
```bash
python rnd\layout_detection\inference\inference.py
python rnd\layout_detection\inference\matching_symbol.py
```

## Output Capabilities

- Electrical symbol detection and localization
- Symbol count statistics
- PDF page classification
- Drawing index extraction
- Interactive visualization of results

## Requirements

- Python 3.8+
- YOLOv8
- Streamlit
- OpenCV
- PyTorch
- Additional dependencies listed in `requirements.txt`