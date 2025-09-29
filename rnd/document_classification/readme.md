# Document Classification with Vision Transformer

A fine-tuned document classifier that categorizes technical documents into three classes: Legend, Circuit, and AutoCAD using Microsoft's DiT (Document Image Transformer) model.

## Quick Setup

1. **Download Dataset:**
   ```bash
   # Download from Google Drive
   https://drive.google.com/drive/folders/16AdadXlGSuYUC02-1cAMklBFU5io0Dsb?usp=sharing
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure
```
dataset/
├── train/
│   ├── Legend/
│   ├── Circuit/
│   └── AutoCAD/
└── val/
    ├── Legend/
    ├── Circuit/
    └── AutoCAD/
```

## Usage

**Training:**
```bash
python train.py
```

**Inference:**
```bash
python inference.py
```

## Features
- GPU accelerated training
- Data augmentation
- Early stopping
- Batch & single image prediction
