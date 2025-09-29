"""
Data validation and processing utilities for Electrical Symbol detection 
"""
import os
import json
import cv2
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup logging
logger = logging.getLogger(__name__)


class SymbolDetectionDataset(Dataset):

    """Custom dataset for symbol detection with YOLO format annotations"""
    
    def __init__(self, data_path: str, split: str = 'train', transform=None, img_size: int = 640):
        self.data_path = Path(data_path)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # Get image and label paths
        self.images_dir = self.data_path / split / 'images'
        self.labels_dir = self.data_path / split / 'labels'
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        # Filter files that have corresponding labels
        self.valid_files = []
        for img_file in self.image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                self.valid_files.append(img_file)
        
        logger.info(f"Found {len(self.valid_files)} valid {split} samples")
        
        # Setup transforms if not provided
        if self.transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                    A.OneOf([
                        A.RandomScale(scale_limit=(-0.4, -0.1), p=0.7),  # Zoom out 
                        A.RandomScale(scale_limit=(0.1, 0.3), p=0.3),   # Zoom in 
                    ], p=0.8),
                    A.OneOf([
                        A.GaussNoise(var_limit=(5, 20), p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
                        A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.2),
                    ], p=0.4),
                    A.OneOf([
                        A.Blur(blur_limit=3, p=0.3),
                        A.MotionBlur(blur_limit=3, p=0.2),
                    ], p=0.3),
                    A.ElasticTransform(alpha=1, sigma=5, p=0.2),
                    A.GridDistortion(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            else:
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.valid_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            boxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
        
        # Apply transforms
        if self.transform and boxes:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        elif self.transform:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = []
            class_labels = []
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert to tensor format
        if boxes:
            targets = torch.zeros(len(boxes), 6)  # batch_idx, class_id, x, y, w, h
            for i, (bbox, cls) in enumerate(zip(boxes, class_labels)):
                targets[i] = torch.tensor([0, cls, bbox[0], bbox[1], bbox[2], bbox[3]])
        else:
            targets = torch.zeros(0, 6)
        
        return image, targets, str(img_path)


def create_data_loaders(data_path: str, batch_size: int = 16, img_size: int = 640, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test sets"""
    
    # Create datasets
    train_dataset = SymbolDetectionDataset(data_path, 'train', img_size=img_size)
    val_dataset = SymbolDetectionDataset(data_path, 'val', img_size=img_size)
    test_dataset = SymbolDetectionDataset(data_path, 'test', img_size=img_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function for batching variable number of objects"""
    images, targets, paths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Process targets
    for i, target in enumerate(targets):
        target[:, 0] = i  # Set batch index
    
    targets = torch.cat(targets, 0)
    
    return images, targets, paths


class DataSplitter:
    """Split dataset into train/val/test sets"""
    
    def __init__(self, data_path: str, output_path: str, split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.split_ratio = split_ratio
        
        # Ensure split ratios sum to 1
        if abs(sum(split_ratio) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratio)}")
    
    def split_data(self):
        """Split data into train/val/test sets"""
        logger.info(f"Splitting dataset with ratio {self.split_ratio}...")
        
        # Get all image files from raw data
        image_files = []
        images_dir = self.data_path / 'images'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(list(images_dir.glob(ext)))
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        logger.info(f"Found {len(image_files)} images to split")
        
        # Shuffle for random distribution
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_end = int(total * self.split_ratio[0])
        val_end = train_end + int(total * self.split_ratio[1])
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Create output directories
        for split in splits.keys():
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy files
        import shutil
        for split, files in splits.items():
            logger.info(f"Creating {split} set with {len(files)} files")
            
            copied_images = 0
            copied_labels = 0
            
            for img_file in files:
                try:
                    # Copy image
                    dest_img = self.output_path / split / 'images' / img_file.name
                    shutil.copy2(img_file, dest_img)
                    copied_images += 1
                    
                    # Copy corresponding label
                    label_file = self.data_path / 'labels' / f"{img_file.stem}.txt"
                    if label_file.exists():
                        dest_label = self.output_path / split / 'labels' / f"{img_file.stem}.txt"
                        shutil.copy2(label_file, dest_label)
                        copied_labels += 1
                    else:
                        logger.warning(f"Label file not found for {img_file.name}")
                        
                except Exception as e:
                    logger.error(f"Error copying {img_file.name}: {str(e)}")
            
            logger.info(f"{split}: {copied_images} images, {copied_labels} labels copied")
        
        logger.info("Dataset split completed")
        return splits


def validate_dataset_structure(data_path: str) -> Dict:
    """Quick validation of dataset structure"""
    data_path = Path(data_path)
    issues = []
    
    # Check split directories
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            issues.append(f"Missing {split} directory")
            continue
            
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            issues.append(f"Missing {split}/images directory")
        if not labels_dir.exists():
            issues.append(f"Missing {split}/labels directory")
        
        # Count files
        if images_dir.exists() and labels_dir.exists():
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(list(images_dir.glob(ext)))
            
            label_files = list(labels_dir.glob('*.txt'))
            
            logger.info(f"{split}: {len(image_files)} images, {len(label_files)} labels")
    
    return {
        'validation_passed': len(issues) == 0,
        'issues': issues
    }


def check_and_split_data(raw_data_path: str, split_data_path: str) -> bool:
    """Check if split data exists, if not create it from raw data"""
    split_path = Path(split_data_path)
    raw_path = Path(raw_data_path)
    
    # Check if split directories exist and have data
    split_exists = True
    for split in ['train', 'val', 'test']:
        split_dir = split_path / split / 'images'
        if not split_dir.exists() or not any(split_dir.iterdir()):
            split_exists = False
            break
    
    if not split_exists:
        logger.info("Split data not found or empty. Creating from raw data...")
        
        # Check if raw data exists
        if not (raw_path / 'images').exists() or not (raw_path / 'labels').exists():
            raise ValueError(f"Raw data not found at {raw_path}. Expected 'images' and 'labels' folders.")
        
        # Create split
        splitter = DataSplitter(str(raw_path), str(split_path))
        splitter.split_data()
        logger.info("Data split completed!")
        return True
    
    logger.info("Using existing split data")
    return False

def generate_training_report(output_dir: Path, config: Dict, timestamp: str):
    """Generate comprehensive training report"""
    logger.info("Generating training report...")
    
    train_dir = output_dir / 'train'
    
    # Create detailed report for current run
    report_data = {
        'run_info': {
            'timestamp': timestamp,
            'config': config,
            'output_dir': str(output_dir)
        },
        'training_summary': {},
        'model_info': {},
        'performance_metrics': {}
    }
    
    # Read training results if available
    results_csv = train_dir / 'results.csv'
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            
            # Comprehensive metrics summary
            if len(df) > 0:
                final_row = df.iloc[-1]
                best_map50_idx = df['metrics/mAP50(B)'].idxmax() if 'metrics/mAP50(B)' in df.columns else 0
                best_row = df.iloc[best_map50_idx]
                
                report_data['training_summary'] = {
                    'total_epochs': int(final_row['epoch']) if 'epoch' in final_row else 0,
                    'best_epoch': int(best_row['epoch']) if 'epoch' in best_row else 0,
                    'final_metrics': {
                        'mAP50': float(final_row.get('metrics/mAP50(B)', 0)),
                        'mAP50_95': float(final_row.get('metrics/mAP50-95(B)', 0)),
                        'precision': float(final_row.get('metrics/precision(B)', 0)),
                        'recall': float(final_row.get('metrics/recall(B)', 0)),
                        'box_loss': float(final_row.get('val/box_loss', 0)),
                        'cls_loss': float(final_row.get('val/cls_loss', 0))
                    },
                    'best_metrics': {
                        'mAP50': float(best_row.get('metrics/mAP50(B)', 0)),
                        'mAP50_95': float(best_row.get('metrics/mAP50-95(B)', 0)),
                        'precision': float(best_row.get('metrics/precision(B)', 0)),
                        'recall': float(best_row.get('metrics/recall(B)', 0))
                    }
                }
            
            # Store dataset info
            if 'path' in config:
                data_path = Path(config['path'])
                train_images = len(list((data_path / 'train' / 'images').glob('*'))) if (data_path / 'train' / 'images').exists() else 0
                val_images = len(list((data_path / 'val' / 'images').glob('*'))) if (data_path / 'val' / 'images').exists() else 0
                test_images = len(list((data_path / 'test' / 'images').glob('*'))) if (data_path / 'test' / 'images').exists() else 0
                
                report_data['dataset_info'] = {
                    'train_images': train_images,
                    'val_images': val_images,
                    'test_images': test_images,
                    'total_images': train_images + val_images + test_images,
                    'split_ratio': f"{train_images}:{val_images}:{test_images}"
                }
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plots
            if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
                axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
                axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
                axes[0, 1].set_title('Classification Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Metrics plots
            if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:
                axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
                axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
                axes[1, 1].set_title('mAP Metrics')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('mAP')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = output_dir / 'training_metrics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Training metrics plot saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error generating training plots: {str(e)}")
    
    # Check model files
    best_model = train_dir / 'weights' / 'best.pt'
    last_model = train_dir / 'weights' / 'last.pt'
    
    report_data['model_info'] = {
        'best_model_path': str(best_model),
        'last_model_path': str(last_model),
        'best_exists': best_model.exists(),
        'last_exists': last_model.exists(),
        'model_size': os.path.getsize(best_model) if best_model.exists() else 0
    }
    
    # Save detailed report
    report_path = output_dir / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Create summary text file for easy reading
    summary_path = output_dir / 'TRAINING_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {report_data.get('dataset_info', {}).get('total_images', 'N/A')} images\n")
        f.write(f"Training Images: {report_data.get('dataset_info', {}).get('train_images', 'N/A')}\n")
        f.write(f"Validation Images: {report_data.get('dataset_info', {}).get('val_images', 'N/A')}\n\n")
        
        if 'training_summary' in report_data:
            summary = report_data['training_summary']
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Epochs: {summary.get('total_epochs', 'N/A')}\n")
            f.write(f"Best Epoch: {summary.get('best_epoch', 'N/A')}\n\n")
            
            best_metrics = summary.get('best_metrics', {})
            f.write("BEST METRICS:\n")
            f.write(f"mAP@0.5: {best_metrics.get('mAP50', 0):.4f}\n")
            f.write(f"mAP@0.5:0.95: {best_metrics.get('mAP50_95', 0):.4f}\n")
            f.write(f"Precision: {best_metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {best_metrics.get('recall', 0):.4f}\n\n")
            
            final_metrics = summary.get('final_metrics', {})
            f.write("FINAL METRICS:\n")
            f.write(f"mAP@0.5: {final_metrics.get('mAP50', 0):.4f}\n")
            f.write(f"mAP@0.5:0.95: {final_metrics.get('mAP50_95', 0):.4f}\n")
            f.write(f"Precision: {final_metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {final_metrics.get('recall', 0):.4f}\n")
    
    logger.info(f"Training report saved to {report_path}")
    logger.info(f"Training summary saved to {summary_path}")
    
    return report_data