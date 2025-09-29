"""
Training script for architectural symbol detection with PyTorch DataLoader integration
"""
import os
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO

from data_utils import create_data_loaders, validate_dataset_structure, generate_training_report, check_and_split_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SymbolDetectionTrainer:
    """Training pipeline for architectural symbol detection with PyTorch DataLoader"""
    
    def __init__(self, config_path: str, data_path: str, raw_data_path: str = None):
        self.config_path = Path(config_path)
        self.data_path = Path(data_path)  # This will be the split data path
        self.raw_data_path = Path(raw_data_path) if raw_data_path else None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available! Make sure your Vast.ai instance has GPU enabled.")

        self.device = torch.device('cuda')
        logger.info(f"Using device: {self.device} - {torch.cuda.get_device_name(0)}")
        
        # Create output directory
        self.output_dir = Path('runs/train') / f"experiment_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and data loaders
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def prepare_data(self, batch_size: int = 16, img_size: int = 640, num_workers: int = 4):
        """Prepare PyTorch DataLoaders with automatic data splitting"""
        logger.info("Preparing PyTorch DataLoaders...")
        
        # Check and split data if needed
        if self.raw_data_path:
            check_and_split_data(str(self.raw_data_path), str(self.data_path))
        
        # Validate dataset structure
        validation_result = validate_dataset_structure(self.data_path)
        if not validation_result['validation_passed']:
            logger.error("Dataset validation failed!")
            for issue in validation_result['issues']:
                logger.error(f"  - {issue}")
            raise ValueError("Dataset validation failed")
        
        logger.info("Dataset validation passed")
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            str(self.data_path), batch_size, img_size, num_workers
        )
        
        logger.info(f"DataLoaders created - Train: {len(self.train_loader)}, "
                   f"Val: {len(self.val_loader)}, Test: {len(self.test_loader)} batches")
        
        # Update config with correct path
        self.config['path'] = str(self.data_path.absolute())
        
        # Save updated config
        config_file = self.output_dir / 'data.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        return str(config_file)
    
    def load_model(self, model_size: str = 'n', pretrained_weights: str = None):
        """Load YOLOv8 model"""
        logger.info(f"Loading YOLOv8{model_size} model...")
        
        if pretrained_weights and Path(pretrained_weights).exists():
            self.model = YOLO(pretrained_weights)
            logger.info(f"Loaded pretrained weights from {pretrained_weights}")
        else:
            self.model = YOLO(f"yolov8{model_size}.pt")
            logger.info(f"Loaded YOLOv8{model_size} base model")
    
    def train(self, epochs: int = 100, batch_size: int = 16, image_size: int = 640, 
              learning_rate: float = 0.01, model_size: str = 'n', 
              pretrained_weights: str = None, num_workers: int = 4, **kwargs):
        """Train the model with PyTorch DataLoader integration"""
        
        # Prepare data loaders
        data_config = self.prepare_data(batch_size, image_size, num_workers)
        
        # Load model
        self.load_model(model_size, pretrained_weights)
        
        # Training parameters
        train_params = {
            'data': data_config,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'lr0': learning_rate,
            'device': self.device,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,
            'cache': False,
            'workers': num_workers,
            'seed': 42
        }
        
        # Add any additional parameters from kwargs
        train_params.update(kwargs)
        
        logger.info("Starting training with PyTorch DataLoader integration...")
        logger.info(f"Training parameters: {train_params}")
        
        # Log DataLoader statistics
        if self.train_loader:
            logger.info(f"Training batches per epoch: {len(self.train_loader)}")
            logger.info(f"Validation batches per epoch: {len(self.val_loader)}")
        
        # Train model
        results = self.model.train(**train_params)
        
        logger.info("Training completed successfully!")
        return results
    
    def evaluate(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """Evaluate model on test set using PyTorch DataLoader"""
        logger.info("Evaluating model on test set...")
        
        if self.model is None:
            best_model = self.output_dir / 'train' / 'weights' / 'best.pt'
            if best_model.exists():
                self.model = YOLO(best_model)
            else:
                logger.error("No trained model found for evaluation")
                return None
        
        # Evaluate on test set
        test_results = self.model.val(
            data=self.output_dir / 'data.yaml',
            split='test',
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True,
            plots=True
        )
        
        logger.info("Evaluation completed")
        return test_results
    
    def export_model(self, formats: list = ['onnx']):
        """Export model to different formats"""
        logger.info("Exporting model...")
        
        if self.model is None:
            best_model = self.output_dir / 'train' / 'weights' / 'best.pt'
            if best_model.exists():
                self.model = YOLO(best_model)
            else:
                logger.error("No trained model found for export")
                return
        
        for fmt in formats:
            try:
                export_path = self.model.export(format=fmt)
                logger.info(f"Model exported to {fmt}: {export_path}")
            except Exception as e:
                logger.error(f"Failed to export to {fmt}: {str(e)}")
    
    def generate_report(self):
        """Generate training report"""
        return generate_training_report(self.output_dir, self.config, self.timestamp)
    
    def inspect_data_loaders(self):
        """Inspect data loaders for debugging"""
        if self.train_loader:
            logger.info("Inspecting training data loader...")
            for batch_idx, (images, targets, paths) in enumerate(self.train_loader):
                logger.info(f"Batch {batch_idx}: Images shape: {images.shape}, "
                           f"Targets shape: {targets.shape}, Paths: {len(paths)}")
                if batch_idx >= 2:  # Only show first 3 batches
                    break


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Symbol Detection Training with PyTorch DataLoader')
    parser.add_argument('--config', type=str, default='config/data.yaml', 
                       help='Path to data configuration file')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to split dataset (will be created if not exists)')
    parser.add_argument('--raw_data_path', type=str, default=None,
                       help='Path to raw dataset (with images/ and labels/ folders) for splitting')
    parser.add_argument('--epochs', type=int, default=150, 
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=9999, help='Early stopping patience (set high to disable)')

    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=640, 
                       help='Input image size')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Initial learning rate')
    parser.add_argument('--model_size', type=str, default='m', 
                       choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLOv8 model size')
    parser.add_argument('--pretrained_weights', type=str, default=None, 
                       help='Path to pretrained weights')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--validate_only', action='store_true', 
                       help='Only validate dataset structure')
    parser.add_argument('--split_only', action='store_true', 
                       help='Only split raw data into train/val/test')
    parser.add_argument('--inspect_data', action='store_true', 
                       help='Inspect data loaders for debugging')
    parser.add_argument('--export_formats', nargs='+', default=['onnx'], 
                       help='Export formats')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SymbolDetectionTrainer(args.config, args.data_path, args.raw_data_path)
    
    # Split data only mode
    if args.split_only:
        logger.info("Split data mode - splitting raw data into train/val/test...")
        if not args.raw_data_path:
            logger.error("--raw_data_path is required for split_only mode")
            return
        
        check_and_split_data(args.raw_data_path, args.data_path)
        logger.info(f"Data split completed! Split data saved to: {args.data_path}")
        return
    
    # Validation only mode
    if args.validate_only:
        logger.info("Validation mode - checking dataset structure only...")
        result = validate_dataset_structure(args.data_path)
        if result['validation_passed']:
            logger.info("Dataset structure validation passed!")
        else:
            logger.error("Dataset structure validation failed!")
            for issue in result['issues']:
                logger.error(f"  - {issue}")
        return
    
    # Data inspection mode
    if args.inspect_data:
        logger.info("Data inspection mode - creating and inspecting data loaders...")
        trainer.prepare_data(args.batch_size, args.image_size, args.num_workers)
        trainer.inspect_data_loaders()
        return
    
    try:
        # Train model
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            model_size=args.model_size,
            pretrained_weights=args.pretrained_weights,
            num_workers=args.num_workers,
            patience=args.patience 
        )
        
        # Evaluate model
        eval_results = trainer.evaluate()
        
        # Export model
        trainer.export_model(args.export_formats)
        
        # Generate report
        report = trainer.generate_report()
        
        # Final summary
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to: {trainer.output_dir}")
        logger.info(f"Best model: {trainer.output_dir / 'train' / 'weights' / 'best.pt'}")
        logger.info(f"Training report: {trainer.output_dir / 'training_report.json'}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()