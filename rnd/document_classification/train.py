import os
import json
from PIL import Image
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

os.environ["WANDB_DISABLED"] = "true"

# Dataset class
class DocumentDataset(Dataset):
    def __init__(self, data_dir, processor, is_train=True):
        self.data_dir = data_dir
        self.processor = processor
        self.is_train = is_train
        
        # Load data
        self.data = []
        self.labels = []
        self.label_to_id = {"Legend": 0, "Circuit": 1, "AutoCAD": 2, "Others": 3}
        
        for label_name, label_id in self.label_to_id.items():
            label_dir = os.path.join(data_dir, label_name)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append(os.path.join(label_dir, img_name))
                        self.labels.append(label_id)
        
        # Data augmentation for training
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224))
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Process with model processor
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Model setup
    model_name = "microsoft/dit-base-finetuned-rvlcdip"
    
    print("Loading model and processor...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    
    # Dataset paths
    train_dir = "dataset/train"
    val_dir = "dataset/val"
    
    print("Loading datasets...")
    train_dataset = DocumentDataset(train_dir, processor, is_train=True)
    val_dataset = DocumentDataset(val_dir, processor, is_train=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # GPU optimized training arguments
    with open("config.json", "r") as f:
        config = json.load(f)
    training_args = TrainingArguments(**config)
    
    # Early stopping callback - more aggressive
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    
    # Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],  # Add early stopping callback
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    model.save_pretrained("./fine_tuned_dit")
    processor.save_pretrained("./fine_tuned_dit")
    
    # Final evaluation
    print("Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()
