import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import subprocess
import sys
import os
from pathlib import Path
from dataset import CTScanDataset

def ensure_data_downloaded():
    """Check if data exists, if not, download it using download_lidc.py"""
    data_dir = Path("data/lidc/processed")
    if data_dir.exists() and any(data_dir.iterdir()):
        print("Data already exists. Skipping download.")
        return True
    
    print("Data not found. Starting download...")
    try:
        # Import the download function directly
        from download_lidc import main as download_data
        download_data()
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please run download_lidc.py manually to download the data.")
        return False

# Configuration
DATA_DIR = Path("data/lidc/processed")
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
DATA_DIR = Path("data/lidc/processed")
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    print("\n" + "=" * 50)
    print("DATA LOADER CREATION")
    print("=" * 50)
    
    # Check data directory first
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        return None, None
        
    series_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(series_dirs)} series directories in {DATA_DIR}")
    
    if len(series_dirs) == 0:
        print("ERROR: No series directories found!")
        return None, None
    
    # Check for DICOM and NPY files
    print("\nSampling directory contents:")
    dicom_count = 0
    npy_count = 0
    for i, series_dir in enumerate(series_dirs[:5]):
        dcm_files = list(series_dir.glob("*.dcm"))
        npy_files = list(series_dir.glob("*.npy"))
        dicom_count += len(dcm_files)
        npy_count += len(npy_files)
        print(f"  {series_dir.name}: {len(dcm_files)} DICOM files, {len(npy_files)} NPY files")
    
    print(f"\nTotal from sample: {dicom_count} DICOM files, {npy_count} NPY files")
        
    # Data augmentation and normalization for training
    print("\nCreating data transforms...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),  # Simplified for diagnostics
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),  # Simplified for diagnostics
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets with detailed error handling
    print("\nCreating training dataset...")
    try:
        train_dataset = CTScanDataset(DATA_DIR, transform=data_transforms['train'], split='train')
        print(f"Training dataset created with {len(train_dataset)} samples")
        
        # Try to access a few samples to verify dataset is working
        if len(train_dataset) > 0:
            print("Verifying sample access from training dataset:")
            for i in range(min(3, len(train_dataset))):
                try:
                    img, label = train_dataset[i]
                    print(f"  Sample {i}: shape={img.shape if hasattr(img, 'shape') else 'Unknown'}, label={label}")
                except Exception as e:
                    print(f"  ERROR accessing sample {i}: {e}")
    except Exception as e:
        print(f"ERROR creating training dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    print("\nCreating validation dataset...")
    try:
        val_dataset = CTScanDataset(DATA_DIR, transform=data_transforms['val'], split='val')
        print(f"Validation dataset created with {len(val_dataset)} samples")
    except Exception as e:
        print(f"ERROR creating validation dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nERROR: One or both datasets are empty!")
        return None, None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                epoch: int) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.3f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate the model on the given dataloader.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        
    Returns:
        Tuple of (average loss, accuracy, all predictions, all labels)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray, class_names: Optional[List[str]] = None) -> None:
    """Plot confusion matrix.
    
    Args:
        true_labels: Array of true labels
        pred_labels: Array of predicted labels
        class_names: List of class names for display
    """
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def train_and_evaluate() -> None:
    """Main training and evaluation loop with comprehensive error handling."""
    print("\n" + "=" * 50)
    print("TRAINING AND EVALUATION")
    print("=" * 50)
    
    # Get data loaders
    try:
        print("Getting data loaders...")
        train_loader, val_loader = get_data_loaders()
        if train_loader is None or val_loader is None:
            print("ERROR: Failed to create data loaders. Exiting.")
            return
        print(f"Successfully created data loaders: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    except Exception as e:
        print(f"ERROR during data loader creation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create model
    try:
        print("\nCreating model...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # Binary classification: Normal vs Abnormal
        model = model.to(device)
        print(f"Model created: {model.__class__.__name__} on {device}")
    except Exception as e:
        print(f"ERROR creating model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Limit number of epochs for testing
    actual_epochs = min(3, NUM_EPOCHS)  # Use just 3 epochs for testing
    print(f"\nWill train for {actual_epochs} epochs (limited for testing)")
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    
    try:
        for epoch in range(actual_epochs):
            try:
                # Train for one epoch
                print(f"\nEpoch {epoch+1}/{actual_epochs}:")
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
                
                # Evaluate on validation set
                val_loss, val_acc = validate(model, val_loader, criterion)
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), "best_model.pth")
                    print(f"New best model saved with accuracy: {best_acc:.2f}%")
            except Exception as e:
                print(f"ERROR during epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next epoch
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTraining completed. Best validation accuracy: {best_acc:.2f}%")
    
    # Final evaluation
    try:
        print("\nEvaluating model on validation set...")
        if not Path("best_model.pth").exists():
            print("WARNING: best_model.pth not found, using current model state")
        else:
            try:
                model.load_state_dict(torch.load("best_model.pth"))
                print("Loaded best model weights")
            except Exception as e:
                print(f"ERROR loading best model: {e}")
                print("Continuing with current model state")
            
        val_loss, val_acc, all_preds, all_labels = evaluate_model(model, val_loader, criterion)
        
        print(f"\nFINAL MODEL PERFORMANCE:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Print more detailed metrics
        print("\nClassification Report:")
        try:
            from sklearn.metrics import classification_report
            print(classification_report(all_labels, all_preds, target_names=['Normal', 'Abnormal']))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            
        # Compute and print simple metrics manually
        print("\nManual Metrics Calculation:")
        true_positives = sum((pred == 1 and label == 1) for pred, label in zip(all_preds, all_labels))
        true_negatives = sum((pred == 0 and label == 0) for pred, label in zip(all_preds, all_labels))
        false_positives = sum((pred == 1 and label == 0) for pred, label in zip(all_preds, all_labels))
        false_negatives = sum((pred == 0 and label == 1) for pred, label in zip(all_preds, all_labels))
        
        accuracy = (true_positives + true_negatives) / len(all_labels) if len(all_labels) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"True Positives: {true_positives}, False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}, True Negatives: {true_negatives}")
        
        # Plot confusion matrix if possible
        try:
            plot_confusion_matrix(all_labels, all_preds, class_names=['Normal', 'Abnormal'])
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")
        
    except Exception as e:
        print(f"ERROR during final evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        # Print more detailed metrics
        print("\nClassification Report:")
        try:
            from sklearn.metrics import classification_report
            print(classification_report(all_labels, all_preds, target_names=['Normal', 'Abnormal']))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            
        # Compute and print simple metrics manually
        print("\nManual Metrics Calculation:")
        true_positives = sum((pred == 1 and label == 1) for pred, label in zip(all_preds, all_labels))
        true_negatives = sum((pred == 0 and label == 0) for pred, label in zip(all_preds, all_labels))
        false_positives = sum((pred == 1 and label == 0) for pred, label in zip(all_preds, all_labels))
        false_negatives = sum((pred == 0 and label == 1) for pred, label in zip(all_preds, all_labels))
        
        accuracy = (true_positives + true_negatives) / len(all_labels) if len(all_labels) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"True Positives: {true_positives}, False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}, True Negatives: {true_negatives}")
        
        # Plot confusion matrix if possible
        try:
            plot_confusion_matrix(all_labels, all_preds, class_names=['Normal', 'Abnormal'])
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")
        
    except Exception as e:
        print(f"ERROR during final evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        print("\n" + "=" * 50)
        print("STARTING CT SCAN CLASSIFIER TRAINING AND EVALUATION")
        print("=" * 50)
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: {device}")
        
        if ensure_data_downloaded():
            print("\nData verified. Starting training and evaluation...")
            try:
                train_and_evaluate()
                print("\nTraining and evaluation completed successfully!")
            except Exception as e:
                print(f"\nERROR during training/evaluation: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            print("Failed to download data. Please check the error messages above.")
            sys.exit(1)
    except Exception as e:
        print(f"\nUNHANDLED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
