from pathlib import Path
from dataset import CTScanDataset

# Print working directory for debugging
import os
print(f"Current working directory: {os.getcwd()}")

# Try to load the dataset
data_dir = Path("data/lidc/processed")
print(f"Data directory: {data_dir.absolute()}")
print(f"Data directory exists: {data_dir.exists()}")

try:
    print("\nAttempting to create dataset...")
    dataset = CTScanDataset(data_dir, transform=None, split='train')
    print(f"\nSuccessfully created dataset with {len(dataset)} samples")
    
    # If we have samples, print info about the first few
    if len(dataset) > 0:
        print("\nFirst few samples:")
        for i in range(min(3, len(dataset))):
            img, label = dataset[i]
            print(f"Sample {i}: Shape={img.shape}, Label={label}")
    else:
        print("Dataset is empty, no samples found")
        
except Exception as e:
    print(f"Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
