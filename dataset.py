from pathlib import Path
from typing import List, Tuple, Callable, Optional
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import pydicom
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict

class CTScanDataset(Dataset):
    """PyTorch Dataset for loading CT scan images.
    
    Args:
        data_dir: Directory containing the processed CT scan data
        transform: Optional transform to be applied on a sample
        split: Either 'train' or 'val' to specify the dataset split
        split_ratio: Ratio of data to use for training (rest will be used for validation)
    """
    def __init__(self, data_dir: Path, transform: Optional[Callable] = None, 
                 split: str = 'train', split_ratio: float = 0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.series_dirs = [d for d in data_dir.glob("*") if d.is_dir()]
        
        # Simple train/val split (in a real scenario, you'd want to ensure no patient overlap)
        random.shuffle(self.series_dirs)
        split_idx = int(len(self.series_dirs) * split_ratio)
        
        if split == 'train':
            self.series_dirs = self.series_dirs[:split_idx]
        else:  # val/test
            self.series_dirs = self.series_dirs[split_idx:]
        
        # Load all images and labels
        self.images, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load images and labels from the data directory.
        
        Returns:
            Tuple of (list of image arrays, list of labels)
            Label 0: No nodule or benign nodule (malignancy score < 3)
            Label 1: Suspicious or malignant nodule (malignancy score >= 3)
        """
        print(f"\nStarting data loading from {self.data_dir}")
        print(f"Found {len(self.series_dirs)} series directories")
        
        images = []
        labels = []
        successful_dirs = 0
        
        # First, let's check if there are any pre-processed numpy files
        print("\nChecking for pre-processed data (images.npy files)...")
        npy_files_exist = False
        
        for series_idx, series_dir in enumerate(self.series_dirs, 1):
            npy_file = series_dir / "images.npy"
            if npy_file.exists():
                npy_files_exist = True
                print(f"Found images.npy in {series_dir.name}")
                break
                
        if npy_files_exist:
            print("\nUsing pre-processed numpy files for faster loading")
            
            # Load data from numpy files
            for series_idx, series_dir in enumerate(self.series_dirs, 1):
                try:
                    npy_file = series_dir / "images.npy"
                    if not npy_file.exists():
                        continue
                        
                    print(f"Loading {npy_file}")
                    image_data = np.load(npy_file)
                    
                    # Make sure the loaded data is the right shape
                    if len(image_data.shape) >= 3:
                        # If multiple images in one file, add them individually
                        for img in image_data:
                            # Ensure the image has the right dimensions (convert if needed)
                            if len(img.shape) == 2:  # If grayscale
                                # Convert to 3 channels if needed
                                img = np.stack([img] * 3, axis=-1)
                            elif len(img.shape) == 3 and img.shape[2] != 3:  # If not RGB
                                img = np.stack([img[:,:,0]] * 3, axis=-1)  # Use first channel
                                
                            images.append(img)
                            # For now, assume all are label 1 (has nodule)
                            # This is just a placeholder - in real code we'd need proper labels
                            labels.append(1)
                        
                        successful_dirs += 1
                        print(f"  Successfully loaded {len(image_data)} images from {npy_file}")
                    else:
                        print(f"  Unexpected data shape in {npy_file}: {image_data.shape}")
                        
                except Exception as e:
                    print(f"  Error loading {npy_file}: {e}")
        
        # If we found no numpy files or couldn't load any images, fall back to DICOM processing
        if not images:
            print("\nNo valid pre-processed data found. Trying to load from DICOM files...")
            
            for series_idx, series_dir in enumerate(self.series_dirs, 1):
                try:
                    print(f"\nProcessing series {series_idx}/{len(self.series_dirs)}: {series_dir.name}")
                    
                    # Look for DICOM files in the series directory
                    dicom_files = list(series_dir.glob("**/*.dcm"))
                    if not dicom_files:
                        print(f"No DICOM files found in {series_dir}")
                        continue
                    
                    print(f"Found {len(dicom_files)} DICOM files")
                    
                    # Process DICOM files
                    series_images = []
                    
                    # Import pydicom here to avoid issues if it's not used
                    import pydicom
                    from pydicom.errors import InvalidDicomError
                    
                    for dcm_file in dicom_files[:20]:  # Limit to first 20 files for testing
                        try:
                            ds = pydicom.dcmread(str(dcm_file))
                            if hasattr(ds, 'pixel_array'):
                                # Convert to the right format and add to images
                                img = ds.pixel_array.astype(np.float32)
                                
                                # Normalize to 0-1 range
                                if img.max() > 0:
                                    img = img / img.max()
                                
                                # Convert to RGB if needed
                                if len(img.shape) == 2:  # If grayscale
                                    img = np.stack([img] * 3, axis=-1)  # Convert to 3 channels
                                
                                series_images.append(img)
                        except Exception as e:
                            print(f"  Error reading {dcm_file.name}: {e}")
                    
                    if series_images:
                        # Add the images and corresponding labels
                        images.extend(series_images)
                        # For now, assign label 1 (has nodule) to all images
                        # In a real scenario, we'd extract this from annotations
                        labels.extend([1] * len(series_images))
                        
                        successful_dirs += 1
                        print(f"  Successfully loaded {len(series_images)} images from DICOM files")
                
                except Exception as e:
                    print(f"Error processing directory {series_dir}: {e}")
        
        print(f"\nData loading complete: loaded {len(images)} images from {successful_dirs} directories")
        
        # Split the data according to the split parameter
        if self.split == 'train':
            # Return the first 80% for training
            split_idx = int(0.8 * len(images))
            return images[:split_idx], labels[:split_idx]
        elif self.split == 'val':
            # Return the last 20% for validation
            split_idx = int(0.8 * len(images))
            return images[split_idx:], labels[split_idx:]
        else:
            # Return all data if split is not specified
            return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transformations
        img = Image.fromarray((img * 255).astype('uint8'))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
