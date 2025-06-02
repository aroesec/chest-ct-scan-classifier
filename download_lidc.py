import os
import requests
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
import zipfile

DATA_DIR = Path("data/lidc")
DICOM_DIR = DATA_DIR / "dicom"
PROCESSED_DIR = DATA_DIR / "processed"
TCIA_API_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA/query/"

def setup_directories():
    """Create necessary directories if they don't exist"""
    for directory in [DATA_DIR, DICOM_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_series_data(collection="LIDC-IDRI"):
    """Get list of series from TCIA API"""
    params = {
        "Collection": collection,
        "format": "json"
    }
    response = requests.get(TCIA_API_URL + "getSeries", params=params)
    return response.json()

def download_series(series_instance_uid, output_dir):
    """Download DICOM series from TCIA"""
    url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage?SeriesInstanceUID={series_instance_uid}"
    response = requests.get(url, stream=True)
    zip_path = output_dir / f"{series_instance_uid}.zip"
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return zip_path

def process_dicom_series(zip_path, output_dir):
    """Process DICOM files from zip"""
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Process DICOM files
    dicom_files = list(output_dir.glob("**/*.dcm"))
    images = []
    
    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            if hasattr(ds, 'pixel_array'):
                img = ds.pixel_array
                # Normalize to 0-1
                img = (img - img.min()) / (img.max() - img.min())
                # Convert to 3 channels
                img = np.stack([img] * 3, axis=-1)
                images.append(img)
        except Exception as e:
            print(f"Error processing {dcm_file}: {e}")
    
    return np.array(images)

def main():
    setup_directories()
    
    print("Fetching available series from TCIA...")
    try:
        series_data = get_series_data()
        print(f"Found {len(series_data)} series.")
        
        # Process all available series
        for series in tqdm(series_data, desc="Downloading series"):
            try:
                series_uid = series['SeriesInstanceUID']
                print(f"\nProcessing series: {series_uid}")
                
                # Create series-specific directory
                series_dir = PROCESSED_DIR / series_uid
                series_dir.mkdir(exist_ok=True)
                
                # Skip if already processed
                if (series_dir / "images.npy").exists():
                    print(f"Skipping {series_uid} - already processed")
                    continue
                
                # Download series
                print(f"Downloading {series_uid}...")
                zip_path = download_series(series_uid, DICOM_DIR)
                
                # Process DICOM files
                print(f"Processing {series_uid}...")
                images = process_dicom_series(zip_path, series_dir)
                
                # Save processed images
                if len(images) > 0:
                    np.save(series_dir / "images.npy", images)
                    print(f"Saved {len(images)} images from {series_uid}")
                else:
                    print(f"No valid DICOM images found in {series_uid}")
                
            except Exception as e:
                print(f"Error processing series {series_uid}: {e}")
                continue
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have an internet connection and access to TCIA API.")

if __name__ == "__main__":
    main()
