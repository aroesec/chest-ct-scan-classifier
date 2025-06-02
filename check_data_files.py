#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def print_separator():
    print("\n" + "="*80 + "\n")

# Print basic environment info
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print_separator()

# Check data directory structure
data_dir = Path("data/lidc/processed")
print(f"Data directory: {data_dir.absolute()}")
print(f"Data directory exists: {data_dir.exists()}")

if not data_dir.exists():
    print(f"ERROR: Data directory {data_dir.absolute()} does not exist!")
    sys.exit(1)

# List contents of data directory
print("\nContents of data directory:")
try:
    data_files = list(data_dir.glob("*"))
    if not data_files:
        print("  (directory is empty)")
    else:
        for i, item in enumerate(data_files, 1):
            if item.is_dir():
                # Count files in subdirectory
                files_in_dir = list(item.glob("*"))
                file_count = len(files_in_dir)
                dcm_count = len(list(item.glob("**/*.dcm")))
                npy_count = len(list(item.glob("**/*.npy")))
                print(f"  {i}. {item.name}/ (directory with {file_count} files, {dcm_count} DCM, {npy_count} NPY)")
                
                # List a few sample files in this directory
                if files_in_dir:
                    for j, f in enumerate(files_in_dir[:3], 1):
                        print(f"     {j}. {f.name} ({'dir' if f.is_dir() else 'file'}, {f.stat().st_size/1024:.1f} KB)")
                    if len(files_in_dir) > 3:
                        print(f"     ... and {len(files_in_dir)-3} more files")
            else:
                print(f"  {i}. {item.name} (file, {item.stat().st_size/1024:.1f} KB)")
except Exception as e:
    print(f"Error listing directory contents: {e}")

print_separator()

# Check for DICOM files specifically
print("Looking for DICOM files...")
dcm_files = list(data_dir.glob("**/*.dcm"))
print(f"Found {len(dcm_files)} DICOM files in total")

# Check for NPY files
print("\nLooking for NPY files...")
npy_files = list(data_dir.glob("**/*.npy"))
print(f"Found {len(npy_files)} NPY files in total")

# If found, list the first few of each
if dcm_files:
    print("\nSample DICOM files:")
    for i, f in enumerate(dcm_files[:5], 1):
        print(f"  {i}. {f.relative_to(data_dir)} ({f.stat().st_size/1024:.1f} KB)")
    
if npy_files:
    print("\nSample NPY files:")
    for i, f in enumerate(npy_files[:5], 1):
        print(f"  {i}. {f.relative_to(data_dir)} ({f.stat().st_size/1024:.1f} KB)")

print_separator()
print("Data directory check complete")
