"""
prepare_bulk_dataset.py

Processes massive satellite datasets (e.g. an entire country or state) overnight.
Place all your raw TIFF files into data/raw_bulk/ 
This script will slice them into thousands of training patches.
"""
import os
import sys
import glob
import zipfile
import tempfile

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rasterio
from src.ingestion.preprocess import load_sentinel2_bands, normalize
from src.ingestion.tiler import tile_image
import numpy as np
import cv2

def create_training_pair(hr_patch: np.ndarray, scale_factor: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates medium-resolution input by downsampling and applying slight blur.
    In a real dataset like WorldStrat, LR and HR are physically distinct images.
    """
    C, H, W = hr_patch.shape
    lr_patch = np.zeros((C, H // scale_factor, W // scale_factor), dtype=np.float32)
    
    for i in range(C):
        # Downsample by scale_factor using INTER_AREA to simulate lower sensor resolution
        lr_patch[i] = cv2.resize(hr_patch[i], (W // scale_factor, H // scale_factor), interpolation=cv2.INTER_AREA)
        
    return lr_patch, hr_patch

def process_zip_file(zip_path: str, extract_dir: str):
    """Finds and stacks B02, B03, B04, B08 from a Sentinel-2 ZIP into a (4, H, W) array."""
    band_mapping = {"B02": 0, "B03": 1, "B04": 2, "B08": 3}
    extracted_files = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if file_info.filename.endswith(('.tif', '.tiff', '.jp2')):
                for band_name in band_mapping.keys():
                    if f"_{band_name}" in file_info.filename or file_info.filename.endswith(f"{band_name}.tif"):
                        zip_ref.extract(file_info, extract_dir)
                        extracted_files[band_mapping[band_name]] = os.path.join(extract_dir, file_info.filename)
                        break
    
    if len(extracted_files) < 4:
        return None
        
    bands = []
    with rasterio.open(extracted_files[0]) as src:
        for i in range(4):
            with rasterio.open(extracted_files[i]) as b_src:
                bands.append(b_src.read(1))
    
    stacked = np.stack(bands, axis=0).astype(np.float32)
    return stacked

def process_bulk_data(target_pattern: str | None = None, clear_existing: bool = True):
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_bulk"))
    out_dir_lr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed_bulk", "lr"))
    out_dir_hr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed_bulk", "hr"))
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir_lr, exist_ok=True)
    os.makedirs(out_dir_hr, exist_ok=True)

    if clear_existing:
        print("🧹 Clearing previous patches in data/processed_bulk...")
        for d in [out_dir_lr, out_dir_hr]:
            for f in os.listdir(d):
                if f.endswith(".npy"):
                    os.remove(os.path.join(d, f))

    input_files = glob.glob(os.path.join(raw_dir, "*.tif")) + \
                  glob.glob(os.path.join(raw_dir, "*.tiff")) + \
                  glob.glob(os.path.join(raw_dir, "*.zip"))
    
    if target_pattern:
        input_files = [f for f in input_files if target_pattern.lower() in os.path.basename(f).lower()]
    
    if not input_files:
        print(f"⚠️ No matching TIFF or ZIP files found in {raw_dir} (filter='{target_pattern}')")
        print("Available files:")
        for f in os.listdir(raw_dir):
            print(f"  - {f}")
        return

    print(f"🌍 Found {len(input_files)} matching file(s) for bulk processing: {[os.path.basename(f) for f in input_files]}")
    
    patch_idx = 0
    for file_path in input_files:
        print(f"\nProcessing {os.path.basename(file_path)}...")
        
        if file_path.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                stacked_data = process_zip_file(file_path, tmpdir)
                if stacked_data is None:
                    print(f"  -> Skipping {os.path.basename(file_path)}: Could not find required bands.")
                    continue
                data = stacked_data
        else:
            data, _ = load_sentinel2_bands(file_path)
            
        data = normalize(data)
        print(f"  -> Extracted multi-spectral shape: {data.shape} ({data.shape[2]}x{data.shape[1]} px)")
        
        # Tile the massive image into manageable 256x256 blocks
        patches = tile_image(data, patch_size=256, overlap=16)
        file_patches = 0
        
        for p in patches:
            hr_patch = p["patch"]
            
            # Check for NoData/Black regions (if more than 50% of the patch is 0)
            zero_ratio = np.sum(hr_patch == 0) / hr_patch.size
            if zero_ratio > 0.5:
                continue  # Skip this patch entirely
                
            # Create synthetic degradation (or load real HR pairs if using WorldStrat)
            lr_patch, hr_patch = create_training_pair(hr_patch, scale_factor=3)
            
            np.save(os.path.join(out_dir_lr, f"patch_{patch_idx:06d}.npy"), lr_patch)
            np.save(os.path.join(out_dir_hr, f"patch_{patch_idx:06d}.npy"), hr_patch)
            patch_idx += 1
            file_patches += 1
            
            if patch_idx % 200 == 0:
                print(f"  -> Generated {patch_idx} training patches so far...")
                
        print(f"  -> Generated {file_patches} patches from this scene.")

    print(f"\n✅ Bulk Processing Complete! Total valid patches ready: {patch_idx}")
    print(f"Location: {out_dir_lr}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Bulk Dataset")
    parser.add_argument("--filter", type=str, default=None, help="Filter to specific filename substring (e.g. '10')")
    parser.add_argument("--keep", action="store_true", help="Keep existing patches instead of clearing")
    args = parser.parse_args()

    process_bulk_data(target_pattern=args.filter, clear_existing=not args.keep)
