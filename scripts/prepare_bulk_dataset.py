"""
prepare_bulk_dataset.py

Processes massive satellite datasets (e.g. an entire country or state) overnight.
Place all your raw TIFF files into data/raw_bulk/ 
This script will slice them into thousands of training patches.
"""
import os
import glob
import zipfile
import tempfile
import rasterio
from src.ingestion.preprocess import load_sentinel2_bands, normalize
from src.ingestion.tiler import tile_image
from src.model.train import create_training_pair
import numpy as np

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

def process_bulk_data():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_bulk"))
    out_dir_lr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed_bulk", "lr"))
    out_dir_hr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed_bulk", "hr"))
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir_lr, exist_ok=True)
    os.makedirs(out_dir_hr, exist_ok=True)

    input_files = glob.glob(os.path.join(raw_dir, "*.tif")) + \
                  glob.glob(os.path.join(raw_dir, "*.tiff")) + \
                  glob.glob(os.path.join(raw_dir, "*.zip"))
    
    if not input_files:
        print(f"⚠️ No TIFF or ZIP files found in {raw_dir}")
        print("Please drop your state/country scale files there first.")
        return

    print(f"🌍 Found {len(input_files)} massive satellite files. Starting bulk processing...")
    
    patch_idx = 0
    for file_path in input_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
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
        
        # Tile the massive image into manageable 256x256 blocks
        patches = tile_image(data, patch_size=256, overlap=16)
        
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
            
            if patch_idx % 500 == 0:
                print(f"  -> Generated {patch_idx} training patches so far...")

    print(f"✅ Bulk Processing Complete! Generated {patch_idx} total training patches.")
    print(f"You can now run train_overnight.py to train on this massive dataset.")

if __name__ == "__main__":
    process_bulk_data()
