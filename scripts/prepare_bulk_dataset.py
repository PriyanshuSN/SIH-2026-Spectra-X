"""
prepare_bulk_dataset.py

Processes massive satellite datasets (e.g. an entire country or state) overnight.
Place all your raw TIFF files into data/raw_bulk/ 
This script will slice them into thousands of training patches.
"""
import os
import glob
from src.ingestion.preprocess import load_sentinel2_bands, normalize
from src.ingestion.tiler import tile_image
from src.model.train import create_training_pair
import numpy as np

def process_bulk_data():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_bulk"))
    out_dir_lr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed_bulk", "lr"))
    out_dir_hr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed_bulk", "hr"))
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir_lr, exist_ok=True)
    os.makedirs(out_dir_hr, exist_ok=True)

    tif_files = glob.glob(os.path.join(raw_dir, "*.tif")) + glob.glob(os.path.join(raw_dir, "*.tiff"))
    
    if not tif_files:
        print(f"⚠️ No TIFF files found in {raw_dir}")
        print("Please drop your state/country scale TIFF files there first.")
        return

    print(f"🌍 Found {len(tif_files)} massive satellite images. Starting bulk processing...")
    
    patch_idx = 0
    for tif_path in tif_files:
        print(f"Processing {os.path.basename(tif_path)}...")
        data, _ = load_sentinel2_bands(tif_path)
        data = normalize(data)
        
        # Tile the massive image into manageable 256x256 blocks
        patches = tile_image(data, patch_size=256, overlap=16)
        
        for p in patches:
            hr_patch = p["patch"]
            # Create synthetic degradation (or load real HR pairs if using WorldStrat)
            lr_patch, hr_patch = create_training_pair(hr_patch, scale_factor=3)
            
            np.save(os.path.join(out_dir_lr, f"patch_{patch_idx:06d}.npy"), lr_patch)
            np.save(os.path.join(out_dir_hr, f"patch_{patch_idx:06d}.npy"), hr_patch)
            patch_idx += 1
            
            if patch_idx % 500 == 0:
                print(f"  -> Generated {patch_idx} training patches so far...")

    print(f"✅ Bulk Processing Complete! Generated {patch_idx} total training patches.")
    print(f"You can now point train_overnight.py to data/processed_bulk to train on this massive dataset.")

if __name__ == "__main__":
    process_bulk_data()
