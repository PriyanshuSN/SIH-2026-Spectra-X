"""
train_overnight.py

Automated script for overnight training on your RTX GPU.
Usage: python scripts/train_overnight.py
"""
import os
import subprocess
import time

def run_training_loop():
    print("🚀 Starting Automated Overnight Training Loop for SpectraX")
    print("---------------------------------------------------------")
    
    # Paths
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_script = os.path.join(project_dir, "src", "model", "train.py")
    
    # Use bulk data if available, otherwise fallback to standard processed data
    bulk_dir = os.path.join(project_dir, "data", "processed_bulk")
    if os.path.exists(bulk_dir) and len(os.listdir(os.path.join(bulk_dir, "lr"))) > 0:
        data_dir = bulk_dir
        print("🌍 Bulk Dataset detected! Training on massive dataset.")
    else:
        data_dir = os.path.join(project_dir, "data", "processed")
        print("Using standard dataset (Pune). Add files to data/raw_bulk/ and run prepare_bulk_dataset.py to scale up.")
        
    python_exe = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    
    print(f"Data Directory: {data_dir}")
    print(f"Using Python: {python_exe}\n")
    
    # Configure training run
    epochs = 800      # Total epochs to run overnight
    batch_size = 16   # Max out the 8GB RTX 5050 VRAM for faster training
    
    cmd = [
        python_exe, train_script,
        "--data_dir", data_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", "0.0002"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print("Press Ctrl+C to stop the training at any time. Checkpoints save automatically.")
    print("---------------------------------------------------------\n")
    
    # Set encoding to avoid Windows Unicode errors
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = project_dir
    
    start_time = time.time()
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n\n❌ Training failed with error code {e.returncode}.")
    
    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print(f"\n✅ Training session ended. Duration: {duration:.2f} hours.")
    print("Your latest model weights are in the 'checkpoints/' folder!")

if __name__ == "__main__":
    run_training_loop()
