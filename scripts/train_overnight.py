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
    import argparse
    parser = argparse.ArgumentParser(description="Automated Overnight Training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (0.0001 recommended for fine-tuning)")
    parser.add_argument("--output_dir", type=str, default="checkpoints_bulk", help="Directory to save new checkpoints")
    parser.add_argument("--resume", type=str, default="checkpoints/swinir_epoch342.pth", help="Checkpoint to warm-start from")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch without loading prior weights")
    args = parser.parse_args()

    cmd = [
        python_exe, train_script,
        "--data_dir", data_dir,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
    ]

    resume_target = None if args.scratch else args.resume
    if resume_target and os.path.exists(os.path.join(project_dir, resume_target)):
        cmd.extend(["--resume_from", os.path.join(project_dir, resume_target)])
        print(f"🎯 Warm-starting / fine-tuning from: {resume_target}")
    elif args.scratch:
        print("🌱 Training fresh from scratch (no prior weights loaded).")
    
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
    print(f"Your latest model weights are in the '{args.output_dir}/' folder!")

if __name__ == "__main__":
    run_training_loop()
