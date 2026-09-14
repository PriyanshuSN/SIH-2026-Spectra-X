import os
import subprocess
import time
import argparse

def run_training_loop():
    print("Starting Automated Overnight Training Loop for SpectraX")
    print("---------------------------------------------------------")
    
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_script = os.path.join(project_dir, "src", "model", "train.py")
    
    bulk_dir = os.path.join(project_dir, "data", "processed_bulk")
    if os.path.exists(bulk_dir) and len(os.listdir(os.path.join(bulk_dir, "lr"))) > 0:
        data_dir = bulk_dir
        print("Bulk Dataset detected! Training on massive dataset.")
    else:
        data_dir = os.path.join(project_dir, "data", "processed")
        print("Using standard dataset (Pune). Add files to data/raw_bulk/ and run prepare_bulk_dataset.py to scale up.")
        
    python_exe = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", action="store_true", help="Start from scratch")
    args = parser.parse_args()

    epochs = 600      
    batch_size = 16   
    lr = "0.0001"     # Slower learning rate for fine-tuning the 342 epoch model!
    
    cmd = [
        python_exe, train_script,
        "--data_dir", data_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", lr
    ]

    if not args.scratch:
        chkpt_dir = os.path.join(project_dir, "checkpoints")
        if os.path.exists(chkpt_dir):
            checkpoints = [f for f in os.listdir(chkpt_dir) if f.endswith(".pth") and f.startswith("swinir_epoch")]
            if checkpoints:
                latest_ckpt = sorted(checkpoints)[-1]
                latest_ckpt_path = os.path.join(chkpt_dir, latest_ckpt)
                print(f"Auto-resuming from latest checkpoint: {latest_ckpt}")
                cmd.extend(["--resume_from", latest_ckpt_path])

    print(f"Executing: {' '.join(cmd)}")
    print("Press Ctrl+C to stop the training at any time. Checkpoints save automatically.")
    print("---------------------------------------------------------\n")
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = project_dir
    
    start_time = time.time()
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n\n Training failed with error code {e.returncode}.")
    
    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print(f"\n Training session ended. Duration: {duration:.2f} hours.")

if __name__ == "__main__":
    run_training_loop()
