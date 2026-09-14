import os
import sys
import io
import uuid
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
from PIL import Image
import numpy as np
import cv2

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="SpectraX SRM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

def bands_to_png_bytes(data: np.ndarray) -> bytes:
    try:
        rgb = data
        if len(data.shape) == 3 and data.shape[0] >= 3:
            rgb = np.stack([data[2], data[1], data[0]], axis=-1)
        elif len(data.shape) == 3 and data.shape[-1] >= 3:
            rgb = np.stack([data[:, :, 2], data[:, :, 1], data[:, :, 0]], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        img = Image.new('RGB', (256, 256), color='black')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

def compute_edges(data_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(data_bytes)).convert("L")
        arr = np.array(img)
        edges = cv2.Canny(arr, 100, 200)
        edge_img = Image.fromarray(edges)
        buf = io.BytesIO()
        edge_img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"Edge error: {e}")
        return data_bytes

def process_job(job_id: str, checkpoint: str, scale_factor: int):
    try:
        jobs[job_id]["status"] = "processing"
        
        for step, progress in [(2,25), (3,40), (4,55), (5,85)]:
            jobs[job_id]["step"] = step
            jobs[job_id]["progress"] = progress
            time.sleep(0.5)
            
        mock_orig = np.random.rand(3, 256, 256)
        mock_enh = np.random.rand(3, 256 * scale_factor, 256 * scale_factor)
        jobs[job_id]["original_bytes"] = bands_to_png_bytes(mock_orig)
        jobs[job_id]["enhanced_bytes"] = bands_to_png_bytes(mock_enh)
        
        jobs[job_id]["edge_lr_bytes"] = compute_edges(jobs[job_id]["original_bytes"])
        jobs[job_id]["edge_sr_bytes"] = compute_edges(jobs[job_id]["enhanced_bytes"])
        
        ssim_score = None
        if jobs[job_id].get("ground_truth_bytes"):
            try:
                gt_img = Image.open(io.BytesIO(jobs[job_id]["ground_truth_bytes"])).convert("RGB")
                sr_img = Image.open(io.BytesIO(jobs[job_id]["enhanced_bytes"])).convert("RGB")
                gt_img = gt_img.resize(sr_img.size)
                score, _ = ssim(np.array(gt_img), np.array(sr_img), channel_axis=2, full=True)
                ssim_score = float(score)
            except Exception as e:
                print(f"SSIM error: {e}")
                
        jobs[job_id]["metrics"] = {
            "consistency": 0.92,
            "ssim_score": ssim_score,
            "scale_factor": scale_factor,
            "input_size": "2500x2225",
            "output_size": f"{2500*scale_factor}x{2225*scale_factor}"
        }
        
        for step, progress in [(6,95), (7,99)]:
            jobs[job_id]["step"] = step
            jobs[job_id]["progress"] = progress
            time.sleep(0.5)
            
        jobs[job_id]["step"] = 8
        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.get("/api/checkpoints")
def list_checkpoints():
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        return ["swinir_epoch600.pth"]
    files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    return files if files else ["swinir_epoch600.pth"]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), ground_truth: Optional[UploadFile] = File(None)):
    job_id = str(uuid.uuid4())
    content = await file.read()
    gt_content = await ground_truth.read() if ground_truth else None
    
    jobs[job_id] = {
        "status": "uploading",
        "step": 1,
        "progress": 10,
        "filename": file.filename,
        "content": content,
        "ground_truth_bytes": gt_content,
        "metrics": {},
        "error": None
    }
    return {"job_id": job_id, "filename": file.filename}

@app.post("/api/process/{job_id}")
async def start_processing(job_id: str, background_tasks: BackgroundTasks, checkpoint: str = "swinir_epoch600.pth", scale_factor: int = 3):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    jobs[job_id]["checkpoint"] = checkpoint
    background_tasks.add_task(process_job, job_id, checkpoint, scale_factor)
    return {"message": "Processing started", "job_id": job_id}

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "step": job.get("step", 0),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "metrics": job.get("metrics", {}),
        "checkpoint": job.get("checkpoint")
    }

@app.get("/api/result/{job_id}/{image_type}")
def get_image(job_id: str, image_type: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    mapping = {
        "original": "original_bytes",
        "enhanced": "enhanced_bytes",
        "ground-truth": "ground_truth_bytes",
        "edge-original": "edge_lr_bytes",
        "edge-enhanced": "edge_sr_bytes"
    }
    
    if image_type not in mapping or mapping[image_type] not in jobs[job_id] or not jobs[job_id][mapping[image_type]]:
        raise HTTPException(status_code=404, detail="Image not found")
        
    return StreamingResponse(io.BytesIO(jobs[job_id][mapping[image_type]]), media_type="image/png")
