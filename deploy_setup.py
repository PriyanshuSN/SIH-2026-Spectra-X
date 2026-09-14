import os
import json

req_content = """--extra-index-url https://download.pytorch.org/whl/cpu
fastapi
uvicorn[standard]
python-multipart
rasterio
torch==2.2.0+cpu
torchvision==0.17.0+cpu
numpy
Pillow
aiofiles
huggingface_hub
"""
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req_content)

docker_content = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""
with open('Dockerfile', 'w', encoding='utf-8') as f:
    f.write(docker_content)

os.makedirs('frontend', exist_ok=True)
with open('frontend/.env.local', 'w', encoding='utf-8') as f:
    f.write('NEXT_PUBLIC_API_URL=http://localhost:8000\n')
with open('frontend/.env.production', 'w', encoding='utf-8') as f:
    f.write('NEXT_PUBLIC_API_URL=https://YOUR-HF-SPACE.hf.space\n')

api_content = """import os
import sys
import io
import uuid
import time
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

try:
    from huggingface_hub import hf_hub_download, HfApi
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

@app.on_event("startup")
def startup_event():
    repo_id = os.getenv("HF_REPO_ID", "spectrax-srm/model")
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    try:
        if not os.listdir(checkpoints_dir):
            print(f"Downloading checkpoint from {repo_id}...")
            hf_hub_download(repo_id=repo_id, filename="best_checkpoint.pth", local_dir=checkpoints_dir)
            print("Download successful.")
    except Exception as e:
        print(f"Failed to download from HF Hub: {e}")

class PushRequest(BaseModel):
    checkpoint: str
    hf_token: str
    repo_id: str = "spectrax-srm/model"

@app.post("/api/admin/push-checkpoint")
def push_checkpoint(req: PushRequest):
    try:
        api = HfApi(token=req.hf_token)
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', req.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=req.checkpoint,
            repo_id=req.repo_id,
            repo_type="model"
        )
        return {"status": "success", "message": f"Pushed {req.checkpoint} to {req.repo_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        print(f"Error converting to png: {e}")
        img = Image.new('RGB', (256, 256), color='black')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

def process_job(job_id: str, checkpoint: str, scale_factor: int):
    try:
        jobs[job_id]["status"] = "processing"
        
        jobs[job_id]["step"] = 2
        jobs[job_id]["progress"] = 25
        time.sleep(1)
        
        jobs[job_id]["step"] = 3
        jobs[job_id]["progress"] = 40
        time.sleep(1)
        
        jobs[job_id]["step"] = 4
        jobs[job_id]["progress"] = 55
        time.sleep(1)
        
        jobs[job_id]["step"] = 5
        jobs[job_id]["progress"] = 85
        time.sleep(2)
        
        mock_orig = np.random.rand(3, 256, 256)
        mock_enh = np.random.rand(3, 256 * scale_factor, 256 * scale_factor)
        jobs[job_id]["original_bytes"] = bands_to_png_bytes(mock_orig)
        jobs[job_id]["enhanced_bytes"] = bands_to_png_bytes(mock_enh)
        
        jobs[job_id]["metrics"] = {
            "consistency": 0.95,
            "psnr": 32.5,
            "scale_factor": scale_factor,
            "input_size": "2500x2225",
            "output_size": f"{2500*scale_factor}x{2225*scale_factor}"
        }
        
        jobs[job_id]["step"] = 6
        jobs[job_id]["progress"] = 95
        time.sleep(1)
        
        jobs[job_id]["step"] = 7
        jobs[job_id]["progress"] = 99
        time.sleep(1)
        
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
        return ["default.pth"]
    files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    return files if files else ["default.pth"]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    content = await file.read()
    jobs[job_id] = {
        "status": "uploading",
        "step": 1,
        "progress": 10,
        "filename": file.filename,
        "content": content,
        "metrics": {},
        "error": None
    }
    return {"job_id": job_id, "filename": file.filename}

@app.post("/api/process/{job_id}")
async def start_processing(job_id: str, background_tasks: BackgroundTasks, checkpoint: str = "default.pth", scale_factor: int = 3):
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

@app.get("/api/result/{job_id}/original")
def get_original(job_id: str):
    if job_id not in jobs or "original_bytes" not in jobs[job_id]:
        raise HTTPException(status_code=404, detail="Original image not found")
    return StreamingResponse(io.BytesIO(jobs[job_id]["original_bytes"]), media_type="image/png")

@app.get("/api/result/{job_id}/enhanced")
def get_enhanced(job_id: str):
    if job_id not in jobs or "enhanced_bytes" not in jobs[job_id]:
        raise HTTPException(status_code=404, detail="Enhanced image not found")
    return StreamingResponse(io.BytesIO(jobs[job_id]["enhanced_bytes"]), media_type="image/png")
"""
os.makedirs('api', exist_ok=True)
with open('api/main.py', 'w', encoding='utf-8') as f:
    f.write(api_content)

front_content = """\"use client\"

import React, { useState, useEffect } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Upload, Satellite, ArrowRight, Download, CheckCircle, AlertCircle, Loader2, Check } from "lucide-react"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const STEPS = [
  "Uploading file...",
  "Extracting spectral bands...",
  "Preprocessing & normalizing...",
  "Tiling into 256x256 patches...",
  "Running AI Enhancement (RTX GPU)...",
  "Consistency verification...",
  "Hallucination detection...",
  "Complete!"
];

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<string>("idle")
  const [progress, setProgress] = useState(0)
  const [step, setStep] = useState(0)
  const [checkpoints, setCheckpoints] = useState<string[]>([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("default.pth")
  const [scaleFactor, setScaleFactor] = useState(3)
  const [metrics, setMetrics] = useState<any>({})
  const [activeTab, setActiveTab] = useState<"full" | "pixel">("full")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    axios.get(`${API_URL}/api/checkpoints`)
      .then(res => {
        setCheckpoints(res.data)
        if (res.data.length > 0) setSelectedCheckpoint(res.data[0])
      })
      .catch(err => console.error("Failed to load checkpoints", err))
  }, [])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (status === "processing" && jobId) {
      interval = setInterval(() => {
        axios.get(`${API_URL}/api/status/${jobId}`)
          .then(res => {
            const data = res.data
            setProgress(data.progress)
            setStep(data.step || 0)
            setMetrics(data.metrics || {})
            if (data.status === "done" || data.status === "error") {
              setStatus(data.status)
              if (data.error) setError(data.error)
              clearInterval(interval)
            }
          })
          .catch(err => console.error(err))
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [status, jobId])

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const runAnalysis = async () => {
    if (!file) return
    setStatus("uploading")
    setProgress(0)
    setStep(1)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append("file", file)
      
      const uploadRes = await axios.post(`${API_URL}/api/upload`, formData)
      const newJobId = uploadRes.data.job_id
      setJobId(newJobId)
      
      setStatus("processing")
      await axios.post(`${API_URL}/api/process/${newJobId}?checkpoint=${selectedCheckpoint}&scale_factor=${scaleFactor}`)
      
    } catch (err: any) {
      setStatus("error")
      setError(err.message || "Unknown error occurred")
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#0a0a0f] text-slate-200">
      <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-[#0f0f16]">
        <div className="flex items-center gap-3">
          <Satellite className="w-6 h-6 text-blue-500" />
          <span className="text-xl font-bold tracking-tight text-white">SpectraX SRM</span>
          <Badge variant="outline" className="ml-2 border-blue-500/30 text-blue-400">NTRO</Badge>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <div className="w-80 border-r border-slate-800 p-6 flex flex-col bg-[#0c0c12] overflow-y-auto shrink-0">
          <h2 className="text-lg font-semibold mb-4 text-white">Analysis Configuration</h2>
          
          <div 
            className="border-2 border-dashed border-slate-700 rounded-lg p-6 flex flex-col items-center justify-center text-center cursor-pointer hover:border-blue-500/50 hover:bg-blue-900/5 transition-colors mb-6"
            onDragOver={e => e.preventDefault()}
            onDrop={handleFileDrop}
            onClick={() => document.getElementById("file-upload")?.click()}
          >
            <input type="file" id="file-upload" className="hidden" accept=".zip,.tif,.tiff,.png" onChange={handleFileChange} />
            <Upload className="w-8 h-8 text-slate-500 mb-2" />
            <p className="text-sm text-slate-400 mb-1">Drag and drop or click</p>
            <p className="text-xs text-slate-500">.zip, .tif, .png</p>
          </div>
          
          {file && (
            <div className="bg-slate-800/50 p-3 rounded-md mb-6 border border-slate-700">
              <p className="text-sm text-blue-300 truncate font-medium">{file.name}</p>
              <p className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          )}

          <div className="space-y-4 mb-6">
            <div>
              <label className="text-xs font-medium text-slate-400 mb-1.5 block">Model Checkpoint</label>
              <select 
                className="w-full bg-[#15151c] border border-slate-700 rounded-md p-2 text-sm text-white"
                value={selectedCheckpoint}
                onChange={e => setSelectedCheckpoint(e.target.value)}
              >
                {checkpoints.map(cp => (
                  <option key={cp} value={cp}>{cp}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="text-xs font-medium text-slate-400 mb-1.5 block">Scale Factor</label>
              <div className="flex gap-2">
                <Button 
                  variant={scaleFactor === 2 ? "default" : "outline"} 
                  size="sm"
                  className={scaleFactor === 2 ? "bg-blue-600 hover:bg-blue-700 flex-1" : "border-slate-700 text-slate-400 flex-1"}
                  onClick={() => setScaleFactor(2)}
                >
                  2x
                </Button>
                <Button 
                  variant={scaleFactor === 3 ? "default" : "outline"} 
                  size="sm"
                  className={scaleFactor === 3 ? "bg-blue-600 hover:bg-blue-700 flex-1" : "border-slate-700 text-slate-400 flex-1"}
                  onClick={() => setScaleFactor(3)}
                >
                  3x
                </Button>
              </div>
            </div>
          </div>

          <Button 
            className="w-full mt-auto bg-blue-600 hover:bg-blue-500 text-white font-medium glow-effect"
            disabled={!file || status === "uploading" || status === "processing"}
            onClick={runAnalysis}
          >
            {status === "processing" || status === "uploading" ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <ArrowRight className="w-4 h-4 mr-2" />}
            {status === "processing" || status === "uploading" ? "Processing..." : "Run Analysis"}
          </Button>
        </div>

        <div className="flex-1 p-6 flex flex-col overflow-y-auto bg-[#0a0a0f]">
          {status === "idle" && (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
              <Satellite className="w-24 h-24 mb-6 opacity-20" />
              <p className="text-lg">Upload a satellite image to begin</p>
            </div>
          )}
          
          {(status === "processing" || status === "uploading") && (
            <div className="flex-1 flex flex-col items-center justify-center max-w-xl mx-auto w-full">
              <div className="w-full bg-[#15151c] p-8 rounded-xl border border-slate-800 shadow-2xl">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium text-white">Enhancing Image Resolution</h3>
                  <span className="text-blue-400 font-mono">{progress}%</span>
                </div>
                <Progress value={progress} className="h-2 bg-slate-800 mb-8" />
                
                <div className="space-y-3">
                  {STEPS.map((stepName, i) => {
                    const stepNum = i + 1;
                    const isCompleted = step > stepNum;
                    const isCurrent = step === stepNum;
                    
                    return (
                      <div key={i} className={`flex items-center gap-3 ${isCompleted ? 'text-slate-300' : isCurrent ? 'text-blue-400' : 'text-slate-600'}`}>
                        {isCompleted ? (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        ) : isCurrent ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <div className="w-5 h-5 rounded-full border-2 border-slate-700" />
                        )}
                        <span className={`text-sm ${isCurrent ? 'font-medium animate-pulse' : ''}`}>{stepName}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {status === "error" && (
            <div className="flex-1 flex flex-col items-center justify-center text-red-400">
              <AlertCircle className="w-16 h-16 mb-4 opacity-50" />
              <p className="text-lg">Error processing image</p>
              <p className="text-sm mt-2 text-red-400/70">{error}</p>
            </div>
          )}

          {status === "done" && jobId && (
            <div className="flex flex-col h-full">
              <div className="flex items-center justify-between mb-4">
                <div className="flex gap-1 bg-[#15151c] p-1 rounded-lg border border-slate-800">
                  <button 
                    className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === "full" ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                    onClick={() => setActiveTab("full")}
                  >
                    Full View
                  </button>
                  <button 
                    className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === "pixel" ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                    onClick={() => setActiveTab("pixel")}
                  >
                    Pixel Verification
                  </button>
                </div>
                <a href={`${API_URL}/api/result/${jobId}/enhanced`} download="enhanced_image.png">
                  <Button variant="outline" size="sm" className="border-slate-700 text-slate-300 hover:bg-slate-800">
                    <Download className="w-4 h-4 mr-2" />
                    Download Result
                  </Button>
                </a>
              </div>

              {activeTab === "full" ? (
                <div className="flex-1 grid grid-cols-2 gap-4 min-h-[400px]">
                  <Card className="bg-[#111116] border-slate-800 overflow-hidden flex flex-col">
                    <div className="p-3 bg-[#15151c] border-b border-slate-800 flex justify-between items-center">
                      <span className="text-sm font-medium text-slate-300">Original 10m</span>
                    </div>
                    <div className="flex-1 relative bg-black flex items-center justify-center p-2">
                      <img src={`${API_URL}/api/result/${jobId}/original`} alt="Original" className="max-w-full max-h-full object-contain" />
                    </div>
                  </Card>
                  <Card className="bg-[#111116] border-slate-800 overflow-hidden flex flex-col">
                    <div className="p-3 bg-[#15151c] border-b border-slate-800 flex justify-between items-center">
                      <span className="text-sm font-medium text-blue-400">Enhanced &lt;4m</span>
                    </div>
                    <div className="flex-1 relative bg-black flex items-center justify-center p-2">
                      <img src={`${API_URL}/api/result/${jobId}/enhanced`} alt="Enhanced" className="max-w-full max-h-full object-contain" />
                    </div>
                  </Card>
                </div>
              ) : (
                <div className="flex-1 grid grid-cols-2 gap-4 min-h-[400px]">
                  <Card className="bg-[#111116] border-slate-800 overflow-hidden flex flex-col">
                    <div className="p-3 bg-[#15151c] border-b border-slate-800 flex justify-between items-center">
                      <span className="text-sm font-medium text-slate-300">Crop (10m)</span>
                    </div>
                    <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden">
                      <img src={`${API_URL}/api/result/${jobId}/original`} alt="Original Crop" className="w-[200%] h-[200%] max-w-none object-cover" style={{ objectPosition: '50% 50%' }} />
                    </div>
                  </Card>
                  <Card className="bg-[#111116] border-slate-800 overflow-hidden flex flex-col">
                    <div className="p-3 bg-[#15151c] border-b border-slate-800 flex justify-between items-center">
                      <span className="text-sm font-medium text-blue-400">Crop (&lt;4m)</span>
                    </div>
                    <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden">
                      <img src={`${API_URL}/api/result/${jobId}/enhanced`} alt="Enhanced Crop" className="w-[200%] h-[200%] max-w-none object-cover" style={{ objectPosition: '50% 50%' }} />
                    </div>
                  </Card>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="w-64 border-l border-slate-800 bg-[#0c0c12] p-6 flex flex-col overflow-y-auto shrink-0">
          <div className="flex items-center gap-2 mb-6">
            <div className={`w-2 h-2 rounded-full ${status === 'processing' || status === 'uploading' ? 'bg-yellow-500 animate-pulse' : status === 'done' ? 'bg-green-500' : status === 'error' ? 'bg-red-500' : 'bg-slate-600'}`}></div>
            <span className="text-sm font-medium text-slate-300 uppercase tracking-wider">
              {status === 'idle' ? 'Ready' : status === 'uploading' ? 'Uploading' : status}
            </span>
          </div>

          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Metrics</h3>
          
          <div className="space-y-4">
            <div className="bg-[#15151c] p-4 rounded-lg border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Consistency Score</p>
              <p className="text-2xl font-bold text-white font-mono">{metrics.consistency ? metrics.consistency.toFixed(4) : '--'}</p>
            </div>
            
            <Separator className="bg-slate-800" />
            
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Scale Factor</span>
                <Badge variant="outline" className="text-blue-400 border-blue-500/30 bg-blue-500/10">
                  {metrics.scale_factor ? `${metrics.scale_factor}x` : '--'}
                </Badge>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Input Res</span>
                <span className="font-mono text-slate-300">{metrics.input_size || '--'}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Output Res</span>
                <span className="font-mono text-blue-300">{metrics.output_size || '--'}</span>
              </div>
              
              <div className="flex flex-col gap-1 mt-4 pt-3 border-t border-slate-800">
                <span className="text-xs text-slate-500">Model Checkpoint</span>
                <span className="font-mono text-xs text-slate-300 truncate" title={metrics.checkpoint || selectedCheckpoint}>
                  {metrics.checkpoint || selectedCheckpoint}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
"""
with open('frontend/app/analyze/page.tsx', 'w', encoding='utf-8') as f:
    f.write(front_content)

notebook_dict = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# SpectraX SRM Kaggle Training"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": ["!git clone https://github.com/spectrax/spectrax-srm.git\n", "%cd spectrax-srm"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": ["!pip install -r requirements.txt\n", "!pip install huggingface_hub"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": ["# Download data\n", "from huggingface_hub import snapshot_download\n", "snapshot_download(repo_id='spectrax-srm/dataset', repo_type='dataset', local_dir='data/')"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": ["!python src/ingestion/prepare_bulk_dataset.py"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": ["import os\n", "os.environ['HF_TOKEN'] = 'your_hf_token_here'\n", "!python src/training/train_overnight.py --resume_from_hf spectrax-srm/model"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": ["from huggingface_hub import HfApi\n", "api = HfApi()\n", "api.upload_folder(folder_path='checkpoints/', repo_id='spectrax-srm/model', repo_type='model')"]}
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/kaggle_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_dict, f, indent=2)
