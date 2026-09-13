"""
SpectraX SRM — Streamlit MVP Dashboard

Reconstruct medium-resolution (10m) satellite imagery into sharper (<4m)
imagery with pixel-level uncertainty, hallucination detection,
and self-consistency verification.
"""

import os
import tempfile
import zipfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
import torch
import folium
from streamlit_folium import st_folium
from streamlit_image_comparison import image_comparison
from PIL import Image

from src.api.inference import load_model, run_inference
from src.ingestion.preprocess import load_sentinel2_bands, normalize

# --- Page Setup ---
st.set_page_config(
    page_title="SpectraX SRM | NTRO",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    h1, h2, h3 { color: #f8f9fa; font-weight: 600; }
    .stAlert { border-radius: 8px; }
    .metric-card { 
        background: #161b22; padding: 15px; border-radius: 8px; 
        border-left: 4px solid #3b82f6; margin-bottom: 15px;
        color: #c9d1d9;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# --- Caching ---
@st.cache_resource
def get_model(ckpt_path: str, device: str):
    return load_model(ckpt_path, device=device)

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
        
    # Stack bands
    bands = []
    with rasterio.open(extracted_files[0]) as src:
        meta = src.meta
        for i in range(4):
            with rasterio.open(extracted_files[i]) as b_src:
                bands.append(b_src.read(1))
    
    stacked = np.stack(bands, axis=0).astype(np.float32)
    return stacked

def bands_to_rgb(img_bands: np.ndarray) -> Image.Image:
    """Converts (4, H, W) normalized array to a PIL Image (uint8 RGB) for display."""
    if img_bands.shape[0] >= 3:
        rgb = np.stack([img_bands[2], img_bands[1], img_bands[0]], axis=-1)
    else:
        rgb = np.repeat(img_bands[0:1], 3, axis=0).transpose(1, 2, 0)

    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_st = (rgb - p2) / (p98 - p2 + 1e-8)
    rgb_st = np.clip(rgb_st, 0.0, 1.0)
    return Image.fromarray((rgb_st * 255).astype(np.uint8))


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛰️ SpectraX Platform")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.success(f"**Hardware:** `{device.upper()}` " + (f"({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    ckpt_files = sorted(Path("checkpoints").glob("*.pth"), key=os.path.getmtime, reverse=True)
    ckpt_options = [str(p) for p in ckpt_files] if ckpt_files else ["checkpoints/swinir_epoch025.pth"]
    selected_ckpt = st.selectbox("Model Weights", ckpt_options, index=0)

    st.subheader("Inference Settings")
    scale_factor = st.selectbox("Resolution Upscale Factor", ["2x", "3x (Current Weights)", "4x"], index=1, help="Note: 2x and 4x require separate training weights. Defaulting to 3x.")
    n_passes = st.slider("Uncertainty Passes", 2, 16, 4, 2)
    consistency_threshold = st.slider("Consistency Threshold", 0.70, 0.99, 0.85, 0.01)


# --- Main Header ---
st.title("SpectraX — Super Resolution Engine")
st.markdown("Transform blurry 10m/pixel satellite data into sharp, tactically actionable <4m/pixel imagery.")

# --- Session State ---
if "input_data" not in st.session_state:
    st.session_state.input_data = None
    st.session_state.ref_data = None
    st.session_state.sample_name = ""
    st.session_state.ready_to_process = False

# --- Data Input ---
st.header("1. Data Ingestion")

tab_demo, tab_zip, tab_upload, tab_map = st.tabs(["🚀 Quick Demo", "🗂️ Upload Copernicus ZIP", "🖼️ Upload Image", "🌍 Select on Map (API)"])

with tab_demo:
    st.markdown("Test the engine instantly using pre-processed data from Pune (Patch 0000).")
    if st.button("Load Demo Tile"):
        lr_sample = "data/processed/lr/patch_0000.npy"
        hr_sample = "data/processed/hr/patch_0000.npy"
        if os.path.exists(lr_sample) and os.path.exists(hr_sample):
            st.session_state.input_data = np.load(lr_sample)
            st.session_state.ref_data = np.load(hr_sample)
            st.session_state.sample_name = "Demo: Pune (Patch 0000)"
            st.session_state.ready_to_process = False
        else:
            st.error("Demo files not found!")

with tab_zip:
    st.markdown("Upload a raw `.zip` file from the Copernicus Data Space. We extract and stack the exact spectral bands needed.")
    zip_file = st.file_uploader("Upload Copernicus ZIP", type=["zip"], key="zip_uploader")
    if zip_file and st.button("Extract & Load ZIP Data"):
        with st.spinner("Extracting multi-spectral bands..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.read())
                
                stacked_data = process_zip_file(zip_path, tmpdir)
                if stacked_data is not None:
                    st.session_state.input_data = normalize(stacked_data, method="percentile")
                    st.session_state.sample_name = zip_file.name
                    st.session_state.ref_data = None
                    st.session_state.ready_to_process = False
                    st.success("Successfully extracted and stacked 4 spectral bands! Ready to process.")
                else:
                    st.error("Could not find standard B02, B03, B04, B08 bands in this zip.")

with tab_upload:
    st.markdown("Upload any standard image. SpectraX handles missing infrared bands automatically.")
    up_file = st.file_uploader("Input Image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up_file and st.button("Load Image"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up_file.name).suffix) as tmp:
            tmp.write(up_file.read())
            data, _ = load_sentinel2_bands(tmp.name)
            st.session_state.input_data = normalize(data, method="percentile")
            st.session_state.sample_name = up_file.name
            st.session_state.ref_data = None
            st.session_state.ready_to_process = False
            st.success("Image loaded! Ready to process.")

with tab_map:
    st.markdown("*(API Simulation)* Select an AOI directly from the globe to fetch Sentinel-2 data.")
    m = folium.Map(location=[18.5204, 73.8567], zoom_start=10) # Centered on Pune
    folium.Marker([18.5204, 73.8567], popup="Pune Tile AOI").add_to(m)
    st_folium(m, height=300, width=800)
    
    if st.button("Fetch Data from Copernicus API"):
        st.info("Initiating OAuth2 connection to Sentinel Hub API...")
        time.sleep(1.5)
        st.success("Data fetched successfully! (Simulated for Demo)")
        lr_sample = "data/processed/lr/patch_0000.npy"
        if os.path.exists(lr_sample):
            st.session_state.input_data = np.load(lr_sample)
            st.session_state.ref_data = np.load("data/processed/hr/patch_0000.npy")
            st.session_state.sample_name = "API Fetch: Coordinates [18.52, 73.85]"
            st.session_state.ready_to_process = False

# --- Execution Workflow ---
if st.session_state.input_data is not None:
    st.divider()
    st.subheader(f"Current Target: `{st.session_state.sample_name}`")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ RUN SUPER RESOLUTION PIPELINE", type="primary", use_container_width=True):
            st.session_state.ready_to_process = True

    if st.session_state.ready_to_process:
        
        if not os.path.exists(selected_ckpt):
            st.error("Model weights not found. Please train the model first.")
        else:
            # Workflow Simulation Block
            with st.status("Executing SpectraX Pipeline...", expanded=True) as status:
                st.write("Loading Deep Learning Weights into VRAM...")
                time.sleep(0.5)
                model = get_model(selected_ckpt, device=device)
                st.write("Normalizing spectral tensors...")
                time.sleep(0.5)
                st.write(f"Running SwinIR Inference on {device.upper()}...")
                results = run_inference(
                    model=model,
                    input_image=st.session_state.input_data,
                    reference_image=st.session_state.ref_data,
                    n_uncertainty_passes=n_passes,
                    scale_factor=3,
                    patch_size=256,
                    device=device,
                )
                st.write("Computing Anti-Hallucination maps...")
                time.sleep(0.5)
                status.update(label="Pipeline Complete!", state="complete", expanded=False)

            sr_output = results["sr_output"]
            uncertainty = results["uncertainty_map"]
            hallucination = results["hallucination_map"]
            consistency = results["consistency"]
            metrics = results["metrics"]

            # === 1. NTRO METRICS (Premium Layout) ===
            st.header("NTRO Verification Dashboard")
            
            col_m1, col_m2 = st.columns([2, 1])
            
            with col_m1:
                st.markdown("#### Anti-Hallucination Telemetry")
                map1, map2 = st.columns(2)
                with map1:
                    st.caption("Uncertainty Heatmap (MC-Dropout)")
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    cax1 = ax1.imshow(uncertainty, cmap="viridis")
                    fig1.colorbar(cax1, ax=ax1, shrink=0.7)
                    ax1.axis("off")
                    fig1.patch.set_facecolor('#0b0f19')
                    st.pyplot(fig1)
                    plt.close(fig1)

                with map2:
                    st.caption("Hallucination Risk Map")
                    fig2, ax2 = plt.subplots(figsize=(4, 4))
                    cax2 = ax2.imshow(hallucination, cmap="magma")
                    fig2.colorbar(cax2, ax=ax2, shrink=0.7)
                    ax2.axis("off")
                    fig2.patch.set_facecolor('#0b0f19')
                    st.pyplot(fig2)
                    plt.close(fig2)

            with col_m2:
                st.markdown("#### Quantitative Benchmarks")
                
                if consistency["passed"]:
                    st.success(f"✅ PASSED Anti-Hallucination\n\nConsistency: **{consistency['consistency_score']:.4f}**")
                else:
                    st.error(f"⚠️ FLAGGED for Hallucination\n\nConsistency: **{consistency['consistency_score']:.4f}**")

                if metrics is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>PSNR:</b> {metrics['psnr']:.2f} dB <br>
                        <b>SSIM:</b> {metrics['ssim']:.4f} <br>
                        <b>Edge F1:</b> {metrics.get('edge_f1', 0.0):.4f} <br>
                        <b>Fabrication Rate:</b> {metrics.get('hallucination_rate', 0.0)*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Detailed metrics require a ground truth reference image.")

            st.divider()

            # === 2. INTERACTIVE SWIPE SLIDER ===
            with st.expander("🔍 Interactive Enhancement Viewer (Swipe to Compare)", expanded=False):
                st.caption("Drag the slider left and right to compare the original 10m resolution to SpectraX's <4m resolution.")
                
                img_lr = bands_to_rgb(st.session_state.input_data)
                img_sr = bands_to_rgb(sr_output)
                img_lr_resized = img_lr.resize(img_sr.size, Image.NEAREST)

                image_comparison(
                    img1=img_lr_resized,
                    img2=img_sr,
                    label1="Original (10m/px)",
                    label2="SpectraX Output (<4m/px)",
                    width=850,
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )

st.divider()
st.caption("SpectraX SRM | AI-Powered Satellite Enhancement | SIH 2026")
