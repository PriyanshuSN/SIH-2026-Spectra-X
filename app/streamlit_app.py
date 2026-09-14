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
    h1, h2, h3, h4 { color: #f8f9fa; font-weight: 600; }
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
    /* Ensure images take full width of columns and can be expanded */
    img {
        border-radius: 8px;
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
        
    bands = []
    with rasterio.open(extracted_files[0]) as src:
        for i in range(4):
            with rasterio.open(extracted_files[i]) as b_src:
                bands.append(b_src.read(1))
    
    stacked = np.stack(bands, axis=0).astype(np.float32)
    return stacked

def load_large_demo_tile():
    """Loads a massive 1024x1024 crop from the raw stacked Pune TIF to show full capabilities."""
    path = "data/raw/pune_stacked.tif"
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        # Read a 1000x1000 chunk from the middle of the image
        data = src.read(window=((500, 1500), (500, 1500)))
    return data

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

    ckpt_dirs = [Path("checkpoints"), Path("checkpoints_bulk")]
    ckpt_files = []
    for cd in ckpt_dirs:
        if cd.exists():
            ckpt_files.extend(cd.glob("*.pth"))
    ckpt_files = sorted(ckpt_files, key=os.path.getmtime, reverse=True)
    ckpt_options = [str(p) for p in ckpt_files] if ckpt_files else ["checkpoints/swinir_epoch342.pth"]
    selected_ckpt = st.selectbox("Model Weights", ckpt_options, index=0)

    st.subheader("Inference Settings")
    scale_factor = st.selectbox("Resolution Upscale Factor", ["2x", "3x (Current Weights)", "4x"], index=1, help="2x and 4x require different weights. Defaulting to 3x.")
    n_passes = st.slider("Uncertainty Passes", 2, 16, 4, 2, help="Higher = more accurate hallucination detection but slower.")
    consistency_threshold = st.slider("Consistency Threshold", 0.70, 0.99, 0.85, 0.01)
    
    st.divider()
    st.info("SpectraX SRM uses SwinIR-Lite with global skip connections to ensure physics-based self-consistency.")


# --- Main Header ---
st.title("SpectraX — Super Resolution Engine")
st.markdown("Transform massive 10m/pixel satellite landscapes into sharp, tactically actionable <4m/pixel imagery.")

# --- Session State ---
if "input_data" not in st.session_state:
    st.session_state.input_data = None
    st.session_state.sample_name = ""
    st.session_state.ready_to_process = False

# --- Data Input ---
st.header("1. Data Ingestion")

tab_demo, tab_zip, tab_upload = st.tabs(["🚀 Massive Landscape Demo", "🗂️ Upload Copernicus ZIP", "🖼️ Upload Custom Image"])

with tab_demo:
    st.markdown("Test the engine on a **MASSIVE 1000x1000 pixel landscape** (100 square kilometers) from Pune. The AI will tile and process the entire region automatically.")
    if st.button("Load Full Pune Landscape"):
        data = load_large_demo_tile()
        if data is not None:
            st.session_state.input_data = normalize(data, method="percentile")
            st.session_state.sample_name = "Full Pune Landscape (100 sq km)"
            st.session_state.ready_to_process = False
        else:
            st.error("Demo file 'pune_stacked.tif' not found!")

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
                    st.session_state.ready_to_process = False
                    st.success("Successfully extracted and stacked 4 spectral bands! Ready to process.")
                else:
                    st.error("Could not find standard B02, B03, B04, B08 bands in this zip.")

with tab_upload:
    st.markdown("Upload any standard image (TIFF, PNG, JPG). SpectraX handles missing infrared bands automatically.")
    up_file = st.file_uploader("Input Image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up_file and st.button("Load Image"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up_file.name).suffix) as tmp:
            tmp.write(up_file.read())
            data, _ = load_sentinel2_bands(tmp.name)
            st.session_state.input_data = normalize(data, method="percentile")
            st.session_state.sample_name = up_file.name
            st.session_state.ready_to_process = False
            st.success("Image loaded! Ready to process.")


# --- Execution Workflow ---
if st.session_state.input_data is not None:
    st.divider()
    st.subheader(f"Current Target: `{st.session_state.sample_name}`")
    st.caption(f"Image Resolution: {st.session_state.input_data.shape[2]}x{st.session_state.input_data.shape[1]} pixels (Coverage: ~{(st.session_state.input_data.shape[2]*10/1000):.1f}km x {(st.session_state.input_data.shape[1]*10/1000):.1f}km)")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ RUN FULL SUPER RESOLUTION PIPELINE", type="primary", use_container_width=True):
            st.session_state.ready_to_process = True

    if st.session_state.ready_to_process:
        
        if not os.path.exists(selected_ckpt):
            st.error("Model weights not found. Please train the model first.")
        else:
            # Display real-time progress for massive images
            st.markdown("### AI Processing Pipeline")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                if total > 0:
                    pct = int((current / total) * 100)
                    progress_bar.progress(pct)
                    status_text.text(f"Processing chunk {current} of {total} via RTX GPU...")

            model = get_model(selected_ckpt, device=device)
            
            start_time = time.time()
            results = run_inference(
                model=model,
                input_image=st.session_state.input_data,
                reference_image=None,
                n_uncertainty_passes=n_passes,
                scale_factor=3,
                patch_size=256,
                device=device,
                progress_callback=update_progress
            )
            elapsed = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.success(f"✅ Full landscape processed successfully in {elapsed:.1f} seconds!")

            sr_output = results["sr_output"]
            uncertainty = results["uncertainty_map"]
            hallucination = results["hallucination_map"]
            consistency = results["consistency"]

            st.divider()

            # === 1. MASSIVE VISUALIZATION ===
            st.header("2. High-Resolution Output Viewer")
            
            img_lr = bands_to_rgb(st.session_state.input_data)
            img_sr = bands_to_rgb(sr_output)

            # Prepare download buffer
            import io
            buf = io.BytesIO()
            img_sr.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            view_tab_main, view_tab_zoom = st.tabs([
                "🖼️ Full Landscape Comparison",
                "🔍 Raw 1:1 Pixel Verification"
            ])
            
            with view_tab_main:
                st.markdown("Hover over either image and click **Arrows ⤢ in the top right** for **Full Screen Mode**. You can zoom in freely.")
                col_img1, col_img2 = st.columns(2)
                
                with col_img1:
                    st.subheader("Original (10m/px)")
                    st.image(img_lr, use_container_width=True)
                    
                with col_img2:
                    st.subheader("SpectraX Output (<4m/px)")
                    st.image(img_sr, use_container_width=True)
                    
                    st.download_button(
                        label="💾 Download Enhanced Output (PNG)",
                        data=byte_im,
                        file_name="spectrax_enhanced_output.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_btn_side"
                    )

            with view_tab_zoom:
                st.subheader("Raw Pixel Verification (1:1 Physical Scale)")
                st.markdown("Web browsers often shrink massive images to fit your monitor, making them look identical. To prove the true physical resolution enhancement, here is a raw **1:1 pixel crop** from the exact center of the map. No shrinking, no downsampling.")
                
                # Center crops
                w, h = img_lr.size
                cx, cy = w // 2, h // 2
                # Take a 150x150 chunk from original
                crop_lr = img_lr.crop((cx - 75, cy - 75, cx + 75, cy + 75))
                
                # Take the corresponding 450x450 chunk from SR
                sw, sh = img_sr.size
                scx, scy = sw // 2, sh // 2
                crop_sr = img_sr.crop((scx - 225, scy - 225, scx + 225, scy + 225))
                
                col_z1, col_z2 = st.columns(2)
                with col_z1:
                    st.markdown("**Original Crop (150x150 pixels)**")
                    st.image(crop_lr, use_container_width=False)
                with col_z2:
                    st.markdown("**SpectraX Crop (450x450 pixels) — Physically 3x larger**")
                    st.image(crop_sr, use_container_width=False)

            st.divider()

            # === 2. NTRO METRICS (Premium Layout) ===
            st.header("3. NTRO Verification Dashboard")
            
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
                
                st.info("Full fidelity metrics (PSNR, SSIM, SAM, ERGAS) are omitted because a ground-truth reference for this massive area was not provided.")

st.divider()
st.caption("SpectraX SRM | AI-Powered Satellite Enhancement | SIH 2026")
