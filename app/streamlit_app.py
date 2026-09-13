"""
SpectraX SRM — Streamlit MVP Dashboard

Reconstruct medium-resolution (10m) satellite imagery into sharper (<4m)
imagery with pixel-level uncertainty, hallucination detection,
and self-consistency verification.
"""

import os
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
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
    .main { background-color: #0E1117; }
    h1 { color: #4DA8DA; font-weight: 700; }
    h2, h3 { color: #E0E6ED; font-weight: 400; }
    .stAlert { border-radius: 8px; }
    .metric-card { 
        background: #1E2329; padding: 15px; border-radius: 8px; 
        border-left: 4px solid #4DA8DA; margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Caching ---
@st.cache_resource
def get_model(ckpt_path: str, device: str):
    return load_model(ckpt_path, device=device)

def extract_bands_from_zip(zip_path: str, extract_dir: str):
    """Finds B02, B03, B04, B08 inside a ZIP and extracts them."""
    extracted_files = {}
    band_mapping = {"B02": 0, "B03": 1, "B04": 2, "B08": 3}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if file_info.filename.endswith(('.tif', '.tiff', '.jp2')):
                for band_name in band_mapping.keys():
                    # Look for band name in the filename (e.g. T43PFP_20240101_B02.jp2)
                    if f"_{band_name}" in file_info.filename or file_info.filename.endswith(f"{band_name}.tif"):
                        # Extract it
                        zip_ref.extract(file_info, extract_dir)
                        extracted_files[band_mapping[band_name]] = os.path.join(extract_dir, file_info.filename)
                        break
    return extracted_files

def bands_to_rgb(img_bands: np.ndarray) -> Image.Image:
    """Converts (4, H, W) normalized array to a PIL Image (uint8 RGB) for display."""
    if img_bands.shape[0] >= 3:
        rgb = np.stack([img_bands[2], img_bands[1], img_bands[0]], axis=-1)
    else:
        rgb = np.repeat(img_bands[0:1], 3, axis=0).transpose(1, 2, 0)

    # Contrast stretch for vibrant colors
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_st = (rgb - p2) / (p98 - p2 + 1e-8)
    rgb_st = np.clip(rgb_st, 0.0, 1.0)
    
    return Image.fromarray((rgb_st * 255).astype(np.uint8))


# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/ISRO_Logo.svg/200px-ISRO_Logo.svg.png", width=80)
    st.title("⚙️ Control Panel")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.success(f"**Hardware:** `{device.upper()}` " + (f"({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # Checkpoint
    ckpt_files = sorted(Path("checkpoints").glob("*.pth"), key=os.path.getmtime, reverse=True)
    ckpt_options = [str(p) for p in ckpt_files] if ckpt_files else ["checkpoints/swinir_epoch025.pth"]
    selected_ckpt = st.selectbox("Model Weights", ckpt_options, index=0)

    st.divider()
    st.subheader("Inference Settings")
    n_passes = st.slider("Uncertainty Passes (MC-Dropout)", 2, 16, 4, 2)
    consistency_threshold = st.slider("Consistency Threshold", 0.70, 0.99, 0.85, 0.01)
    
    st.divider()
    st.info("SpectraX SRM leverages an RTX-accelerated SwinIR-Lite backbone with global skip connections.")


# --- Main Header ---
st.title("🛰️ SpectraX — Super Resolution Mapping")
st.markdown("Transform blurry 10m/pixel satellite data into sharp, tactically actionable <4m/pixel imagery.")

# --- Data Input ---
st.header("1. Upload Satellite Data")

tab_demo, tab_upload, tab_zip = st.tabs(["🚀 Quick Demo", "🖼️ Upload Image (JPG/PNG/TIF)", "🗂️ Upload Copernicus ZIP"])

input_data = None
ref_data = None
sample_name = ""

with tab_demo:
    st.markdown("Test the engine instantly using pre-processed data from Pune (Patch 0000).")
    if st.button("Load Demo (Pune Tile)", type="primary"):
        lr_sample = "data/processed/lr/patch_0000.npy"
        hr_sample = "data/processed/hr/patch_0000.npy"
        if os.path.exists(lr_sample) and os.path.exists(hr_sample):
            input_data = np.load(lr_sample)
            ref_data = np.load(hr_sample)
            sample_name = "Demo: Pune (Patch 0000)"
            st.session_state["input_data"] = input_data
            st.session_state["ref_data"] = ref_data
            st.session_state["sample_name"] = sample_name
        else:
            st.error("Demo files not found!")

with tab_upload:
    st.markdown("Upload any standard image. SpectraX handles missing infrared bands automatically.")
    up_file = st.file_uploader("Input Image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up_file.name).suffix) as tmp:
            tmp.write(up_file.read())
            data, _ = load_sentinel2_bands(tmp.name)
            input_data = normalize(data, method="percentile")
            sample_name = up_file.name
            st.session_state["input_data"] = input_data
            st.session_state["ref_data"] = None
            st.session_state["sample_name"] = sample_name

with tab_zip:
    st.markdown("Upload a raw `.zip` file from the Copernicus Data Space. We will extract the exact spectral bands needed.")
    zip_file = st.file_uploader("Upload Copernicus ZIP", type=["zip"])
    if zip_file:
        with st.spinner("Extracting multi-spectral bands..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.read())
                
                extracted = extract_bands_from_zip(zip_path, tmpdir)
                if len(extracted) > 0:
                    st.success(f"Successfully extracted {len(extracted)} spectral bands!")
                    # In a full implementation, we'd stack these. For the UI placeholder, we'll request JPG for now.
                    st.info("ZIP Extraction successful! (Full stacking pipeline to be connected). Use Image Upload for now.")
                else:
                    st.error("Could not find standard B02, B03, B04, B08 bands in this zip.")

# Restore session state
if "input_data" in st.session_state and input_data is None:
    input_data = st.session_state["input_data"]
    ref_data = st.session_state.get("ref_data", None)
    sample_name = st.session_state.get("sample_name", "")


# --- Inference & Results ---
if input_data is not None:
    st.divider()
    st.subheader(f"Current Target: `{sample_name}`")

    if not os.path.exists(selected_ckpt):
        st.error("Model weights not found. Please train the model first.")
    else:
        with st.spinner("Initializing Deep Neural Network..."):
            model = get_model(selected_ckpt, device=device)
            results = run_inference(
                model=model,
                input_image=input_data,
                reference_image=ref_data,
                n_uncertainty_passes=n_passes,
                scale_factor=3,
                patch_size=256,
                device=device,
            )

        sr_output = results["sr_output"]
        uncertainty = results["uncertainty_map"]
        hallucination = results["hallucination_map"]
        consistency = results["consistency"]
        metrics = results["metrics"]

        # === 1. INTERACTIVE SWIPE SLIDER ===
        st.header("2. Interactive Enhancement Viewer")
        st.caption("Drag the slider left and right to compare the original 10m resolution to SpectraX's <4m resolution.")
        
        # Convert arrays to PIL Images for the slider
        img_lr = bands_to_rgb(input_data)
        img_sr = bands_to_rgb(sr_output)

        # Resize LR to match SR for the slider component
        img_lr_resized = img_lr.resize(img_sr.size, Image.NEAREST)

        # Slider component
        image_comparison(
            img1=img_lr_resized,
            img2=img_sr,
            label1="Original (10m/px)",
            label2="SpectraX Output (<4m/px)",
            width=800,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

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
                fig1.patch.set_facecolor('#0E1117')
                st.pyplot(fig1)
                plt.close(fig1)

            with map2:
                st.caption("Hallucination Risk Map")
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                cax2 = ax2.imshow(hallucination, cmap="magma")
                fig2.colorbar(cax2, ax=ax2, shrink=0.7)
                ax2.axis("off")
                fig2.patch.set_facecolor('#0E1117')
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
                if ref_data is not None:
                    st.image(bands_to_rgb(ref_data), caption="Ground Truth Reference")
            else:
                st.info("Metrics require a ground truth reference image.")

st.divider()
st.caption("SpectraX SRM | AI-Powered Satellite Enhancement | SIH 2026")
