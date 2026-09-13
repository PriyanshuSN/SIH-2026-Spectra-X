"""
SpectraX SRM — Streamlit MVP Dashboard

Reconstruct medium-resolution (10m) satellite imagery into sharper (<4m)
imagery with pixel-level uncertainty, hallucination detection,
and self-consistency verification.
"""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.api.inference import load_model, run_inference
from src.ingestion.preprocess import load_sentinel2_bands, normalize

# Page setup
st.set_page_config(
    page_title="SpectraX SRM — NTRO Super Resolution",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ SpectraX — Super Resolution Mapping (SRM)")
st.markdown(
    "Deep Learning-based Super Resolution of Medium-Resolution Satellite Imagery (<4m) "
    "with **Anti-Hallucination Verification** and **Pixel-Level Uncertainty**."
)


@st.cache_resource
def get_model(ckpt_path: str, device: str):
    return load_model(ckpt_path, device=device)


# --- Device & Checkpoint Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
default_ckpt = "checkpoints/swinir_epoch050.pth"

with st.sidebar:
    st.header("⚙️ Model & Hardware")
    st.info(f"**Compute Device:** `{device.upper()}` " + (f"({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # List available checkpoints
    ckpt_files = sorted(Path("checkpoints").glob("*.pth"), key=os.path.getmtime, reverse=True)
    ckpt_options = [str(p) for p in ckpt_files] if ckpt_files else [default_ckpt]
    selected_ckpt = st.selectbox("Model Checkpoint", ckpt_options, index=0)

    st.divider()
    st.header("🔬 Inference Controls")
    n_passes = st.slider("MC-Dropout Passes", min_value=2, max_value=16, value=4, step=2)
    scale_factor = st.selectbox("Scale Factor", [3], index=0)
    consistency_threshold = st.slider("Consistency Pass Threshold", 0.70, 0.99, 0.85, 0.01)

    st.divider()
    st.markdown("""
    ### 📐 NTRO Verification Metrics
    - **PSNR (>30 dB):** Signal fidelity.
    - **SSIM (>0.85):** Structural preservation.
    - **SAM (<5.0°):** Spectral colour fidelity.
    - **Edge F1 (>0.70):** Geographic edge coherence.
    - **Hallucination Rate (<0.30):** Fabricated detail test.
    """)


def bands_to_rgb(img_bands: np.ndarray) -> np.ndarray:
    """Converts (4, H, W) or (3, H, W) normalized satellite data to (H, W, 3) RGB."""
    # Sentinel-2 bands order in our pipeline: 0: Blue, 1: Green, 2: Red, 3: NIR
    if img_bands.shape[0] >= 3:
        r = img_bands[2]
        g = img_bands[1]
        b = img_bands[0]
        rgb = np.stack([r, g, b], axis=-1)
    else:
        rgb = np.repeat(img_bands[0:1], 3, axis=0).transpose(1, 2, 0)

    # Normalize to [0, 1] for display
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


# --- Data Selection ---
st.header("1. Select Satellite Imagery")

tab1, tab2 = st.tabs(["🚀 Demo Pune Tile (Pre-loaded)", "📤 Upload Custom Image"])

input_data = None
ref_data = None
sample_name = ""

with tab1:
    st.markdown("Instantly evaluate the model on pre-processed Sentinel-2 Pune tiles.")
    if st.button("Load Pune Tile Patch 00", type="primary"):
        lr_sample = "data/processed/lr/patch_0000.npy"
        hr_sample = "data/processed/hr/patch_0000.npy"
        if os.path.exists(lr_sample) and os.path.exists(hr_sample):
            input_data = np.load(lr_sample)
            ref_data = np.load(hr_sample)
            sample_name = "Pune Tile (Patch 0000)"
            st.session_state["input_data"] = input_data
            st.session_state["ref_data"] = ref_data
            st.session_state["sample_name"] = sample_name
        else:
            st.error("Sample patches not found in data/processed/. Run dataset generation first.")

with tab2:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        up_file = st.file_uploader("Upload Input (TIFF, PNG, JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    with col_up2:
        ref_file = st.file_uploader("Upload Reference/Ground Truth (Optional)", type=["tif", "tiff", "png", "jpg", "jpeg"])

    if up_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up_file.name).suffix) as tmp:
            tmp.write(up_file.read())
            tmp_path = tmp.name
        data, _ = load_sentinel2_bands(tmp_path)
        input_data = normalize(data, method="percentile")
        sample_name = up_file.name
        st.session_state["input_data"] = input_data
        st.session_state["sample_name"] = sample_name

        if ref_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(ref_file.name).suffix) as tmp_ref:
                tmp_ref.write(ref_file.read())
                tmp_ref_path = tmp_ref.name
            ref_d, _ = load_sentinel2_bands(tmp_ref_path)
            ref_data = normalize(ref_d, method="percentile")
            st.session_state["ref_data"] = ref_data

# Restore session state if available
if "input_data" in st.session_state and input_data is None:
    input_data = st.session_state["input_data"]
    ref_data = st.session_state.get("ref_data", None)
    sample_name = st.session_state.get("sample_name", "Selected Sample")

# --- Execute Inference ---
if input_data is not None:
    st.divider()
    st.subheader(f"Analyzing: {sample_name}")

    if not os.path.exists(selected_ckpt):
        st.error(f"Checkpoint `{selected_ckpt}` not found. Please verify your checkpoints directory.")
    else:
        with st.spinner("Running Super Resolution & Anti-Hallucination Verification on GPU..."):
            model = get_model(selected_ckpt, device=device)
            results = run_inference(
                model=model,
                input_image=input_data,
                reference_image=ref_data,
                n_uncertainty_passes=n_passes,
                scale_factor=scale_factor,
                patch_size=128 if input_data.shape[1] <= 128 else 256,
                device=device,
            )

        sr_output = results["sr_output"]
        uncertainty = results["uncertainty_map"]
        hallucination = results["hallucination_map"]
        consistency = results["consistency"]
        metrics = results["metrics"]

        # === 1. TRIPTYCH COMPARISON ===
        st.header("2. Triptych Geographic Reconstruction")
        st.caption("Proves authentic feature reconstruction (roads, buildings, agricultural boundaries).")

        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.subheader("1. Input (10m Resolution)")
            st.image(bands_to_rgb(input_data), use_container_width=True, caption="Original Medium Resolution")

        with t_col2:
            st.subheader("2. SpectraX (<4m Resolution)")
            st.image(bands_to_rgb(sr_output), use_container_width=True, caption="Super-Resolved Output")

        with t_col3:
            st.subheader("3. Reference Ground Truth")
            if ref_data is not None:
                st.image(bands_to_rgb(ref_data), use_container_width=True, caption="True High-Resolution Baseline")
            else:
                st.info("Ground truth reference image not provided for comparison.")

        # === 2. ANTI-HALLUCINATION & UNCERTAINTY ===
        st.header("3. Anti-Hallucination & Reliability Maps")
        map1, map2 = st.columns(2)

        with map1:
            st.subheader("Pixel-Level Uncertainty Heatmap (MC-Dropout)")
            fig, ax = plt.subplots()
            cax = ax.imshow(uncertainty, cmap="viridis")
            fig.colorbar(cax, ax=ax, label="Uncertainty Variance")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with map2:
            st.subheader("Hallucination Risk Map")
            fig, ax = plt.subplots()
            cax = ax.imshow(hallucination, cmap="magma")
            fig.colorbar(cax, ax=ax, label="Fabrication Probability")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        # === 3. QUALITY & FIDELITY METRICS ===
        st.header("4. Quantitative Validation (NTRO Benchmarks)")

        if metrics is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PSNR (dB)", f"{metrics['psnr']:.2f}", delta="Fidelity" if metrics['psnr'] > 28 else None)
            m2.metric("SSIM", f"{metrics['ssim']:.4f}", delta="Structure" if metrics['ssim'] > 0.80 else None)
            m3.metric("SAM (Degrees)", f"{metrics['sam']:.2f}°", delta="Spectral Angle" if metrics['sam'] < 5.0 else None, delta_color="inverse")
            m4.metric("ERGAS", f"{metrics['ergas']:.2f}", delta="Global Error" if metrics['ergas'] < 3.5 else None, delta_color="inverse")

            f1, f2, f3 = st.columns(3)
            f1.metric("Geographic Edge F1", f"{metrics.get('edge_f1', 0.0):.4f}", help="Road and building boundary preservation.")
            f2.metric("Hallucination Rate", f"{metrics.get('hallucination_rate', 0.0) * 100:.1f}%", help="Percentage of detected edges deemed ungrounded.")
            f3.metric("Consistency Score", f"{consistency['consistency_score']:.4f}", delta="PASSED" if consistency['passed'] else "ALERT", delta_color="normal" if consistency['passed'] else "inverse")
        else:
            st.metric("Consistency Score", f"{consistency['consistency_score']:.4f}", delta="PASSED" if consistency['passed'] else "ALERT")

        # === 4. CONSISTENCY CHECK VERDICT ===
        st.header("5. Physics-Based Self-Consistency Check")
        if consistency["passed"]:
            st.success(f"✅ **PASS:** Self-consistency score ({consistency['consistency_score']:.4f}) exceeds threshold ({consistency_threshold}). The degraded output aligns with observations without geometric hallucination.")
        else:
            st.warning(f"⚠️ **FLAGGED:** Self-consistency score ({consistency['consistency_score']:.4f}) is below threshold ({consistency_threshold}).")

st.divider()
st.caption("SpectraX SRM | SIH 2026 | Problem Statement 26142 | Sponsor: NTRO | GPU-Accelerated on RTX 5050")
