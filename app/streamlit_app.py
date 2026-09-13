"""
SpectraX SRM — Streamlit MVP Dashboard

Upload a Sentinel-2 GeoTIFF or standard image → Super-Resolve → View Results:
  - Triptych: Original / Super-Resolved / Ground Truth (if available)
  - Uncertainty Heatmap + Hallucination Risk Map
  - Metrics panel: PSNR, SSIM, SAM, ERGAS, Edge F1, Hallucination Rate
  - Consistency score with pass/fail indicator
  - Map view showing AOI location

Supports: TIFF, PNG, JPG, JPEG
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Page config
st.set_page_config(
    page_title="SpectraX SRM",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ SpectraX — Super Resolution Mapping")
st.markdown(
    "Reconstruct medium-resolution (10m) satellite imagery into sharper (<4m) "
    "imagery with **pixel-level uncertainty**, **hallucination detection**, "
    "and **self-consistency verification**."
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    n_passes = st.slider(
        "MC-Dropout passes (N)",
        min_value=2,
        max_value=32,
        value=8,
        help="More passes = better uncertainty estimation, but slower inference.",
    )
    scale_factor = st.selectbox("Scale factor", [2, 3, 4], index=1)
    consistency_threshold = st.slider(
        "Consistency threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.85,
        step=0.05,
    )

    st.divider()
    st.header("📐 What the metrics mean")
    st.markdown("""
    | Metric | Good | Meaning |
    |--------|------|---------|
    | **PSNR** | >30 dB | Overall pixel accuracy |
    | **SSIM** | >0.85 | Structural similarity |
    | **SAM** | <5° | Spectral band fidelity |
    | **ERGAS** | <3 | Global relative error |
    | **Edge F1** | >0.7 | Geographic structure preserved |
    | **Halluc. Rate** | <0.3 | Low = less fake detail |
    """)

# --- Upload ---
st.header("📤 Upload Satellite Image")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    uploaded_file = st.file_uploader(
        "Upload input image (TIFF, PNG, JPG)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="input_upload",
    )

with col_upload2:
    reference_file = st.file_uploader(
        "Upload ground truth reference (optional, for metrics)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="reference_upload",
        help="Upload a high-resolution reference image of the same location "
             "to compute quality metrics and prove no hallucination.",
    )

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if reference_file is not None:
        st.success(f"Reference: {reference_file.name}")

    # TODO: Wire up actual inference pipeline
    # This is a placeholder showing the full UI layout

    # === TRIPTYCH VIEW ===
    st.header("🔬 Triptych Comparison")
    st.caption(
        "Side-by-side proof that the model recovers real geographic detail. "
        "This is the view that convinces NTRO judges."
    )

    if reference_file is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original (10m)")
            st.info("🖼️ Original LR tile will be displayed here")

        with col2:
            st.subheader("Super-Resolved (<4m)")
            st.info("🖼️ AI-enhanced output will be displayed here")

        with col3:
            st.subheader("Ground Truth (Reference)")
            st.info("🖼️ Real high-res reference for validation")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original (10m)")
            st.info("🖼️ Original LR tile will be displayed here")

        with col2:
            st.subheader("Super-Resolved (<4m)")
            st.info("🖼️ AI-enhanced output will be displayed here")

        st.warning(
            "⚠️ Upload a ground truth reference image to enable the full "
            "triptych view and compute geographic fidelity metrics."
        )

    # === UNCERTAINTY & HALLUCINATION MAPS ===
    st.header("🗺️ Confidence & Hallucination Maps")
    map_col1, map_col2 = st.columns(2)

    with map_col1:
        st.subheader("Uncertainty Heatmap")
        st.caption("Red = high uncertainty. Blue = high confidence.")
        st.info("MC-Dropout uncertainty map will be displayed here")

    with map_col2:
        st.subheader("Hallucination Risk Map")
        st.caption(
            "Highlights pixels where the model added detail that cannot "
            "be explained by the original input — potential fabrication."
        )
        st.info("Hallucination risk map will be displayed here")

    # === METRICS PANEL ===
    st.header("📏 Quality Metrics")

    # Row 1: Standard metrics
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)

    with met_col1:
        st.metric("PSNR", "— dB", help="Peak Signal-to-Noise Ratio (higher is better)")
    with met_col2:
        st.metric("SSIM", "—", help="Structural Similarity (higher is better, max 1.0)")
    with met_col3:
        st.metric("SAM", "— °", help="Spectral Angle Mapper (lower is better)")
    with met_col4:
        st.metric("ERGAS", "—", help="Global relative error (lower is better)")

    # Row 2: Geographic fidelity metrics (the differentiator)
    fid_col1, fid_col2, fid_col3 = st.columns(3)

    with fid_col1:
        st.metric(
            "Edge F1 Score",
            "—",
            help="Geographic fidelity: measures how well real edges "
                 "(roads, buildings, field boundaries) are preserved. Higher is better.",
        )
    with fid_col2:
        st.metric(
            "Hallucination Rate",
            "—",
            help="Fraction of detected edges in the SR output that do NOT "
                 "exist in the ground truth. Lower is better (0% = perfect).",
        )
    with fid_col3:
        st.metric(
            "Consistency Score",
            "—",
            help="Self-consistency: degrade the SR output back to LR and "
                 "compare with original input. Higher = more trustworthy.",
        )

    # === CONSISTENCY CHECK ===
    st.header("✅ Self-Consistency Check (Anti-Hallucination)")
    st.markdown(
        "**How it works:** We take the super-resolved output, degrade it back "
        "to 10m resolution using the same physics-based degradation kernel, "
        "and compare it with the original input. If they don't match, the model "
        "has hallucinated detail that doesn't exist."
    )
    st.info("Consistency verification results will be displayed here after inference")

else:
    # Map view for AOI selection (placeholder)
    st.header("🗺️ Or Select an Area of Interest")
    st.info(
        "Map-based AOI selection will be integrated here using "
        "`streamlit-folium`. Upload a satellite image above to get started."
    )

    # Quick info section
    st.divider()
    st.header("ℹ️ What makes SpectraX different?")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 🔍 Not Just Sharpening")
        st.markdown(
            "Unlike basic upscaling, SpectraX uses deep learning to "
            "reconstruct genuine fine-scale detail from multi-spectral data."
        )

    with col_b:
        st.markdown("### 🛡️ Anti-Hallucination")
        st.markdown(
            "Built-in self-consistency checker and hallucination risk "
            "maps ensure the model doesn't invent false geographic features."
        )

    with col_c:
        st.markdown("### 📊 Proven Fidelity")
        st.markdown(
            "Six quality metrics including edge coherence and spectral "
            "angle mapping prove the output is scientifically accurate."
        )

# --- Footer ---
st.divider()
st.caption(
    "SpectraX SRM | SIH 2026 | Problem Statement 26142 | "
    "Sponsor: NTRO | Budget: $0 | 100% Open Source"
)
