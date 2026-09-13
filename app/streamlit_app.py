"""
SpectraX SRM — Streamlit MVP Dashboard

Upload a Sentinel-2 GeoTIFF → Super-Resolve → View Results:
  - Side-by-side: Original / Reconstructed / Uncertainty Heatmap
  - Metrics panel: PSNR, SSIM, SAM, ERGAS
  - Consistency score with pass/fail indicator
  - Map view showing AOI location
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
    "imagery with **pixel-level uncertainty** and **self-consistency verification**."
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

# --- Upload ---
st.header("📤 Upload Sentinel-2 Tile")
uploaded_file = st.file_uploader(
    "Upload a GeoTIFF file (B/G/R/NIR bands)",
    type=["tif", "tiff"],
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    # TODO: Wire up actual inference pipeline
    # This is a placeholder showing the UI layout

    st.header("📊 Results")

    # Three-column layout for images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original (10m)")
        # TODO: Display original RGB composite
        st.info("Original tile will be displayed here")

    with col2:
        st.subheader("Super-Resolved (<4m)")
        # TODO: Display SR output RGB composite
        st.info("SR output will be displayed here")

    with col3:
        st.subheader("Uncertainty Heatmap")
        # TODO: Display uncertainty heatmap
        st.info("Uncertainty map will be displayed here")

    # Metrics panel
    st.header("📏 Quality Metrics")
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)

    with met_col1:
        st.metric("PSNR", "— dB", help="Peak Signal-to-Noise Ratio (higher is better)")
    with met_col2:
        st.metric("SSIM", "—", help="Structural Similarity (higher is better, max 1.0)")
    with met_col3:
        st.metric("SAM", "— °", help="Spectral Angle Mapper (lower is better)")
    with met_col4:
        st.metric("ERGAS", "—", help="Global relative error (lower is better)")

    # Consistency check
    st.header("✅ Consistency Check")
    st.info("Consistency score will be displayed here after inference")

else:
    # Map view for AOI selection (placeholder)
    st.header("🗺️ Or Select an Area of Interest")
    st.info(
        "Map-based AOI selection will be integrated here using "
        "`streamlit-folium`. Upload a GeoTIFF above to get started."
    )

# --- Footer ---
st.divider()
st.caption(
    "SpectraX SRM | SIH 2026 | Problem Statement 26142 | "
    "Sponsor: NTRO | Budget: $0"
)
