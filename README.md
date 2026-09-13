# SpectraX — Deep Learning Super Resolution Mapping (SRM)

> Reconstruct medium-resolution (10m) Sentinel-2 satellite imagery into sharper (<4m) imagery with **pixel-level uncertainty** and **self-consistency verification**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cost: $0](https://img.shields.io/badge/Cost-%240-green.svg)](#)

## Problem

High-resolution satellite tasking is expensive and slow. Free Sentinel-2 imagery (10m/pixel) is too coarse for fine-grained analysis. Generic AI upscalers hallucinate detail that was never observed — dangerous when outputs inform real decisions.

## What SpectraX Does Differently

- **Super-Resolution**: Upscales 4-band (B/G/R/NIR) Sentinel-2 tiles from 10m to <4m effective resolution
- **Uncertainty Heatmap**: MC-Dropout generates a per-pixel confidence map — know *where* to trust the output
- **Self-Consistency Check**: Degrades the output back to 10m and compares against the original — verifiable, not just "prettier"
- **Spectral Fidelity**: Joint 4-band processing preserves spectral relationships (critical for vegetation indices, NDVI, etc.)
- **Objective Metrics**: PSNR, SSIM, SAM, ERGAS reported for every tile

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/spectrax-srm.git
cd spectrax-srm

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

## Architecture

```
Sentinel-2 Tile (10m, 4-band)
        │
        ▼
  Preprocessing (Rasterio/GDAL — align, tile, normalize)
        │
        ▼
  SR Model (SwinIR-inspired, joint 4-band)
        │
        ├──► MC-Dropout Uncertainty (N stochastic passes → confidence heatmap)
        │
        ├──► Consistency Check (degrade & compare → pass/fail score)
        │
        └──► Metrics Engine (PSNR / SSIM / SAM / ERGAS)
        │
        ▼
  Streamlit Dashboard (upload → view results)
```

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Data Source | Copernicus Data Space (Sentinel-2) | Free |
| Geospatial | Rasterio, GDAL, NumPy | Free |
| Model | PyTorch, SwinIR-inspired backbone | Free |
| Metrics | scikit-image, torchmetrics | Free |
| UI | Streamlit + Leaflet (streamlit-folium) | Free |
| Hosting | Hugging Face Spaces | Free |
| Training | Google Colab / Kaggle (free GPU) | Free |

**Total cost: $0**

## Team

SpectraX — SIH 2026 | Problem Statement 26142 | Sponsor: NTRO

## License

[MIT](LICENSE)
