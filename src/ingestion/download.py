"""
download.py — Fetch Sentinel-2 L2A tiles from Copernicus Data Space Ecosystem.

Usage:
    from src.ingestion.download import download_sentinel2_tile
    download_sentinel2_tile(bbox, date_range, output_dir)
"""

import os
from pathlib import Path


def download_sentinel2_tile(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    output_dir: str | Path,
    max_cloud_cover: float = 20.0,
) -> Path:
    """
    Download a Sentinel-2 L2A tile from Copernicus Data Space.

    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84.
        date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format.
        output_dir: Directory to save the downloaded GeoTIFF.
        max_cloud_cover: Maximum acceptable cloud cover percentage (default 20%).

    Returns:
        Path to the downloaded GeoTIFF file.
    """
    # TODO: Implement Copernicus OData API download
    # Steps:
    # 1. Query Copernicus Data Space catalog for matching tiles
    # 2. Filter by cloud cover
    # 3. Download the best match
    # 4. Save as GeoTIFF to output_dir
    raise NotImplementedError(
        "Implement Copernicus Data Space API download. "
        "See: https://documentation.dataspace.copernicus.eu/"
    )
