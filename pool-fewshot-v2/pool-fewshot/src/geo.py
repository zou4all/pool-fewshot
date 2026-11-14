"""Tile <-> lat/lon conversions and GeoJSON export.

The IGN Geoplateforme "PM" TileMatrixSet is the standard slippy-map grid
(Google/OSM), so zoom levels are directly comparable.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable

TILE_PX = 256
EQUATOR_M = 156543.033928041  # metres per pixel at zoom 0 on the equator


def resolution(zoom: int, lat_deg: float) -> float:
    """Ground resolution in metres per pixel."""
    return EQUATOR_M * math.cos(math.radians(lat_deg)) / (2 ** zoom)


def deg2tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat_deg))) / math.pi) / 2.0 * n)
    return x, y


def tile2deg(x: float, y: float, zoom: int) -> tuple[float, float]:
    """North-west corner of tile (x, y). Fractional x/y address a pixel."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def pixel2deg(tile_x: int, tile_y: int, zoom: int,
              px: float, py: float, tile_px: int = TILE_PX) -> tuple[float, float]:
    """Pixel inside a mosaic -> (lat, lon). This is what geolocates a detection."""
    return tile2deg(tile_x + px / tile_px, tile_y + py / tile_px, zoom)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "pool"


def to_geojson(dets: Iterable[Detection], origin_tile: tuple[int, int],
               zoom: int, tile_px: int = TILE_PX) -> dict:
    """Detections in mosaic pixels -> WGS84 FeatureCollection.

    origin_tile is the (x, y) tile matching pixel (0, 0) of the mosaic.
    The result opens directly in QGIS, uMap or geojson.io.
    """
    ox, oy = origin_tile
    feats = []
    for d in dets:
        ring = []
        for px, py in ((d.x1, d.y1), (d.x2, d.y1), (d.x2, d.y2), (d.x1, d.y2)):
            lat, lon = pixel2deg(ox, oy, zoom, px, py, tile_px)
            ring.append([lon, lat])
        ring.append(ring[0])
        cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        lat_c, lon_c = pixel2deg(ox, oy, zoom, cx, cy, tile_px)
        res = resolution(zoom, lat_c)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "label": d.label,
                "score": round(float(d.score), 4),
                "area_m2": round((d.x2 - d.x1) * (d.y2 - d.y1) * res * res, 1),
                "lat": round(lat_c, 7),
                "lon": round(lon_c, 7),
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def save_geojson(path: str, gj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gj, fh, ensure_ascii=False, indent=1)
