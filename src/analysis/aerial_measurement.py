"""
Aerial Site Measurement — T4-5.

Enter a site address → geocode → fetch satellite tile (Google Maps Static API)
→ OpenCV HSV segmentation → detect footprint / road / laydown areas → sqm counts.

Matches attentive.ai's Automeasure product.

Optional dependencies:
  - GOOGLE_MAPS_API_KEY env var — for live geocode + satellite tiles
  - opencv-python (cv2) — for image segmentation

Graceful fallback: all functions return synthetic zero-result with
confidence="low" when dependencies are absent.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional cv2
HAS_CV2 = False
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# At Google Maps zoom=18, one pixel ≈ 0.597 m at equator.
# For India (lat≈22°N): pixel_size_m ≈ 0.597 * cos(22°) ≈ 0.554 m
# → sqm per pixel ≈ 0.307
_SQM_PER_PIXEL_ZOOM18 = 0.307
_TILE_SIZE = 640  # default Google Maps Static API tile size


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SiteMeasurement:
    address: str
    lat: float = 0.0
    lon: float = 0.0
    zoom: int = 18
    total_site_sqm: float = 0.0
    built_footprint_sqm: float = 0.0
    access_road_sqm: float = 0.0
    laydown_sqm: float = 0.0
    confidence: str = "low"          # "high"|"medium"|"low"
    tile_url: str = ""
    detected_features: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_address(address: str, api_key: str = "") -> Tuple[float, float]:
    """
    Geocode an address string → (lat, lon).

    Uses Google Geocoding API when api_key is provided.
    Falls back to (0.0, 0.0) on missing key or error.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        logger.debug("No GOOGLE_MAPS_API_KEY — geocode fallback")
        return 0.0, 0.0

    try:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({"address": address, "key": api_key})
        url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        if data.get("status") == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
    except Exception as exc:
        logger.debug("Geocode error: %s", exc)
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Satellite tile fetch
# ---------------------------------------------------------------------------

def fetch_satellite_tile(lat: float, lon: float, zoom: int = 18,
                         api_key: str = "", size: str = "640x640") -> Optional[bytes]:
    """
    Fetch a Google Maps Static API satellite tile.

    Returns PNG bytes, or None on error / missing key.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key or (lat == 0.0 and lon == 0.0):
        return None

    try:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": size,
            "maptype": "satellite",
            "key": api_key,
        })
        url = f"https://maps.googleapis.com/maps/api/staticmap?{params}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read()
    except Exception as exc:
        logger.debug("Satellite tile fetch error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

def detect_site_features(image_bytes: bytes, zoom: int = 18) -> dict:
    """
    Segment a satellite tile using OpenCV HSV colour segmentation.

    Colour ranges (HSV):
      - Green  (vegetation)     H: 35-85   S: 40-255  V: 40-255
      - Grey   (impervious)     H: 0-180   S: 0-50    V: 80-200
      - Brown  (bare earth)     H: 5-30    S: 40-200  V: 40-180

    Returns dict:
        {total, footprint, road, laydown, confidence, pixel_scale_sqm}

    When HAS_CV2=False: returns synthetic zero result.
    """
    sqm_per_px = _SQM_PER_PIXEL_ZOOM18 * (18 / max(zoom, 1))

    if not HAS_CV2 or image_bytes is None:
        return {
            "total": 0.0, "footprint": 0.0, "road": 0.0, "laydown": 0.0,
            "confidence": "low", "pixel_scale_sqm": sqm_per_px,
        }

    try:
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        total_pixels = h * w

        # Vegetation (green)
        veg_mask = cv2.inRange(hsv,
                               np.array([35, 40, 40]),
                               np.array([85, 255, 255]))

        # Impervious surfaces (grey — roads, concrete)
        road_mask = cv2.inRange(hsv,
                                np.array([0, 0, 80]),
                                np.array([180, 50, 200]))

        # Bare earth (brown — laydown, excavated areas)
        brown_mask = cv2.inRange(hsv,
                                 np.array([5, 40, 40]),
                                 np.array([30, 200, 180]))

        veg_px  = int(cv2.countNonZero(veg_mask))
        road_px = int(cv2.countNonZero(road_mask))
        earth_px = int(cv2.countNonZero(brown_mask))

        # Built footprint ≈ non-vegetation impervious area
        footprint_px = max(0, total_pixels - veg_px - earth_px)

        confidence = "high" if total_pixels > 100_000 else "medium"

        return {
            "total":      round(total_pixels * sqm_per_px, 1),
            "footprint":  round(footprint_px  * sqm_per_px, 1),
            "road":       round(road_px        * sqm_per_px, 1),
            "laydown":    round(earth_px       * sqm_per_px, 1),
            "confidence": confidence,
            "pixel_scale_sqm": sqm_per_px,
        }

    except Exception as exc:
        logger.debug("detect_site_features error: %s", exc)
        return {
            "total": 0.0, "footprint": 0.0, "road": 0.0, "laydown": 0.0,
            "confidence": "low", "pixel_scale_sqm": sqm_per_px,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def measure_site(address: str, api_key: str = "", zoom: int = 18) -> SiteMeasurement:
    """
    Full aerial measurement pipeline:
      geocode_address → fetch_satellite_tile → detect_site_features
      → SiteMeasurement

    When API key is absent or any step fails, returns confidence="low" with
    zero areas (never crashes).
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    try:
        lat, lon = geocode_address(address, api_key)
    except Exception as exc:
        logger.debug("geocode_address error: %s", exc)
        lat, lon = 0.0, 0.0

    tile_url = ""
    if api_key and lat != 0.0:
        import urllib.parse
        params = urllib.parse.urlencode({
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": "640x640",
            "maptype": "satellite",
            "key": api_key,
        })
        tile_url = f"https://maps.googleapis.com/maps/api/staticmap?{params}"

    image_bytes = fetch_satellite_tile(lat, lon, zoom, api_key)
    features = detect_site_features(image_bytes, zoom)

    detected = []
    if features["footprint"] > 0:
        detected.append({"label": "built_footprint",  "area_sqm": features["footprint"]})
    if features["road"] > 0:
        detected.append({"label": "access_road",      "area_sqm": features["road"]})
    if features["laydown"] > 0:
        detected.append({"label": "laydown_area",     "area_sqm": features["laydown"]})

    return SiteMeasurement(
        address=address,
        lat=lat,
        lon=lon,
        zoom=zoom,
        total_site_sqm=features["total"],
        built_footprint_sqm=features["footprint"],
        access_road_sqm=features["road"],
        laydown_sqm=features["laydown"],
        confidence=features["confidence"],
        tile_url=tile_url,
        detected_features=detected,
    )
