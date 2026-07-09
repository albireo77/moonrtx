import json
import os
import cv2
from typing import Optional

import numpy as np

from moonrtx.shared_types import MoonFeature

from plotoptix.utils import read_image, make_color_2d

# Processed-array disk caches: reading the 7.9 GB elevation TIFF and block-mean
# downscaling it takes about a minute on every start, while np.load of the
# ready-made float32 result takes seconds. A cache is valid when the sidecar
# JSON matches the source file (size, mtime) and the processing parameters;
# any read or write problem silently falls back to the regular path, so a
# broken cache can only cost time, never correctness. Bump the version when
# the processing itself changes.
_CACHE_VERSION = 1


def _cache_fingerprint(filepath: str, **params) -> dict:
    return {
        "version": _CACHE_VERSION,
        "source_size": os.path.getsize(filepath),
        "source_mtime": int(os.path.getmtime(filepath)),
        **params,
    }


def _load_cache(cache_base: str, fingerprint: dict) -> tuple[Optional[np.ndarray], dict]:
    try:
        with open(cache_base + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if all(meta.get(k) == v for k, v in fingerprint.items()):
            return np.load(cache_base + ".npy"), meta
    except Exception:
        pass
    return None, {}


def _save_cache(cache_base: str, array: np.ndarray, meta: dict):
    try:
        np.save(cache_base + ".npy", array)
        with open(cache_base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print(f"  Cached to {cache_base}.npy for faster next start")
    except Exception as e:
        print(f"Warning: could not write cache {cache_base}.npy: {e}")

def load_moon_features(filepath: str) -> list:
    """
    Load Moon features from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file with columns: name, latitude, longitude, angular_size,
        standard_label, spot_label, status_bar, optional feature_id,
        optional www address.
        Separator is ':'
        
    Returns
    -------
    list
        List of MoonFeature entries parsed from the CSV file.
    """
    moon_features = []
    if not os.path.isfile(filepath):
        print(f"Warning: Moon features file {filepath} was not found. Features not loaded.")
        return moon_features
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':')
                if len(parts) >= 7:
                    name = parts[0].strip()
                    # Handle Unicode minus sign (−) and regular minus (-)
                    lat_str = parts[1].strip().replace('−', '-')
                    lon_str = parts[2].strip().replace('−', '-')
                    diameter_km_str = parts[3].strip()
                    standard_label = parts[4].strip().lower() == 'true'
                    spot_label = parts[5].strip().lower() == 'true'
                    status_bar = parts[6].strip().lower() == 'true'
                    feature_id_str = parts[7].strip() if len(parts) >= 8 else ''
                    www_address_str = parts[8].strip() if len(parts) >= 9 else ''
                    try:
                        lat = float(lat_str)
                        diameter_km = float(diameter_km_str)
                        moon_feature = MoonFeature(
                            name=name,
                            lat=lat,
                            lon=float(lon_str),
                            angular_radius=diameter_km / 60.647,
                            cos_lat=np.cos(np.radians(lat)),
                            diameter_km=diameter_km,
                            standard_label=standard_label,
                            spot_label=spot_label,
                            status_bar=status_bar,
                            feature_id=int(feature_id_str) if feature_id_str else None,
                            www_address=www_address_str or None,
                        )
                        moon_features.append(moon_feature)
                    except ValueError as e:
                        print(f"Warning: Could not load Moon feature named {name}: {e}")
                        continue
    except Exception as e:
        print(f"Warning: Could not load Moon features file: {e}")
    
    return moon_features

# LOLA LDEM products store elevation as signed 16-bit integers, 0.5 m per unit,
# relative to the reference Moon radius of 1737.4 km.
LDEM_METERS_PER_UNIT = 0.5
MOON_REFERENCE_RADIUS_M = 1_737_400.0


def load_elevation_data(filepath: str, downscale: int) -> tuple[np.ndarray, float]:
    """
    Load and process the Moon elevation data (LOLA LDEM TIFF).

    Parameters
    ----------
    filepath : str
        Path to the elevation TIFF file
    downscale : int
        Downscale factor (2-3 recommended for most GPUs)

    Returns
    -------
    tuple[np.ndarray, float]
        Elevation as displacement factors using the physical LDEM value scale of
        0.5 m per unit, so the relief amplitude is exact, and the radius scale
        needed to convert the factors back to physical elevation. The factors are
        normalized so that the highest peak is exactly 1.0: the displaced surface
        must not extend beyond the geometry bounding sphere, otherwise ray
        intersection tests miss the terrain (light leaks onto the night side).
    """
    print(f"Loading elevation data from {filepath}...")

    # Disk cache of the processed result (skipped at downscale 1, where the
    # cache would be a ~16 GB file for little gain over reading the source)
    cache_base = f"{filepath}.ds{downscale}"
    fingerprint = None
    if downscale > 1:
        fingerprint = _cache_fingerprint(filepath, downscale=downscale)
        elevation, meta = _load_cache(cache_base, fingerprint)
        if elevation is not None:
            print(f"  Loaded from cache: {cache_base}.npy, dimensions {elevation.shape}")
            return elevation, float(meta["radius_scale"])

    elev_src = read_image(filepath)

    if elev_src is None:
        raise ValueError(f"Failed to read elevation file: {filepath}")

    print(f"  Original dimensions: {elev_src.shape}")
    print(f"  Size: {elev_src.nbytes / (1024**3):.2f} GB")

    # Reinterpret as signed 16-bit and convert to displacement factor of the radius
    elev_src.dtype = np.int16
    scale = LDEM_METERS_PER_UNIT / MOON_REFERENCE_RADIUS_M

    if downscale == 1:
        # No downscaling needed, just convert to float
        elevation = elev_src.astype(np.float32) * scale
    else:
        # Downscale by averaging
        h = elev_src.shape[0] // downscale
        w = elev_src.shape[1] // downscale
        elevation = elev_src.reshape(1, h, downscale, w, downscale).mean(
            4, dtype=np.float32).mean(2, dtype=np.float32).reshape(h, w)
        elevation *= scale

    # Release source memory
    elev_src = None

    elevation += 1.0

    print(f"  Downscaled dimensions: {elevation.shape}")
    print(f"  Downscaled size: {elevation.nbytes / (1024**3):.2f} GB")
    print("  Relief range: {:.0f} m to {:+.0f} m relative to the 1737.4 km reference radius".format(
        (elevation.min() - 1.0) * MOON_REFERENCE_RADIUS_M,
        (elevation.max() - 1.0) * MOON_REFERENCE_RADIUS_M))

    # Keep the surface inside the bounding sphere: highest peak = exactly 1.0
    radius_scale = float(elevation.max())
    elevation /= radius_scale

    if fingerprint is not None:
        _save_cache(cache_base, elevation, {**fingerprint, "radius_scale": radius_scale})

    return elevation, radius_scale


def load_color_data(filepath: str, gamma: float = 2.2) -> np.ndarray:
    """
    Load and process the Moon color/albedo data.
    
    Parameters
    ----------
    filepath : str
        Path to the color TIFF file
    gamma : float
        Gamma correction value
        
    Returns
    -------
    np.ndarray
        Processed color data ready for texturing
    """
    print(f"Loading color data from {filepath}...")
    color_src = cv2.imread(filepath)
    
    if color_src is None:
        raise ValueError(f"Failed to read color file: {filepath}")
    
    # Convert BGR to RGB and normalize
    color_src = color_src[..., ::-1].astype(np.float32)
    color_src = 0.2 + (0.75 / 255) * color_src
    
    print(f"  Dimensions: {color_src.shape}")
    print(f"  Size: {color_src.nbytes / (1024**3):.2f} GB")
    
    # Prepare for texture
    color_data = make_color_2d(color_src, gamma=gamma, channel_order="RGBA")
    color_data *= 255
    
    return color_data.astype(np.uint8)


def load_starmap(filepath: str, target_width: int) -> Optional[np.ndarray]:
    """
    Load and process the star map for background.
    
    Parameters
    ----------
    filepath : str
        Path to the star map TIFF file
    target_width : int
        Target width for downscaling (to save memory)
        
    Returns
    -------
    np.ndarray or None
        Processed star map, or None if file not found
    """
    if not os.path.isfile(filepath):
        print(f"Star map not found: {filepath}")
        return None

    print(f"Loading star map from {filepath}...")

    # Disk cache of the processed result, keyed by the target width
    # (screen-dependent), so the 16k source is decoded and resized only once
    cache_base = f"{filepath}.w{target_width}"
    fingerprint = _cache_fingerprint(filepath, target_width=target_width)
    star_map, _ = _load_cache(cache_base, fingerprint)
    if star_map is not None:
        print(f"  Loaded from cache: {cache_base}.npy, dimensions {star_map.shape}")
        return star_map

    star_src = cv2.imread(filepath)

    if star_src is None:
        print(f"Failed to read star map: {filepath}")
        return None
    
    # Convert BGR to RGB and normalize
    star_src = star_src[..., ::-1].astype(np.float32)
    star_src *= 1 / 255
    
    # Downscale if needed
    if target_width < star_src.shape[1]:
        target_height = int(star_src.shape[0] * target_width / star_src.shape[1])
        star_map = cv2.resize(star_src, (target_width, target_height), 
                             interpolation=cv2.INTER_CUBIC)
        np.clip(star_map, 0, 1, out=star_map)
    else:
        star_map = star_src

    print(f"  Dimensions: {star_map.shape}")

    _save_cache(cache_base, star_map, fingerprint)

    return star_map