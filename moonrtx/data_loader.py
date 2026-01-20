import os
import cv2
from typing import Optional

import numpy as np

from moonrtx.shared_types import MoonFeature

from plotoptix.utils import read_image, make_color_2d

def load_moon_features(filepath: str) -> list:
    """
    Load Moon features from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file with columns: name, latitude, longitude, angular_size, standard_label, spot_label, status_bar
        Separator is ':'
        
    Returns
    -------
    list
        List of dicts with keys: name, lat, lon, angular_size, standard_label, spot_label, status_bar
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
                    angle_str = parts[3].strip().replace('−', '-')
                    standard_label = parts[4].strip().lower() == 'true'
                    spot_label = parts[5].strip().lower() == 'true'
                    status_bar = parts[6].strip().lower() == 'true'
                    try:
                        moon_feature = MoonFeature(
                            name=name,
                            lat=float(lat_str),
                            lon=float(lon_str),
                            half_angle=float(angle_str) / 2,
                            cos_lat=np.cos(np.radians(float(lat_str))),
                            size_km=float(angle_str) * 30.34,
                            standard_label=standard_label,
                            spot_label=spot_label,
                            status_bar=status_bar
                        )
                        moon_features.append(moon_feature)
                    except ValueError as e:
                        print(f"Warning: Could not load Moon feature named {name}: {e}")
                        continue
    except Exception as e:
        print(f"Warning: Could not load Moon features file: {e}")
    
    return moon_features

def load_elevation_data(filepath: str, downscale: int) -> np.ndarray:
    """
    Load and process the Moon elevation data.
    
    Parameters
    ----------
    filepath : str
        Path to the elevation TIFF file
    downscale : int
        Downscale factor (2-3 recommended for most GPUs)
        
    Returns
    -------
    np.ndarray
        Processed elevation data normalized for displacement mapping
    """
    print(f"Loading elevation data from {filepath}...")
    elev_src = read_image(filepath)
    
    if elev_src is None:
        raise ValueError(f"Failed to read elevation file: {filepath}")
    
    print(f"  Original dimensions: {elev_src.shape}")
    print(f"  Size: {elev_src.nbytes / (1024**3):.2f} GB")
    
    # Convert to signed 16-bit and normalize
    elev_src.dtype = np.int16
    scale = 1. / np.iinfo(np.int16).max
    
    h = elev_src.shape[0] // downscale
    w = elev_src.shape[1] // downscale
    
    # Downscale by averaging
    elevation = elev_src.reshape(1, h, downscale, w, downscale).mean(
        4, dtype=np.float32).mean(2, dtype=np.float32).reshape(h, w)
    elevation *= scale
    
    # Release source memory
    elev_src = None
    
    print(f"  Downscaled dimensions: {elevation.shape}")
    print(f"  Downscaled size: {elevation.nbytes / (1024**3):.2f} GB")
    
    # Normalize for displacement mapping
    # Real Moon: radius ~1737 km, max relief ~20 km = ~1.15% of radius
    # Increase this value for more dramatic terrain, decrease for flatter appearance
    displacement_range = 0.0115  # 1.15% of radius
    
    rmin = np.min(elevation)
    rmax = np.max(elevation)
    rv = rmax - rmin
    
    elevation += rmin
    elevation *= displacement_range / rv
    elevation += (1.0 - displacement_range)
    
    return elevation


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


def load_starmap(filepath: str, target_width: int = 10240) -> Optional[np.ndarray]:
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
    
    return star_map