import argparse
import os
import re
import sys
import shutil
import struct
import base64
import urllib.request
from datetime import datetime
from typing import NamedTuple, Optional

from plotoptix.utils import get_gpu_architecture
from plotoptix.enums import GpuArchitecture
from plotoptix.install import download_file_from_google_drive

from moonrtx.moon_renderer import run_renderer
from moonrtx.shared_types import CameraParams

APP_NAME = "MoonRTX"

BASE_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)     # frozen attribute from cx_Freeze
DATA_DIRECTORY_PATH = os.path.join(BASE_PATH, "data")

DEFAULT_ELEVATION_FILE_NAME = "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
DEFAULT_ELEVATION_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, DEFAULT_ELEVATION_FILE_NAME)
DEFAULT_ELEVATION_FILE_REMOTE_PATH = "http://planetarymaps.usgs.gov/mosaic/" + DEFAULT_ELEVATION_FILE_NAME
DEFAULT_ELEVATION_FILE_SIZE_GB = 7.91
DEFAULT_ELEVATION_FILE_SIZE_BYTES = DEFAULT_ELEVATION_FILE_SIZE_GB * 1024**3

STARMAP_FILE_NAME = "starmap_16k.tif"
STARMAP_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, STARMAP_FILE_NAME)
STARMAP_FILE_REMOTE_PATH = "https://svs.gsfc.nasa.gov/vis/a000000/a003800/a003895/" + STARMAP_FILE_NAME
STARMAP_FILE_SIZE_MB = 132
STARMAP_FILE_SIZE_BYTES = STARMAP_FILE_SIZE_MB * 1024**2

COLOR_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, "moon_color_10k_8bit.tif")
COLOR_FILE_SIZE_MB = 71.3
COLOR_FILE_SIZE_BYTES = COLOR_FILE_SIZE_MB * 1024**2

MOON_FEATURES_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, "moon_features.csv")

class InitView(NamedTuple):
    """Parsed init-view data for restoring a screenshot view."""
    dt_local: datetime
    lat: float
    lon: float
    eye: list
    target: list
    up: list
    fov: float

def parse_args():

    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - ray-traced Moon observatory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--lat", type=float, default=None,
                        help="Observer latitude in degrees. Examples: 50.0614 (Cracow, Poland), -34.6131 (Buenos Aires, Argentina). "
                             "Mandatory parameter unless --init-view is used.")
    parser.add_argument("--lon", type=float, default=None,
                        help="Observer longitude in degrees. Examples: 19.9365 (Cracow, Poland), -58.3772 (Buenos Aires, Argentina). "
                             "Mandatory parameter unless --init-view is used.")
    parser.add_argument("--time", type=str, default="now",
                        help="Time in ISO format with timezone information. Examples: 2024-01-01T12:00:00Z, 2025-12-26T16:30:00+01:00")
    parser.add_argument("--elevation-file", type=str, default=DEFAULT_ELEVATION_FILE_LOCAL_PATH,
                        help="Path to Moon elevation map file")
    parser.add_argument("--downscale", type=int, default=3,
                        help="Elevation downscale factor. The higher value, the lower GPU memory usage but also lower quality of Moon surface. 1 is no downscaling.")
    parser.add_argument("--light-intensity", type=int, default=120,
                        help="Light intensity")
    parser.add_argument("--init-view", type=str, default=None,
                        help="Initialize view from a screenshot default filename (without extension). "
                             "This restores the exact camera position from time when attempt to take a screenshot was made. ")
    return parser.parse_args()

def check_elevation_file(elevation_file: str) -> bool:
    if not os.path.isfile(elevation_file):
        if elevation_file == DEFAULT_ELEVATION_FILE_LOCAL_PATH:
            _, _, free = shutil.disk_usage(os.getcwd())
            if free < DEFAULT_ELEVATION_FILE_SIZE_BYTES * 1.02:
                print(f"Not enough disk space to download default elevation file ({DEFAULT_ELEVATION_FILE_SIZE_GB} GB required).")
                return False
            print(f"Downloading default elevation file (size {DEFAULT_ELEVATION_FILE_SIZE_GB} GB). It can take some time but must be done only once.")
            try:
                os.makedirs(os.path.dirname(elevation_file), exist_ok=True)
                urllib.request.urlretrieve(DEFAULT_ELEVATION_FILE_REMOTE_PATH, elevation_file)
            except Exception as e:
                print(f"Error downloading default elevation file: {e}")
                return False
        else:
            return False
    return True

def check_starmap_file() -> bool:
    if not os.path.isfile(STARMAP_FILE_LOCAL_PATH):
        _, _, free = shutil.disk_usage(os.getcwd())
        if free < STARMAP_FILE_SIZE_BYTES * 1.02:
            print(f"Not enough disk space to download starmap file ({STARMAP_FILE_SIZE_MB} MB required).")
            return False
        print(f"Downloading starmap file (size {STARMAP_FILE_SIZE_MB} MB). It can take some time but must be done only once.")
        try:
            os.makedirs(os.path.dirname(STARMAP_FILE_LOCAL_PATH), exist_ok=True)
            urllib.request.urlretrieve(STARMAP_FILE_REMOTE_PATH, STARMAP_FILE_LOCAL_PATH)
        except Exception as e:
            print(f"Error downloading starmap file: {e}")
            return False
    return True

def check_color_file() -> bool:
    if not os.path.isfile(COLOR_FILE_LOCAL_PATH):
        _, _, free = shutil.disk_usage(os.getcwd())
        if free < COLOR_FILE_SIZE_BYTES * 1.02:
            print(f"Not enough disk space to download color file ({COLOR_FILE_SIZE_MB} MB required).")
            return False
        print(f"Downloading color file (size {COLOR_FILE_SIZE_MB} MB). It can take some time but must be done only once.")
        try:
            os.makedirs(os.path.dirname(COLOR_FILE_LOCAL_PATH), exist_ok=True)
            download_file_from_google_drive("1gJeVic597BUAkpz1GgCYRMJVninKEDKB", COLOR_FILE_LOCAL_PATH)
        except Exception as e:
            print(f"Error downloading color file: {e}")
            return False
    return True

def check_gpu_architecture() -> bool:
    try:
        gpu_arch = get_gpu_architecture()
        return gpu_arch is not None and gpu_arch.value >= GpuArchitecture.Compute_75.value
    except ValueError:
        print("WARNING: Unrecognized GPU RTX architecture")
        return True
    
def get_date_time_local(time_iso: str):
    if time_iso.endswith("Z"):
        time_iso = time_iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(time_iso)
    except ValueError as e:
        return None, e
    if dt.tzinfo is None:
        return None, ValueError("Time without timezone information.")
    return dt, None

def decode_camera_params(encoded: str) -> Optional[tuple]:
    """
    Decode camera parameters from a base64 string.
    
    Parameters
    ----------
    encoded : str
        Base64-encoded camera parameters
        
    Returns
    -------
    tuple or None
        (eye, target, up, fov) or None if decoding fails
    """
    try:
        # Add padding if needed
        padding = 4 - (len(encoded) % 4)
        if padding != 4:
            encoded += '=' * padding
        
        packed = base64.urlsafe_b64decode(encoded)
        values = struct.unpack('<10f', packed)
        
        eye = [values[0], values[1], values[2]]
        target = [values[3], values[4], values[5]]
        up = [values[6], values[7], values[8]]
        fov = values[9]
        
        return eye, target, up, fov
    except Exception as e:
        print(f"Error decoding camera params: {e}")
        return None


def parse_init_view(init_view_str: str) -> Optional[InitView]:
    """
    Parse an init-view string (filename without extension) back into its components.
    
    Format: datetime_lat+XX.XXXXXX_lon+XX.XXXXXX_cam<base64>
    
    Parameters
    ----------
    init_view_str : str
        The init-view string to parse
        
    Returns
    -------
    InitView or None
        Parsed data or None if parsing fails
    """
    try:
        pattern = r'^(.+?)_lat([+-]?\d+\.\d+)_lon([+-]?\d+\.\d+)_cam([A-Za-z0-9_-]+)$'
        match = re.match(pattern, init_view_str)
        
        if not match:
            return None
        
        dt_str = match.group(1)
        lat = float(match.group(2))
        lon = float(match.group(3))
        cam_encoded = match.group(4)
        
        decoded = decode_camera_params(cam_encoded)
        if decoded is None:
            return None
        eye, target, up, fov = decoded
        
        dt_local, error = get_date_time_local(dt_str.replace('.', ':'))
        if error is not None:
            print(f"Incorrect time: {error}")
            return None
        
        return InitView(
            dt_local=dt_local,
            lat=lat,
            lon=lon,
            eye=eye,
            target=target,
            up=up,
            fov=fov
        )
    except Exception as e:
        print(f"Error parsing init-view string: {e}")
        return None

def main():

    args = parse_args()

    init_view = None
    init_camera_params = None
    if args.init_view:
        init_view = parse_init_view(args.init_view)
        if init_view is None:
            print(f"Error: Could not parse --init-view value: {args.init_view}")
            sys.exit(1)

    # Use datetime from init_view if provided, otherwise use --time argument
    if init_view is not None:
        dt_local = init_view.dt_local
        lat = init_view.lat
        lon = init_view.lon
        init_camera_params = CameraParams(
            eye=init_view.eye,
            target=init_view.target,
            up=init_view.up,
            fov=init_view.fov
        )
    else:
        time_iso = datetime.now().astimezone().isoformat(timespec="seconds") if args.time == "now" else args.time
        dt_local, error = get_date_time_local(time_iso)
        if error is not None:
            print(f"Incorrect time: {error}")
            sys.exit(1)
        if args.lat is None:
            print("Error: --lat parameter is mandatory.")
            sys.exit(1)
        if args.lon is None:
            print("Error: --lon parameter is mandatory.")
            sys.exit(1)
        lat = args.lat
        lon = args.lon

    if not (lon >= -180.0 and lon <= 180.0):
        print("Invalid longitude. Must be between -180 and 180 degrees.")
        sys.exit(1)

    if not (lat >= -90.0 and lat <= 90.0):
        print("Invalid latitude. Must be between -90 and 90 degrees.")
        sys.exit(1)

    if args.downscale < 1:
        print("Invalid downscale factor. Must be a positive integer.")
        sys.exit(1)

    if not check_gpu_architecture():
        print("No RTX GPU found.")
        sys.exit(1)

    if not check_elevation_file(args.elevation_file):
        sys.exit(1)

    if not check_color_file():
        sys.exit(1)

    if not check_starmap_file():
        sys.exit(1)

    run_renderer(dt_local=dt_local,
                 elevation_file=args.elevation_file,
                 lat=lat,
                 lon=lon,
                 downscale=args.downscale,
                 light_intensity=args.light_intensity,
                 app_name=APP_NAME,
                 color_file=COLOR_FILE_LOCAL_PATH,
                 starmap_file=STARMAP_FILE_LOCAL_PATH,
                 features_file=MOON_FEATURES_FILE_LOCAL_PATH,
                 init_camera_params=init_camera_params)

if __name__ == "__main__":
    main()