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
from moonrtx.orientations import VIEW_ORIENTATION_NSWE, VIEW_ORIENTATION_SNEW, VIEW_ORIENTATIONS
from moonrtx.shared_types import Camera, Observer

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

DEFAULT_COLOR_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, "moon_color_10k_8bit.tif")
DEFAULT_COLOR_FILE_SIZE_MB = 71.3
DEFAULT_COLOR_FILE_SIZE_BYTES = DEFAULT_COLOR_FILE_SIZE_MB * 1024**2

MOON_FEATURES_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, "moon_features.csv")

class InitView(NamedTuple):
    """Parsed init-view data for restoring a screenshot view."""
    dt_local: datetime
    lat: float
    lon: float
    view_orientation: str
    parallactic_mode: bool
    camera: Camera

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
    parser.add_argument("--elevation", type=int, default=0,
                        help="Observer elevation above sea level in meters. Examples: 0 (sea level), 219 (Cracow, Poland).")
    parser.add_argument("--time", type=str, default="now",
                        help="Time in ISO format with timezone information. Examples: 2024-01-01T12:00:00Z, 2025-12-26T16:30:00+01:00")
    parser.add_argument("--elevation-file", type=str, default=DEFAULT_ELEVATION_FILE_LOCAL_PATH,
                        help="Path to Moon elevation map local file")
    parser.add_argument("--color-file", type=str, default=DEFAULT_COLOR_FILE_LOCAL_PATH,
                        help="Path to Moon color map local file. Alternate color files can be downloaded from https://svs.gsfc.nasa.gov/4720")
    parser.add_argument("--downscale", type=int, default=3,
                        help="Elevation downscale factor. The higher value, the lower GPU memory usage but also lower quality of Moon surface. 1 is no downscaling.")
    parser.add_argument("--brightness", type=int, default=80,
                        help="Brightness")
    parser.add_argument("--gamma", type=float, default=2.8,
                        help="Gamma correction value (0.5 - 5.0, default 2.8)")
    parser.add_argument("--parallactic-mode", action="store_true",
                        help="Turn on parallactic mode (maintains Moon aligned to celestial north)")
    parser.add_argument("--time-step-minutes", type=int, default=15,
                        help="Time step in minutes for Q/W keys")
    parser.add_argument("--init-view", type=str, default=None,
                        help="Initialize view from a screenshot default filename (without extension). "
                             "This restores the exact camera position from time when attempt to take a screenshot was made. ")
    parser.add_argument("--init-view-orientation", type=str, default=VIEW_ORIENTATION_NSWE,
                        help=f"View orientation for specific telescope type (e.g. {VIEW_ORIENTATION_SNEW} for refractor). Valid values: {', '.join(VIEW_ORIENTATIONS)}. ")
    return parser.parse_args()

def _urlretrieve(url: str, dest: str):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', APP_NAME)]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest)

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
                _urlretrieve(DEFAULT_ELEVATION_FILE_REMOTE_PATH, elevation_file)
            except Exception as e:
                print(f"Error downloading default elevation file: {e}")
                return False
        else:
            print(f"Elevation file not found: {elevation_file}")
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
            _urlretrieve(STARMAP_FILE_REMOTE_PATH, STARMAP_FILE_LOCAL_PATH)
        except Exception as e:
            print(f"Error downloading starmap file: {e}")
            return False
    return True

def check_color_file(color_file: str) -> bool:
    if not os.path.isfile(color_file):
        if color_file == DEFAULT_COLOR_FILE_LOCAL_PATH:
            _, _, free = shutil.disk_usage(os.getcwd())
            if free < DEFAULT_COLOR_FILE_SIZE_BYTES * 1.02:
                print(f"Not enough disk space to download color file ({DEFAULT_COLOR_FILE_SIZE_MB} MB required).")
                return False
            print(f"Downloading color file (size {DEFAULT_COLOR_FILE_SIZE_MB} MB). It can take some time but must be done only once.")
            try:
                os.makedirs(os.path.dirname(color_file), exist_ok=True)
                download_file_from_google_drive("1gJeVic597BUAkpz1GgCYRMJVninKEDKB", color_file)
            except Exception as e:
                print(f"Error downloading color file: {e}")
                return False
        else:
            print(f"Color file not found: {color_file}")
            return False
    return True

def check_gpu_architecture() -> bool:
    try:
        gpu_arch = get_gpu_architecture()
        return gpu_arch is not None and gpu_arch.value >= GpuArchitecture.Compute_75.value
    except ValueError:
        print("WARNING: Unrecognized GPU RTX architecture")
        return True
    
def get_date_time_local(time_iso: str) -> tuple[Optional[datetime], Optional[Exception]]:
    if time_iso.endswith("Z"):
        time_iso = time_iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(time_iso)
    except ValueError as e:
        return None, e
    if dt.tzinfo is None:
        return None, ValueError("Time without timezone information.")
    return dt, None

def decode_camera(encoded: str) -> Optional[Camera]:
    """
    Decode camera from a base64 string.
    
    Parameters
    ----------
    encoded : str
        Base64-encoded camera parameters
        
    Returns
    -------
    Camera or None
        Camera object or None if decoding fails
    """
    try:
        # Add padding if needed
        padding = 4 - (len(encoded) % 4)
        if padding != 4:
            encoded += '=' * padding
        
        packed = base64.urlsafe_b64decode(encoded)
        values = struct.unpack('<10f', packed)
    
        return Camera(
            eye=[values[0], values[1], values[2]],
            target=[values[3], values[4], values[5]],
            up=[values[6], values[7], values[8]],
            fov=values[9]
            )
    except Exception as e:
        print(f"Error decoding camera: {e}")
        return None

def parse_init_view(init_view_str: str) -> Optional[InitView]:
    """
    Parse an init-view string (filename without extension) back into its components.
    
    Format: datetime_lat+XX.XXXXXX_lon+XX.XXXXXX_view<orientation>[_par<0|1>]_cam<base64>

    The _par<0|1> segment is optional for backwards compatibility with
    filenames saved before the parallactic-mode flag was introduced; when
    absent it defaults to OFF.

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
        pattern = (
            r'^(.+?)_lat([+-]?\d+\.\d+)_lon([+-]?\d+\.\d+)'
            r'_view([A-Z]+)(?:_par([01]))?_cam([A-Za-z0-9_-]+)$'
        )
        match = re.match(pattern, init_view_str)

        if not match:
            return None

        dt_str = match.group(1)
        lat = float(match.group(2))
        lon = float(match.group(3))
        view_orientation = match.group(4)
        par_flag = match.group(5)
        camera_encoded = match.group(6)

        # Validate view orientation
        if view_orientation not in VIEW_ORIENTATIONS:
            print(f"Invalid view orientation in init-view: {view_orientation}")
            return None

        parallactic_mode = par_flag == '1'

        camera = decode_camera(camera_encoded)
        if camera is None:
            return None

        dt_local, error = get_date_time_local(dt_str.replace('.', ':'))
        if error is not None:
            print(f"Incorrect time: {error}")
            return None

        return InitView(dt_local, lat, lon, view_orientation, parallactic_mode, camera)
    
    except Exception as e:
        print(f"Error parsing init-view string: {e}")
        return None

def main():

    args = parse_args()

    initial_camera = None
    init_view_orientation = args.init_view_orientation.upper()
    parallactic_mode = args.parallactic_mode
    lat = args.lat
    lon = args.lon
    if args.init_view:
        init_view = parse_init_view(args.init_view)
        if init_view is None:
            print(f"Error: Could not parse --init-view value: {args.init_view}")
            sys.exit(1)
        dt_local = init_view.dt_local
        lat = init_view.lat
        lon = init_view.lon
        init_view_orientation = init_view.view_orientation
        parallactic_mode = init_view.parallactic_mode
        initial_camera = init_view.camera
    else:
        time_iso = datetime.now().astimezone().isoformat(timespec="seconds") if args.time == "now" else args.time
        dt_local, error = get_date_time_local(time_iso)
        if error is not None:
            print(f"Incorrect time: {error}")
            sys.exit(1)
        if lat is None:
            print("Error: --lat parameter is mandatory.")
            sys.exit(1)
        if lon is None:
            print("Error: --lon parameter is mandatory.")
            sys.exit(1)

    if not (-180.0 <= lon <= 180.0):
        print("Invalid longitude. Must be between -180 and 180 degrees.")
        sys.exit(1)

    if not (-90.0 <= lat <= 90.0):
        print("Invalid latitude. Must be between -90 and 90 degrees.")
        sys.exit(1)

    if args.downscale < 1:
        print("Invalid downscale factor. Must be a positive integer.")
        sys.exit(1)

    if not (0 <= args.brightness <= 500):
        print("Invalid brightness. Must be between 0 and 500.")
        sys.exit(1)

    if not (0.5 <= args.gamma <= 5.0):
        print("Invalid gamma. Must be between 0.5 and 5.0.")
        sys.exit(1)

    if not (0 <= args.elevation <= 100000):
        print("Invalid elevation. Must be between 0 and 100000 meters.")
        sys.exit(1)

    if not (1 <= args.time_step_minutes <= 1440):
        print("Invalid time step. Must be between 1 and 1440 minutes.")
        sys.exit(1)

    if init_view_orientation not in VIEW_ORIENTATIONS:
        print(f"Invalid view orientation '{init_view_orientation}'. Must be one of: {', '.join(VIEW_ORIENTATIONS)}")
        sys.exit(1)

    if not check_gpu_architecture():
        print("No RTX GPU found.")
        sys.exit(1)

    if not check_elevation_file(args.elevation_file):
        sys.exit(1)

    if not check_color_file(args.color_file):
        sys.exit(1)

    if not check_starmap_file():
        sys.exit(1)

    run_renderer(dt_local=dt_local,
                 elevation_file=args.elevation_file,
                 observer=Observer(lat, lon, args.elevation),
                 downscale=args.downscale,
                 brightness=args.brightness,
                 color_file=args.color_file,
                 starmap_file=STARMAP_FILE_LOCAL_PATH,
                 features_file=MOON_FEATURES_FILE_LOCAL_PATH,
                 initial_camera=initial_camera,
                 time_step_minutes=args.time_step_minutes,
                 init_view_orientation=init_view_orientation,
                 gamma=args.gamma,
                 parallactic_mode=parallactic_mode)

if __name__ == "__main__":
    main()