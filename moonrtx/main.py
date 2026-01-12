import argparse
import os
import sys
import shutil
import urllib.request
from datetime import datetime
from datetime import timezone

from plotoptix.utils import get_gpu_architecture
from plotoptix.enums import GpuArchitecture
from plotoptix.install import download_file_from_google_drive

from moonrtx.moon_renderer import run_renderer, APP_NAME

BASE_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__) 
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

MOON_FEATURES_FILE_PATH = os.path.join(DATA_DIRECTORY_PATH, "moon_features.csv")

def parse_args():
    parser = argparse.ArgumentParser(
        description=APP_NAME + " - ray-traced Moon observatory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--lat", type=float, required=True,
                        help="Observer latitude in degrees. Examples: 50.0614 (Cracow, Poland), -34.6131 (Buenos Aires, Argentina)")
    parser.add_argument("--lon", type=float, required=True,
                        help="Observer longitude in degrees. Examples: 19.9365 (Cracow, Poland), -58.3772 (Buenos Aires, Argentina)")
    parser.add_argument("--time", type=str, default="now",
                        help="Time in ISO format with timezone information. Examples: 2024-01-01T12:00:00Z, 2025-12-26T16:30:00+01:00")
    parser.add_argument("--elevation-file", type=str, default=DEFAULT_ELEVATION_FILE_LOCAL_PATH,
                        help="Path to Moon elevation map file")
    parser.add_argument("--downscale", type=int, default=3,
                        help="Elevation downscale factor. The higher value, the lower GPU memory usage but also lower quality of Moon surface. 1 is no downscaling.")
    parser.add_argument("--light-intensity", type=int, default=200,
                        help="Light intensity")
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

def get_date_time(time_iso: str):
    if time_iso.endswith("Z"):
        time_iso = time_iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(time_iso)
    except ValueError as e:
        return None, e
    if dt.tzinfo is None:
        return None, ValueError("Time without timezone information.")
    return dt, None

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
    features = []
    if not os.path.isfile(filepath):
        print(f"Warning: Moon features file {filepath} was not found. Features not loaded.")
        return features
    
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
                    angular_size_str = parts[3].strip().replace('−', '-')
                    standard_label = parts[4].strip().lower() == 'true'
                    spot_label = parts[5].strip().lower() == 'true'
                    status_bar = parts[6].strip().lower() == 'true'
                    try:
                        features.append({
                            'name': name,
                            'lat': float(lat_str),
                            'lon': float(lon_str),
                            'angular_size': float(angular_size_str),
                            'standard_label': standard_label,
                            'spot_label': spot_label,
                            'status_bar': status_bar
                        })
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load Moon features file: {e}")
    
    return features

def main():

    args = parse_args()

    if not (args.lon >= -180.0 and args.lon <= 180.0):
        print("Invalid longitude. Must be between -180 and 180 degrees.")
        sys.exit(1)

    if not (args.lat >= -90.0 and args.lat <= 90.0):
        print("Invalid latitude. Must be between -90 and 90 degrees.")
        sys.exit(1)

    if args.downscale < 1:
        print("Invalid downscale factor. Must be a positive integer.")
        sys.exit(1)

    time_iso = datetime.now().astimezone().isoformat(timespec="seconds") if args.time == "now" else args.time
    date_time, error = get_date_time(time_iso)
    if error:
        print(f"Incorrect time: {error}")
        sys.exit(1)    

    gpu_arch = get_gpu_architecture()
    if gpu_arch is None or gpu_arch.value < GpuArchitecture.Compute_75.value:
        print("No RTX GPU found.")
        sys.exit(1)


    if not check_elevation_file(args.elevation_file):
        sys.exit(1)

    if not check_color_file():
        sys.exit(1)

    if not check_starmap_file():
        sys.exit(1)

    print(f"\nStarting renderer with parameters:")
    print(f"  Geographical Location: Lat {args.lat}°, Lon {args.lon}°")
    print(f"  Time: {date_time}")
    print(f"  Elevation Map File: {args.elevation_file}")
    print(f"  Light Intensity: {args.light_intensity}")
    print(f"  Downscale Factor: {args.downscale}\n")

    moon_features = load_moon_features(MOON_FEATURES_FILE_PATH)

    run_renderer(date_time_utc=date_time.astimezone(timezone.utc),
                 elevation_file=args.elevation_file,
                 lat=args.lat,
                 lon=args.lon,
                 downscale=args.downscale,
                 color_file=COLOR_FILE_LOCAL_PATH,
                 starmap_file=STARMAP_FILE_LOCAL_PATH,
                 light_intensity=args.light_intensity,
                 moon_features=moon_features
                 )

if __name__ == "__main__":
    main()