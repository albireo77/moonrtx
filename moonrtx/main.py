import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tzlocal import get_localzone
from typing import Optional

import plotoptix
from plotoptix.utils import get_gpu_architecture
from plotoptix.enums import GpuArchitecture
from plotoptix.install import download_file_from_google_drive

from moonrtx.data_loader import (COLOR_DOWNSCALE_FACTORS, downscale_cache_available,
                                 elevation_map_problem, free_space, starmap_cache_available)
from moonrtx.display import make_dpi_aware, starmap_target_width
from moonrtx.moon_renderer import run_renderer
from moonrtx.view_orientation import VIEW_ORIENTATION_NSWE, VIEW_ORIENTATION_SNEW, VIEW_ORIENTATIONS
from moonrtx.shared_types import InitView, MapTooLargeError, Observer
from moonrtx.skyfield_utils import SKYFIELD_MOON_FRAME_END_UTC, SKYFIELD_MOON_FRAME_START_UTC

APP_NAME = "MoonRTX"

BASE_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)     # frozen attribute from cx_Freeze
DATA_DIRECTORY_PATH = os.path.join(BASE_PATH, "data")

DEFAULT_ELEVATION_FILE_NAME = "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
DEFAULT_ELEVATION_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, DEFAULT_ELEVATION_FILE_NAME)
DEFAULT_ELEVATION_FILE_REMOTE_PATH = "https://planetarymaps.usgs.gov/mosaic/" + DEFAULT_ELEVATION_FILE_NAME
DEFAULT_ELEVATION_FILE_SIZE_GB = 7.91
DEFAULT_ELEVATION_FILE_SIZE_BYTES = DEFAULT_ELEVATION_FILE_SIZE_GB * 1024**3
# Height and width in samples, known before the map is fetched, so a downscale
# that cannot divide it is refused ahead of the download (see elevation_problem)
DEFAULT_ELEVATION_FILE_SHAPE = (46080, 92160)

STARMAP_FILE_NAME = "starmap_16k.tif"
STARMAP_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, STARMAP_FILE_NAME)
STARMAP_FILE_REMOTE_PATH = "https://svs.gsfc.nasa.gov/vis/a000000/a003800/a003895/" + STARMAP_FILE_NAME
STARMAP_FILE_SIZE_MB = 132
STARMAP_FILE_SIZE_BYTES = STARMAP_FILE_SIZE_MB * 1024**2

DEFAULT_COLOR_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, "moon_color_10k_8bit.tif")
DEFAULT_COLOR_FILE_SIZE_MB = 71.3
DEFAULT_COLOR_FILE_SIZE_BYTES = DEFAULT_COLOR_FILE_SIZE_MB * 1024**2

MOON_FEATURES_FILE_LOCAL_PATH = os.path.join(DATA_DIRECTORY_PATH, "moon_features.csv")

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
                        help="Local time to start at, on the clock of --timezone. Example: 2025-12-26T16:30:00. "
                             "Omit to start at the current time.")
    parser.add_argument("--timezone", type=str, default=None,
                        help="Timezone the observation time is given in, as an IANA name. Examples: Europe/Warsaw, America/New_York. "
                             "Omit to use this computer's own timezone.")
    parser.add_argument("--elevation-file", type=str, default=DEFAULT_ELEVATION_FILE_LOCAL_PATH,
                        help="Path to Moon elevation map local file")
    parser.add_argument("--color-file", type=str, default=DEFAULT_COLOR_FILE_LOCAL_PATH,
                        help="Path to Moon color map local file. Alternate color files can be downloaded from https://svs.gsfc.nasa.gov/4720")
    parser.add_argument("--downscale", type=int, default=2,
                        help="Elevation downscale factor. The higher value, the lower GPU memory usage but also lower quality of Moon surface. 1 is no downscaling.")
    parser.add_argument("--color-downscale", type=int, default=1, choices=COLOR_DOWNSCALE_FACTORS,
                        help="Color map downscale factor. The map is decoded straight at this fraction of its "
                             "size, so RAM needed to load it falls with the square of the factor. Raise it if a "
                             "large color map fails to load. 1 is no downscaling.")
    parser.add_argument("--brightness", type=int, default=100,
                        help="Brightness")
    parser.add_argument("--gamma", type=float, default=2.2,
                        help="Gamma correction value (0.5 - 5.0, default 2.2)")
    parser.add_argument("--parallactic-mode", action="store_true",
                        help="Turn on parallactic mode (maintains Moon aligned to celestial north)")
    parser.add_argument("--no-stars", action="store_true",
                        help="Black background (saves GPU memory)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Start in full screen. F11 toggles it, Escape leaves it.")
    parser.add_argument("--time-step-minutes", type=int, default=15,
                        help="Time step in minutes for Q/W keys")
    parser.add_argument("--init-view", type=str, default=None,
                        help="Initialize view from the default filename a saved image or an exported "
                             "video was offered (without the extension). This restores exact camera "
                             "position along with observation time and location of the moment it was saved. ")
    parser.add_argument("--init-view-orientation", type=str, default=VIEW_ORIENTATION_NSWE,
                        help=f"View orientation for specific telescope type (e.g. {VIEW_ORIENTATION_SNEW} for refractor). Valid values: {', '.join(VIEW_ORIENTATIONS)}. ")
    return parser.parse_args()

class _DownloadProgress:
    """
    Says how far a download has got, from the bytes done and the whole size

    On a console the figure is rewritten in place on one line. Into a file or a
    pipe, where a carriage return would only leave junk, a line is written each
    tenth of the way. When the server gives no length, the megabytes that have
    come are said every hundred of them instead.
    """

    def __init__(self):
        self.out = sys.stdout
        self.in_place = self.out is not None and getattr(self.out, "isatty", lambda: False)()
        self.shown = -1          # the last percentage, or hundred megabytes, written
        self.started = False

    def report(self, done: int, total: int):
        if self.out is None:     # a frozen build with no console
            return
        if total > 0:
            done = min(done, total)
            mark = done * 100 // total
            step = 1 if self.in_place else 10
            text = f"{mark:3d}%  {done / 1024**2:,.0f} of {total / 1024**2:,.0f} MB"
        else:
            mark = done // (100 * 1024**2)
            step = 1
            text = f"{done / 1024**2:,.0f} MB"
        if self.shown >= 0 and mark < self.shown + step:
            return
        self.shown = mark
        self.started = True
        if self.in_place:
            print(f"\r  {text}", end="", flush=True)
        else:
            print(f"  {text}", flush=True)

    def finish(self):
        """End the line rewritten in place, so what is printed next starts afresh."""
        if self.in_place and self.started:
            print(flush=True)


def _download_whole(dest: str, fetch, keep_part: bool = False):
    """
    Have fetch(path) write the file under a ".part" name beside dest, and move it
    to dest only once it has finished.

    Written straight to dest, a download stopped part way - Ctrl+C, a dropped
    connection - left a truncated file where the whole one belongs, and every
    later run found it there, took it as complete, and failed loading it with
    nothing to say why. Now dest only ever exists whole. A failure is passed on,
    and removes the part-file unless keep_part: a fetch that can continue from
    what the part-file holds (see _fetch_resuming) wants it left for the next
    run. One the process does not live through leaves it behind either way,
    under a name nothing but the next download reads.

    os.replace moves it in one step, the two names being in the same folder and
    so on the same drive.
    """
    part = dest + ".part"
    try:
        fetch(part)
        os.replace(part, dest)
    except BaseException:           # KeyboardInterrupt too, which is no Exception
        if not keep_part:
            try:
                os.remove(part)
            except OSError:
                pass
        raise


# Seconds a download may wait for the server before it is given up, its
# part-file kept for the next run to continue. Without one, a connection that
# went quiet held the launcher at the same percentage for ever.
DOWNLOAD_TIMEOUT_SEC = 60

# "bytes 100-199/200" for a part of a file, "bytes */200" when none was sent
_CONTENT_RANGE = re.compile(r"bytes (?:(\d+)-\d+|\*)/(\d+)")


class _CannotResume(Exception):
    """The server will not send a file's rest from where its part-file ends."""


def _file_size(path: str) -> int:
    """Bytes in the file at path, or 0 when there is none."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _open_download(url: str, start: int) -> tuple:
    """
    Ask for url from byte start on, the whole of it when start is 0.

    Returns the response, positioned at start, and the whole file's size (-1
    when the server gives none), or None for the response when the file ends
    at start, so there is nothing left to send. Raises _CannotResume when the
    server answers a resume with anything but the rest: the whole file again, a
    part from elsewhere, or a size other than the part-file's own.
    """
    headers = {"User-Agent": APP_NAME}
    if start:
        headers["Range"] = f"bytes={start}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                          timeout=DOWNLOAD_TIMEOUT_SEC)
    except urllib.error.HTTPError as e:
        if e.code != 416 or not start:      # 416: nothing lies at or past start
            raise
        found = _CONTENT_RANGE.match(e.headers.get("Content-Range", ""))
        if found:
            size = int(found.group(2))
        else:
            # S3 and NASA's server say no size with it, so it is read from the
            # whole file's answer, which is closed before any of it comes
            response, size = _open_download(url, 0)
            response.close()
        if size == start:
            return None, start
        raise _CannotResume() from e
    if not start:
        length = response.headers.get("Content-Length", "")
        return response, int(length) if length.isdigit() else -1
    found = _CONTENT_RANGE.match(response.headers.get("Content-Range", ""))
    if response.status == 206 and found and found.group(1) and int(found.group(1)) == start:
        return response, int(found.group(2))
    response.close()
    raise _CannotResume()


def _fetch_resuming(url: str, part: str, report):
    """
    Download url into the file part, continuing from what it already holds.

    A download that broke off - a dropped connection, Ctrl+C, the window closed -
    leaves its bytes in part, and the next one asks the server only for the
    rest, so a break at 8 GB of the elevation map's 8.5 does not cost the 8 GB
    again. A server that will not send only the rest has the file written over
    from the start. report(done_bytes, total_bytes) is told how far it has got,
    total being -1 when the server gives no size.

    Raises when the connection ends before the whole size has come; what did
    come stays in part.
    """
    start = _file_size(part)
    if start:
        try:
            response, total = _open_download(url, start)
        except _CannotResume:
            start = 0
        else:
            print(f"  Continuing from {start / 1024**2:,.0f} MB downloaded before.", flush=True)
    if not start:
        response, total = _open_download(url, 0)
    done = start
    report(done, total)
    if response is None:
        return
    with response, open(part, "ab" if start else "wb") as f:
        while True:
            chunk = response.read(1024**2)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            report(done, total)
    if total > 0 and done != total:
        raise OSError(f"the download ended after {done:,} of {total:,} bytes")


def _urlretrieve(url: str, dest: str, on_progress=None):
    """
    Download url to dest, saying on the console how far it has got, and handing
    the same to on_progress(done_bytes, total_bytes) if one is given - total
    being 0 or less when the server gives no length. The launcher passes one, so
    its window can say it too; it may be called on a thread other than Tk's.
    A download that broke off before is continued, not begun again.
    """
    progress = _DownloadProgress()

    def report(done: int, total: int):
        progress.report(done, total)
        if on_progress is not None:
            on_progress(done, total)

    try:
        _download_whole(dest, lambda part: _fetch_resuming(url, part, report), keep_part=True)
    finally:
        progress.finish()

def elevation_problem(elevation_file: str, downscale: int) -> Optional[str]:
    """
    Why this elevation map cannot be used at this downscale - samples that are
    not heights, or a factor that cannot divide it - or None. Asked before
    anything is fetched or read: the default map's size is known without it
    being there, so a missing one is not downloaded only to be refused.
    """
    expected = (DEFAULT_ELEVATION_FILE_SHAPE
                if elevation_file == DEFAULT_ELEVATION_FILE_LOCAL_PATH else None)
    return elevation_map_problem(elevation_file, downscale, expected)


def check_elevation_file(elevation_file: str, downscale: int, on_progress=None) -> bool:
    if not os.path.isfile(elevation_file):
        # The downscaled cache holds everything the renderer reads, so the
        # source it was made from can be deleted to reclaim its gigabytes and
        # need not be fetched again to stand unused beside it
        if downscale_cache_available(elevation_file, downscale):
            print(f"Elevation file {elevation_file} is not present; "
                  f"using the cached data downscaled by {downscale}.")
            return True
        if elevation_file == DEFAULT_ELEVATION_FILE_LOCAL_PATH:
            # What an earlier download left need not fit a second time
            free = free_space(elevation_file) + _file_size(elevation_file + ".part")
            if free < DEFAULT_ELEVATION_FILE_SIZE_BYTES * 1.02:
                print(f"Not enough disk space to download default elevation file ({DEFAULT_ELEVATION_FILE_SIZE_GB} GB required).")
                return False
            print(f"Downloading default elevation file (size {DEFAULT_ELEVATION_FILE_SIZE_GB} GB)...")
            try:
                os.makedirs(os.path.dirname(elevation_file), exist_ok=True)
                _urlretrieve(DEFAULT_ELEVATION_FILE_REMOTE_PATH, elevation_file, on_progress)
            except Exception as e:
                print(f"Error downloading default elevation file: {e}")
                print("What came is kept; run again to continue from there.")
                return False
        else:
            print(f"Elevation file not found: {elevation_file}")
            return False
    return True

def get_starmap_file(on_progress=None) -> Optional[str]:
    if not os.path.isfile(STARMAP_FILE_LOCAL_PATH):
        # As with the elevation and color maps, what the renderer reads is the
        # cache, so a source deleted to reclaim its megabytes stays deleted.
        # The star map cache is keyed by screen width rather than a downscale,
        # hence the screen lookup here (see moon_renderer.starmap_target_width).
        if starmap_cache_available(STARMAP_FILE_LOCAL_PATH, starmap_target_width()):
            print(f"Starmap file {STARMAP_FILE_LOCAL_PATH} is not present; using the cached star map.")
            return STARMAP_FILE_LOCAL_PATH
        free = free_space(STARMAP_FILE_LOCAL_PATH) + _file_size(STARMAP_FILE_LOCAL_PATH + ".part")
        if free < STARMAP_FILE_SIZE_BYTES * 1.02:
            print(f"Not enough disk space to download starmap file ({STARMAP_FILE_SIZE_MB} MB required).")
            return None
        print(f"Downloading starmap file (size {STARMAP_FILE_SIZE_MB} MB)...")
        try:
            os.makedirs(os.path.dirname(STARMAP_FILE_LOCAL_PATH), exist_ok=True)
            _urlretrieve(STARMAP_FILE_REMOTE_PATH, STARMAP_FILE_LOCAL_PATH, on_progress)
        except Exception as e:
            print(f"Error downloading starmap file: {e}")
            print("What came is kept; run again to continue from there.")
            return None
    return STARMAP_FILE_LOCAL_PATH

def check_color_file(color_file: str, color_downscale: int) -> bool:
    if not os.path.isfile(color_file):
        # As for the elevation map: the downscaled cache is all the renderer
        # reads, so a source deleted to reclaim its gigabytes stays deleted
        if downscale_cache_available(color_file, color_downscale):
            print(f"Color file {color_file} is not present; "
                  f"using the cached data downscaled by {color_downscale}.")
            return True
        if color_file == DEFAULT_COLOR_FILE_LOCAL_PATH:
            free = free_space(color_file)
            if free < DEFAULT_COLOR_FILE_SIZE_BYTES * 1.02:
                print(f"Not enough disk space to download color file ({DEFAULT_COLOR_FILE_SIZE_MB} MB required).")
                return False
            print(f"Downloading color file (size {DEFAULT_COLOR_FILE_SIZE_MB} MB)...")
            try:
                os.makedirs(os.path.dirname(color_file), exist_ok=True)
                _download_whole(color_file, lambda part: download_file_from_google_drive(
                    "1gJeVic597BUAkpz1GgCYRMJVninKEDKB", part))
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

# MoonRTX relies on the marching_step/scene_epsilon decoupling introduced in
# PlotOptiX 0.19.2 (exact terminator shadows) and on the >= 0.19.1 accumulation
# semantics that the interactive-preview mode is built around
MIN_PLOTOPTIX_VERSION = (0, 19, 2)

def check_plotoptix_version() -> bool:
    version = plotoptix.__version__
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        print(f"WARNING: Unrecognized PlotOptiX version string: {version}")
        return True
    if tuple(int(g) for g in match.groups()) < MIN_PLOTOPTIX_VERSION:
        required = ".".join(str(v) for v in MIN_PLOTOPTIX_VERSION)
        print(f"PlotOptiX {version} is too old: {APP_NAME} requires {required} or newer. "
              "Update with: pip install --upgrade -r requirements.txt")
        return False
    return True
    
def resolve_timezone(name: Optional[str]):
    """
    The named IANA timezone, or this computer's own when no name is given.

    A zone carries the rules, so every date of a session gets the offset that
    really applied on it - daylight saving, the historical changes before it,
    and the rules of another country when planning for one. That is why a name
    is asked for rather than an offset, which can only ever describe one
    instant and knows nothing of the dates around it.

    Returns
    -------
    tuple
        (tzinfo, None), or (None, error) when the name is not a known zone
    """
    if not name:
        try:
            return get_localzone(), None
        except Exception as e:
            return None, e
    try:
        return ZoneInfo(name), None
    except (ZoneInfoNotFoundError, ValueError) as e:
        return None, e


def get_date_time_local(time_iso: str, zone) -> tuple[Optional[datetime], Optional[Exception]]:
    """
    Read a starting time in the session's timezone.

    A plain wall clock ("2026-12-01T20:00:00") is the expected form and is read
    in `zone`, which decides the offset for that date. A value that still
    carries a UTC offset names an instant instead, and is re-expressed in the
    zone, so command lines and screenshot names written before zones keep
    working and keep meaning the same moment.
    """
    if time_iso.endswith("Z"):
        time_iso = time_iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(time_iso)
    except ValueError as e:
        return None, e
    if dt.tzinfo is None:
        return dt.replace(tzinfo=zone), None
    return dt.astimezone(zone), None


def date_problem(dt_local: datetime) -> Optional[str]:
    """
    Why the renderer cannot start at this moment, or None when it can.

    Asked before the renderer starts: past the years the bundled Moon data
    covers, the first position worked out fails, and that is only after the
    maps have been loaded - a wait that ends in a traceback.
    """
    dt_utc = dt_local.astimezone(timezone.utc)
    if SKYFIELD_MOON_FRAME_START_UTC <= dt_utc <= SKYFIELD_MOON_FRAME_END_UTC:
        return None
    return (f"The date must be from {SKYFIELD_MOON_FRAME_START_UTC.year} to "
            f"{SKYFIELD_MOON_FRAME_END_UTC.year - 1}: the Moon data {APP_NAME} comes "
            f"with covers only those years.")

def main():

    make_dpi_aware()

    args = parse_args()

    initial_camera = None
    init_view_orientation = args.init_view_orientation.upper()
    parallactic_mode = args.parallactic_mode
    lat = args.lat
    lon = args.lon

    timezone, error = resolve_timezone(args.timezone)
    if error is not None:
        print(f"Unknown timezone {args.timezone!r}: {error}")
        print("Names are IANA ones, e.g. Europe/Warsaw or America/New_York.")
        sys.exit(1)

    if args.init_view:
        init_view = InitView.decode(args.init_view, timezone)
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
        if args.time == "now":
            dt_local = datetime.now(timezone)
        else:
            dt_local, error = get_date_time_local(args.time, timezone)
            if error is not None:
                print(f"Incorrect time: {error}")
                sys.exit(1)
        if lat is None:
            print("Error: --lat parameter is mandatory.")
            sys.exit(1)
        if lon is None:
            print("Error: --lon parameter is mandatory.")
            sys.exit(1)

    problem = date_problem(dt_local)
    if problem is not None:
        print(problem)
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

    problem = elevation_problem(args.elevation_file, args.downscale)
    if problem is not None:
        print(problem)
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

    if not check_plotoptix_version():
        sys.exit(1)

    if not check_gpu_architecture():
        print("No RTX GPU found.")
        sys.exit(1)

    if not check_elevation_file(args.elevation_file, args.downscale):
        sys.exit(1)

    if not check_color_file(args.color_file, args.color_downscale):
        sys.exit(1)

    starmap_file = None if args.no_stars else get_starmap_file()

    try:
        run_renderer(dt_local=dt_local,
                     elevation_file=args.elevation_file,
                     observer=Observer(lat, lon, args.elevation),
                     downscale=args.downscale,
                     brightness=args.brightness,
                     color_file=args.color_file,
                     color_downscale=args.color_downscale,
                     starmap_file=starmap_file,
                     features_file=MOON_FEATURES_FILE_LOCAL_PATH,
                     initial_camera=initial_camera,
                     time_step_minutes=args.time_step_minutes,
                     init_view_orientation=init_view_orientation,
                     gamma=args.gamma,
                     parallactic_mode=parallactic_mode,
                     fullscreen=args.fullscreen)
    except MapTooLargeError as e:
        # Says which map and what to change, so a traceback would only bury it
        print(f"\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()