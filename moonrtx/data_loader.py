import json
import os
import shutil
import cv2
from contextlib import contextmanager
from typing import Optional

import numpy as np

from moonrtx.shared_types import MapTooLargeError, MoonFeature

from plotoptix.utils import read_image

# Processed-array disk caches: reading the 7.9 GB elevation TIFF and block-mean
# downscaling it takes about a minute on every start, while np.load of the
# ready-made float32 result takes seconds. A cache is valid when the sidecar
# JSON matches the source file size and the processing parameters;
# any read or write problem silently falls back to the regular path, so a
# broken cache can only cost time, never correctness. Bump the version when
# the processing itself changes.
_CACHE_VERSION = 1


def _cache_fingerprint(filepath: str, **params) -> dict:
    """
    What a cache has to match before it is used: the processing parameters
    always, and the source file's size whenever the source is still there to
    compare against. A source deleted to reclaim its several gigabytes - the
    cache being a small fraction of it - leaves nothing to check against, so
    what was cached from it is then taken on trust rather than discarded.

    Size only, deliberately: modification time also changes when the very same
    bytes arrive again - a re-download of a deleted source, a restore from
    backup, a copy to another machine - and keying on it threw away good caches
    in exactly the case this app invites, deleting a source and fetching it back
    later. What it would have caught in exchange is an edit that leaves the byte
    count identical, which for these multi-gigabyte survey products is not a
    thing that happens. Bump _CACHE_VERSION for changes in the processing.
    """
    fingerprint = {"version": _CACHE_VERSION, **params}
    if os.path.isfile(filepath):
        fingerprint["source_size"] = os.path.getsize(filepath)
    return fingerprint


def _cache_meta(cache_base: str, fingerprint: dict) -> Optional[dict]:
    """
    The sidecar metadata of a cache that matches the fingerprint and has its
    array beside it, or None. Reads only the small JSON, so it answers whether
    a cache is usable without loading the hundreds of megabytes it describes.
    """
    try:
        with open(cache_base + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if not all(meta.get(k) == v for k, v in fingerprint.items()):
        return None
    return meta if os.path.isfile(cache_base + ".npy") else None


def _load_cache(cache_base: str, fingerprint: dict,
                mmap_mode: Optional[str] = None) -> tuple[Optional[np.ndarray], dict]:
    """
    The cached array and its metadata, or (None, {}) to fall back to reading the
    source. Running out of memory is not such a case and is left to propagate:
    the source is bigger than the cache, so re-reading it could only fail again,
    more slowly and with a less useful message.

    mmap_mode : str, optional
        Passed to np.load. "r" leaves the array in its file and pages it in as
        it is read, which is how the full-size elevation map is used - it is
        larger than the RAM of the machines that can still render it.
    """
    meta = _cache_meta(cache_base, fingerprint)
    if meta is None:
        return None, {}
    try:
        return np.load(cache_base + ".npy", mmap_mode=mmap_mode), meta
    except MemoryError:
        raise
    except Exception:
        return None, {}


def downscale_cache_available(filepath: str, downscale: int) -> bool:
    """
    Whether the downscaled form of an elevation or color map is already on disk,
    in which case the source file is not needed at all and need not be
    downloaded to replace one that has been deleted (see main.check_elevation_file
    and main.check_color_file).

    Parameters
    ----------
    filepath : str
        Path the source TIFF would have; the cache sits beside it
    downscale : int
        The factor the cache must have been made with

    Returns
    -------
    bool
        True when that cache is present and usable
    """
    return _cache_meta(f"{filepath}.ds{downscale}",
                       _cache_fingerprint(filepath, downscale=downscale)) is not None


def starmap_cache_available(filepath: str, target_width: int) -> bool:
    """
    Whether the star map processed for this screen is already on disk, in which
    case the source is not needed and need not be downloaded again to stand
    unused beside it (see main.check_starmap_file).

    Parameters
    ----------
    filepath : str
        Path the source TIFF would have; the cache sits beside it
    target_width : int
        The width the cache must have been made for, which depends on the
        screen (moon_renderer.starmap_target_width)

    Returns
    -------
    bool
        True when that cache is present and usable
    """
    return _cache_meta(f"{filepath}.w{target_width}",
                       _cache_fingerprint(filepath, target_width=target_width)) is not None


def _save_cache(cache_base: str, array: np.ndarray, meta: dict):
    try:
        np.save(cache_base + ".npy", array)
        with open(cache_base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print(f"  Cached to {cache_base}.npy for faster next start")
    except Exception as e:
        print(f"Warning: could not write cache {cache_base}.npy: {e}")

def _is_out_of_memory(e: BaseException) -> bool:
    """
    Whether a failure is a failed allocation.

    Numpy raises MemoryError, but OpenCV does not let one through: it allocates
    inside its own decoder and resizer and raises StsNoMem instead. Its other
    errors say the image could not be decoded, which is not a size problem.
    """
    if isinstance(e, MemoryError):
        return True
    return isinstance(e, cv2.error) and getattr(e, "code", None) == cv2.Error.StsNoMem


@contextmanager
def _fits_in_memory(what: str, remedy: str):
    """
    Turn running out of system RAM while preparing a map into an error that
    names the map and the parameter to change.

    Numpy's own message says how much it wanted and for what shape, which is
    worth keeping, but on its own it surfaces as a bare traceback from whatever
    line happened to allocate last.

    Parameters
    ----------
    what : str
        Name of the map, for the message
    remedy : str
        What the user can change to make it fit
    """
    try:
        yield
    except (MemoryError, cv2.error) as e:
        if not _is_out_of_memory(e):
            raise
        raise MapTooLargeError(
            f"Not enough memory to prepare {what}: {e}\n{remedy}") from e


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
                        diameter_km = float(diameter_km_str)
                        moon_feature = MoonFeature(
                            name=name,
                            lat=float(lat_str),
                            lon=float(lon_str),
                            angular_radius=diameter_km / 60.647,
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

ELEVATION_DOWNSCALE_REMEDY = "Raise --downscale (Elevation downscale in the launcher)."


# Rows converted at a time when the full-size map is built (see
# _build_full_size_cache), as for the color texture: large enough that the
# per-band overhead is lost in the work, small enough to stay out of the way.
_ELEVATION_BLOCK_ROWS = 1024


def _build_full_size_cache(elev_src: np.ndarray, scale: float,
                           cache_base: str, fingerprint: dict) -> Optional[tuple[np.ndarray, float]]:
    """
    Write the undownscaled map to its cache a band of rows at a time and hand it
    back memory-mapped, rather than built in RAM.

    Written this way for memory: at downscale 1 the float32 map is twice the
    size of the source - about 16 GB against 8 GB for the 46080 x 92160 LDEM -
    and holding both at once needs some 24 GB of RAM, which puts the map out of
    reach of machines that could otherwise render it. Here only the source and
    one band are ever resident, and the renderer reads the rest from the file as
    it uploads it.

    Returns None when the file cannot be written - no room for it, or nowhere to
    put it - for the caller to fall back to building the map in memory.
    """
    height, width = elev_src.shape
    path = cache_base + ".npy"
    needed = height * width * 4
    free = shutil.disk_usage(os.path.dirname(path) or ".").free
    if free < needed * 1.02:
        print(f"Not enough disk space for {path} "
              f"({needed / (1024**3):.2f} GB needed, {free / (1024**3):.2f} GB free)")
        return None

    print(f"  Building {path} ({needed / (1024**3):.2f} GB on disk), "
          f"{_ELEVATION_BLOCK_ROWS} rows at a time")
    try:
        elevation = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                              shape=(height, width))
    except Exception as e:
        print(f"Warning: could not write {path}: {e}")
        return None

    low, peak = np.inf, -np.inf
    for y in range(0, height, _ELEVATION_BLOCK_ROWS):
        band = elevation[y:y + _ELEVATION_BLOCK_ROWS]
        band[:] = elev_src[y:y + _ELEVATION_BLOCK_ROWS].astype(np.float32)
        band *= scale
        band += 1.0
        low = min(low, float(band.min()))
        peak = max(peak, float(band.max()))

    print(f"  Dimensions: {elevation.shape}")
    print("  Relief range: {:.0f} m to {:+.0f} m relative to the 1737.4 km reference radius".format(
        (low - 1.0) * MOON_REFERENCE_RADIUS_M, (peak - 1.0) * MOON_REFERENCE_RADIUS_M))

    # Keep the surface inside the bounding sphere: highest peak = exactly 1.0
    radius_scale = peak
    for y in range(0, height, _ELEVATION_BLOCK_ROWS):
        elevation[y:y + _ELEVATION_BLOCK_ROWS] /= radius_scale
    elevation.flush()
    del elevation                       # reopened read-only below

    try:
        with open(cache_base + ".json", "w", encoding="utf-8") as f:
            json.dump({**fingerprint, "radius_scale": radius_scale}, f)
        print(f"  Cached to {path} for faster next start")
    except Exception as e:
        # The map is built and usable; only the next start pays, rebuilding it
        print(f"Warning: could not write {cache_base}.json: {e}")

    return np.load(path, mmap_mode="r"), radius_scale


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

    # Disk cache of the processed result. The undownscaled map is left in its
    # file and paged in as it is read: at ~16 GB it is bigger than the RAM of
    # machines that can still render it (see _build_full_size_cache).
    cache_base = f"{filepath}.ds{downscale}"
    fingerprint = _cache_fingerprint(filepath, downscale=downscale)
    mmap_mode = "r" if downscale == 1 else None
    with _fits_in_memory("the elevation map", ELEVATION_DOWNSCALE_REMEDY):
        elevation, meta = _load_cache(cache_base, fingerprint, mmap_mode)
    if elevation is not None:
        print(f"  Loaded from cache: {cache_base}.npy, dimensions {elevation.shape}")
        return elevation, float(meta["radius_scale"])

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Elevation file not found: {filepath}, and no cache of it downscaled by "
            f"{downscale} beside it. Restore the file, or start with a downscale one "
            f"has already been cached for.")

    with _fits_in_memory("the elevation map", ELEVATION_DOWNSCALE_REMEDY):
        elev_src = read_image(filepath)

        if elev_src is None:
            raise ValueError(f"Failed to read elevation file: {filepath}")

        print(f"  Original dimensions: {elev_src.shape}")
        print(f"  Size: {elev_src.nbytes / (1024**3):.2f} GB")

        # Reinterpret as signed 16-bit and convert to displacement factor of the radius
        elev_src.dtype = np.int16
        scale = LDEM_METERS_PER_UNIT / MOON_REFERENCE_RADIUS_M

        if downscale == 1:
            built = _build_full_size_cache(elev_src, scale, cache_base, fingerprint)
            if built is not None:
                return built            # already scaled and normalized, on disk
            elevation = elev_src.astype(np.float32)
        else:
            # Downscale by averaging
            h = elev_src.shape[0] // downscale
            w = elev_src.shape[1] // downscale
            elevation = elev_src.reshape(1, h, downscale, w, downscale).mean(
                4, dtype=np.float32).mean(2, dtype=np.float32).reshape(h, w)

        # Scaled in place, not a copy
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

    if downscale > 1:
        _save_cache(cache_base, elevation, {**fingerprint, "radius_scale": radius_scale})

    return elevation, radius_scale


# Color map downscale factors OpenCV can decode straight to, so the full-size
# image is never allocated. Anything else would mean decoding in full first,
# which is exactly what the factor is there to avoid.
COLOR_DOWNSCALE_FACTORS = (1, 2, 4, 8)

COLOR_DOWNSCALE_REMEDY = "Raise --color-downscale (Color downscale in the launcher)."

_REDUCED_COLOR_FLAGS = {
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}

# Albedo range the 0-255 source is mapped onto: dark maria at 0.2, brightest
# highlands at 0.95.
COLOR_ALBEDO_MIN = 0.2
COLOR_ALBEDO_RANGE = 0.75

# Rows converted at a time. Large enough that the per-block overhead is lost in
# the noise, small enough that the temporary each channel lookup produces stays
# a few tens of megabytes rather than a copy of the whole image.
_COLOR_BLOCK_ROWS = 1024


def _albedo_lut(gamma: float) -> np.ndarray:
    """
    The whole color pipeline as a 256-entry table.

    Every step - albedo mapping, the inverse gamma that plotoptix.utils.make_color_2d
    applies so the Gamma postprocessing returns the intended color, and the scale
    back to bytes - is the same function of one 8-bit source value, and cv2.imread
    always hands back 8-bit channels. So the table gives bit-for-bit what running
    the arithmetic over the whole image gave, at 256 elements instead of billions
    (verified equal on the 10k default map and on 374-1062 Mpx 8- and 16-bit maps).
    """
    lut = np.arange(256, dtype=np.float32)
    lut = COLOR_ALBEDO_MIN + (COLOR_ALBEDO_RANGE / 255) * lut
    lut = np.power(lut, gamma, dtype=np.float32)
    lut *= 255
    return lut.astype(np.uint8)


def load_color_data(filepath: str, gamma: float = 2.2, downscale: int = 1) -> np.ndarray:
    """
    Load and process the Moon color/albedo data.

    Parameters
    ----------
    filepath : str
        Path to the color TIFF file
    gamma : float
        Gamma correction value
    downscale : int
        Decode the map at 1/downscale of its size, one of COLOR_DOWNSCALE_FACTORS.
        Peak memory falls with the square of it, which is what makes maps beyond
        a few hundred megapixels loadable at all.

    Returns
    -------
    np.ndarray
        Processed color data ready for texturing, RGBA bytes
    """
    print(f"Loading color data from {filepath}...")

    # Disk cache of the decoded, downscaled image (skipped at downscale 1, where
    # it would be larger than the compressed source for little gain). Gamma is
    # applied after it is read, so changing gamma does not invalidate it and a
    # source deleted to reclaim its gigabytes stays deleted.
    cache_base = f"{filepath}.ds{downscale}"
    fingerprint = None
    if downscale > 1:
        fingerprint = _cache_fingerprint(filepath, downscale=downscale)
        with _fits_in_memory("the color map", COLOR_DOWNSCALE_REMEDY):
            color_src, _ = _load_cache(cache_base, fingerprint)
        if color_src is not None:
            print(f"  Loaded from cache: {cache_base}.npy, dimensions {color_src.shape}")
            return _moon_texture(color_src, gamma)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Color file not found: {filepath}, and no cache of it downscaled by "
            f"{downscale} beside it. Restore the file, or start with a color downscale "
            f"one has already been cached for.")

    with _fits_in_memory("the color map", COLOR_DOWNSCALE_REMEDY):
        color_src = cv2.imread(filepath, _REDUCED_COLOR_FLAGS.get(downscale, cv2.IMREAD_COLOR))

    if color_src is None:
        raise ValueError(f"Failed to read color file: {filepath}")

    print(f"  Dimensions: {color_src.shape}"
          + (f" (decoded at 1/{downscale})" if downscale > 1 else ""))

    if fingerprint is not None:
        _save_cache(cache_base, color_src, fingerprint)

    return _moon_texture(color_src, gamma)


def _moon_texture(color_src: np.ndarray, gamma: float) -> np.ndarray:
    """
    Turn the decoded BGR bytes into the RGBA texture, a band of rows at a time.

    Written this way for memory: the straightforward whole-image form went
    through several float32 copies of three and four channels, peaking at about
    ten times the size of the finished texture and putting large color maps out
    of reach. Here only the source and the result are ever live.
    """
    lut = _albedo_lut(gamma)
    height, width = color_src.shape[:2]
    with _fits_in_memory("the color map texture", COLOR_DOWNSCALE_REMEDY):
        color_data = np.empty((height, width, 4), dtype=np.uint8)

    for y in range(0, height, _COLOR_BLOCK_ROWS):
        block = color_src[y:y + _COLOR_BLOCK_ROWS]
        texture = color_data[y:y + _COLOR_BLOCK_ROWS]
        texture[..., 0] = lut[block[..., 2]]    # source is BGR, texture is RGBA
        texture[..., 1] = lut[block[..., 1]]
        texture[..., 2] = lut[block[..., 0]]
        texture[..., 3] = 255                   # opaque

    print(f"  Texture size: {color_data.nbytes / (1024**3):.2f} GB")

    return color_data


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
    # Disk cache of the processed result, keyed by the target width
    # (screen-dependent), so the 16k source is decoded and resized only once.
    # Tried before the source is looked for, so that a source deleted to reclaim
    # its megabytes keeps the sky it was cached into (see _cache_fingerprint).
    cache_base = f"{filepath}.w{target_width}"
    fingerprint = _cache_fingerprint(filepath, target_width=target_width)
    try:
        star_map, _ = _load_cache(cache_base, fingerprint)
        if star_map is not None:
            print(f"Loaded star map from cache: {cache_base}.npy, dimensions {star_map.shape}")
            return star_map

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
    except (MemoryError, cv2.error) as e:
        if not _is_out_of_memory(e):
            raise
        # Background decoration only: a starless sky beats not starting at all
        print(f"Not enough memory to prepare the star map: {e}".rstrip())
        print("Starting without the star map.")
        return None

    print(f"  Dimensions: {star_map.shape}")

    _save_cache(cache_base, star_map, fingerprint)

    return star_map