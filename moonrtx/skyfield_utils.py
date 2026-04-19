import ssl
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from skyfield.api import Loader, PlanetaryConstants, load_file


SKYFIELD_EPHEMERIS_NAME = "de421.bsp"
SKYFIELD_EPHEMERIS_URL = "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp"
SKYFIELD_MOON_FRAME_NAME = "MOON_ME_DE421"
SKYFIELD_MOON_TEXT_KERNEL_NAME = "moon_080317.tf"
SKYFIELD_MOON_TEXT_KERNEL_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf"
SKYFIELD_MOON_PCK_NAME = "pck00008.tpc"
SKYFIELD_MOON_PCK_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/pck00008.tpc"
SKYFIELD_MOON_BINARY_PCK_NAME = "moon_pa_de421_1900-2050.bpc"
SKYFIELD_MOON_BINARY_PCK_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc"
SKYFIELD_DOWNLOAD_TIMEOUT_SEC = 120
SKYFIELD_MOON_FRAME_START_UTC = datetime(1900, 1, 1, tzinfo=timezone.utc)
SKYFIELD_MOON_FRAME_END_UTC = datetime(2051, 1, 1, tzinfo=timezone.utc)


def _skyfield_data_dir() -> Path:
    from moonrtx.main import DATA_DIRECTORY_PATH
    data_dir = Path(DATA_DIRECTORY_PATH) / "skyfield"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_file_with_tls_fallback(url: str, destination: Path, description: str) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    ssl_error = None

    for verify_tls in (True, False):
        try:
            context = ssl.create_default_context() if verify_tls else ssl._create_unverified_context()
            with urlopen(url, context=context, timeout=SKYFIELD_DOWNLOAD_TIMEOUT_SEC) as response, temp_path.open("wb") as fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)

            temp_path.replace(destination)
            if not verify_tls:
                print(f"WARNING: Downloaded {description} without TLS certificate verification.")
            return
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            if verify_tls and isinstance(reason, ssl.SSLError):
                ssl_error = exc
                continue
            if temp_path.exists():
                temp_path.unlink()
            raise
        except ssl.SSLError as exc:
            if verify_tls:
                ssl_error = exc
                continue
            if temp_path.exists():
                temp_path.unlink()
            raise

    if temp_path.exists():
        temp_path.unlink()
    raise RuntimeError(f"Unable to download {description}: {ssl_error}")


def _ensure_skyfield_support_file(file_name: str, url: str) -> Path:
    path = _skyfield_data_dir() / file_name
    if not path.exists():
        _download_file_with_tls_fallback(url, path, file_name)
    return path


@lru_cache(maxsize=1)
def skyfield_timescale():
    return Loader(str(_skyfield_data_dir()), verbose=False).timescale()


@lru_cache(maxsize=1)
def skyfield_ephemeris():
    ephemeris_path = _skyfield_data_dir() / SKYFIELD_EPHEMERIS_NAME
    if not ephemeris_path.exists():
        _download_file_with_tls_fallback(
            SKYFIELD_EPHEMERIS_URL,
            ephemeris_path,
            SKYFIELD_EPHEMERIS_NAME,
        )
    return load_file(str(ephemeris_path))


@lru_cache(maxsize=1)
def skyfield_moon_frame():
    planetary_constants = PlanetaryConstants()

    with _ensure_skyfield_support_file(
        SKYFIELD_MOON_TEXT_KERNEL_NAME,
        SKYFIELD_MOON_TEXT_KERNEL_URL,
    ).open("rb") as fh:
        planetary_constants.read_text(fh)

    with _ensure_skyfield_support_file(
        SKYFIELD_MOON_PCK_NAME,
        SKYFIELD_MOON_PCK_URL,
    ).open("rb") as fh:
        planetary_constants.read_text(fh)

    binary_fh = _ensure_skyfield_support_file(
        SKYFIELD_MOON_BINARY_PCK_NAME,
        SKYFIELD_MOON_BINARY_PCK_URL,
    ).open("rb")
    planetary_constants.read_binary(binary_fh)
    planetary_constants._moonrtx_binary_fh = binary_fh
    return planetary_constants.build_frame_named(SKYFIELD_MOON_FRAME_NAME)