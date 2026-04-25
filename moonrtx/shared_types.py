from typing import NamedTuple, Optional

from numpy.typing import NDArray

class MoonEphemeris(NamedTuple):
    az: float
    alt: float
    ra: float
    dec: float
    distance: int
    phase_angle: float
    pa: float
    q: float
    libr_long_geo: float
    libr_lat_geo: float
    libr_long_topo: float
    libr_lat_topo: float
    sun_separation: float  # Topocentric angular separation between Sun and Moon centers (degrees)
    delta_long: float  # Ecliptical longitude difference (degrees, 0-360)
    colongitude: float  # Selenographic colongitude of the Sun (degrees, 0-360)
    rotation_matrix: NDArray

class MoonFeature(NamedTuple):
    name: str
    lat: float
    lon: float
    angular_radius: float
    cos_lat: float
    diameter_km: float
    standard_label: bool
    spot_label: bool
    status_bar: bool
    feature_id: Optional[int]
    www_address: Optional[str]

class CameraParams(NamedTuple):
    eye: list
    target: list
    up: list
    fov: float

class MoonLabel(NamedTuple):
    segments: list[list]
    anchor_point: tuple[float, float]
