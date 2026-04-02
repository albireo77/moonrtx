from typing import NamedTuple

class MoonEphemeris(NamedTuple):
    az: float
    alt: float
    ra: float
    dec: float
    distance: int
    phase: float
    pa: float
    pa_axis_view: float
    q: float
    libr_long: float
    libr_lat: float
    sun_separation: float  # Topocentric angular separation between Sun and Moon centers (degrees)

class MoonFeature(NamedTuple):
    name: str
    lat: float
    lon: float
    angular_radius: float
    cos_lat: float
    size_km: float
    standard_label: bool
    spot_label: bool
    status_bar: bool

class CameraParams(NamedTuple):
    eye: list
    target: list
    up: list
    fov: float

class MoonLabel(NamedTuple):
    segments: list[list]
    anchor_point: tuple[float, float]
