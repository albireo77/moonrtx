from typing import NamedTuple, Optional

from numpy.typing import NDArray

class MoonEphemeris(NamedTuple):
    az: float
    alt: float
    ra: float
    dec: float
    distance: float
    sun_distance: float
    phase_angle: float
    age_days: float
    bright_limb_angle: float
    libr_long_geo: float
    libr_lat_geo: float
    libr_long_topo: float
    libr_lat_topo: float
    elongation: float
    phase_name: str
    colongitude: float
    rotation_matrix: NDArray

class MoonFeature(NamedTuple):
    name: str
    lat: float
    lon: float
    angular_radius: float
    diameter_km: float
    standard_label: bool
    spot_label: bool
    status_bar: bool
    feature_id: Optional[int]
    www_address: Optional[str]

class Camera(NamedTuple):
    eye: list
    target: list
    up: list
    fov: float
    type: str = "Pinhole"
    # The three lens parameters below are used only by the lens-simulating
    # camera types (DoF/ThinLens, Fisheye, ...); the Pinhole camera used by the
    # app has infinite depth of field and ignores them. Note: if the type is
    # ever switched to a lens camera, focal_scale 0.7 puts the focal plane well
    # in front of the Moon (0.7 x eye-target distance) and blurs the surface.
    aperture_radius: float = 0.01
    aperture_fract: float = 0.2
    focal_scale: float = 0.7

class Observer(NamedTuple):
    lat: float
    lon: float
    elevation_m: int

class MoonLabel(NamedTuple):
    segments: list[list]
    anchor_point: tuple[float, float]
