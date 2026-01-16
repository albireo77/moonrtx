from typing import NamedTuple
from numpy.typing import NDArray

class MoonEphemeris(NamedTuple):
    az: float
    alt: float
    ra: float
    dec: float
    distance: int
    illum: float
    phase: float
    pa: float
    pa_axis_view: float
    q: float
    libr_long: float
    libr_lat: float

class MoonFeature(NamedTuple):
    name: str
    lat: float
    lon: float
    angle: float
    standard_label: bool
    spot_label: bool
    status_bar: bool

class MoonGrid(NamedTuple):
    lat_lines: list
    lon_lines: list
    lat_labels: list
    lat_label_values: list
    lon_labels: list
    lon_label_values: list
    N: list

class CameraParams(NamedTuple):
    eye: list
    target: list
    up: list
    fov: float

class Scene(NamedTuple):
    eye: NDArray
    target: NDArray
    up: NDArray
    light_pos: NDArray