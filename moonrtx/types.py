from typing import NamedTuple

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