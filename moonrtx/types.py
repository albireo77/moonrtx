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