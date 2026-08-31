from datetime import datetime
from typing import NamedTuple, Optional

from numpy.typing import NDArray

# Exit code run_renderer_process leaves when a map did not fit, so the GUI
# launcher can tell that apart from any other failure and say what to change.
MAP_TOO_LARGE_EXIT_CODE = 2


class MapTooLargeError(RuntimeError):
    """
    A Moon map did not fit in memory - system RAM while it was being prepared,
    or GPU memory when it was uploaded. Carries a message naming the map and
    the downscale parameter that makes it smaller.
    """


class VisibilityChart(NamedTuple):
    """
    When the Moon and the Sun are above the observer's horizon over a span of
    days, as spells rather than as daily rise and set times: a spell that runs
    past either end of the span is clipped to it, so a Moon that stays up for
    days near the pole is one long spell instead of a gap in a table. Times are
    UTC. `transits` pairs each upper meridian crossing with the altitude the
    Moon reaches there, which is the highest it stands that time round.
    """
    start: datetime
    end: datetime
    moon_up: list[tuple[datetime, datetime]]
    sun_up: list[tuple[datetime, datetime]]
    sun_twilight: list[tuple[datetime, datetime]]
    transits: list[tuple[datetime, float]]
    illumination: list[tuple[datetime, float]]

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
    subsolar_lat: float
    subsolar_lon: float
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
    feature_id: Optional[str]
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

class ClairObscurEvent(NamedTuple):
    """
    A light-and-shadow pattern that forms when the terminator lights only the
    high ground of a formation while the ground around it is still dark.

    Defined by a representative point and the range of Sun altitude over that
    point in which the pattern stands, rather than by the colongitude the usual
    sources quote: the altitude is what the terrain responds to, and it tracks
    the effect a couple of degrees better at high latitudes, where the subsolar
    latitude of the month shifts the sunrise line. Peaks catch the light before
    the ground does, so a formation whose interest is its summits (the Jewelled
    Handle) has a negative window.
    """
    name: str
    lat: float
    lon: float
    sun_alt_min: float
    sun_alt_max: float
    rising: bool            # True at local sunrise, False at local sunset
    description: str

class Observer(NamedTuple):
    lat: float
    lon: float
    elevation_m: int

class MoonLabel(NamedTuple):
    segments: list[list]
    anchor_point: tuple[float, float]
