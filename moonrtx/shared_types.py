import base64
import struct
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

    # How the view is packed into the name a saved image or an exported video is
    # offered: eye, target and up, three floats each, then the field of view -
    # ten little-endian float32s, written in url-safe base64 with the padding
    # taken off. Both ends of that name read it from here: get_default_filename
    # writing it (renderer_dialogs), parse_init_view reading it back (main).
    FORMAT = '<10f'
    # How many characters that makes, which is what tells the camera apart from
    # anything after it in a name: base64 spells itself with digits and
    # underscores as well as letters, so a video's "_x120" could otherwise be
    # read as more of the camera. Worked out from FORMAT rather than written
    # down, so the two cannot disagree.
    TEXT_LENGTH = len(base64.urlsafe_b64encode(bytes(struct.calcsize(FORMAT))).rstrip(b'='))

    def encode(self) -> str:
        """
        The view as text for a file name, url-safe base64 without padding.

        The view only - eye, target, up and field of view. The type and the lens
        parameters are not written, the app using the Pinhole camera throughout,
        so decode gives those back at their defaults: the two round-trip the
        view, not the lens.
        """
        packed = struct.pack(self.FORMAT, *self.eye, *self.target, *self.up, self.fov)
        return base64.urlsafe_b64encode(packed).decode('ascii').rstrip('=')

    @classmethod
    def decode(cls, text: str) -> Optional["Camera"]:
        """
        The view read back from encode's text, the lens at its defaults (see
        encode), or None when the text is not one.
        """
        try:
            packed = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))
            values = struct.unpack(cls.FORMAT, packed)
        except Exception as e:
            print(f"Error decoding camera: {e}")
            return None
        return cls(eye=list(values[0:3]), target=list(values[3:6]),
                   up=list(values[6:9]), fov=values[9])

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
