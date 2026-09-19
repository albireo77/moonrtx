import base64
import re
import struct
from datetime import datetime
from typing import NamedTuple, Optional

from numpy.typing import NDArray

from moonrtx.view_orientation import VIEW_ORIENTATIONS

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
    # taken off, as part of the name InitView writes (encode) and reads back
    # (decode).
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


class InitView(NamedTuple):
    """
    A view as it is written into the name a saved image or an exported video is
    offered, and read back from that name by --init-view and the launcher: the
    moment, the observer's place, the view orientation, parallactic mode and the
    camera. The name is laid out as

        <time>_lat<+dd.dddddd>_lon<+ddd.dddddd>_view<orientation>_par<0|1>_cam<camera>[_x<frames>]

    the time in ISO form to the second, its colons written as dots, which a file
    name cannot hold. encode writes it and decode reads it back, side by side
    here so that a part added to one is not missed by the other - as Camera does
    for the camera within it.
    """
    dt_local: datetime
    lat: float
    lon: float
    view_orientation: str
    parallactic_mode: bool
    camera: Optional[Camera]

    _NAME_PATTERN = (r'^(.+?)_lat([+-]?\d+\.\d+)_lon([+-]?\d+\.\d+)'
                     r'_view([A-Z]+)(?:_par([01]))?'
                     r'_cam([A-Za-z0-9_-]{%d})(?:_x\d+)?$' % Camera.TEXT_LENGTH)

    def encode(self) -> str:
        """
        The name, without an extension. A camera the renderer could not give is
        written as "nocam", which decode does not read back: such a name
        still says when and where, but cannot be returned to.
        """
        # To the second: decode turns every dot back into a colon, so a
        # fractional part would come back as "SS:ffffff", which parses only
        # through a leniency of the older ISO reader
        time = self.dt_local.isoformat(timespec='seconds').replace(':', '.')
        camera = f"cam{self.camera.encode()}" if self.camera is not None else "nocam"
        return (f"{time}_lat{self.lat:+.6f}_lon{self.lon:+.6f}"
                f"_view{self.view_orientation}_par{1 if self.parallactic_mode else 0}_{camera}")

    @classmethod
    def decode(cls, text: str, zone) -> Optional["InitView"]:
        """
        The view a name was written from, its time on the clock of `zone`, or
        None when the name is not one - saying why wherever it is more than the
        name simply not matching.

        Two parts are optional. _par<0|1> came in with the parallactic-mode
        flag, and a name from before it is taken as OFF. _x<frames> is what the
        video export adds, so an exported video says how long it is; it is read
        past, a video carrying the same view a screenshot does. The camera ahead
        of it is taken by its length (Camera.TEXT_LENGTH), which is what tells
        the two apart - base64 could otherwise end in "_x120" of its own accord.

        The time carries its offset, as encode writes it, and so names an
        instant, re-expressed in `zone`: it means the same moment wherever it is
        opened. One without an offset is read as a wall clock in `zone`.
        """
        try:
            match = re.match(cls._NAME_PATTERN, text)
            if not match:
                return None
            time_text, lat, lon, orientation, par_flag, camera_text = match.groups()
            if orientation not in VIEW_ORIENTATIONS:
                print(f"Invalid view orientation in init-view: {orientation}")
                return None
            camera = Camera.decode(camera_text)
            if camera is None:
                return None
            try:
                dt = datetime.fromisoformat(time_text.replace('.', ':'))
            except ValueError as e:
                print(f"Incorrect time: {e}")
                return None
            dt_local = dt.replace(tzinfo=zone) if dt.tzinfo is None else dt.astimezone(zone)
            return cls(dt_local, float(lat), float(lon), orientation, par_flag == '1', camera)
        except Exception as e:
            print(f"Error parsing init-view string: {e}")
            return None

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
