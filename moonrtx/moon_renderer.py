"""
MoonRenderer: core renderer class (composing mixins) and run_renderer entry point.
"""

import sys
import tkinter as tk
import numpy as np
from contextlib import contextmanager
from typing import Optional
from datetime import datetime, timedelta, timezone

import plotoptix
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse

from moonrtx import astro
from moonrtx.shared_types import (Camera, MAP_TOO_LARGE_EXIT_CODE,
                                  MapTooLargeError, Observer)
from moonrtx.data_loader import load_moon_features, load_elevation_data, load_color_data, load_starmap
from moonrtx.view_orientation import VIEW_ORIENTATION_NSWE, VIEW_ORIENTATION_NSEW, VIEW_ORIENTATION_SNEW, VIEW_ORIENTATION_SNWE

# Mixins – each adds a focused group of methods
from moonrtx.renderer_status import StatusMixin, timezone_name
from moonrtx.renderer_dialogs import DialogsMixin
from moonrtx.renderer_labels import LabelsMixin
from moonrtx.renderer_pins import PinsMixin
from moonrtx.renderer_navigation import NavigationMixin
from moonrtx.renderer_video import VideoMixin
from moonrtx.renderer_fov import FovMixin
from moonrtx.renderer_subpoints import SubPointsMixin
from moonrtx.renderer_compass import CompassMixin


# The star map is loaded several times wider than the window: it wraps the whole
# sky, so only a fraction of it is ever on screen at once.
STARMAP_WIDTH_FACTOR = 6


def screen_size() -> tuple[int, int]:
    """
    Screen width and usable height (less the taskbar), from a hidden root window.
    """
    _tmp = tk.Tk()
    _tmp.withdraw()
    size = (_tmp.winfo_screenwidth(), _tmp.winfo_screenheight() - 40)
    _tmp.destroy()
    return size


def starmap_target_width() -> int:
    """
    The width load_starmap is asked for, and so the key its cache is stored
    under. A module-level function because main.check_starmap_file needs it
    before a renderer (and its window) exists, to tell whether the source has
    to be downloaded at all.
    """
    return screen_size()[0] * STARMAP_WIDTH_FACTOR


class MoonRenderer(StatusMixin, DialogsMixin, LabelsMixin, PinsMixin, NavigationMixin,
                   VideoMixin, FovMixin, SubPointsMixin, CompassMixin):
    """
    Renders the Moon surface as seen from a specific location on Earth
    at a specific time, with accurate solar illumination.
    """

    # Scene geometry
    MOON_RADIUS = 10.0          # Radius of Moon sphere in scene units
    MOON_RADIUS_KM = 1737.4     # The real radius that scene radius stands for,
                                # used wherever the render has to match a real
                                # angle or length: apparent size, the Sun disk,
                                # surface distances and elevation differences
    MOON_FILL_FRACTION = 0.9    # Moon fills 90% of window height (5% margins top/bottom)
                                # at MOON_REFERENCE_DISTANCE (see moon_camera_distance)
    # Reference camera distance in scene units. Larger distance renders the limb closer
    # to what a real observer sees (at 30 radii the visible cap reaches 88.1 degrees
    # from the disk center vs 84.3 at 10 radii and 89.7 in reality). The value is a
    # trade-off: much larger distances degrade float32 ray precision and produce
    # contour/tessellation artifacts on the displaced surface (visible at ~220 radii).
    CAMERA_DISTANCE = MOON_RADIUS * 30
    # The Moon's real apparent size varies by ~14% over an anomalistic month
    # (perigee 356500 km vs apogee 406700 km), plus up to 1.7% within a single
    # night as the topocentric distance drops by one Earth radius on the way to
    # the zenith. The render follows it by moving the camera (moon_camera_distance),
    # never by changing the FOV: the star background is an environment texture
    # sampled by ray direction, so any FOV change would zoom the whole sky along
    # with the Moon, while a shift of the eye along the view axis leaves every
    # ray direction - and therefore the sky - exactly as it was.
    # CAMERA_DISTANCE renders the Moon at the size it has at this distance from
    # the observer; the mean geocentric one. Ephemeris distances are topocentric,
    # i.e. on average about half an Earth radius shorter, so the disk typically
    # fills slightly more than MOON_FILL_FRACTION of the window height (0.91),
    # between 0.84 for an apogee Moon near the horizon and 0.99 for a perigee
    # Moon in the zenith - always inside the window.
    MOON_REFERENCE_DISTANCE = 384_400.0     # km
    # Sun light distance and radius keep the real solar angular size seen from the
    # Moon: arcsin(100/21460) = 0.267 degrees, so penumbra softness is realistic.
    # The distance also sets the terminator parallax error: a light at distance D
    # pulls the terminator toward the subsolar point by arcsin(MOON_RADIUS/D).
    # At 2146 units that was 0.267 degrees of selenographic longitude (~30 minutes
    # of crater sunrise/sunset timing at 0.508 deg/hour of colongitude); at 21460
    # it is 0.027 degrees (~3 minutes), below other error sources of the app.
    SUN_LIGHT_DISTANCE = 21460
    # Radius of the light (not of the visible Sun disk) at the mean Sun distance;
    # update_view rescales it to the true Sun distance of the date, so penumbra
    # softness and illumination follow the +/-1.7% annual angular size variation
    SUN_RADIUS = 100
    # The light color is the emitting sphere's radiance: surface illumination
    # depends only on radiance x angular size, NOT on light distance (verified
    # against PlotOptiX 0.19.0), so this calibration constant must not change
    # when SUN_LIGHT_DISTANCE changes. Value maps the user brightness setting
    # (default 100) to a well-exposed surface; kept from the original tuning
    # at light distance 2146.
    SUN_BRIGHTNESS_SCALE = (2146.0 / 100.0) ** 2

    # Displaced-surface ray tracing settings for PlotOptiX >= 0.19.2, which
    # decoupled the ray-marching step from the self-intersection epsilon
    # (added upstream for MoonRTX, see
    # https://github.com/rnd-team-dev/plotoptix/issues/71). scene_epsilon now
    # only lifts hit points and shadow-ray origins off the terrain: 1e-4 scene
    # units = 17 m (~0.3 km of shadow-tip error at 3 deg sun altitude, below
    # perception; the old coupled default caused 265 m of lift = 5-7 km of
    # missing shadow near the terminator, ~2 h of shadow evolution). Do not go
    # below 1e-4: rays start leaking under the surface and darken the terrain.
    # marching_step stays coarse for speed; marching_step_eps 3e-4 is the
    # near-surface refinement sweet spot (1e-4 renders 3.6x slower with no
    # visible gain). Measured on the Piazzi Smyth mount shadow at 2.8 deg sun:
    # 18.1 km rendered vs 18.9 km geometric, at 2.4x the cost of the fastest
    # (but badly truncating) settings - exact shadows no longer need a toggle.
    SCENE_EPSILON = 1.0e-4
    MARCHING_STEP = 5.0e-3
    MARCHING_STEP_EPS = 3.0e-4

    # Visible Sun disk, decoupled from the light source (see calculate_sun_disk).
    # It sits closer than the light, but its material lets shadow rays pass
    # through (see init_renderer), so it never shadows the Moon.
    SUN_RADIUS_KM = 695_700.0
    SUN_DISK_NAME = "sun_disk"
    SUN_DISK_DISTANCE = 3100    # distance from the default camera position
    # Flat radiance: >= 1.12 renders as pure white for any gamma in the 0.5-5.0 range,
    # while keeping the stray light the disk bounces onto the Moon negligible
    SUN_DISK_COLOR = 2.0
    # Radius that effectively hides the disk: the size it is created with before
    # the first update_view gives it a real position and size, and the size it is
    # parked at when the Sun is too far from the Moon to share a view with it
    # (see calculate_sun_disk)
    SUN_DISK_PARKED_RADIUS = 0.01

    # Accumulation settings. Since PlotOptiX 0.19.1 the displayed image is
    # presented once per completed accumulation cycle (max_accumulation_frames),
    # so any scene change (time stepping, brightness, overlays, navigation)
    # would only appear after a full cycle converges - held-key Q/W animation
    # would barely refresh at all. During interactive changes the cycle is
    # therefore shortened to a single frame (immediate but slightly noisy
    # preview, ~20 steps/s measured at full screen with exact shadows) and the
    # converged setting is restored shortly after the last change.
    # 64 frames settle in ~1.5 s at full screen: since interactivity uses the
    # single-frame preview, this value only sets the quiet-image quality. 64
    # keeps path-tracing grain low in the shadow/terminator regions (which
    # zero ambient no longer masks) at diminishing returns beyond it.
    ACCUMULATION_FRAMES = 64
    PREVIEW_ACCUMULATION_FRAMES = 1
    PREVIEW_RESTORE_DELAY_MS = 500

    CAMERA_NAME = "cam1"
    LIGHT_NAME = "sun"
    MOON_OBJECT_NAME = "moon"

    def __init__(self,
                 elevation_file: str,
                 color_file: str,
                 features_file: str,
                 brightness: int,
                 observer: Observer,
                 initial_camera: Optional[Camera],
                 dt_local: datetime,
                 starmap_file: Optional[str],
                 downscale: int = 3,
                 color_downscale: int = 1,
                 time_step_minutes: int = 15,
                 init_view_orientation: str = VIEW_ORIENTATION_NSWE,
                 gamma: float = 2.2,
                 parallactic_mode: bool = False):
        """
        Initialize the planetarium.

        Parameters
        ----------
        elevation_file : str
            Path to Moon elevation data TIFF
        color_file : str
            Path to Moon color data TIFF
        features_file : str
            Moon features CSV file with craters, mounts etc.
        brightness : int
            Brightness
        initial_camera : Optional[Camera]
            Initial camera for resets with R key (if None, a default camera will be calculated from ephemeris)
        dt_local : datetime
            Local datetime for the view 
        starmap_file : Optional[str]
            Path to star map TIFF for background (if None, black background is used)
        downscale : int
            Elevation map downscale factor
        color_downscale : int
            Color map downscale factor
        time_step_minutes : int
            Time step in minutes for Q/W keys
        init_view_orientation : str
            Initial view orientation
        observer : Observer
            Observer latitude, longitude, and elevation
        gamma : float
            Gamma correction value (default 2.2)
        parallactic_mode : bool
            Whether to use parallactic projection mode (default False)
        """
        self.downscale = downscale
        self.color_downscale = color_downscale
        self.gamma = gamma
        self.time_step_minutes = time_step_minutes
        self.parallactic_mode = parallactic_mode
        self.observer = observer

        # Load data (color and star map are loaded in init_renderer, where they
        # are uploaded to GPU textures and not needed afterwards)
        self.color_file = color_file
        self.starmap_file = starmap_file
        self.elevation, self.elevation_radius_scale = load_elevation_data(elevation_file, downscale)
        # Sort features by angular_radius (smallest first) for efficient lookup
        self.moon_features = sorted(load_moon_features(features_file), key=lambda f: f.angular_radius)
        self._init_feature_lookup()
        self.width, self.height = screen_size()

        self.brightness = brightness

        # Renderer
        self.rt = None
        self.moon_ephem = None
        self.moon_rotation = None
        self.moon_rotation_inv = None

        # Grid settings
        self.moon_grid_visible = False
        self.moon_grid = None
        # Merged grid graphs: body-frame vertices and edge indices
        self._grid_lines_pos = None
        self._grid_lines_edges = None
        self._grid_labels_pos = None
        self._grid_labels_edges = None

        # Markers at the points the Sun and Earth stand over (see SubPointsMixin)
        self.sub_points_visible = False

        self.view_orientation = init_view_orientation
        self.initial_view_orientation = init_view_orientation  # For reset with R/V keys

        self.dt_local = dt_local

        # Initial time for reset with R key
        self.initial_dt_local = self.dt_local

        # Initial camera for reset with R key. When none is given it is the
        # whole-disk view of the initial date, which needs the ephemeris and is
        # therefore resolved in init_astro.
        self.initial_camera = initial_camera

        # Moon angular radius the current camera distance was set for; kept in
        # sync by update_view (see _move_camera_to_apparent_size)
        self._apparent_radius = None

        # Flag to track if window has been maximized
        self._window_maximized = False

        # Standard labels settings
        self.standard_labels_visible = False
        self.standard_labels = None
        self.standard_label_features = []
        # Merged label graph: body-frame vertices, edges, per-label vertex
        # counts and feature unit vectors (for vectorized illumination checks)
        self._standard_labels_pos = None
        self._standard_labels_edges = None
        self._standard_labels_counts = None
        self._standard_units = None

        # Spot labels settings
        self.spot_labels_visible = False
        self.spot_labels = None
        self.spot_label_features = []
        self._spot_labels_pos = None
        self._spot_labels_edges = None
        self._spot_labels_counts = None
        self._spot_units = None

        # Light position in scene coordinates (set on first update_view)
        self.light_pos = None

        # Flag to track if search dialog is open
        self.search_dialog_open = False

        # Datetime dialog tracking
        self.datetime_dialog = None
        self._datetime_dialog_show = None  # set while the date/time window is open
        self.datetime_dialog_focused = False

        # Pins settings
        self.pins_visible = True  # Pins visible by default
        self.pins = {}  # dict mapping digit (1-9) to (lat, lon, body-frame graph vertices)

        # Distance measurement settings
        self.measuring = False
        self.measure_start_canvas = None
        self.measure_start_coords = None
        self.leading_line_id = None
        self.measured_distance = None
        self.measured_height_diff = None

        # Status bar panel variables (set up as StringVars after renderer is created)
        self._status_parallactic_var = None
        self._status_view_var = None
        self._status_time_var = None
        self._status_measured_var = None
        self._status_feature_var = None
        self._status_brightness_var = None
        self._status_gamma_var = None
        self._status_pins_var = None
        self._status_coords_var = None
        self._status_coords_alt_var = None
        self._status_coords_sun_label = None    # blanked by colour, see _update_info_coords
        self._status_coords_sun_fg = None
        self._status_coords_sun_bg = None
        self._status_feature = None

        # Interactive-preview state (short accumulation cycles during scene changes)
        self._preview_active = False
        self._preview_restore_id = None

        # Time-lapse video export state (see renderer_video.VideoMixin)
        self._init_video_export()

        # Field-of-view overlay state (see renderer_fov.FovMixin)
        self._init_fov_overlay()

        # View-orientation globe state (see renderer_compass.CompassMixin)
        self._init_compass_overlay()

        # Auto-advance (real-time playback) settings
        self._auto_advance_var = None
        self._auto_advance_id = None
        self._auto_advance_elapsed = 0
        self._auto_advance_interval = 1000  # tick interval in ms
        self._auto_advance_target_ms = time_step_minutes * 60 * 1000

        # Info panel variables (bottom-left overlay)
        self._info_frame = None
        self.show_info_panel = True
        self._info_az_var = None
        self._info_alt_var = None
        self._info_ra_var = None
        self._info_dec_var = None
        self._info_phase_var = None
        self._info_age_var = None
        self._info_elongation_var = None
        self._info_distance_var = None
        self._info_diameter_var = None
        self._info_illum_var = None
        self._info_libr_l_var = None
        self._info_libr_b_var = None
        self._info_colong_var = None

    # ---- brightness / time-step / auto-advance ----

    def change_brightness(self, delta: int):
        if delta == 0:
            return
        new_brightness = max(0, min(500, self.brightness + delta))
        if new_brightness == self.brightness:
            return
        self.brightness = new_brightness
        self.rt.update_light(self.LIGHT_NAME, color=self.brightness * self.SUN_BRIGHTNESS_SCALE)
        self._update_status_brightness()

    def change_gamma(self, delta: float):
        """
        Change the gamma correction value by a given amount.

        Parameters
        ----------
        delta : float
            Amount to add (positive) or subtract (negative) from gamma
        """
        if delta == 0:
            return
        new_gamma = self.gamma + delta
        new_gamma = round(new_gamma, 1)  # Avoid floating-point drift
        new_gamma = max(0.5, min(5.0, new_gamma))
        if new_gamma == self.gamma:
            return
        self.gamma = new_gamma
        self.rt.set_float("tonemap_gamma", self.gamma)
        self._update_status_gamma()

    def change_time_step(self, delta: int):
        """
        Change the time step value by a given amount.

        Parameters
        ----------
        delta : int
            Amount to add (positive) or subtract (negative) from time_step_minutes
        """
        if delta == 0:
            return
        new_step = max(1, min(1440, self.time_step_minutes + delta))
        if new_step == self.time_step_minutes:
            return
        self.time_step_minutes = new_step
        self._auto_advance_target_ms = new_step * 60 * 1000
        # Reset auto-advance counter when time step changes while active
        if self._auto_advance_var and self._auto_advance_var.get():
            self._auto_advance_elapsed = 0
        self._update_status_time()

    def _on_auto_advance_toggle(self):
        """Called when the auto-advance checkbox is toggled."""
        if self._auto_advance_var.get():
            self._auto_advance_elapsed = 0
            self._schedule_auto_advance()
        else:
            if self._auto_advance_id is not None:
                self.rt._root.after_cancel(self._auto_advance_id)
                self._auto_advance_id = None

    def _schedule_auto_advance(self):
        """Schedule the next auto-advance tick."""
        if self.rt is not None and self.rt._root is not None:
            self._auto_advance_id = self.rt._root.after(
                self._auto_advance_interval, self._auto_advance_tick)

    def _auto_advance_tick(self):
        """Periodic tick for auto-advance."""
        if not self._auto_advance_var.get():
            self._auto_advance_id = None
            return
        self._auto_advance_elapsed += self._auto_advance_interval
        if self._auto_advance_elapsed >= self._auto_advance_target_ms:
            self._auto_advance_elapsed = 0
            self.change_time(self.time_step_minutes)
        self._schedule_auto_advance()

    def set_time_to_now(self):
        """Set the observation time to the current (now) time."""

        self.update_view(datetime.now().astimezone())

        if self._auto_advance_var and self._auto_advance_var.get():
            self._auto_advance_elapsed = 0

        self._update_all_status_panels()

    def set_time_to_now_and_auto_advance(self):
        """Set time to now and start auto-advance to keep in sync with real time."""
        self.set_time_to_now()
        if self._auto_advance_var and not self._auto_advance_var.get():
            self._auto_advance_var.set(True)
            self._on_auto_advance_toggle()

    def change_time(self, delta_minutes: int):
        """
        Change the observation time by a given number of minutes.

        Parameters
        ----------
        delta_minutes : int
            Number of minutes to add (positive) or subtract (negative)
        """
        if delta_minutes == 0:
            return

        if self._auto_advance_var and self._auto_advance_var.get():
            self._auto_advance_elapsed = 0

        new_dt_local = self.shifted_time(delta_minutes)

        self.update_view(new_dt_local)

        self._update_status_time()
        self._update_info_moon()

    def _begin_interactive_preview(self):
        """
        Switch to single-frame accumulation cycles for the duration of a burst
        of interactive scene changes (held Q/W, brightness, navigation etc.),
        so every change is displayed immediately. Re-arms the timer that
        restores converged rendering after the burst.

        Note: change_time() itself does not call this, so programmatic time
        steps (auto-advance ticks) render straight to the converged image.
        """
        if self.rt is None or self.rt._root is None:
            return
        # Never drop to single-pass cycles while a video export runs: the
        # encoder would capture the noisy preview frames
        if self._video_export is not None:
            return
        if not self._preview_active:
            self._preview_active = True
            self.rt.set_param(max_accumulation_frames=self.PREVIEW_ACCUMULATION_FRAMES)
        if self._preview_restore_id is not None:
            self.rt._root.after_cancel(self._preview_restore_id)
        self._preview_restore_id = self.rt._root.after(
            self.PREVIEW_RESTORE_DELAY_MS, self._end_interactive_preview)

    def _end_interactive_preview(self):
        """Restore converged accumulation after the last interactive change."""
        self._preview_restore_id = None
        if self.rt is None or not self._preview_active:
            return
        self._preview_active = False
        self.rt.set_param(max_accumulation_frames=self.ACCUMULATION_FRAMES)
        self.rt.refresh_scene()

    # ---- renderer setup ----

    def _mouse_wheel_handler(self, event):
        """Handle mouse wheel events for zooming."""
        self._begin_interactive_preview()
        self.zoom_with_wheel(event)

    def init_astro(self):
        astro.init(self.observer)
        # The ephemeris of the initial date is needed before the renderer
        # exists: the camera distance depends on the Moon's apparent size on
        # that date. update_view recomputes it from then on.
        self.moon_ephem = astro.calculate_moon_ephemeris(self.dt_local, self.parallactic_mode)
        self._apparent_radius = self.moon_apparent_radius()
        if self.initial_camera is None:
            self.initial_camera = self.default_camera

    @property
    def default_camera(self) -> Camera:
        """
        Whole-disk view of the currently rendered date (reset with the V key):
        Moon centered and shown at the apparent size it has on that date.
        """
        visible_height = 2 * self.MOON_RADIUS / self.MOON_FILL_FRACTION
        fov = np.degrees(2 * np.arctan(visible_height / (2 * self.CAMERA_DISTANCE)))
        return Camera(
            eye=[0, -self.moon_camera_distance(), 0],
            target=[0, 0, 0],
            up=[0, 0, 1],
            fov=max(1, min(90, fov))
        )

    def moon_apparent_radius(self, distance_km: Optional[float] = None) -> float:
        """
        Angular radius of the Moon in radians as seen by the observer, from the
        topocentric distance of the current ephemeris unless one is given.
        """
        if distance_km is None:
            distance_km = self.moon_ephem.distance
        return float(np.arcsin(self.MOON_RADIUS_KM / distance_km))

    def moon_camera_distance(self, distance_km: Optional[float] = None) -> float:
        """
        Camera distance in scene units that shows the Moon at the apparent size
        it really has, with the FOV left untouched (see MOON_REFERENCE_DISTANCE).

        The rendered angular size is proportional to MOON_RADIUS / distance, so
        the scene distance is scaled by the inverse ratio of the real angular
        radii: 27.3 Moon radii for the closest possible Moon, 32.2 for the most
        distant one, against 30 at MOON_REFERENCE_DISTANCE. The camera stays
        well inside the range where the limb geometry and the float32 ray
        precision are good (see CAMERA_DISTANCE).
        """
        return self.CAMERA_DISTANCE * (self.moon_apparent_radius(self.MOON_REFERENCE_DISTANCE) /
                                       self.moon_apparent_radius(distance_km))

    def _move_camera_to_apparent_size(self):
        """
        Follow the Moon's changing apparent size when the rendered date changes.

        The eye is moved along the view direction by the inverse ratio of the
        Moon's angular radius before and after the change. Nothing else is
        touched: the target, the up vector and above all the FOV stay as they
        are, so the star background renders exactly as before (its rays keep
        their directions) and any zoom, pan or roll the user has set is
        preserved - only the Moon grows and shrinks. A camera set elsewhere
        (startup, a restored view, the R and V resets) already stands at the
        distance of the date it was made for and is picked up here unchanged.
        """
        prev_radius = self._apparent_radius
        self._apparent_radius = self.moon_apparent_radius()

        if self.rt is None or prev_radius is None or prev_radius == self._apparent_radius:
            return

        cam = self.rt.get_camera(self.CAMERA_NAME)
        target = np.array(cam["Target"])
        eye_rel = (np.array(cam["Eye"]) - target) * (prev_radius / self._apparent_radius)
        self.rt.update_camera(self.CAMERA_NAME, eye=(target + eye_rel).tolist())

    @contextmanager
    def _gpu_upload(self, what: str, size_bytes: int, remedy: str):
        """
        Upload a large array to the GPU, failing loudly if it does not fit.

        PlotOptiX reports a failed upload only in its log and otherwise carries
        on (_raise_on_error is False by default), so a map too big for the card
        would leave the Moon rendered without it - the wrong image rather than
        an error. The flag is turned on for the upload and put back afterwards,
        and the resulting exception is given the size and the parameter to
        change. Same reasoning as the encoder_is_open check in
        start_video_export.

        Parameters
        ----------
        what : str
            Name of the map, for the message
        size_bytes : int
            Its size, for the message
        remedy : str
            What the user can change to make it fit
        """
        previous = self.rt._raise_on_error
        self.rt._raise_on_error = True
        try:
            yield
        except (RuntimeError, ValueError) as e:
            raise MapTooLargeError(
                f"Could not upload {what} ({size_bytes / (1024**3):.2f} GB) to GPU memory: {e}\n"
                f"{remedy}\nDetails are in the console output above.") from e
        finally:
            self.rt._raise_on_error = previous

    def init_renderer(self):
        self.rt = TkOptiX(
            width=self.width,
            height=self.height,
            on_launch_finished=self._on_launch_finished
        )

        # Rendering parameters
        self.rt.set_param(min_accumulation_step=1, max_accumulation_frames=self.ACCUMULATION_FRAMES)

        # Single diffuse body with one light: long multi-bounce paths add mostly
        # noise, so cap path length for faster, cleaner frames. Trade-off is
        # slightly darker shadowed crater floors (less bounced light).
        self.rt.set_uint("path_seg_range", 2, 4)

        # Exact terminator shadows at interactive speed (see SCENE_EPSILON comment)
        self.rt.set_float("scene_epsilon", self.SCENE_EPSILON)
        self.rt.set_float("marching_step", self.MARCHING_STEP)
        self.rt.set_float("marching_step_eps", self.MARCHING_STEP_EPS)

        # No ambient light: in space the Moon's night side and shadow interiors
        # receive no atmospheric skylight, only sunlight bounced from nearby
        # sunlit terrain (path_seg_range above). PlotOptiX's default ambient
        # (~0.45 gray) would otherwise wash the whole disk to a flat gray,
        # washing out the night side of a crescent and lifting shadow floors.
        self.rt.set_ambient(0)

        # Tone mapping
        self.rt.set_float("tonemap_exposure", 0.9)
        self.rt.set_float("tonemap_gamma", self.gamma)
        self.rt.add_postproc("Gamma")

        # Background (stars). Loaded locally: uploaded to a GPU texture here and
        # released when this method returns (the host copy is ~760 MB)
        if self.starmap_file is not None:
            star_map = load_starmap(self.starmap_file, self.width * STARMAP_WIDTH_FACTOR)
        else:
            star_map = None
        if star_map is not None:
            self.rt.set_background_mode("TextureEnvironment")
            with self._gpu_upload("the star map", star_map.nbytes,
                                  "Free GPU memory, or raise --downscale / --color-downscale "
                                  "to leave room for it."):
                self.rt.set_background(star_map, gamma=self.gamma, rt_format="UByte4")
        else:
            self.rt.set_background(0)  # Black background

        # Setup material with Moon texture (local for the same reason, ~200 MB).
        # Copy the material so the shared plotoptix module dict stays untouched.
        color_data = load_color_data(self.color_file, self.gamma, self.color_downscale)
        with self._gpu_upload("the color map texture", color_data.nbytes,
                              "Raise --color-downscale, or use a smaller color map."):
            self.rt.set_texture_2d("moon_color", color_data)
        moon_material = m_diffuse.copy()
        moon_material["ColorTextures"] = ["moon_color"]
        self.rt.update_material("diffuse", moon_material)

        # Create Moon sphere with displacement
        self.rt.set_data(self.MOON_OBJECT_NAME, geom="ParticleSetTextured", geom_attr="DisplacedSurface",
                        pos=[0, 0, 0], u=[0, 0, 1], v=[0, -1, 0], r=self.MOON_RADIUS)

        # Apply displacement map (no refresh: the renderer is not started yet)
        with self._gpu_upload("the elevation displacement map", self.elevation.nbytes,
                              "Raise --downscale."):
            self.rt.set_displacement(self.MOON_OBJECT_NAME, self.elevation, refresh=False)

        cam = self.initial_camera
        self.rt.setup_camera(self.CAMERA_NAME,
                             cam_type=cam.type,
                             eye=cam.eye,
                             target=cam.target,
                             up=cam.up,
                             fov=cam.fov,
                             aperture_radius=cam.aperture_radius,
                             aperture_fract=cam.aperture_fract,
                             focal_scale=cam.focal_scale)
        
        # The light itself is hidden: its radius is chosen for correct illumination
        # (shadow softness), not for the Sun's visible size. The visible Sun is the
        # separate flat-shaded disk below.
        self.rt.setup_light(self.LIGHT_NAME, color=self.brightness * self.SUN_BRIGHTNESS_SCALE,
                            radius=self.SUN_RADIUS, in_geometry=False)

        # Visible Sun disk: unlit white sphere; position and radius are set on
        # update_view. Flat material with transparent occlusion (same recipe as
        # the overlays), so the disk stays visible but never shadows the Moon
        # even though it is closer than the light source.
        self.rt.setup_material("flat", self._no_shadow_flat_material())
        self.rt.set_data(self.SUN_DISK_NAME, geom="ParticleSet", mat="flat",
                         pos=[[0.0, self.SUN_DISK_DISTANCE, 0.0]],
                         r=self.SUN_DISK_PARKED_RADIUS, c=self.SUN_DISK_COLOR)


    def calculate_light_pos(self) -> list:
        """
        Calculate light direction for the renderer.
        
        Scene coordinate system:
        - Moon is at origin
        - Camera looks along +Y axis toward the Moon
        - +X is to the RIGHT in the view
        - +Z is UP in the view (toward zenith)
        """
        
        # Calculate bright limb angle in observer's view
        # Position angle: direction from Moon to Sun, measured from celestial North toward East
        # Parallactic angle: how much celestial North is rotated from zenith
        # bright_limb_angle = position_angle - parallactic_angle
        # This gives us the angle from ZENITH (top of view) to the bright limb
        # Positive angles go toward EAST (counterclockwise as seen from behind camera)
        
        # The surface is rotated by (parallactic - PA_axis) around Y.
        # The light direction in celestial coords is PA (from celestial north).
        # To get light direction in view coords (from zenith), subtract parallactic.
        # This puts light in the same reference frame as the rotated surface.
        
        bright_limb_angle = np.radians(self.moon_ephem.bright_limb_angle)
        phase_angle = np.radians(self.moon_ephem.phase_angle)
        light_distance = self.SUN_LIGHT_DISTANCE
        
        # The bright limb angle tells us which edge of the Moon is illuminated
        # The LIGHT source is in the OPPOSITE direction from the dark side
        # 
        # If bright_limb_angle = 0°: bright limb at TOP, Sun is ABOVE Moon
        #    -> Light from +Z direction (above)
        # If bright_limb_angle = 90°: bright limb on LEFT (east), Sun is to the LEFT
        #    -> Light from -X direction (left)
        # If bright_limb_angle = -90°: bright limb on RIGHT (west), Sun is to the RIGHT
        #    -> Light from +X direction (right)  
        # If bright_limb_angle = ±180°: bright limb at BOTTOM, Sun is BELOW
        #    -> Light from -Z direction (below)
        #
        # In our scene, looking along +Y:
        # Light X = -sin(angle) maps: 0° -> 0, 90° -> -1 (left), -90° -> +1 (right)
        # Light Z = cos(angle) maps: 0° -> +1 (up), ±180° -> -1 (down)
        
        # Calculate light direction using proper 3D geometry
        # 
        # The Sun's position relative to the Moon-Earth line can be described as:
        # - phase angle: angle between Sun-Moon and Earth-Moon directions (at Moon vertex)
        #   This is the "elongation" of the Sun from Earth as seen from Moon
        #   phase = 0° means Sun is in same direction as Earth (full moon for us)
        #   phase = 180° means Sun is opposite to Earth (new moon for us)
        # - bright_limb_angle: direction of Sun in the observer's view plane (XZ)
        #   measured from +Z (up) toward +X (right) - but note the sign conventions
        #
        # In our scene coordinate system:
        # - Camera at -Y looking toward Moon at origin
        # - The Sun is at angle 'phase' from the -Y axis (camera direction)
        # - The azimuthal direction of Sun in the XZ plane is given by bright_limb_angle
        #
        # Using spherical coordinates with -Y as the pole:
        # - theta = phase (angle from -Y axis, 0° = behind camera, 180° = behind Moon)
        # - phi = bright_limb_angle (angle in XZ plane, 0° = +Z direction)
        #
        # Converting to Cartesian:
        # Y = -cos(theta) = -cos(phase)  [negative because -Y is our reference]
        # X = sin(theta) * sin(phi) = sin(phase) * sin(bright_limb_angle)
        # Z = sin(theta) * cos(phi) = sin(phase) * cos(bright_limb_angle)
        #
        # But bright_limb_angle convention: 0° = up (+Z), 90° = left (-X), -90° = right (+X)
        # So: X = -sin(bright_limb_angle), Z = cos(bright_limb_angle)
        
        light_x = -np.sin(bright_limb_angle) * np.sin(phase_angle) * light_distance
        light_z = np.cos(bright_limb_angle) * np.sin(phase_angle) * light_distance
        light_y = -np.cos(phase_angle) * light_distance

        return [light_x, light_y, light_z]


    def calculate_sun_disk(self) -> tuple[list, float]:
        """
        Calculate position and radius of the visible Sun disk.

        The disk is decoupled from the light source: the light keeps the Sun's real
        angular size as seen from the Moon (correct illumination and shadow softness),
        while this disk reproduces what the observer would see. The rendered Moon is
        magnified (it fills the window although the real Moon subtends only ~0.5
        degree), so the disk's apparent size and its apparent separation from
        the Moon are scaled by the same magnification, as in a telescope view. This
        keeps solar eclipse views (Sun size, coverage, total vs annular character)
        consistent with reality.

        The magnification is the same on every date: the camera stands where the
        Moon renders at its true apparent size (moon_camera_distance), so the Sun
        disk keeps a constant screen size while the Moon grows and shrinks against
        it, exactly as in a fixed eyepiece. Only the Sun's own apparent size still
        varies with the Sun distance.
        """
        camera_distance = self.moon_camera_distance()

        # Magnification of the rendered Moon relative to its real apparent size
        magnification = np.arcsin(self.MOON_RADIUS / camera_distance) / \
            self.moon_apparent_radius()

        sun_angular_radius = magnification * np.arcsin(self.SUN_RADIUS_KM / self.moon_ephem.sun_distance)

        # Apparent Moon-Sun separation, seen from the default camera position
        separation = magnification * np.radians(self.moon_ephem.elongation)

        # Beyond 90 degrees the disk cannot be in any view together with the Moon and
        # would start facing the Moon's night side, brightening it with bounced light
        # and producing speckle noise. Park it behind the camera with negligible size.
        in_view = separation <= np.pi / 2
        if not in_view:
            separation = np.radians(175.0)

        # Same view-plane direction convention as in calculate_light_pos
        bright_limb_angle = np.radians(self.moon_ephem.bright_limb_angle)
        sin_sep = np.sin(separation)
        direction = np.array([
            -np.sin(bright_limb_angle) * sin_sep,
            np.cos(separation),
            np.cos(bright_limb_angle) * sin_sep,
        ])
        center = np.array([0.0, -camera_distance, 0.0]) + self.SUN_DISK_DISTANCE * direction
        radius = (self.SUN_DISK_DISTANCE * np.tan(sun_angular_radius) if in_view
                  else self.SUN_DISK_PARKED_RADIUS)
        return center.tolist(), float(radius)


    def update_overlays(self):
        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()
        if self.spot_labels_visible:
            self.update_spot_labels_orientation()
        if self.pins_visible:
            self.update_pins_orientation()
        if self.sub_points_visible:
            self.update_sub_points()


    def shifted_time(self, minutes: int) -> datetime:
        """
        The observation time advanced by that many minutes of real time.

        The arithmetic is done on the instant, not on the wall clock: adding to
        a zone-aware datetime moves the clock reading, which at a daylight
        saving change is not the same thing. Stepping through the autumn change
        that way would skip an hour of real time, and through the spring one it
        would step backwards - the Moon must move by the minutes asked for.
        update_view puts the result back on the observer's clock.
        """
        return self.dt_local.astimezone(timezone.utc) + timedelta(minutes=minutes)

    def in_observer_clock(self, instant: datetime) -> datetime:
        """
        The given instant on the observer's clock.

        The session's timezone carries its own rules, so this is simply a
        conversion: the offset that applied on that date - daylight saving,
        the historical changes before it, and the rules of the observer's own
        country when it is not this computer's - comes out of the zone.
        """
        return instant.astimezone(self.dt_local.tzinfo)

    def from_observer_clock(self, wall_clock: datetime) -> datetime:
        """
        A wall-clock reading (a naive datetime) as an instant on the observer's
        clock: the counterpart of in_observer_clock, so the hour typed into the
        date/time dialog is the hour shown afterwards on any date.
        """
        return wall_clock.replace(tzinfo=self.dt_local.tzinfo)

    def update_view(self, dt_local: Optional[datetime] = None):

        # Compute the ephemeris before committing the new time. Dates outside
        # the range of the bundled kernels are rejected here (see astro), and a
        # rejected date must leave the renderer on the last valid time: with the
        # time already committed, the status bar would keep showing the old time
        # while dt_local held the rejected one, and every further step would
        # build on it - the view would appear frozen.
        # Re-expressed on the observer's clock, so a step or a jump that crosses
        # a daylight saving change lands on the wall-clock time really in force
        # there; the instant itself is untouched
        target_dt = self.in_observer_clock(self.dt_local if dt_local is None else dt_local)
        moon_ephem = astro.calculate_moon_ephemeris(target_dt, self.parallactic_mode)

        self.dt_local = target_dt
        self.moon_ephem = moon_ephem
        self.moon_rotation = self.moon_ephem.rotation_matrix
        self.moon_rotation_inv = self.moon_rotation.T
        self.light_pos = self.calculate_light_pos()

        u_new = self.moon_rotation[:, 2]        # Z axis of the rotated surface
        v_new = -self.moon_rotation[:, 1]       # Invert Y axis to match our convention of v pointing down in the texture

        sun_disk_pos, sun_disk_radius = self.calculate_sun_disk()

        # Hold the render padlock across all scene updates: the render thread
        # cannot launch frames on a half-updated scene, and accumulation
        # restarts once instead of once per update call.
        with self.rt._padlock:
            self._move_camera_to_apparent_size()
            self.rt.update_data(self.MOON_OBJECT_NAME, u=u_new, v=v_new)
            self.rt.update_data(self.SUN_DISK_NAME, pos=[sun_disk_pos], r=sun_disk_radius)
            # Light radius follows the true solar angular size seen from the Moon.
            # Light color is radiance, so illumination scales with angular size
            # squared, reproducing the real annual 1/d^2 brightness variation.
            sun_light_radius = float(self.SUN_LIGHT_DISTANCE * self.SUN_RADIUS_KM / self.moon_ephem.sun_distance)
            self.rt.update_light(self.LIGHT_NAME, pos=self.light_pos, radius=sun_light_radius)
            self.update_overlays()

        # Keys reach the main window while the date/time window has focus, so
        # the clock can move under it; its fields follow rather than going stale
        self.sync_datetime_dialog()

        # Since 0.19.1 updates applied while the accumulation cycle is idle do
        # not restart rendering on their own; force a new cycle so the change
        # is displayed immediately
        if self.rt._is_started:
            self.rt.refresh_scene()

    # ---- lifecycle ----

    def start(self):
        """Start the renderer."""
        if self.rt is not None:
            self.rt.start()

    def close(self):
        """Close the renderer."""
        if self.rt is not None:
            self.rt.close()
            self.rt = None

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

# PlotOptiX binds the key handler with bind_all, so it hears keys typed into the
# dialogs too. The date/time window only wants what its spinboxes are made of -
# digits, and the keys that move about or edit what is already in them.
DATETIME_DIALOG_KEYSYMS = frozenset({
    'BackSpace', 'Delete', 'Left', 'Right', 'Up', 'Down',
    'Home', 'End', 'Tab', 'ISO_Left_Tab', 'Return', 'KP_Enter',
})


def datetime_dialog_takes_key(event) -> bool:
    """
    Whether a key typed while the date/time window has focus belongs to it.

    Anything it does not take is meant for the main window and falls through to
    the usual handler, so the Moon can be driven with the window still open and
    the spinboxes, which refuse anything but digits anyway, stay out of the way.
    """
    return event.char.isdigit() or event.keysym in DATETIME_DIALOG_KEYSYMS


def run_renderer(dt_local: datetime,
                 observer: Observer,
                 elevation_file: str,
                 color_file: str,
                 starmap_file: Optional[str],
                 features_file: str,
                 downscale: int,
                 brightness: int,
                 initial_camera: Optional[Camera],
                 time_step_minutes: int = 15,
                 init_view_orientation: str = VIEW_ORIENTATION_NSWE,
                 gamma: float = 2.2,
                 parallactic_mode: bool = False,
                 color_downscale: int = 1) -> TkOptiX:
    """
    Quick function to render the Moon for a specific time and location.

    Parameters
    ----------
    dt_local : datetime
        Local time
    observer : Observer
        Observer latitude, longitude, and elevation
    elevation_file, color_file, starmap_file, features_file : str
        Paths to data files
    downscale : int
        Elevation downscale factor
    brightness : int
        Brightness
    initial_camera : Camera, optional
        Initial camera to restore a specific view
    time_step_minutes : int
        Time step in minutes for Q/W keys (default 15)
    init_view_orientation : str
        Initial view orientation mode.
    gamma : float
        Gamma correction value (default 2.2)
    parallactic_mode : bool
        Whether to use parallactic projection mode (default False)
    color_downscale : int
        Color map downscale factor (default 1)

    Returns
    -------
    TkOptiX
        The renderer instance
    """
    print()
    print("Used PlotOptiX version:", plotoptix.__version__)
    print("Renderer started with parameters:")
    print(f"  Observer Location: Lat {observer.lat}°, Lon {observer.lon}°, Elevation {observer.elevation_m} m")
    print(f"  Local Time: {dt_local}")
    print(f"  Timezone: {timezone_name(dt_local)}")
    print(f"  Elevation File: {elevation_file}")
    print(f"  Color File: {color_file}")
    print(f"  Starmap File: {starmap_file}")
    print(f"  Brightness: {brightness}")
    print(f"  Gamma: {gamma}")
    print(f"  Downscale Factor: {downscale}")
    print(f"  Color Downscale Factor: {color_downscale}")
    print(f"  Time Step (minutes): {time_step_minutes}")
    print(f"  Initial View Orientation: {init_view_orientation}")
    print(f"  Parallactic Mode: {'ON' if parallactic_mode else 'OFF'}")
    if initial_camera is not None:
        print("  Location, time and view set from --init-view parameter value")
    print()

    moon_renderer = MoonRenderer(
        elevation_file=elevation_file,
        color_file=color_file,
        starmap_file=starmap_file,
        downscale=downscale,
        color_downscale=color_downscale,
        features_file=features_file,
        brightness=brightness,
        time_step_minutes=time_step_minutes,
        init_view_orientation=init_view_orientation,
        observer=observer,
        gamma=gamma,
        parallactic_mode=parallactic_mode,
        dt_local=dt_local,
        initial_camera=initial_camera
    )

    moon_renderer.init_astro()
    moon_renderer.init_renderer()

    moon_renderer.update_view()

    original_key_handler = moon_renderer.rt._gui_key_pressed

    # Held-repeat keys that animate the scene (time stepping, view navigation
    # and roll, brightness and gamma sweeps): use single-frame preview cycles so
    # rapid autorepeat stays responsive (see ACCUMULATION_FRAMES). One-shot keys
    # (orientation, toggles, resets, pins, set-time-now) are deliberately left
    # out: a single scene update finishes its accumulation cycle undisturbed and
    # renders straight to the converged image, like the date/time dialog, with
    # no noisy intermediate frame.
    preview_keysyms = {'Left', 'Right', 'Up', 'Down'}
    preview_letters = set('qwazedhj')

    # Keys that reach update_view: time stepping (Q/W), the resets that restore
    # the initial time (R), the dialogs that jump to a time (T, the planners K
    # and X, and the rise and set chart U, which goes to whatever moment in it
    # is clicked), the parallactic toggle (F4) and the set-time-now keys
    # (F9/F10). A running video export drives update_view from the raytracing
    # thread, so these are ignored while it lasts - see the export guard in
    # custom_key_handler.
    update_view_keysyms = {'F4', 'F9', 'F10'}
    update_view_letters = set('qwrtkxu')

    def custom_key_handler(event):
        # The search dialog wants every key, being a place to type a name; the
        # date/time window takes only its own (see datetime_dialog_takes_key)
        if moon_renderer.search_dialog_open:
            return
        if moon_renderer.datetime_dialog_focused and datetime_dialog_takes_key(event):
            return
        # The video export owns the clock until it finishes: letting a key
        # change the time here would run update_view on the Tk main thread while
        # the export runs it on the raytracing thread, and the two would race
        # over dt_local, the ephemeris and the scene
        if moon_renderer._video_export is not None and (
                event.keysym in update_view_keysyms
                or event.keysym.lower() in update_view_letters):
            return
        if event.keysym in preview_keysyms or event.keysym.lower() in preview_letters:
            moon_renderer._begin_interactive_preview()
        if event.keysym.lower() == 'g':
            moon_renderer.toggle_grid()
        elif event.keysym.lower() == 'l':
            moon_renderer.toggle_standard_labels()
        elif event.keysym.lower() == 's':
            moon_renderer.toggle_spot_labels()
        elif event.keysym == 'F4':
            moon_renderer.toggle_parallactic_mode()
        elif event.keysym == 'F5':
            moon_renderer.set_view_orientation(VIEW_ORIENTATION_NSWE)
            original_key_handler(event)
        elif event.keysym == 'F6':
            moon_renderer.set_view_orientation(VIEW_ORIENTATION_NSEW)
            original_key_handler(event)
        elif event.keysym == 'F7':
            moon_renderer.set_view_orientation(VIEW_ORIENTATION_SNEW)
            original_key_handler(event)
        elif event.keysym == 'F8':
            moon_renderer.set_view_orientation(VIEW_ORIENTATION_SNWE)
            original_key_handler(event)
        elif event.keysym.lower() == 'r':
            moon_renderer.reset_camera_position()
        elif event.keysym.lower() == 'c':
            moon_renderer.toggle_compass()
        elif event.keysym == 'F3':
            moon_renderer.fov_overlay_dialog()
        elif event.keysym.lower() == 'b':
            moon_renderer.toggle_fov_overlay()
        elif event.keysym == 'F11':
            moon_renderer.export_video_dialog()
        elif event.keysym == 'F12':
            moon_renderer.save_image_dialog()
        elif event.keysym.lower() == 'f':
            moon_renderer.search_feature_dialog()
        elif event.keysym.lower() == 'k':
            moon_renderer.observation_planner_dialog(moon_renderer._status_feature)
        elif event.keysym.lower() == 'x':
            moon_renderer.clair_obscur_dialog()
        elif event.keysym.lower() == 'u':
            moon_renderer.visibility_chart_dialog()
        elif event.keysym.lower() == 'i':
            moon_renderer.open_status_feature_usgs_page()
        elif event.keysym.lower() == 'o':
            moon_renderer.open_status_feature_www_page()
        elif event.keysym.lower() == 'h':
            moon_renderer.rotate_around_view_direction('ccw')
        elif event.keysym.lower() == 'j':
            moon_renderer.rotate_around_view_direction('cw')
        elif event.keysym in ('Left', 'Right', 'Up', 'Down'):
            if event.state & 0x4:  # Ctrl key pressed
                moon_renderer.rotate_around_moon_axis(event.keysym)
            else:
                moon_renderer.navigate_view(event.keysym)
        elif event.keysym.lower() == 'v':
            moon_renderer.reset_to_default_view()
        elif event.keysym.lower() == 'a':
            moon_renderer.change_brightness(10)
        elif event.keysym.lower() == 'z':
            moon_renderer.change_brightness(-10)
        elif event.keysym.lower() == 'e':
            moon_renderer.change_gamma(0.1)
        elif event.keysym.lower() == 'd':
            moon_renderer.change_gamma(-0.1)
        elif event.keysym.lower() == 'm':
            step = 60 if event.state & 0x1 else 1
            moon_renderer.change_time_step(step)
        elif event.keysym.lower() == 'n':
            step = 60 if event.state & 0x1 else 1
            moon_renderer.change_time_step(-step)
        elif event.keysym == 'F2':
            moon_renderer.toggle_info_panel()
        elif event.keysym.lower() == 'p':
            moon_renderer.toggle_pins()
        elif event.keysym.lower() == 'y':
            moon_renderer.toggle_sub_points()
        elif event.keysym == 'space':
            moon_renderer.center_view_on_cursor(event)
        elif event.keysym.lower() == 'q':
            moon_renderer.change_time(-moon_renderer.time_step_minutes)
        elif event.keysym.lower() == 'w':
            moon_renderer.change_time(moon_renderer.time_step_minutes)
        elif event.keysym.lower() == 't':
            moon_renderer.open_datetime_dialog()
        elif event.keysym == 'F1':
            moon_renderer.show_help_dialog()
        elif event.keysym == 'F9':
            moon_renderer.set_time_to_now()
        elif event.keysym == 'F10':
            moon_renderer.set_time_to_now_and_auto_advance()
        elif event.keysym in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
            moon_renderer.toggle_pin_at_cursor(event, int(event.keysym))
        else:
            original_key_handler(event)

    moon_renderer.rt._gui_key_pressed = custom_key_handler

    # Override mouse motion handler to show selenographic coordinates
    original_motion_handler = moon_renderer.rt._gui_motion

    def custom_motion_handler(event):
        original_motion_handler(event)
        if not (moon_renderer.rt._any_mouse or moon_renderer.rt._any_key):
            x, y = moon_renderer.rt._get_image_xy(event.x, event.y)
            hx, hy, hz, hd = moon_renderer.rt._get_hit_at(x, y)
            lat = None
            lon = None
            feature = None
            if hd > 0:
                lat, lon = moon_renderer.hit_to_selenographic(hx, hy, hz)
                if lat is not None and lon is not None:
                    feature = moon_renderer.find_feature_for_status_bar(lat, lon)
            moon_renderer.rt._status_action_text.set('')
            moon_renderer._update_info_coords(lat, lon)
            moon_renderer._update_status_feature(feature)

    moon_renderer.rt._gui_motion = custom_motion_handler

    # Override mouse handlers for distance measurement (Ctrl+drag)
    original_pressed_left = moon_renderer.rt._gui_pressed_left
    original_released_left = moon_renderer.rt._gui_released_left
    original_motion_pressed = moon_renderer.rt._gui_motion_pressed

    def custom_pressed_left(event):
        if event.state & 0x4:
            moon_renderer.start_measurement(event)
            return
        original_pressed_left(event)

    def custom_released_left(event):
        if moon_renderer.measuring:
            moon_renderer.finish_measurement(event)
            return
        original_released_left(event)

    def custom_motion_pressed(event):
        if moon_renderer.measuring:
            moon_renderer.update_leading_line(event)
            return
        original_motion_pressed(event)

    moon_renderer.rt._gui_pressed_left = custom_pressed_left
    moon_renderer.rt._gui_released_left = custom_released_left
    moon_renderer.rt._gui_motion_pressed = custom_motion_pressed

    # Override camera pan/tilt (right mouse drag, no modifier keys): the built-in
    # handler rotates by fixed angles per pixel, which is far too sensitive with a
    # narrow FOV. pan_tilt_view scales the rotation to the current FOV instead.
    # All other gestures are passed to the original handler.
    original_apply_scene_edits = moon_renderer.rt._gui_apply_scene_edits

    def custom_apply_scene_edits(*args):
        rt = moon_renderer.rt
        # Mouse-driven view manipulation benefits from immediate preview too
        if rt._any_mouse:
            moon_renderer._begin_interactive_preview()
        if rt._selection_handle == -1 and rt._right_mouse and not rt._any_key:
            dx = rt._mouse_to_x - rt._mouse_from_x
            dy = rt._mouse_to_y - rt._mouse_from_y
            if dx != 0 or dy != 0:
                rt._status_action_text.set("camera pan/tilt")
                moon_renderer.pan_tilt_view(dx, dy)
            rt._mouse_from_x = rt._mouse_to_x
            rt._mouse_from_y = rt._mouse_to_y
            return
        original_apply_scene_edits(*args)

    moon_renderer.rt._gui_apply_scene_edits = custom_apply_scene_edits

    moon_renderer.start()
    return moon_renderer.rt


def run_renderer_process(*args, **kwargs):
    """
    run_renderer as the target of a spawned process (see main_gui_launcher).

    A map that does not fit - in system RAM while it is prepared, or in GPU
    memory when it is uploaded - ends the process with its own message and
    MAP_TOO_LARGE_EXIT_CODE rather than a traceback, which the launcher turns
    back into something the user can act on.
    """
    try:
        run_renderer(*args, **kwargs)
    except MapTooLargeError as e:
        print(f"\n{e}")
        sys.exit(MAP_TOO_LARGE_EXIT_CODE)
