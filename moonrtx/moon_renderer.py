"""
MoonRenderer: core renderer class (composing mixins) and run_renderer entry point.
"""

import math
import sys
import threading
import numpy as np
from contextlib import contextmanager
from typing import Optional
from datetime import datetime, timedelta, timezone

import plotoptix
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse, m_flat

from moonrtx import astro
from moonrtx.shared_types import (Camera, MAP_TOO_LARGE_EXIT_CODE,
                                  MapTooLargeError, Observer)
from moonrtx.data_loader import (load_moon_features, load_elevation_data, load_color_data,
                                 load_starmap, MOON_REFERENCE_RADIUS_M, SRGB_GAMMA)
from moonrtx.view_orientation import VIEW_ORIENTATION_NSWE, VIEW_ORIENTATION_NSEW, VIEW_ORIENTATION_SNEW, VIEW_ORIENTATION_SNWE
from moonrtx.display import make_dpi_aware, screen_size, starmap_target_width

# Mixins – each adds a focused group of methods
from moonrtx.renderer_status import StatusMixin, timezone_name
from moonrtx.renderer_fullscreen import FullScreenMixin
from moonrtx.renderer_dialogs import DialogsMixin
from moonrtx.renderer_planning import PlanningMixin
from moonrtx.renderer_labels import LabelsMixin
from moonrtx.renderer_pins import PinsMixin
from moonrtx.renderer_navigation import NavigationMixin
from moonrtx.renderer_video import VideoMixin
from moonrtx.renderer_fov import FovMixin
from moonrtx.renderer_subpoints import SubPointsMixin
from moonrtx.renderer_overlay import CanvasOverlayMixin
from moonrtx.renderer_compass import CompassMixin
from moonrtx.renderer_locator import LocatorMixin
from moonrtx.renderer_catalogue import CatalogueMixin
from moonrtx.renderer_profile import ProfileMixin
from moonrtx.renderer_settings import SettingsMixin
from moonrtx.renderer_scalebar import ScaleBarMixin


class MoonRenderer(StatusMixin, FullScreenMixin, DialogsMixin, PlanningMixin,
                   LabelsMixin, PinsMixin, NavigationMixin, VideoMixin,
                   FovMixin, SubPointsMixin, CanvasOverlayMixin,
                   CompassMixin, LocatorMixin, CatalogueMixin, ProfileMixin,
                   SettingsMixin, ScaleBarMixin):
    """
    Renders the Moon surface as seen from a specific location on Earth
    at a specific time, with accurate solar illumination.
    """

    # Scene geometry
    MOON_RADIUS = 10.0          # Radius of Moon sphere in scene units
    MOON_RADIUS_KM = MOON_REFERENCE_RADIUS_M / 1000.0
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

    # The Earth's shadow, which is a lunar eclipse. It is not left to the ray
    # tracer to find by sending shadow rays at random points of the Sun past
    # an Earth: near the umbra only a sliver of the Sun is in sight, a ray finds
    # it lit only now and then, and the penumbra came out as heavy grain -
    # worst of all in the one-frame preview of a held Q or W. It is worked out
    # instead, ring by ring - how much of the Sun the Earth hides from each
    # point of the Moon - and laid on a flat square standing across the line
    # to the Sun as its see-through texture. While any of the Moon is in the
    # shadow the Sun is made a point (see SUN_POINT_RADIUS), so every shadow
    # ray goes through the square at the one place that answers for it, and
    # the shadow is exact in a single frame. The square lies behind the camera,
    # never seen, and away from an eclipse no shadow ray comes near it.
    #
    # Seen from the Moon the umbra is some 2% wider than the solid Earth makes
    # it, the atmosphere adding to the body; the 1% of Danjon's rule is the
    # one the eclipse tables use.
    EARTH_SHADOW_NAME = "earth_shadow"
    EARTH_RADIUS_KM = 6378.137
    EARTH_SHADOW_ENLARGEMENT = 1.01
    # Texels across the square, and how far past the penumbra's edge it
    # reaches, so the texture is fully clear at its rim
    EARTH_SHADOW_TEXELS = 512
    EARTH_SHADOW_REACH = 1.02
    # The square's side away from an eclipse: put out of the way rather than
    # kept where the shadow is, so a step of the clock costs nothing for it.
    # Not merely left where an eclipse ended - at the next full Moon the light
    # comes round behind much the same place, and would cast that shadow again
    EARTH_SHADOW_PARKED = 1.0e-6
    # The light reaching the umbra, as a share of full sunlight in red, green
    # and blue: what the atmosphere bends round the Earth, reddened by it.
    # Chosen to be seen rather than measured - the totally eclipsed Moon is
    # thousands of times fainter than the full one, far past what A can make
    # up for - and coppery, as a middling eclipse is
    EARTH_UMBRA_LIGHT = (0.06, 0.02, 0.008)
    # The light's radius while the Moon is in the Earth's shadow: a point, as
    # far as a shadow is concerned - a second of arc seen from the Moon - its
    # radiance raised by the square of the shrinking, so it lights the surface
    # exactly as brightly. The terrain's own shadows harden with it, which at
    # the full Moon an eclipse needs is out of sight.
    SUN_POINT_RADIUS = 0.1

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
                 downscale: int = 2,
                 color_downscale: int = 1,
                 time_step_minutes: int = 15,
                 init_view_orientation: str = VIEW_ORIENTATION_NSWE,
                 gamma: float = 2.2,
                 parallactic_mode: bool = False,
                 fullscreen: bool = False):
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
            Initial camera for resets with Home key (if None, a default camera will be calculated from ephemeris)
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
        fullscreen : bool
            Open with the window frame and the status bar already gone. It is a
            starting state and nothing more: from then on the window answers to
            F11 and Escape, and this is not consulted again.
        parallactic_mode : bool
            Whether to use parallactic projection mode (default False)
        """
        self.downscale = downscale
        self.color_downscale = color_downscale
        self.gamma = gamma
        self.time_step_minutes = time_step_minutes
        self.parallactic_mode = parallactic_mode
        # Acted on once the window exists, in StatusMixin._on_launch_finished
        self.initial_fullscreen = fullscreen
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
        self.initial_view_orientation = init_view_orientation  # For reset with Home/End keys

        self.dt_local = dt_local

        # Initial time for reset with Home key
        self.initial_dt_local = self.dt_local

        # Initial camera for reset with Home key. When none is given it is the
        # whole-disk view of the initial date, which needs the ephemeris and is
        # therefore resolved in init_astro.
        self.initial_camera = initial_camera

        # Moon angular radius the current camera distance was set for; kept in
        # sync by update_view (see _move_camera_to_apparent_size)
        self._apparent_radius = None

        # Flag to track if window has been maximized
        self._window_maximized = False

        # What was along the bottom of the window before full screen took it
        # away (see renderer_status._hide_status_row)
        self._windowed_status = []

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

        # Count of dialogs currently open that want every key (see
        # DialogsMixin._dialog_window) - a count rather than a flag because a
        # planning dialog can open another on top of it (the observation
        # planner's graph) and closing one must not clear this while the
        # other is still up.
        self.search_dialog_open = 0

        # Datetime dialog tracking
        self.datetime_dialog = None
        self._datetime_dialog_show = None  # set while the date/time window is open
        self.datetime_dialog_focused = False

        # Pins settings
        self.pins_visible = True  # Pins visible by default
        self.pins = {}  # dict mapping digit (1-9) to (lat, lon, body-frame graph vertices)
        # The place the Find window went to, and its mark's body-frame vertices
        # (see PinsMixin.mark_place)
        self._place_mark = None
        self._place_mark_pos = None

        # Distance measurement settings
        self.measuring = False
        self.measure_start_canvas = None
        self.measure_start_coords = None
        self.leading_line_id = None
        self.measured_distance = None
        self.measured_angle = None          # what the line spans on the sky, radians
        self.measured_height_diff = None
        # Whether the measurement under way was begun with the right Ctrl, which
        # draws its profile as well (see ProfileMixin), and whether that key is
        # down now: a mouse event says only that some Ctrl is held, not which
        self.measure_with_profile = False
        self._measure_serial = 0            # names each measurement's line (see start_measurement)
        self._right_ctrl_down = False

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

        # The window with nothing on it but the Moon
        # (see renderer_fullscreen.FullScreenMixin)
        self._init_full_screen()

        # Time-lapse video export state (see renderer_video.VideoMixin)
        self._init_video_export()

        # Field-of-view overlay state (see renderer_fov.FovMixin)
        self._init_fov_overlay()

        # View-orientation globe state (see renderer_compass.CompassMixin)
        self._init_compass_overlay()
        self._init_locator()

        # How far a stretch of the picture is on the ground (see renderer_scalebar)
        self._init_scale_bar()

        # Drawing all three into a picture, for F12 and for video frames
        # (see renderer_overlay.CanvasOverlayMixin.overlay_image)
        self._init_overlay_raster()

        # Size of the lettering on the surface (see renderer_labels.LabelsMixin)
        self._init_label_scale()

        # Names of everything else in view (see renderer_catalogue.CatalogueMixin)
        self._init_catalogue()

        # The ground along a measurement (see renderer_profile.ProfileMixin)
        self._init_profile()

        # The windows' choices the last run left, the field-of-view setup among
        # them - so after that setup's defaults (see renderer_settings)
        self._load_settings()

        # Auto-advance (real-time playback) settings
        self._auto_advance_var = None
        self._auto_advance_id = None
        self._auto_advance_elapsed = 0
        self._auto_advance_interval = 1000  # tick interval in ms
        self._auto_advance_target_ms = time_step_minutes * 60 * 1000

        # Info panel variables (bottom-left overlay)
        self._info_frame = None
        self.show_info_panel = True

        # The status panel (F3): what the status bar says, drawn on the
        # canvas so that it survives full screen, F12 and the video export.
        # Off to begin with - see StatusMixin.toggle_status_panel
        self.show_status_panel = False
        self._status_panel_frame = None
        self._status_panel_datetime_var = None
        self._status_panel_step_var = None
        self._status_panel_distance_var = None
        self._status_panel_height_var = None
        self._status_panel_lat_var = None
        self._status_panel_lon_var = None
        self._status_panel_sun_var = None
        self._status_panel_feature_var = None
        self._info_az_var = None
        self._info_alt_var = None
        self._info_ra_var = None
        self._info_dec_var = None
        self._info_phase_var = None
        self._info_phase_name_var = None
        self._info_age_var = None
        self._info_elongation_var = None
        self._info_distance_var = None
        self._info_diameter_var = None
        self._info_illum_var = None
        self._info_geo_libr_l_var = None
        self._info_geo_libr_b_var = None
        self._info_topo_libr_l_var = None
        self._info_topo_libr_b_var = None
        self._info_colong_var = None

    # ---- brightness / time-step / auto-advance ----

    def change_brightness(self, delta: int):
        if delta == 0:
            return
        new_brightness = max(0, min(500, self.brightness + delta))
        if new_brightness == self.brightness:
            return
        self.brightness = new_brightness
        self.rt.update_light(self.LIGHT_NAME, color=self._sun_light_colour())

    def _sun_light_colour(self) -> float:
        """
        The light's radiance: the brightness setting, raised by the square of
        however much the light has been shrunk towards a point for an eclipse
        (see SUN_POINT_RADIUS), so the surface is lit the same either way.
        """
        return self.brightness * self.SUN_BRIGHTNESS_SCALE * getattr(self, "_sun_light_boost", 1.0)
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
        # The eclipse shadow the held key left as it was (see _place_earth_shadow)
        with self.rt._padlock:
            self._redraw_earth_shadow()
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
        Whole-disk view of the currently rendered date (reset with the End key):
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

    @property
    def reference_radius(self) -> float:
        """
        The scene radius of the 1737.4 km reference surface - the Moon's mean
        limb as it is drawn.

        Not MOON_RADIUS: the surface is scaled so its highest peak reaches that
        (see data_loader.load_elevation_data), which puts the reference radius
        elevation_radius_scale - some 0.6% - inside it. Anything that sets the
        drawn Moon against a true angle or a true distance has to measure it
        here, or it comes out that much off.
        """
        return self.MOON_RADIUS / getattr(self, "elevation_radius_scale", 1.0)

    def moon_radius_in_pixels(self, eye_distance: float = None,
                              fov_deg: float = None) -> Optional[float]:
        """
        Radius of the rendered Moon in pixels - its mean limb, the reference
        surface - for the camera now or for one given as a distance and a
        vertical field.

        The disk subtends arcsin(radius / distance) at the eye, and the window
        covers half the field either side of its middle, so the two tangents
        give the fraction of the half-height the disk reaches across.
        """
        if self.rt is None:
            return None
        if fov_deg is None:
            fov_deg = self.rt._optix.get_camera_fov(0)
        if eye_distance is None:
            eye_distance = float(np.linalg.norm(
                self.rt.get_camera(self.CAMERA_NAME)["Eye"]))
        if fov_deg <= 0.0 or eye_distance <= self.MOON_RADIUS or self.rt._height <= 0:
            return None

        return (self.rt._height / 2) * np.tan(np.arcsin(self.reference_radius / eye_distance)) \
            / np.tan(np.radians(fov_deg) / 2)

    def surface_magnification(self) -> float:
        """
        How much larger the surface is drawn than in the default view of the
        moment - 1 at that view, 2 when a crater is drawn twice the size.

        Both the wheel, which changes the field, and Shift with the right button,
        which moves the eye, are in it: the disk radius in pixels answers to both.
        The default it is measured against is the one of the date, so the
        annual swing in apparent size does not read as magnification.
        """
        default = self.default_camera
        now = self.moon_radius_in_pixels()
        at_default = self.moon_radius_in_pixels(
            eye_distance=float(np.linalg.norm(default.eye)), fov_deg=default.fov)
        if not now or not at_default:
            return 1.0
        return float(now / at_default)

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

        # Direct sunlight only, no light bounced from one part of the surface to another.
        self.rt.set_uint("path_seg_range", 1, 1)

        # Exact terminator shadows at interactive speed (see SCENE_EPSILON comment)
        self.rt.set_float("scene_epsilon", self.SCENE_EPSILON)
        self.rt.set_float("marching_step", self.MARCHING_STEP)
        self.rt.set_float("marching_step_eps", self.MARCHING_STEP_EPS)

        # No ambient light: in space the Moon's night side and shadow interiors
        # receive no atmospheric skylight. PlotOptiX's default ambient
        # (~0.45 gray) would otherwise wash the whole disk to a flat gray,
        # washing out the night side of a crescent and lifting shadow floors.
        self.rt.set_ambient(0)

        # Tone mapping. This is the only place the viewer's gamma acts, which is
        # what lets E/D reach exactly the picture starting at that value gives:
        # the textures below are decoded with SRGB_GAMMA, a property of the files.
        self.rt.set_float("tonemap_exposure", 0.9)
        self.rt.set_float("tonemap_gamma", self.gamma)
        self.rt.add_postproc("Gamma")

        # Background (stars). Loaded locally: uploaded to a GPU texture here and
        # released when this method returns (the host copy is ~760 MB)
        if self.starmap_file is not None:
            star_map = load_starmap(self.starmap_file, starmap_target_width())
        else:
            star_map = None
        if star_map is not None:
            self.rt.set_background_mode("TextureEnvironment")
            with self._gpu_upload("the star map", star_map.nbytes,
                                  "Free GPU memory, or raise --downscale / --color-downscale "
                                  "to leave room for it."):
                self.rt.set_background(star_map, gamma=SRGB_GAMMA, rt_format="UByte4")
        else:
            self.rt.set_background(0)  # Black background

        # Setup material with Moon texture (local for the same reason, ~200 MB).
        # Copy the material so the shared plotoptix module dict stays untouched.
        color_data = load_color_data(self.color_file, self.color_downscale)
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

        # The square the Earth's shadow is laid on (see EARTH_SHADOW_NAME). A
        # shadow ray crossing it is passed on in the colour of its texture
        # there, which is clear until update_view first draws the shadow, and
        # it is placed and sized by update_view as well
        self._earth_shadow_shape = None
        self._earth_shadow_parked = True
        self._earth_shadow = None
        self.rt.set_texture_2d(self.EARTH_SHADOW_NAME, np.ones((2, 2, 4), dtype=np.float32) * [1, 1, 1, 0])
        shadow_material = m_flat.copy()
        shadow_material["OcclusionProgram"] = (
            "chit7_occlusion_transp.ptx::__closesthit__occlusion_transparency")
        shadow_material["ColorTextures"] = [self.EARTH_SHADOW_NAME]
        self.rt.setup_material(self.EARTH_SHADOW_NAME, shadow_material)
        self.rt.set_data(self.EARTH_SHADOW_NAME, geom="Parallelograms", mat=self.EARTH_SHADOW_NAME,
                         pos=[[0.0, -10.0 * self.CAMERA_DISTANCE, 0.0]],
                         u=[self.EARTH_SHADOW_PARKED, 0.0, 0.0], v=[0.0, 0.0, self.EARTH_SHADOW_PARKED],
                         c=[1.0, 1.0, 1.0])


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
        magnification = np.arcsin(self.reference_radius / camera_distance) / \
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


    def calculate_earth_shadow(self) -> dict:
        """
        Where the Earth's shadow falls in the scene, as much of it as the
        renderer needs: its axis, its two radii where it crosses the Moon's
        centre, and whether the Moon is in it.

        The radii are the real ones, the umbra and penumbra cones from the
        Sun's limb past the Earth's, scaled to the scene. The axis runs from
        the light's centre past a point placed for the light as it stands in
        the scene: the light has the Sun's true apparent size but stands only
        some ten Earth-Moon distances off, so that point is not the Earth's
        true place but the one an Earth would need to cast those two radii
        here - a little nearer and a little smaller than the true one, along
        the true direction to the Earth's centre, which is the geocentric
        libration. The same placing puts the shadow's centre where the real
        one falls (the eclipse magnitudes come out within 0.005 of the tables).

        Returns
        -------
        dict
            "centre" (that point), "axis" (unit vector from the light along
            the shadow), "umbra" and "penumbra" (radii at the Moon's centre,
            scene units), "along" (from the centre to the Moon's along the
            axis), "span" (from the light to the centre), and "inside": the
            Moon is in the penumbra, or near enough that a shadow ray aimed
            at any part of the Sun could cross the shadow
        """
        km = self.reference_radius / self.MOON_RADIUS_KM            # scene units to a km
        e = self.moon_ephem

        earth_km = self.EARTH_RADIUS_KM * self.EARTH_SHADOW_ENLARGEMENT
        umbra_km = earth_km - e.earth_distance * math.tan(
            math.asin((self.SUN_RADIUS_KM - earth_km) / e.sun_distance))
        penumbra_km = earth_km + e.earth_distance * math.tan(
            math.asin((self.SUN_RADIUS_KM + earth_km) / e.sun_distance))

        # The Earth that would cast those two against a light of this size and
        # distance. With k its distance over its distance from the light, the
        # two radii are its own (1 + k) less and more k light radii
        light_radius = self.SUN_LIGHT_DISTANCE * self.SUN_RADIUS_KM / e.sun_distance
        umbra, penumbra = umbra_km * km, penumbra_km * km
        k = (penumbra - umbra) / (2 * light_radius)
        distance = k * self.SUN_LIGHT_DISTANCE / (1 + k)

        lat, lon = np.radians(e.libr_lat_geo), np.radians(e.libr_long_geo)
        towards = self.moon_rotation @ np.array([np.cos(lat) * np.sin(lon),
                                                 -np.cos(lat) * np.cos(lon),
                                                 np.sin(lat)])
        centre = towards * distance
        light = np.asarray(self.light_pos, dtype=float)
        span = float(np.linalg.norm(centre - light))
        axis = (centre - light) / span
        along = float(-centre @ axis)
        miss = float(np.linalg.norm(-centre - along * axis))
        # A ray from the Moon to the Sun's limb, not its centre, crosses the
        # square up to this much further in: the light's radius brought down
        # to the Moon past the square
        slack = light_radius * along / span
        inside = along > 0.0 and miss < penumbra + self.reference_radius + 1.1 * slack
        return {"centre": centre, "axis": axis, "umbra": umbra, "penumbra": penumbra,
                "along": along, "span": span, "inside": inside}

    @classmethod
    def earth_shadow_texture(cls, umbra: float, penumbra: float) -> np.ndarray:
        """
        The shadow drawn as the light let through, ring by ring: the share of
        the Sun's disk each point of the Moon still sees past the Earth's, and
        in the share the Earth hides, the light the atmosphere bends into the
        umbra (EARTH_UMBRA_LIGHT). The square reaches EARTH_SHADOW_REACH times
        the penumbra's radius from the axis, measured where the shadow crosses
        the Moon, and is clear at its rim.

        Seen from a point of the Moon in the shadow's plane, at a distance d
        from the axis, the Earth's disk and the Sun's are two circles there of
        radii (penumbra plus and minus umbra) / 2, d apart; the Sun hidden is
        the part of its circle the Earth's covers.
        """
        n = cls.EARTH_SHADOW_TEXELS
        across = ((np.arange(n) + 0.5) / n * 2.0 - 1.0) * cls.EARTH_SHADOW_REACH * penumbra
        d = np.hypot(across[None, :], across[:, None])
        earth, sun = (penumbra + umbra) / 2, (penumbra - umbra) / 2

        hidden = np.where(d <= earth - sun, 1.0, 0.0)
        partly = (d > earth - sun) & (d < earth + sun)
        dp = d[partly]
        lens = (sun ** 2 * np.arccos(np.clip((dp ** 2 + sun ** 2 - earth ** 2) / (2 * dp * sun), -1, 1))
                + earth ** 2 * np.arccos(np.clip((dp ** 2 + earth ** 2 - sun ** 2) / (2 * dp * earth), -1, 1))
                - 0.5 * np.sqrt(np.clip((-dp + sun + earth) * (dp + sun - earth)
                                        * (dp - sun + earth) * (dp + sun + earth), 0, None)))
        hidden[partly] = lens / (np.pi * sun ** 2)

        texture = np.zeros((n, n, 4), dtype=np.float32)
        texture[..., :3] = (1.0 - hidden)[..., None] + hidden[..., None] * np.array(cls.EARTH_UMBRA_LIGHT)
        return texture                  # alpha 0: a crossing passes the colour as it is

    def _place_earth_shadow(self, shadow: dict):
        """
        Stand the square across the shadow's axis while the Moon is in the
        shadow, sized so that a ray from the Moon to the light crosses it where
        the texture answers for that point - and away from an eclipse, shrink
        it to nothing once and leave it be (see EARTH_SHADOW_PARKED).

        A point at a distance from the axis where the shadow crosses the
        Moon's centre is seen past the square span / (span + along) of that
        distance from its centre, the light being a point by then.
        """
        self._earth_shadow = shadow
        if not shadow["inside"]:
            if not self._earth_shadow_parked:
                self._earth_shadow_parked = True
                self.rt.update_data(self.EARTH_SHADOW_NAME,
                                    u=[self.EARTH_SHADOW_PARKED, 0.0, 0.0],
                                    v=[0.0, 0.0, self.EARTH_SHADOW_PARKED])
            return
        self._earth_shadow_parked = False

        # Drawn the first time the Moon comes into the shadow, and again as
        # the shadow's shape drifts - but not while a key is held, a redraw
        # costing a step's worth of time: the drift over a whole eclipse comes
        # to a couple of pixels, and the preview's end catches it up
        if self._earth_shadow_shape is None or not self._preview_active:
            self._redraw_earth_shadow()

        penumbra = shadow["penumbra"]
        axis = shadow["axis"]
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        across = np.cross(axis, helper)
        across /= np.linalg.norm(across)
        up = np.cross(axis, across)
        half = (self.EARTH_SHADOW_REACH * penumbra * shadow["span"]
                / (shadow["span"] + shadow["along"]))
        corner = shadow["centre"] - half * (across + up)
        self.rt.update_data(self.EARTH_SHADOW_NAME, pos=[corner.tolist()],
                            u=(2 * half * across).tolist(), v=(2 * half * up).tolist())

    def _redraw_earth_shadow(self) -> bool:
        """
        Draw the shadow into the square's texture if its shape has moved on
        since it was last drawn, and say whether it was. The texture depends on
        the shape alone - its size is the square's - which an eclipse moves by
        a hair.
        """
        shadow = self._earth_shadow
        if shadow is None or not shadow["inside"]:
            return False
        shape = round(shadow["umbra"] / shadow["penumbra"], 4)
        if shape == self._earth_shadow_shape:
            return False
        self._earth_shadow_shape = shape
        self.rt.set_texture_2d(self.EARTH_SHADOW_NAME,
                               self.earth_shadow_texture(shadow["umbra"], shadow["penumbra"]),
                               refresh=False)
        return True

    def update_overlays(self):
        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()
        if self.spot_labels_visible:
            self.update_spot_labels_orientation()
        if self.pins_visible:
            self.update_pins_orientation()
        if self._place_mark is not None:
            self.update_place_mark_orientation()
        if self.sub_points_visible:
            self.update_sub_points()
        if self._catalogue_active():
            self.update_catalogue(moved=True)


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
        shadow = self.calculate_earth_shadow()

        # Hold the render padlock across all scene updates: the render thread
        # cannot launch frames on a half-updated scene, and accumulation
        # restarts once instead of once per update call.
        with self.rt._padlock:
            self._move_camera_to_apparent_size()
            self.rt.update_data(self.MOON_OBJECT_NAME, u=u_new, v=v_new)
            self.rt.update_data(self.SUN_DISK_NAME, pos=[sun_disk_pos], r=sun_disk_radius)
            self._place_earth_shadow(shadow)
            # Light radius follows the true solar angular size seen from the Moon.
            # Light color is radiance, so illumination scales with angular size
            # squared, reproducing the real annual 1/d^2 brightness variation.
            # In the Earth's shadow the light is a point instead, its radiance
            # raised to light the surface as brightly (see SUN_POINT_RADIUS)
            sun_light_radius = float(self.SUN_LIGHT_DISTANCE * self.SUN_RADIUS_KM / self.moon_ephem.sun_distance)
            radius = self.SUN_POINT_RADIUS if shadow["inside"] else sun_light_radius
            self._sun_light_boost = (sun_light_radius / radius) ** 2
            self.rt.update_light(self.LIGHT_NAME, pos=self.light_pos, radius=radius,
                                 color=self._sun_light_colour())
            self.update_overlays()
            # A video export burning the canvas overlays into its frames draws
            # them here, where the new time is in force and the cycle that
            # carries the frame has not yet started (see renderer_video)
            self._refresh_export_overlay()

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

    def quit_window(self):
        """
        What closing the window does: write the settings down, stop the
        render, and end the Tk loop - in an order that cannot hang.

        PlotOptiX's own quit takes the render padlock on this, the Tk thread,
        and waits there in stop_rt for the render thread to finish. But the
        render thread hands every finished frame to Tk (TkOptiX posts
        <<LaunchFinished>> with when="now"), and a call into Tk from another
        thread waits for this one to take it. Closed while frames were still
        being drawn - just after a label is put on the Moon, say - each thread
        was left waiting on the other and the window stopped responding: a bare
        PlotOptiX window closed mid-render hung 2 times in 16.

        So the render thread is waited for from a thread of its own while this
        one goes on taking Tk's calls, and the padlock is not held meanwhile,
        the render thread taking it after every frame as well. Only once it has
        stopped is the scene destroyed and the loop ended, as PlotOptiX would.
        """
        rt = self.rt
        if rt is None or getattr(self, "_quitting", False) or rt._is_closed:
            return
        self._quitting = True
        self.save_settings()

        stopping = threading.Thread(target=rt._optix.stop_rt, daemon=True)
        stopping.start()
        while stopping.is_alive():
            rt._root.update()           # take whatever the render thread is waiting in
            stopping.join(0.01)

        with rt._padlock:
            rt._is_scene_created = False
            rt._is_started = False
            rt._optix.destroy_scene()
            rt._is_closed = True
        rt._root.quit()

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
                 color_downscale: int = 1,
                 fullscreen: bool = False) -> TkOptiX:
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
    fullscreen : bool
        Start with no window frame and no status bar (default False)

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
    print(f"  Fullscreen: {'ON' if fullscreen else 'OFF'}")
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
        fullscreen=fullscreen,
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
    # the initial time (Home), the dialogs that jump to a time (T, the planners K
    # and X, and the rise and set chart U, which goes to whatever moment in it
    # is clicked), the parallactic toggle (F4) and the set-time-now keys
    # (F9/F10). A running video export drives update_view from the raytracing
    # thread, so these are ignored while it lasts - see the export guard in
    # custom_key_handler.
    update_view_keysyms = {'F4', 'F9', 'F10', 'Home'}
    update_view_letters = set('qwtkxu')

    def custom_key_handler(event):
        # Ahead of every early return below: a dialog holding the keys must not
        # leave the renderer believing the right Ctrl is up while it is down
        if event.keysym == 'Control_R':
            moon_renderer._right_ctrl_down = True
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
        if event.keysym == 'F1':
            moon_renderer.show_help_dialog()
        elif event.keysym == 'F2':
            moon_renderer.toggle_info_panel()
        elif event.keysym == 'F3':
            moon_renderer.toggle_status_panel()
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
        elif event.keysym == 'F9':
            moon_renderer.set_time_to_now()
        elif event.keysym == 'F10':
            moon_renderer.set_time_to_now_and_auto_advance()
        elif event.keysym == 'F11':
            moon_renderer.toggle_full_screen()
        elif event.keysym == 'F12':
            moon_renderer.save_image_dialog()
        elif event.keysym == 'Escape':
            moon_renderer.exit_full_screen()
        elif event.keysym == 'Home':
            moon_renderer.reset_camera_position()
        elif event.keysym == 'Delete':
            moon_renderer.hide_pinned_labels()
        elif event.keysym == 'End':
            moon_renderer.reset_to_default_view()
        elif event.keysym == 'Insert':
            moon_renderer.toggle_scale_bar()
        elif event.keysym.lower() == 'v':
            moon_renderer.export_video_dialog()
        elif event.keysym.lower() == 'g':
            moon_renderer.toggle_grid()
        elif event.keysym.lower() == 'l':
            moon_renderer.toggle_standard_labels()
        elif event.keysym.lower() == 's':
            moon_renderer.toggle_spot_labels()
        elif event.keysym.lower() == 'r':
            moon_renderer.toggle_locator()
        elif event.keysym.lower() == 'c':
            moon_renderer.toggle_compass()
        elif event.keysym.lower() == 'b':
            # Shift sets the frame up, B alone turns it on and off
            if event.state & 0x1:
                moon_renderer.fov_overlay_dialog()
            else:
                moon_renderer.toggle_fov_overlay()
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
        elif event.keysym.lower() == 'p':
            moon_renderer.toggle_catalogue()
        elif event.keysym.lower() == 'h':
            moon_renderer.rotate_around_view_direction('ccw')
        elif event.keysym.lower() == 'j':
            moon_renderer.rotate_around_view_direction('cw')
        elif event.keysym in ('Left', 'Right', 'Up', 'Down'):
            if event.state & 0x4:  # Ctrl key pressed
                moon_renderer.rotate_around_moon_axis(event.keysym)
            else:
                moon_renderer.navigate_view(event.keysym)
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
        elif event.keysym == '0':
            moon_renderer.toggle_pins()
        elif event.keysym in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
            moon_renderer.toggle_pin_at_cursor(event, int(event.keysym))
        else:
            original_key_handler(event)

    moon_renderer.rt._gui_key_pressed = custom_key_handler

    original_key_released = moon_renderer.rt._gui_key_released

    def custom_key_released(event):
        if event.keysym == 'Control_R':
            moon_renderer._right_ctrl_down = False
        original_key_released(event)

    moon_renderer.rt._gui_key_released = custom_key_released

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
            if lat is None:
                moon_renderer.clear_measurement()

    moon_renderer.rt._gui_motion = custom_motion_handler

    # Override mouse handlers for distance measurement (Ctrl+drag)
    original_pressed_left = moon_renderer.rt._gui_pressed_left
    original_released_left = moon_renderer.rt._gui_released_left
    original_motion_pressed = moon_renderer.rt._gui_motion_pressed

    # Where a plain press of the left button began, so that letting it go in
    # the same place can be told from the end of a drag (see name_feature_at).
    # Only a press with no key held: Shift with the left button zooms, and a
    # held key gives the drag another meaning in PlotOptiX
    click_from = None

    def custom_pressed_left(event):
        nonlocal click_from
        click_from = None
        if event.state & 0x4:
            # Left Ctrl measures; right Ctrl measures and draws the profile
            moon_renderer.start_measurement(event, with_profile=moon_renderer._right_ctrl_down)
            return
        if not (event.state & 0x1 or moon_renderer.rt._any_key):
            click_from = (event.x, event.y)
        original_pressed_left(event)

    def custom_released_left(event):
        nonlocal click_from
        if moon_renderer.measuring:
            moon_renderer.finish_measurement(event)
            return
        original_released_left(event)
        pressed_at, click_from = click_from, None
        if pressed_at is not None:
            slop = moon_renderer._overlay_px(moon_renderer.CLICK_SLOP_PX)
            if (abs(event.x - pressed_at[0]) <= slop
                    and abs(event.y - pressed_at[1]) <= slop):
                moon_renderer.name_feature_at(event)

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

    # Closing the window writes the windows' choices down for the next run and
    # stops the render without PlotOptiX's own quit, which could hang (see
    # quit_window). PlotOptiX binds its quit to the window when it starts, so
    # it takes this one in its place
    moon_renderer.rt._gui_quit_callback = lambda *args: moon_renderer.quit_window()

    moon_renderer.start()
    return moon_renderer.rt


def run_renderer_process(*args, **kwargs):
    """
    run_renderer as the target of a spawned process (see main_gui_launcher).

    Spawned means a fresh interpreter, which has made no declaration about
    display scaling of its own and must make one before it opens a window: the
    launcher process it came from cannot make it on this one's behalf.

    A map that does not fit - in system RAM while it is prepared, or in GPU
    memory when it is uploaded - ends the process with its own message and
    MAP_TOO_LARGE_EXIT_CODE rather than a traceback, which the launcher turns
    back into something the user can act on.
    """
    make_dpi_aware()
    try:
        run_renderer(*args, **kwargs)
    except MapTooLargeError as e:
        print(f"\n{e}")
        sys.exit(MAP_TOO_LARGE_EXIT_CODE)
