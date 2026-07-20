"""
MoonRenderer: core renderer class (composing mixins) and run_renderer entry point.
"""

import tkinter as tk
import numpy as np
from typing import Optional
from datetime import datetime, timedelta

import plotoptix
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse

from moonrtx import astro
from moonrtx.shared_types import Camera, Observer
from moonrtx.data_loader import load_moon_features, load_elevation_data, load_color_data, load_starmap
from moonrtx.view_orientation import VIEW_ORIENTATION_NSWE, VIEW_ORIENTATION_NSEW, VIEW_ORIENTATION_SNEW, VIEW_ORIENTATION_SNWE

# Mixins – each adds a focused group of methods
from moonrtx.renderer_status import StatusMixin
from moonrtx.renderer_dialogs import DialogsMixin
from moonrtx.renderer_labels import LabelsMixin
from moonrtx.renderer_pins import PinsMixin
from moonrtx.renderer_navigation import NavigationMixin


class MoonRenderer(StatusMixin, DialogsMixin, LabelsMixin, PinsMixin, NavigationMixin):
    """
    Renders the Moon surface as seen from a specific location on Earth
    at a specific time, with accurate solar illumination.
    """

    # Scene geometry
    MOON_RADIUS = 10.0          # Radius of Moon sphere in scene units
    MOON_FILL_FRACTION = 0.9    # Moon fills 90% of window height (5% margins top/bottom)
    # Default camera distance in scene units. Larger distance renders the limb closer
    # to what a real observer sees (at 30 radii the visible cap reaches 88.1 degrees
    # from the disk center vs 84.3 at 10 radii and 89.7 in reality). The value is a
    # trade-off: much larger distances degrade float32 ray precision and produce
    # contour/tessellation artifacts on the displaced surface (visible at ~220 radii).
    CAMERA_DISTANCE = MOON_RADIUS * 30
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

    # Accumulation settings. Since PlotOptiX 0.19.1 the displayed image is
    # presented once per completed accumulation cycle (max_accumulation_frames),
    # so any scene change (time stepping, brightness, overlays, navigation)
    # would only appear after a full 32-frame cycle converges - held-key Q/W
    # animation would barely refresh at all. During interactive changes the
    # cycle is therefore shortened to a single frame (immediate but slightly
    # noisy preview, ~20 steps/s measured at full screen with exact shadows)
    # and the converged setting is restored shortly after the last change.
    ACCUMULATION_FRAMES = 32
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
        initial_camera : Camera
            Optional initial camera for resets with R key (if None, a default camera will be calculated from ephemeris)
        dt_local : datetime
            Local datetime for the view 
        starmap_file : str, optional
            Path to star map TIFF for background
        downscale : int
            Elevation map downscale factor
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
        _tmp = tk.Tk()
        _tmp.withdraw()
        self.width = _tmp.winfo_screenwidth()
        self.height = _tmp.winfo_screenheight() - 40
        _tmp.destroy()

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

        self.view_orientation = init_view_orientation
        self.initial_view_orientation = init_view_orientation  # For reset with R/V keys

        # Default camera calculated from ephemeris (for reset with V key)
        visible_height = 2 * self.MOON_RADIUS / self.MOON_FILL_FRACTION
        fov = np.degrees(2 * np.arctan(visible_height / (2 * self.CAMERA_DISTANCE)))
        self.default_camera = Camera(
            eye=[0, -self.CAMERA_DISTANCE, 0],
            target=[0, 0, 0],
            up=[0, 0, 1],
            fov=max(1, min(90, fov))
        )

        self.dt_local = dt_local

        # Initial time for reset with R key
        self.initial_dt_local = self.dt_local

        # Initial camera for reset with R key
        self.initial_camera = self.default_camera if initial_camera is None else initial_camera

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
        self.datetime_dialog_focused = False

        # Pins settings
        self.pins_visible = True  # Pins visible by default
        self.pins = {}  # dict mapping digit (1-9) to body-frame graph vertices

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
        self._status_feature = None

        # Interactive-preview state (short accumulation cycles during scene changes)
        self._preview_active = False
        self._preview_restore_id = None

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

        new_dt_local = self.dt_local + timedelta(minutes=delta_minutes)

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

        # Tone mapping
        self.rt.set_float("tonemap_exposure", 0.9)
        self.rt.set_float("tonemap_gamma", self.gamma)
        self.rt.add_postproc("Gamma")

        # Background (stars). Loaded locally: uploaded to a GPU texture here and
        # released when this method returns (the host copy is ~760 MB)
        star_map = load_starmap(self.starmap_file, self.width * 6) if self.starmap_file else None
        if star_map is not None:
            self.rt.set_background_mode("TextureEnvironment")
            self.rt.set_background(star_map, gamma=self.gamma, rt_format="UByte4")
        else:
            self.rt.set_background(0)  # Black background

        # Setup material with Moon texture (local for the same reason, ~200 MB).
        # Copy the material so the shared plotoptix module dict stays untouched.
        color_data = load_color_data(self.color_file, self.gamma)
        self.rt.set_texture_2d("moon_color", color_data)
        moon_material = m_diffuse.copy()
        moon_material["ColorTextures"] = ["moon_color"]
        self.rt.update_material("diffuse", moon_material)

        # Create Moon sphere with displacement
        self.rt.set_data(self.MOON_OBJECT_NAME, geom="ParticleSetTextured", geom_attr="DisplacedSurface",
                        pos=[0, 0, 0], u=[0, 0, 1], v=[0, -1, 0], r=self.MOON_RADIUS)

        # Apply displacement map (no refresh: the renderer is not started yet)
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
                         r=self.SUN_RADIUS, c=self.SUN_DISK_COLOR)


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

        Both angles vary with the date: the magnification with the real Moon distance,
        the Sun's apparent size with the Sun distance.
        """
        # Magnification of the rendered Moon relative to its real apparent size
        magnification = np.arcsin(self.MOON_RADIUS / self.CAMERA_DISTANCE) / \
            np.arcsin(self.MOON_RADIUS_KM / self.moon_ephem.distance)

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
        center = np.array([0.0, -self.CAMERA_DISTANCE, 0.0]) + self.SUN_DISK_DISTANCE * direction
        radius = self.SUN_DISK_DISTANCE * np.tan(sun_angular_radius) if in_view else 0.01
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


    def update_view(self, dt_local: Optional[datetime] = None):

        if dt_local is not None:
            self.dt_local = dt_local

        self.moon_ephem = astro.calculate_moon_ephemeris(self.dt_local, self.parallactic_mode)
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
            self.rt.update_data(self.MOON_OBJECT_NAME, u=u_new, v=v_new)
            self.rt.update_data(self.SUN_DISK_NAME, pos=[sun_disk_pos], r=sun_disk_radius)
            # Light radius follows the true solar angular size seen from the Moon.
            # Light color is radiance, so illumination scales with angular size
            # squared, reproducing the real annual 1/d^2 brightness variation.
            sun_light_radius = float(self.SUN_LIGHT_DISTANCE * self.SUN_RADIUS_KM / self.moon_ephem.sun_distance)
            self.rt.update_light(self.LIGHT_NAME, pos=self.light_pos, radius=sun_light_radius)
            self.update_overlays()

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

def run_renderer(dt_local: datetime,
                 observer: Observer,
                 elevation_file: str,
                 color_file: str,
                 starmap_file: str,
                 features_file: str,
                 downscale: int,
                 brightness: int,
                 initial_camera: Optional[Camera],
                 time_step_minutes: int = 15,
                 init_view_orientation: str = VIEW_ORIENTATION_NSWE,
                 gamma: float = 2.2,
                 parallactic_mode: bool = False) -> TkOptiX:
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
    print(f"  Elevation File: {elevation_file}")
    print(f"  Color File: {color_file}")
    print(f"  Brightness: {brightness}")
    print(f"  Gamma: {gamma}")
    print(f"  Downscale Factor: {downscale}")
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

    # Keys that modify the rendered scene: switch to single-frame preview
    # cycles so the change is displayed immediately (see ACCUMULATION_FRAMES)
    preview_keysyms = {'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
                       'Left', 'Right', 'Up', 'Down',
                       '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    preview_letters = set('glsrchjvazedqwp')

    def custom_key_handler(event):
        # Ignore key events when search dialog or datetime dialog is focused
        if moon_renderer.search_dialog_open:
            return
        if moon_renderer.datetime_dialog_focused:
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
            moon_renderer.parallactic_mode = not moon_renderer.parallactic_mode
            moon_renderer.update_view()
            moon_renderer._update_status_parallactic()
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
            moon_renderer.center_view_on_cursor(event)
        elif event.keysym == 'F12':
            moon_renderer.save_image_dialog()
        elif event.keysym.lower() == 'f':
            moon_renderer.search_feature_dialog()
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