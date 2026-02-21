"""
MoonRenderer: core renderer class (composing mixins) and run_renderer entry point.
"""

import numpy as np
from typing import Optional
from datetime import datetime, timezone, timedelta

import plotoptix
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse

from moonrtx.shared_types import MoonEphemeris, MoonFeature, CameraParams
from moonrtx.astro import calculate_moon_ephemeris
from moonrtx.data_loader import load_moon_features, load_elevation_data, load_color_data, load_starmap

from moonrtx.constants import (
    MOON_FILL_FRACTION, SUN_RADIUS, MOON_RADIUS,
    CAMERA_TYPE,
    ORIENTATION_NSWE, ORIENTATION_NSEW, ORIENTATION_SNEW, ORIENTATION_SNWE,
)
from moonrtx.scene_math import Scene, calculate_rotation, calculate_camera_and_light, encode_camera_params

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

    def __init__(self,
                 app_name: str,
                 elevation_file: str,
                 color_file: str,
                 features_file: str,
                 brightness: int,
                 starmap_file: Optional[str] = None,
                 downscale: int = 3,
                 width: int = 1400,
                 height: int = 900,
                 time_step_minutes: int = 15,
                 init_view_orientation: str = ORIENTATION_NSWE,
                 observer_elevation: int = 0,
                 gamma: float = 3.2):
        """
        Initialize the planetarium.

        Parameters
        ----------
        app_name : str
            Application name
        elevation_file : str
            Path to Moon elevation data TIFF
        color_file : str
            Path to Moon color data TIFF
        features_file : str
            Moon features CSV file with craters, mounts etc.
        brightness : int
            Brightness
        starmap_file : str, optional
            Path to star map TIFF for background
        downscale : int
            Elevation map downscale factor
        width, height : int
            Render window size
        time_step_minutes : int
            Time step in minutes for Q/W keys
        init_view_orientation : str
            Initial view orientation (ORIENTATION_NSWE, ORIENTATION_NSEW, etc.)
        observer_elevation : int
            Observer elevation in meters above sea level
        gamma : float
            Gamma correction value (default 2.2)
        """
        self.width = width
        self.height = height
        self.downscale = downscale
        self.gamma = gamma
        self.time_step_minutes = time_step_minutes

        # Load data
        self.elevation = load_elevation_data(elevation_file, downscale)
        self.color_data = load_color_data(color_file, self.gamma)
        # Sort features by angular_radius (smallest first) for efficient lookup
        self.moon_features = sorted(load_moon_features(features_file), key=lambda f: f.angular_radius)
        self.star_map = load_starmap(starmap_file) if starmap_file else None

        self.app_name = app_name
        self.brightness = brightness

        # Renderer
        self.rt = None
        self.moon_ephem = None
        self.moon_rotation = None
        self.moon_rotation_inv = None

        # Grid settings
        self.moon_grid_visible = False
        self.moon_grid = None
        self.moon_radius = MOON_RADIUS

        self.orientation_mode = init_view_orientation
        self.initial_orientation_mode = init_view_orientation  # For reset with R/V keys

        # Initial camera parameters (for reset with R key)
        self.initial_camera_params = None

        # Initial time for reset with R key
        self.initial_dt_local = None

        # Default camera parameters calculated from ephemeris (for reset with V key)
        self.default_camera_params = None

        # Flag to track if window has been maximized
        self._window_maximized = False

        # Standard labels settings
        self.standard_labels_visible = False
        self.standard_labels = None
        self.standard_label_features = []

        # Spot labels settings
        self.spot_labels_visible = False
        self.spot_labels = None
        self.spot_label_features = []

        # Light position in scene coordinates (set on first update_view)
        self.light_pos = None

        # Store view parameters for filename generation
        self.dt_local = None
        self.observer_lat = None
        self.observer_lon = None
        self.observer_elevation = observer_elevation

        # Flag to track if search dialog is open
        self.search_dialog_open = False

        # Datetime dialog tracking
        self.datetime_dialog = None
        self.datetime_dialog_focused = False

        # Pins settings
        self.pins_visible = True  # Pins visible by default
        self.pins = {}  # dict mapping digit (1-9) to pin segments

        # Distance measurement settings
        self.measuring = False
        self.measure_start_canvas = None
        self.measure_start_coords = None
        self.leading_line_id = None
        self.measured_distance = None

        # Status bar panel variables (set up as StringVars after renderer is created)
        self._status_observer_var = None
        self._status_view_var = None
        self._status_time_var = None
        self._status_measured_var = None
        self._status_feature_var = None
        self._status_brightness_var = None
        self._status_gamma_var = None
        self._status_pins_var = None
        self._status_coords_var = None

        # Auto-advance (real-time playback) settings
        self._auto_advance_var = None
        self._auto_advance_id = None
        self._auto_advance_elapsed = 0
        self._auto_advance_interval = 1000  # tick interval in ms

        # Info panel variables (bottom-left overlay)
        self._info_frame = None
        self.show_info_panel = True
        self._info_az_var = None
        self._info_alt_var = None
        self._info_ra_var = None
        self._info_dec_var = None
        self._info_phase_var = None
        self._info_sun_sep_var = None
        self._info_distance_var = None
        self._info_libr_l_var = None
        self._info_libr_b_var = None

    # ---- brightness / time-step / auto-advance ----

    def change_brightness(self, delta: int):
        if delta == 0:
            return
        if self.brightness <= 0 and delta < 0:
            return
        if self.brightness >= 500 and delta > 0:
            return
        self.brightness += delta
        self.brightness = max(0, min(500, self.brightness))
        self.rt.setup_light("sun", color=self.brightness)
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
        if self.time_step_minutes <= 1 and delta < 0:
            return
        if self.time_step_minutes >= 1440 and delta > 0:
            return
        self.time_step_minutes += delta
        self.time_step_minutes = max(1, min(1440, self.time_step_minutes))
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
        target_ms = self.time_step_minutes * 60 * 1000
        if self._auto_advance_elapsed >= target_ms:
            self._auto_advance_elapsed = 0
            self.change_time(self.time_step_minutes)
        self._schedule_auto_advance()

    def set_time_to_now(self):
        """Set the observation time to the current (now) time."""
        now_local = datetime.now().astimezone()

        self.update_moon_for_time(now_local, self.observer_lat, self.observer_lon, self.observer_elevation)

        if self._auto_advance_var and self._auto_advance_var.get():
            self._auto_advance_elapsed = 0

        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()
        if self.spot_labels_visible:
            self.update_spot_labels_orientation()

        self.update_pins_orientation()
        self._update_all_status_panels()

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

        self.update_moon_for_time(new_dt_local, self.observer_lat, self.observer_lon, self.observer_elevation)

        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()
        if self.spot_labels_visible:
            self.update_spot_labels_orientation()

        self.update_pins_orientation()

        self._update_status_time()
        self._update_info_moon()

    # ---- renderer setup ----

    def _mouse_wheel_handler(self, event):
        """Handle mouse wheel events for zooming."""
        self.zoom_with_wheel(event)

    def setup_renderer(self):
        """Initialize the PlotOptiX renderer."""
        self.rt = TkOptiX(
            width=self.width,
            height=self.height,
            on_launch_finished=self._on_launch_finished
        )

        # Rendering parameters
        self.rt.set_param(min_accumulation_step=1, max_accumulation_frames=100)

        # Tone mapping
        self.rt.set_float("tonemap_exposure", 0.9)
        self.rt.set_float("tonemap_gamma", self.gamma)
        self.rt.add_postproc("Gamma")

        # Background (stars)
        if self.star_map is not None:
            self.rt.set_background_mode("TextureEnvironment")
            self.rt.set_background(self.star_map, gamma=self.gamma, rt_format="UByte4")
        else:
            self.rt.set_background(0)  # Black background

        # Setup material with Moon texture
        self.rt.set_texture_2d("moon_color", self.color_data)
        m_diffuse["ColorTextures"] = ["moon_color"]
        self.rt.update_material("diffuse", m_diffuse)

        # Create Moon sphere with displacement
        self.rt.set_data("moon", geom="ParticleSetTextured", geom_attr="DisplacedSurface",
                        pos=[0, 0, 0], u=[0, 0, 1], v=[0, -1, 0], r=10)

        # Apply displacement map
        self.rt.set_displacement("moon", self.elevation, refresh=True)

    # ---- view update ----

    def update_view(self, dt_local: datetime, lat: float, lon: float, elevation: int = 0, zoom: float = 1000):
        """
        Update the view for a specific time and location.

        Parameters
        ----------
        dt_local : datetime
            Local time
        lat, lon : float
            Observer latitude and longitude in degrees
        elevation : int
            Observer elevation in meters above sea level
        zoom : float
            Camera zoom factor
        """
        dt_utc = dt_local.astimezone(timezone.utc)
        eph = calculate_moon_ephemeris(dt_utc, lat, lon, elevation)
        self.moon_rotation = calculate_rotation(-eph.libr_long, eph.libr_lat, eph.pa_axis_view)
        self.moon_rotation_inv = self.moon_rotation.T
        self.moon_ephem = eph

        self.dt_local = dt_local
        self.observer_lat = lat
        self.observer_lon = lon
        self.observer_elevation = elevation

        scene = calculate_camera_and_light(self.moon_ephem, zoom, self.moon_radius)
        self.light_pos = scene.light_pos

        u_new = self.moon_rotation @ np.array([0.0, 0.0, 1.0])
        v_new = self.moon_rotation @ np.array([0.0, -1.0, 0.0])

        self.rt.update_data("moon", u=u_new.tolist(), v=v_new.tolist())

        # Calculate FOV so moon fills MOON_FILL_FRACTION of window height
        camera_distance = self.moon_radius * (zoom / 100)
        moon_diameter = 2 * self.moon_radius
        visible_height = moon_diameter / MOON_FILL_FRACTION
        fov = np.degrees(2 * np.arctan(visible_height / (2 * camera_distance)))
        fov = max(1, min(90, fov))

        self.rt.setup_camera("cam1",
                             cam_type=CAMERA_TYPE,
                             eye=scene.eye.tolist(),
                             target=scene.target.tolist(),
                             up=scene.up.tolist(),
                             aperture_radius=0.01,
                             aperture_fract=0.2,
                             focal_scale=0.7,
                             fov=fov)

        self.default_camera_params = CameraParams(
            eye=scene.eye.tolist(), target=scene.target.tolist(),
            up=scene.up.tolist(), fov=fov)

        if self.initial_camera_params is None:
            self.initial_camera_params = self.default_camera_params
            self.initial_dt_local = dt_local

        self.rt.setup_light("sun", pos=scene.light_pos.tolist(),
                            color=self.brightness, radius=SUN_RADIUS)

        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()

    def update_moon_for_time(self, dt_local: datetime, lat: float, lon: float, elevation: int = 0):
        """
        Update Moon orientation and lighting for a new time without changing camera.

        Parameters
        ----------
        dt_local : datetime
            Local time
        lat, lon : float
            Observer latitude and longitude in degrees
        elevation : int
            Observer elevation in meters above sea level
        """
        dt_utc = dt_local.astimezone(timezone.utc)
        eph = calculate_moon_ephemeris(dt_utc, lat, lon, elevation)
        self.moon_rotation = calculate_rotation(-eph.libr_long, eph.libr_lat, eph.pa_axis_view)
        self.moon_rotation_inv = self.moon_rotation.T
        self.moon_ephem = eph

        self.dt_local = dt_local
        self.observer_lat = lat
        self.observer_lon = lon
        self.observer_elevation = elevation

        scene = calculate_camera_and_light(self.moon_ephem, 1000, self.moon_radius)
        self.light_pos = scene.light_pos

        current_fov = self.default_camera_params.fov if self.default_camera_params else 45.0
        self.default_camera_params = CameraParams(
            eye=scene.eye.tolist(),
            target=scene.target.tolist(),
            up=scene.up.tolist(),
            fov=current_fov
        )

        u_new = self.moon_rotation @ np.array([0.0, 0.0, 1.0])
        v_new = self.moon_rotation @ np.array([0.0, -1.0, 0.0])
        self.rt.update_data("moon", u=u_new.tolist(), v=v_new.tolist())

        self.rt.setup_light("sun", pos=scene.light_pos.tolist(),
                            color=self.brightness, radius=SUN_RADIUS)

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

    def save_image(self, filename: str):
        """Save the current render to file."""
        if self.rt is not None:
            self.rt.save_image(filename)
            print(f"Saved: {filename}")

    def apply_camera_params(self, params: CameraParams):
        """
        Apply camera parameters to restore a specific view.

        Parameters
        ----------
        params : CameraParams
            Camera parameters (eye, target, up, fov)
        """
        if self.rt is None:
            return

        self.rt.setup_camera("cam1",
                             eye=params.eye,
                             target=params.target,
                             up=params.up,
                             fov=params.fov)

        self.initial_camera_params = params

        print(f"Applied camera params: eye={params.eye}, target={params.target}, "
              f"up={params.up}, fov={params.fov:.2f}")


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_renderer(dt_local: datetime,
                 lat: float,
                 lon: float,
                 observer_elevation: int,
                 elevation_file: str,
                 color_file: str,
                 starmap_file: str,
                 features_file: str,
                 downscale: int,
                 brightness: int,
                 app_name: str,
                 init_camera_params: Optional[CameraParams] = None,
                 time_step_minutes: int = 15,
                 init_view_orientation: str = ORIENTATION_NSWE,
                 gamma: float = 3.2) -> TkOptiX:
    """
    Quick function to render the Moon for a specific time and location.

    Parameters
    ----------
    dt_local : datetime
        Local time
    lat, lon : float
        Observer latitude and longitude in degrees
    observer_elevation : int
        Observer elevation in meters above sea level
    elevation_file, color_file, starmap_file, features_file : str
        Paths to data files
    downscale : int
        Elevation downscale factor
    brightness : int
        Brightness
    app_name : str
        Application name
    init_camera_params : CameraParams, optional
        Initial camera parameters to restore a specific view
    time_step_minutes : int
        Time step in minutes for Q/W keys (default 15)
    init_view_orientation : str
        Initial view orientation mode.
    gamma : float
        Gamma correction value (default 2.2)

    Returns
    -------
    TkOptiX
        The renderer instance
    """
    print()
    print("Used PlotOptiX version:", plotoptix.__version__)
    print("Renderer started with parameters:")
    print(f"  Geographical Location: Lat {lat}°, Lon {lon}°, Elevation {observer_elevation} m")
    print(f"  Local Time: {dt_local}")
    print(f"  Elevation Map File: {elevation_file}")
    print(f"  Brightness: {brightness}")
    print(f"  Gamma: {gamma}")
    print(f"  Downscale Factor: {downscale}")
    print(f"  Time Step (minutes): {time_step_minutes}")
    print(f"  Initial View Orientation: {init_view_orientation}")
    if init_camera_params:
        print("  Init View: Restoring camera from screenshot filename")
    print()

    moon_renderer = MoonRenderer(
        app_name=app_name,
        elevation_file=elevation_file,
        color_file=color_file,
        starmap_file=starmap_file,
        downscale=downscale,
        features_file=features_file,
        brightness=brightness,
        time_step_minutes=time_step_minutes,
        init_view_orientation=init_view_orientation,
        observer_elevation=observer_elevation,
        gamma=gamma
    )

    # Setup renderer
    moon_renderer.setup_renderer()

    # Set view
    moon_renderer.update_view(dt_local=dt_local, lat=lat, lon=lon, elevation=observer_elevation)

    # Apply custom camera parameters if provided (to restore a saved view)
    if init_camera_params is not None:
        moon_renderer.apply_camera_params(init_camera_params)

    original_key_handler = moon_renderer.rt._gui_key_pressed

    def custom_key_handler(event):
        # Ignore key events when search dialog or datetime dialog is focused
        if moon_renderer.search_dialog_open:
            return
        if moon_renderer.datetime_dialog_focused:
            return
        if event.keysym.lower() == 'g':
            moon_renderer.toggle_grid()
        elif event.keysym.lower() == 'l':
            moon_renderer.toggle_standard_labels()
        elif event.keysym.lower() == 's':
            moon_renderer.toggle_spot_labels()
        elif event.keysym == 'F5':
            moon_renderer.set_orientation(ORIENTATION_NSWE)
            original_key_handler(event)
        elif event.keysym == 'F6':
            moon_renderer.set_orientation(ORIENTATION_NSEW)
            original_key_handler(event)
        elif event.keysym == 'F7':
            moon_renderer.set_orientation(ORIENTATION_SNEW)
            original_key_handler(event)
        elif event.keysym == 'F8':
            moon_renderer.set_orientation(ORIENTATION_SNWE)
            original_key_handler(event)
        elif event.keysym.lower() == 'r':
            moon_renderer.reset_camera_position()
        elif event.keysym.lower() == 'c':
            moon_renderer.center_view_on_cursor(event)
        elif event.keysym == 'F12':
            moon_renderer.save_image_dialog()
        elif event.keysym.lower() == 'f':
            moon_renderer.search_feature_dialog()
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
            feature_text = ""
            if hd > 0:
                lat, lon = moon_renderer.hit_to_selenographic(hx, hy, hz)
                if lat is not None and lon is not None:
                    feature = moon_renderer.find_feature_for_status_bar(lat, lon)
                    if feature is not None:
                        feature_text = f"{feature.name} (size = {feature.size_km:.1f} km)"
            moon_renderer.rt._status_action_text.set('')
            moon_renderer._update_info_coords(lat, lon)
            moon_renderer._update_status_feature(feature_text)

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

    moon_renderer.start()
    return moon_renderer.rt
