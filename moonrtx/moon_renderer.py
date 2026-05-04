"""
MoonRenderer: core renderer class (composing mixins) and run_renderer entry point.
"""

import numpy as np
from typing import Optional
from datetime import datetime, timedelta

import plotoptix
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse

from moonrtx.shared_types import Camera, Observer
from moonrtx.astro import calculate_moon_ephemeris
from moonrtx.data_loader import load_moon_features, load_elevation_data, load_color_data, load_starmap

from moonrtx.constants import (
    CAMERA_NAME, LIGHT_NAME, MOON_FILL_FRACTION, MOON_OBJECT_NAME, SUN_RADIUS, MOON_RADIUS,
    ORIENTATION_NSWE, ORIENTATION_NSEW, ORIENTATION_SNEW, ORIENTATION_SNWE,
)

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
                 window_title: str,
                 elevation_file: str,
                 color_file: str,
                 features_file: str,
                 brightness: int,
                 observer: Observer,
                 initial_camera: Optional[Camera],
                 dt_local: datetime,
                 starmap_file: Optional[str],
                 downscale: int = 3,
                 width: int = 1400,
                 height: int = 900,
                 time_step_minutes: int = 15,
                 init_view_orientation: str = ORIENTATION_NSWE,
                 gamma: float = 2.8,
                 parallactic_mode: bool = False):
        """
        Initialize the planetarium.

        Parameters
        ----------
        window_title : str
            Window title
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
        width, height : int
            Render window size
        time_step_minutes : int
            Time step in minutes for Q/W keys
        init_view_orientation : str
            Initial view orientation (ORIENTATION_NSWE, ORIENTATION_NSEW, etc.)
        observer : Observer
            Observer latitude, longitude, and elevation
        gamma : float
            Gamma correction value (default 2.8)
        parallactic_mode : bool
            Whether to use parallactic projection mode (default False)
        """
        self.width = width
        self.height = height
        self.downscale = downscale
        self.gamma = gamma
        self.time_step_minutes = time_step_minutes
        self.parallactic_mode = parallactic_mode

        # Load data
        self.elevation, self._elev_scale, self._elev_rv, self._elev_displacement_range = load_elevation_data(elevation_file, downscale)
        self.color_data = load_color_data(color_file, self.gamma)
        # Sort features by angular_radius (smallest first) for efficient lookup
        self.moon_features = sorted(load_moon_features(features_file), key=lambda f: f.angular_radius)
        self.star_map = load_starmap(starmap_file) if starmap_file else None

        self.window_title = window_title
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
        self.camera_distance = self.moon_radius * 10

        self.orientation_mode = init_view_orientation
        self.initial_orientation_mode = init_view_orientation  # For reset with R/V keys

        # Default camera calculated from ephemeris (for reset with V key)
        visible_height = 2 * self.moon_radius / MOON_FILL_FRACTION
        fov = np.degrees(2 * np.arctan(visible_height / (2 * self.camera_distance)))
        self.default_camera = Camera(
            eye=[0, -self.camera_distance, 0],
            target=[0, 0, 0],
            up=[0, 0, 1],
            fov=max(1, min(90, fov))
        )

        self.dt_local = dt_local

        # Initial camera for reset with R key
        self.initial_camera = self.default_camera if initial_camera is None else initial_camera

        # Initial time for reset with R key
        self.initial_dt_local = self.dt_local

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

        self.observer = observer

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
        self._info_illum_var = None
        self._info_libr_l_var = None
        self._info_libr_b_var = None
        self._info_colong_var = None

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
        self.rt.update_light(LIGHT_NAME, color=self.brightness)
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

        self.update_view(datetime.now().astimezone())

        if self._auto_advance_var and self._auto_advance_var.get():
            self._auto_advance_elapsed = 0

        self.update_overlays()
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

        self.update_overlays()
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
        self.rt.set_data(MOON_OBJECT_NAME, geom="ParticleSetTextured", geom_attr="DisplacedSurface",
                        pos=[0, 0, 0], u=[0, 0, 1], v=[0, -1, 0], r=self.moon_radius)

        # Apply displacement map
        self.rt.set_displacement(MOON_OBJECT_NAME, self.elevation, refresh=True)

        self.rt.setup_camera(CAMERA_NAME,
                             cam_type=self.initial_camera.type,
                             eye=self.initial_camera.eye,
                             target=self.initial_camera.target,
                             up=self.initial_camera.up,
                             fov=self.initial_camera.fov,
                             aperture_radius=0.01,
                             aperture_fract=0.2,
                             focal_scale=0.7)
        
        self.rt.setup_light(LIGHT_NAME, color=self.brightness, radius=SUN_RADIUS)


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
        #
        # In parallactic-mount mode the view frame keeps celestial north up
        # (rotation matrix was built with q=0), so PA is already measured from
        # the view's "up" direction and no q subtraction is needed.
        if self.parallactic_mode:
            bright_limb_angle_deg = self.moon_ephem.pa
        else:
            bright_limb_angle_deg = self.moon_ephem.pa - self.moon_ephem.q
        
        # Normalize to -180 to 180
        while bright_limb_angle_deg > 180: bright_limb_angle_deg -= 360
        while bright_limb_angle_deg < -180: bright_limb_angle_deg += 360
        
        bright_limb_angle = np.radians(bright_limb_angle_deg)
        phase_angle = np.radians(self.moon_ephem.phase_angle)
        light_distance = 100  # Far away for parallel rays
        
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
        
        # When phase is very small (near full moon), sin(phase) approaches 0,
        # which would place the light exactly behind the camera. This causes
        # the Moon to appear completely dark in ray tracing because no light
        # rays can illuminate the visible surface.
        #
        # To fix this, we ensure a minimum offset angle so the light is always
        # slightly to the side, providing proper illumination even at full moon.
        #
        # This offset is only applied near full moon (phase < 6°), not near new moon
        # (phase ≈ 180°) where the Moon should actually be dark.
        #
        # Consequences of 6 degree offset:
        # - Only affects when phase_angle < 6.0° (very close to full moon)
        # - At 6° phase, illumination = (1 + cos(6.0°))/2 ≈ 99.7% (vs 100% at true full)
        # - A ~0.3% sliver at the Moon's edge would be in shadow, visually imperceptible
        # - Most of the lunar cycle (phase_angle > 6.0°) is completely unaffected
        min_phase_offset = np.radians(6.0)
        # Only apply minimum offset near full moon (phase_angle < 6.0°), not near new moon
        effective_sin_phase = np.sin(min_phase_offset if phase_angle < min_phase_offset else phase_angle)
        
        light_x = -np.sin(bright_limb_angle) * effective_sin_phase * light_distance
        light_z = np.cos(bright_limb_angle) * effective_sin_phase * light_distance
        light_y = -np.cos(phase_angle) * light_distance

        return [light_x, light_y, light_z]
    

    def update_overlays(self):
        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()
        if self.spot_labels_visible:
            self.update_spot_labels_orientation()
        if self.pins_visible:
            self.update_pins_orientation()


    def update_view(self, dt_local: datetime = None):

        if dt_local is not None:
            self.dt_local = dt_local

        self.moon_ephem = calculate_moon_ephemeris(self.dt_local, self.observer, self.parallactic_mode)
        self.moon_rotation = self.moon_ephem.rotation_matrix
        self.moon_rotation_inv = self.moon_rotation.T

        self.light_pos = self.calculate_light_pos()

        u_new = self.moon_rotation[:, 2]        # Z axis of the rotated surface
        v_new = -self.moon_rotation[:, 1]       # Invert Y axis to match our convention of v pointing down in the texture

        self.rt.update_data(MOON_OBJECT_NAME, u=u_new, v=v_new)
        self.rt.update_light(LIGHT_NAME, pos=self.light_pos)
        self.update_overlays()


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
                 window_title: str,
                 initial_camera: Optional[Camera],
                 time_step_minutes: int = 15,
                 init_view_orientation: str = ORIENTATION_NSWE,
                 gamma: float = 2.8,
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
    window_title : str
        Window title
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
        window_title=window_title,
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

    # Setup renderer
    moon_renderer.setup_renderer()

    # Set view
    moon_renderer.update_view()

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
        elif event.keysym == 'F4':
            moon_renderer.parallactic_mode = not moon_renderer.parallactic_mode
            moon_renderer.update_view()
            moon_renderer.update_overlays()
            moon_renderer._update_status_parallactic()
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

    moon_renderer.start()
    return moon_renderer.rt
