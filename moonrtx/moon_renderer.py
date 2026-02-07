import numpy as np
import os
import struct
import base64
import tkinter as tk
from tkinter import filedialog
from typing import NamedTuple, Optional
from datetime import datetime, timezone, timedelta
from numpy.typing import NDArray

from moonrtx.shared_types import MoonEphemeris, MoonFeature, CameraParams
from moonrtx.astro import calculate_moon_ephemeris
from moonrtx.data_loader import load_moon_features, load_elevation_data, load_color_data, load_starmap
from moonrtx.moon_grid import create_moon_grid, create_standard_labels, create_spot_labels, create_single_digit_on_sphere, create_grid_labels_for_orientation

import plotoptix
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse, m_flat

GRID_COLOR = [0.50, 0.50, 0.50]
MOON_FILL_FRACTION = 0.9    # Moon fills 90% of window height (5% margins top/bottom)
SUN_RADIUS = 10             # affects Moon surface illumination
MOON_RADIUS = 10.0          # Radius of Moon sphere in scene units
PIN_COLOR = [1.0, 0.0, 0.0]
GRID_LINE_RADIUS = 0.006    # Thin lines for grid
GRID_LABEL_RADIUS = 0.012   # Slightly thicker lines for grid labels
STANDARD_LABEL_RADIUS = 0.008  # Standard feature label thickness
SPOT_LABEL_RADIUS = 0.008   # Spot feature label thickness
PIN_LABEL_RADIUS = 0.012    # Pin digit label thickness
CAMERA_TYPE = "Pinhole"

# View orientation modes for different telescope configurations
# Each mode specifies: (vertical_flip, horizontal_flip)
# vertical_flip=True means S is up (N is down)
# horizontal_flip=True means E is left (W is right)
ORIENTATION_NSWE = "NSWE"  # Default: N up, S down, W left, E right
ORIENTATION_NSEW = "NSEW"  # N up, S down, E left, W right (horizontal flip)
ORIENTATION_SNEW = "SNEW"  # S up, N down, E left, W right (both flips so same as 180° rotation)
ORIENTATION_SNWE = "SNWE"  # S up, N down, W left, E right (vertical flip)

class Scene(NamedTuple):
    eye: NDArray
    target: NDArray
    up: NDArray
    light_pos: NDArray

def _rot_x(angle_deg: float) -> NDArray:
    """Rotation matrix around X axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle_deg: float) -> NDArray:
    """Rotation matrix around Y axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle_deg: float) -> NDArray:
    """Rotation matrix around Z axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def encode_camera_params(eye: list, target: list, up: list, fov: float) -> str:
    """
    Encode camera parameters into a compact base64 string.
    
    Packs 10 floats (eye[3], target[3], up[3], fov) into binary and base64 encodes.
    Uses URL-safe base64 (- and _ instead of + and /) for filename compatibility.
    
    Parameters
    ----------
    eye : list
        Camera eye position [x, y, z]
    target : list
        Camera target position [x, y, z]
    up : list
        Camera up vector [x, y, z]
    fov : float
        Field of view in degrees
        
    Returns
    -------
    str
        Base64-encoded camera parameters (URL-safe, no padding)
    """
    # Pack 10 floats: eye(3) + target(3) + up(3) + fov(1)
    packed = struct.pack('<10f', 
                         eye[0], eye[1], eye[2],
                         target[0], target[1], target[2],
                         up[0], up[1], up[2],
                         fov)
    # URL-safe base64 without padding (= chars)
    encoded = base64.urlsafe_b64encode(packed).decode('ascii').rstrip('=')
    return encoded

def run_renderer(dt_local: datetime,
                 lat: float,
                 lon: float,
                 elevation_file: str,
                 color_file: str,
                 starmap_file: str,
                 features_file: str,
                 downscale: int,
                 brightness: int,
                 app_name: str,
                 init_camera_params: Optional[CameraParams] = None,
                 time_step_minutes: int = 15,
                 init_view_orientation: str = ORIENTATION_NSWE) -> TkOptiX:
    """
    Quick function to render the Moon for a specific time and location.
    
    Parameters
    ----------
    dt_local : datetime
        Local time
    lat, lon : float
        Observer latitude and longitude in degrees
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
        Initial view orientation mode. One of:
        ORIENTATION_NSWE (default), ORIENTATION_NSEW, ORIENTATION_SNEW, ORIENTATION_SNWE
    Returns
    -------
    TkOptiX
        The renderer instance
    """

    print()
    print("Used PlotOptiX version:", plotoptix.__version__)
    print("Renderer started with parameters:")
    print(f"  Geographical Location: Lat {lat}°, Lon {lon}°")
    print(f"  Local Time: {dt_local}")
    print(f"  Elevation Map File: {elevation_file}")
    print(f"  Brightness: {brightness}")
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
        init_view_orientation=init_view_orientation
    )
    
    # Setup renderer
    moon_renderer.setup_renderer()
    
    # Set view
    moon_renderer.update_view(dt_local=dt_local, lat=lat, lon=lon)
    
    # Apply custom camera parameters if provided (to restore a saved view)
    if init_camera_params is not None:
        moon_renderer.apply_camera_params(init_camera_params)
    
    # Print info
    print("\n" + moon_renderer.get_info())
    print("\nKeys and mouse:")
    print("  G - Toggle selenographic grid")
    print("  L - Toggle standard labels")
    print("  S - Toggle spot labels")
    print("  P - Toggle pins ON/OFF")
    print("  Y - Toggle Moon data panel")
    print("  F5-F8 - Switch view orientation (NSWE, NSEW, SNEW, SNWE)")
    print("  1-9 - Create/Remove pin (when pins are ON)")
    print("  R - Reset view and time to initial state")
    print("  V - Reset view to that based on current time (useful after starting with --init-view parameter)")
    print("  C - Center and fix view on point under cursor")
    print("  F - Search for Moon features (craters, mounts etc.)")
    print("  T - Open date/time window")
    print("  F12 - Save image")
    print("  Arrows - Navigate view")
    print("  A/Z - Increase/Decrease brightness")
    print("  Q/W - Go back/forward in time by step minutes")
    print("  M/N - Increase/Decrease time step by 1 minute (max is 1440 - 1 day)")
    print("  Shift + M/N - Increase/Decrease time step by 60 minutes (max is 1440 - 1 day)")
    print("  Ctrl + Left/Right - Rotate view around Moon's polar axis")
    print("  Ctrl + Up/Down - Rotate view around Moon's equatorial axis")
    print("  Hold and drag left mouse button - Rotate the eye around Moon")
    print("  Hold and drag right mouse button - Rotate Moon around the eye")
    print("  Hold shift + right mouse button and drag up/down - Move eye backward/forward")
    print("  Hold shift + left mouse button and drag up/down - Zoom out/in (more reliable)")
    print("  Hold ctrl + drag left mouse button - Measure distance on Moon surface")
    print("  Mouse wheel up/down - Zoom in/out (less reliable)")
    
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
        elif event.keysym.lower() == 'm':
            step = 60 if event.state & 0x1 else 1  # Shift key pressed
            moon_renderer.change_time_step(step)
        elif event.keysym.lower() == 'n':
            step = 60 if event.state & 0x1 else 1  # Shift key pressed
            moon_renderer.change_time_step(-step)
        elif event.keysym.lower() == 'y':
            moon_renderer.toggle_info_panel()
        elif event.keysym.lower() == 'p':
            moon_renderer.toggle_pins()
        elif event.keysym.lower() == 'q':
            moon_renderer.change_time(-moon_renderer.time_step_minutes)
        elif event.keysym.lower() == 'w':
            moon_renderer.change_time(moon_renderer.time_step_minutes)
        elif event.keysym.lower() == 't':
            moon_renderer.open_datetime_dialog()
        elif event.keysym in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
            moon_renderer.toggle_pin_at_cursor(event, int(event.keysym))
        else:
            original_key_handler(event)
    moon_renderer.rt._gui_key_pressed = custom_key_handler
    
    # Override mouse motion handler to show selenographic coordinates
    original_motion_handler = moon_renderer.rt._gui_motion
    def custom_motion_handler(event):
        # Call original handler first to update default status
        original_motion_handler(event)
        
        # Only update if not dragging
        if not (moon_renderer.rt._any_mouse or moon_renderer.rt._any_key):
            x, y = moon_renderer.rt._get_image_xy(event.x, event.y)
            
            # Get hit position using the internal method
            hx, hy, hz, hd = moon_renderer.rt._get_hit_at(x, y)
            
            lat = None
            lon = None
            feature_text = ""
            # Check if we hit something (distance > 0 means valid hit)
            if hd > 0:
                lat, lon = moon_renderer.hit_to_selenographic(hx, hy, hz)
                if lat is not None and lon is not None:
                    # Check if hovering over a named feature
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
        """Handle left mouse button press - start measurement if Ctrl is held."""
        if event.state & 0x4:  # Ctrl key pressed
            moon_renderer.start_measurement(event)
            return  # Don't call original handler when measuring
        original_pressed_left(event)
    
    def custom_released_left(event):
        """Handle left mouse button release - finish measurement if measuring."""
        if moon_renderer.measuring:
            moon_renderer.finish_measurement(event)
            return  # Don't call original handler when finishing measurement
        original_released_left(event)
    
    def custom_motion_pressed(event):
        """Handle mouse motion while pressed - update leading line if measuring."""
        if moon_renderer.measuring:
            moon_renderer.update_leading_line(event)
            return  # Don't call original handler when measuring
        original_motion_pressed(event)
    
    moon_renderer.rt._gui_pressed_left = custom_pressed_left
    moon_renderer.rt._gui_released_left = custom_released_left
    moon_renderer.rt._gui_motion_pressed = custom_motion_pressed
    
    moon_renderer.start()
    return moon_renderer.rt

def calculate_rotation(z: float, x: float, y: float) -> NDArray:
    """Calculate combined rotation matrix by applying rotations in order: Z, X, Y.
    
    Parameters
    ----------
    z, x, y : float
        Rotation angles in degrees
        
    Returns
    -------
    NDArray
        Combined rotation matrix
    """
    return _rot_y(y) @ _rot_x(x) @ _rot_z(z)

def calculate_camera_and_light(moon_ephem: MoonEphemeris, zoom: float, moon_radius: float) -> dict:
    """
    Calculate camera position and light direction for the renderer.
    
    Scene coordinate system:
    - Moon is at origin
    - Camera looks along +Y axis toward the Moon
    - +X is to the RIGHT in the view
    - +Z is UP in the view (toward zenith)
    
    Parameters
    ----------
    moon_ephem : MoonEphemeris
        Moon ephemeris
    zoom : float
        Camera zoom factor (distance multiplier)
    moon_radius : float
        Radius of the Moon
        
    Returns
    -------
    dict
        Camera eye position, target, up vector, and light position
    """
    camera_distance = moon_radius * (zoom / 100)
    
    # Camera setup - looking along +Y axis toward Moon at origin
    camera_eye = np.array([0, -camera_distance, 0])
    camera_target = np.array([0, 0, 0])
    camera_up = np.array([0, 0, 1])
    
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
    bright_limb_angle_deg = moon_ephem.pa - moon_ephem.q
    
    # Normalize to -180 to 180
    while bright_limb_angle_deg > 180: bright_limb_angle_deg -= 360
    while bright_limb_angle_deg < -180: bright_limb_angle_deg += 360
    
    bright_limb_angle = np.radians(bright_limb_angle_deg)
    phase = np.radians(moon_ephem.phase)
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
    # - Only affects when phase < 6° (very close to full moon)
    # - At 6° phase, illumination = (1 + cos(6°))/2 ≈ 99.7% (vs 100% at true full)
    # - A ~0.3% sliver at the Moon's edge would be in shadow, visually imperceptible
    # - Most of the lunar cycle (phase > 6°) is completely unaffected
    min_phase_offset = np.radians(6.0)
    # Only apply minimum offset near full moon (phase < 6°), not near new moon
    effective_sin_phase = np.sin(min_phase_offset if phase < min_phase_offset else phase)
    
    light_x = -np.sin(bright_limb_angle) * effective_sin_phase * light_distance
    light_z = np.cos(bright_limb_angle) * effective_sin_phase * light_distance
    light_y = -np.cos(phase) * light_distance
    
    light_pos = np.array([light_x, light_y, light_z])

    return Scene(eye=camera_eye, target=camera_target, up=camera_up, light_pos=light_pos)

class MoonRenderer:
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
                 init_view_orientation: str = ORIENTATION_NSWE):
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
            Elevation downscale factor
        width, height : int
            Render window size
        time_step_minutes : int
            Time step in minutes for Q/W keys
        init_view_orientation : str
            Initial view orientation (ORIENTATION_NSWE, ORIENTATION_NSEW, etc.)
        """
        self.width = width
        self.height = height
        self.downscale = downscale
        self.gamma = 2.2
        self.time_step_minutes = time_step_minutes
        
        # Load data
        self.elevation = load_elevation_data(elevation_file, downscale)
        self.color_data = load_color_data(color_file, self.gamma)
        # Sort features by angular_radius (smallest first) for efficient lookup in find_feature_for_status_bar()
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
        # This is the view without any --init-view override
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
        
        # Flag to track if search dialog is open
        self.search_dialog_open = False
        
        # Datetime dialog tracking
        self.datetime_dialog = None
        self.datetime_dialog_focused = False
        
        # Pins settings
        self.pins_visible = True  # Pins visible by default
        self.pins = {}  # dict mapping digit (1-9) to pin segments

        # Distance measurement settings
        self.measuring = False  # True when Ctrl+drag is in progress
        self.measure_start_canvas = None  # Start position (x, y) on canvas for leading line
        self.measure_start_coords = None  # Start selenographic coordinates (lat, lon)
        self.leading_line_id = None  # Canvas line ID for the leading line
        self.measured_distance = None  # Last measured distance in km

        # Status bar panel variables (set up as StringVars after renderer is created)
        self._status_observer_var = None
        self._status_view_var = None
        self._status_time_var = None
        self._status_measured_var = None
        self._status_feature_var = None
        self._status_brightness_var = None
        self._status_pins_var = None

        # Info panel variables (bottom-left overlay, set up as StringVars after renderer is created)
        self._info_frame = None
        self.show_info_panel = True
        self._info_az_var = None
        self._info_alt_var = None
        self._info_ra_var = None
        self._info_dec_var = None
        self._info_phase_var = None
        self._info_distance_var = None
        self._info_libr_l_var = None
        self._info_libr_b_var = None
        self._info_lat_var = None
        self._info_lon_var = None

    # ---- Status panel update methods ----

    def _update_status_observer(self):
        if self._status_observer_var:
            if self.observer_lat is not None and self.observer_lon is not None:
                lat_dir = 'N' if self.observer_lat >= 0 else 'S'
                lon_dir = 'E' if self.observer_lon >= 0 else 'W'
                self._status_observer_var.set(
                    f"Observer: {abs(self.observer_lat):.3f}\u00b0{lat_dir} {abs(self.observer_lon):.3f}\u00b0{lon_dir}")
            else:
                self._status_observer_var.set("Observer:")

    def _update_status_view(self):
        if self._status_view_var:
            self._status_view_var.set(f"View: {self.orientation_mode}")

    def _update_status_time(self):
        if self._status_time_var and self.dt_local:
            offset = self.dt_local.strftime('%z')
            offset_fmt = f"{offset[:3]}:{offset[3:]}" if offset else ""
            self._status_time_var.set(
                f"Time: {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}{offset_fmt} (step {self.time_step_minutes} min)")

    def _update_info_moon(self):
        """Update the info panel with current Moon ephemeris data."""
        if self.moon_ephem is None:
            return
        e = self.moon_ephem
        if self._info_az_var:
            self._info_az_var.set(f"Azimuth:  {e.az:6.2f}°")
        if self._info_alt_var:
            self._info_alt_var.set(f"Altitude: {e.alt:+6.2f}°")
        if self._info_ra_var:
            ra_total_h = e.ra / 15.0
            ra_h = int(ra_total_h)
            ra_m = int((ra_total_h - ra_h) * 60)
            ra_s = (ra_total_h - ra_h - ra_m / 60) * 3600
            self._info_ra_var.set(f"RA:  {ra_h:02d}h{ra_m:02d}m{ra_s:04.1f}s")
        if self._info_dec_var:
            dec_sign = '+' if e.dec >= 0 else '-'
            dec_abs = abs(e.dec)
            dec_d = int(dec_abs)
            dec_m = int((dec_abs - dec_d) * 60)
            dec_s = (dec_abs - dec_d - dec_m / 60) * 3600
            self._info_dec_var.set(f"DEC: {dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\"")
        if self._info_phase_var:
            self._info_phase_var.set(f"Phase ∠: {e.phase:6.2f}°")
        if self._info_distance_var:
            self._info_distance_var.set(f"Dist: {e.distance:,.0f} km".replace(",", " "))
        if self._info_libr_l_var:
            self._info_libr_l_var.set(f"Libr L: {e.libr_long:+5.2f}°")
        if self._info_libr_b_var:
            self._info_libr_b_var.set(f"Libr B: {e.libr_lat:+5.2f}°")

    def _update_status_measured(self):
        if self._status_measured_var:
            if self.measured_distance is not None:
                self._status_measured_var.set(f"Measured: {self.measured_distance:7.2f} km")
            else:
                self._status_measured_var.set("")

    def _update_info_coords(self, lat=None, lon=None):
        """Update latitude/longitude in the info panel."""
        if self._info_lat_var:
            if lat is not None:
                lat_dir = 'N' if lat >= 0 else 'S'
                self._info_lat_var.set(f"Lat: {abs(lat):5.2f}° {lat_dir}")
            else:
                self._info_lat_var.set("Lat:")
        if self._info_lon_var:
            if lon is not None:
                lon_dir = 'E' if lon >= 0 else 'W'
                self._info_lon_var.set(f"Lon: {abs(lon):6.2f}° {lon_dir}")
            else:
                self._info_lon_var.set("Lon:")

    def _update_status_feature(self, feature_text: str = ""):
        """Update feature name in the status bar."""
        if self._status_feature_var:
            self._status_feature_var.set(feature_text)

    def _update_status_brightness(self):
        if self._status_brightness_var:
            self._status_brightness_var.set(f"Brightness: {self.brightness}")

    def _update_status_pins(self):
        if self._status_pins_var:
            self._status_pins_var.set(f"Pins {'ON' if self.pins_visible else 'OFF'}")

    def _update_all_status_panels(self):
        self._update_status_observer()
        self._update_status_view()
        self._update_status_time()
        self._update_status_measured()
        self._update_status_feature()
        self._update_status_brightness()
        self._update_status_pins()
        self._update_info_moon()
        self._update_info_coords()
    
    def set_orientation(self, orientation: str):
        """
        Set the view orientation mode and update the status bar.
        
        Called when F5-F8 keys are pressed to match plotoptix internal orientation change.
        
        Parameters
        ----------
        orientation : str
            One of ORIENTATION_NSWE, ORIENTATION_NSEW, ORIENTATION_SNEW, ORIENTATION_SNWE
        """
        self.orientation_mode = orientation
        
        # Update grid labels if grid is visible
        if self.moon_grid is not None and self.moon_grid_visible:
            self.update_grid_labels_for_orientation()
        
        # Update standard labels if visible
        if self.standard_labels is not None and self.standard_labels_visible:
            self.update_standard_labels_for_view_orientation()
        
        # Update spot labels if visible
        if self.spot_labels is not None and self.spot_labels_visible:
            self.update_spot_labels_for_view_orientation()
        
        self._update_status_view()

    def update_grid_labels_for_orientation(self):
        """
        Update grid number labels to match current view orientation.
        
        Regenerates latitude and longitude number labels so they are
        always readable (not upside down) in the current view orientation.
        """
        if self.rt is None or self.moon_grid is None:
            return
        
        # Determine flip flags based on orientation
        # NSWE (default): N up, W left - no flips
        # NSEW: N up, E left - horizontal flip
        # SNEW: S up, E left - both flips (180° rotation)
        # SNWE: S up, W left - vertical flip
        flip_horizontal = self.orientation_mode in (ORIENTATION_NSEW, ORIENTATION_SNEW)
        flip_vertical = self.orientation_mode in (ORIENTATION_SNEW, ORIENTATION_SNWE)
        
        # Generate new labels with proper orientation
        lat_labels, lat_label_values, lon_labels, lon_label_values = create_grid_labels_for_orientation(
            moon_radius=self.moon_radius,
            lat_step=15.0,
            lon_step=15.0,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        
        # Update the moon_grid with new labels
        self.moon_grid = self.moon_grid._replace(
            lat_labels=lat_labels,
            lat_label_values=lat_label_values,
            lon_labels=lon_labels,
            lon_label_values=lon_label_values
        )
        
        # Get rotation matrix
        R = self.moon_rotation
        
        # Update latitude labels in renderer
        for i, segments in enumerate(self.moon_grid.lat_labels):
            for j, seg in enumerate(segments):
                name = f"grid_lat_label_{i}_{j}"
                if R is not None:
                    rotated = (R @ seg.T).T
                else:
                    rotated = seg
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
        
        # Update longitude labels in renderer
        for i, segments in enumerate(self.moon_grid.lon_labels):
            for j, seg in enumerate(segments):
                name = f"grid_lon_label_{i}_{j}"
                if R is not None:
                    rotated = (R @ seg.T).T
                else:
                    rotated = seg
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass

    def update_standard_labels_for_view_orientation(self):
        """
        Update standard labels to match current view orientation.
        
        Regenerates standard labels so they are always readable
        (not upside down) in the current view orientation.
        """
        if self.rt is None or self.standard_labels is None or self.standard_label_features is None:
            return
        
        # Determine flip flags based on orientation
        flip_horizontal = self.orientation_mode in (ORIENTATION_NSEW, ORIENTATION_SNEW)
        flip_vertical = self.orientation_mode in (ORIENTATION_SNEW, ORIENTATION_SNWE)
        
        # Regenerate labels with proper orientation
        self.standard_labels = create_standard_labels(
            self.standard_label_features,
            moon_radius=self.moon_radius,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        
        # Get rotation matrix
        R = self.moon_rotation
        
        # Update labels in renderer
        for i, label in enumerate(self.standard_labels):
            feature = self.standard_label_features[i]
            label_radius = STANDARD_LABEL_RADIUS if self._is_feature_illuminated(feature) else 0.0
            for j, seg in enumerate(label.segments):
                name = f"standard_label_{i}_{j}"
                if R is not None:
                    rotated = (R @ seg.T).T
                else:
                    rotated = seg
                try:
                    self.rt.update_data(name, pos=rotated, r=label_radius)
                except:
                    pass

    def update_spot_labels_for_view_orientation(self):
        """
        Update spot labels to match current view orientation.
        
        Regenerates spot labels so they are always readable
        (not upside down) in the current view orientation.
        """
        if self.rt is None or self.spot_labels is None or self.spot_label_features is None:
            return
        
        # Determine flip flags based on orientation
        flip_horizontal = self.orientation_mode in (ORIENTATION_NSEW, ORIENTATION_SNEW)
        flip_vertical = self.orientation_mode in (ORIENTATION_SNEW, ORIENTATION_SNWE)
        
        # Regenerate labels with proper orientation
        self.spot_labels = create_spot_labels(
            self.spot_label_features,
            moon_radius=self.moon_radius,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        
        # Get rotation matrix
        R = self.moon_rotation
        
        # Update labels in renderer
        for i, label in enumerate(self.spot_labels):
            feature = self.spot_label_features[i]
            label_radius = SPOT_LABEL_RADIUS if self._is_feature_illuminated(feature) else 0.0
            for j, seg in enumerate(label.segments):
                name = f"spot_label_{i}_{j}"
                if R is not None:
                    rotated = (R @ seg.T).T
                else:
                    rotated = seg
                try:
                    self.rt.update_data(name, pos=rotated, r=label_radius)
                except:
                    pass

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
        self._update_status_time()

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
        
        # Calculate new time
        new_dt_local = self.dt_local + timedelta(minutes=delta_minutes)
        
        # Update Moon orientation and lighting for new time (without moving camera)
        self.update_moon_for_time(new_dt_local, self.observer_lat, self.observer_lon)
        
        # Regenerate grid and labels with new orientation
        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()
        if self.spot_labels_visible:
            self.update_spot_labels_orientation()
        
        # Update pins positions
        self.update_pins_orientation()
        
        # Update status bar
        self._update_status_time()
        self._update_info_moon()
        
    def _on_launch_finished(self, rt):
        """Callback to maximize window and set title on first launch."""
        if not self._window_maximized:
            self._window_maximized = True
            # Schedule maximize and title change on the main thread
            def init_window():
                rt._root.state('zoomed')
                rt._root.title(self.app_name)

                # Hide FPS panel from status bar
                if hasattr(rt, '_status_fps'):
                    rt._status_fps.grid_remove()

                # Build multi-panel status bar replacing the single label
                if hasattr(rt, '_status_action'):
                    grid_info = rt._status_action.grid_info()
                    parent = rt._status_action.master
                    rt._status_action.grid_remove()

                    status_frame = tk.Frame(parent)

                    self._status_observer_var = tk.StringVar()
                    self._status_view_var = tk.StringVar()
                    self._status_time_var = tk.StringVar()
                    self._status_measured_var = tk.StringVar()
                    self._status_feature_var = tk.StringVar()
                    self._status_brightness_var = tk.StringVar()
                    self._status_pins_var = tk.StringVar()

                    font = ("Consolas", 9)
                    panels = [
                        (self._status_pins_var,        8),
                        (self._status_brightness_var, 15),
                        (self._status_feature_var,    50),
                        (self._status_measured_var,   20),
                        (self._status_time_var,       47),
                        (self._status_view_var,       10),
                        (self._status_observer_var,   27)
                    ]
                    for var, w in panels:
                        tk.Label(
                            status_frame,
                            textvariable=var,
                            font=font,
                            anchor='w',
                            width=w,
                            relief='sunken',
                            borderwidth=1,
                        ).pack(side='right', padx=24)

                # Build info panel (bottom-left overlay on canvas)
                if hasattr(rt, '_canvas'):
                    info_font = ("Consolas", 9)
                    info_fg = "#808080"
                    info_bg = "#010104"
                    info_width = 17  # Fixed width in chars (fits DEC: +89°59'59.9")

                    self._info_az_var = tk.StringVar(value="Az:")
                    self._info_alt_var = tk.StringVar(value="Alt:")
                    self._info_ra_var = tk.StringVar(value="RA:")
                    self._info_dec_var = tk.StringVar(value="DEC:")
                    self._info_phase_var = tk.StringVar(value="Ph:")
                    self._info_distance_var = tk.StringVar(value="Dist:")
                    self._info_libr_l_var = tk.StringVar(value="LbL:")
                    self._info_libr_b_var = tk.StringVar(value="LbB:")
                    self._info_lat_var = tk.StringVar(value="Lat:")
                    self._info_lon_var = tk.StringVar(value="Lon:")

                    info_frame = tk.Frame(rt._canvas, bg=info_bg, padx=6, pady=4)
                    self._info_frame = info_frame
                    info_vars = [
                        self._info_az_var,
                        self._info_alt_var,
                        self._info_ra_var,
                        self._info_dec_var,
                        self._info_phase_var,
                        self._info_distance_var,
                        self._info_libr_l_var,
                        self._info_libr_b_var,
                        self._info_lat_var,
                        self._info_lon_var,
                    ]
                    for var in info_vars:
                        tk.Label(
                            info_frame,
                            textvariable=var,
                            font=info_font,
                            fg=info_fg,
                            bg=info_bg,
                            anchor='w',
                            width=info_width,
                        ).pack(anchor='w')
                    info_frame.place(relx=0.0, rely=1.0, anchor='sw', x=6, y=-6)

                # Add 4-char left padding to shift panels right
                status_frame.grid(
                    row=int(grid_info['row']),
                    column=int(grid_info['column']),
                    columnspan=int(grid_info.get('columnspan', 1)),
                    sticky='we',
                    padx=(4, 0), pady=0
                )

                # Bind mouse wheel for zoom
                if hasattr(rt, '_canvas'):
                    rt._canvas.bind('<MouseWheel>', self._mouse_wheel_handler)
                
                # Apply initial view orientation to plotoptix
                if self.orientation_mode != ORIENTATION_NSWE:
                    rt._view_orientation = self.orientation_mode
                    # Update grid labels for initial orientation if grid exists
                    if self.moon_grid is not None and self.moon_grid_visible:
                        self.update_grid_labels_for_orientation()
                
                self._update_all_status_panels()
            rt._root.after_idle(init_window)
    
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
        # u = Moon's north pole direction (rotation axis)
        # v = direction where longitude 0° (center of nearside) faces
        # Camera is at -Y, looking toward origin
        # v = [0, -1, 0] so longitude 0° (nearside) faces camera at -Y
        self.rt.set_data("moon", geom="ParticleSetTextured", geom_attr="DisplacedSurface",
                        pos=[0, 0, 0], u=[0, 0, 1], v=[0, -1, 0], r=10)
        
        # Apply displacement map
        self.rt.set_displacement("moon", self.elevation, refresh=True)

    def update_view(self, dt_local: datetime, lat: float, lon: float, zoom: float = 1000):
        """
        Update the view for a specific time and location.
        
        Parameters
        ----------
        dt_local : datetime
            Local time
        lat, lon : float
            Observer latitude and longitude in degrees
        zoom : float
            Camera zoom factor
        """

        dt_utc = dt_local.astimezone(timezone.utc)
        eph = calculate_moon_ephemeris(dt_utc, lat, lon)
        self.moon_rotation = calculate_rotation(-eph.libr_long, eph.libr_lat, eph.pa_axis_view)
        self.moon_rotation_inv = self.moon_rotation.T  # For orthogonal matrices, inverse = transpose
        self.moon_ephem = eph
        
        # Store view parameters for filename generation
        self.dt_local = dt_local
        self.observer_lat = lat
        self.observer_lon = lon
        
        scene = calculate_camera_and_light(self.moon_ephem, zoom, self.moon_radius)
        # Cache light position for label illumination decisions
        self.light_pos = scene.light_pos

        # The u,v vectors define how the texture is mapped onto the sphere
        # u = north pole direction
        # v = longitude 0° direction (orthogonalized to u)
        # Base orientation (no libration, looking from -Y toward Moon at origin):
        # - Moon's north pole points to +Z (up in view)
        # - Moon's prime meridian (lon=0) faces toward -Y (toward camera)
        # - u = [0, 0, 1] (north pole)
        # - v = [0, -1, 0] (prime meridian faces -Y)
        u_new = self.moon_rotation @ np.array([0.0, 0.0, 1.0])
        v_new = self.moon_rotation @ np.array([0.0, -1.0, 0.0])

        # Update Moon geometry with new orientation
        self.rt.update_data("moon", u=u_new.tolist(), v=v_new.tolist())
        
        # Update camera
        # Calculate FOV so moon fills MOON_FILL_FRACTION (90%) of window height
        # Moon diameter = 2 * moon_radius, visible_height = diameter / fill_fraction
        # FOV = 2 * atan(visible_height / (2 * camera_distance))
        camera_distance = self.moon_radius * (zoom / 100)
        moon_diameter = 2 * self.moon_radius
        visible_height = moon_diameter / MOON_FILL_FRACTION
        fov = np.degrees(2 * np.arctan(visible_height / (2 * camera_distance)))
        fov = max(1, min(90, fov))  # Clamp to valid range
        
        self.rt.setup_camera("cam1",
                             cam_type=CAMERA_TYPE,
                             eye=scene.eye.tolist(),
                             target=scene.target.tolist(),
                             up=scene.up.tolist(),
                             aperture_radius=0.01,
                             aperture_fract=0.2,
                             focal_scale=0.7,
                             fov=fov)
        
        # Always store default camera params (the view calculated from ephemeris)
        self.default_camera_params = CameraParams(eye=scene.eye.tolist(), target=scene.target.tolist(), up=scene.up.tolist(), fov=fov)

        # Store initial camera parameters and time for reset functionality
        if self.initial_camera_params is None:
            self.initial_camera_params = self.default_camera_params
            self.initial_dt_local = dt_local
        
        # Brightness based on phase - full moon is brighter
        # brightness = 40 + 20 * np.cos(np.radians(self.moon_ephem.phase))
        
        self.rt.setup_light("sun", pos=scene.light_pos.tolist(), color=self.brightness, radius=SUN_RADIUS)
        
        # Update grid orientation if visible
        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        
        # Update standard labels orientation if visible
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()

    def update_moon_for_time(self, dt_local: datetime, lat: float, lon: float):
        """
        Update Moon orientation and lighting for a new time without changing camera.
        
        This method updates the Moon's libration, position angle, and Sun illumination
        for a new observation time, but preserves the current camera position/orientation.
        
        Parameters
        ----------
        dt_local : datetime
            Local time
        lat, lon : float
            Observer latitude and longitude in degrees
        """
        dt_utc = dt_local.astimezone(timezone.utc)
        eph = calculate_moon_ephemeris(dt_utc, lat, lon)
        self.moon_rotation = calculate_rotation(-eph.libr_long, eph.libr_lat, eph.pa_axis_view)
        self.moon_rotation_inv = self.moon_rotation.T
        self.moon_ephem = eph
        
        # Store view parameters for filename generation
        self.dt_local = dt_local
        self.observer_lat = lat
        self.observer_lon = lon
        
        # Calculate new light position based on current ephemeris
        scene = calculate_camera_and_light(self.moon_ephem, 1000, self.moon_radius)
        self.light_pos = scene.light_pos
        
        # Update default camera params for V key reset
        # Use previously stored FOV (from initial setup or last update_view)
        current_fov = self.default_camera_params.fov if self.default_camera_params else 45.0
        self.default_camera_params = CameraParams(
            eye=scene.eye.tolist(), 
            target=scene.target.tolist(), 
            up=scene.up.tolist(), 
            fov=current_fov
        )
        
        # Update Moon orientation (u,v vectors)
        u_new = self.moon_rotation @ np.array([0.0, 0.0, 1.0])
        v_new = self.moon_rotation @ np.array([0.0, -1.0, 0.0])
        self.rt.update_data("moon", u=u_new.tolist(), v=v_new.tolist())
        
        # Update light position
        self.rt.setup_light("sun", pos=scene.light_pos.tolist(), color=self.brightness, radius=SUN_RADIUS)
        
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
        
        # Update initial camera params so reset works correctly
        self.initial_camera_params = params
        
        print(f"Applied camera params: eye={params.eye}, target={params.target}, up={params.up}, fov={params.fov:.2f}")
    
    def get_default_filename(self) -> str:
        """
        Generate a default filename for saving screenshots.
        
        Format: datetime_lat+XX.XXXXXX_lon+XX.XXXXXX_view<orientation>_cam<base64>
        
        The camera parameters (eye, target, up, fov) are encoded into a compact
        base64 string for a shorter filename while remaining fully reversible.
        
        Returns
        -------
        str
            Default filename (without extension)
        """
        parts = []
        
        # 1. Local time in ISO format (replace colons with dots for filename compatibility)
        if self.dt_local is not None:
            # Format: YYYY-MM-DDTHH.MM.SS+HH.MM (colons replaced with dots)
            iso_str = self.dt_local.isoformat()
            iso_str = iso_str.replace(':', '.')
            parts.append(iso_str)
        else:
            parts.append("notime")
        
        # 2. Latitude
        if self.observer_lat is not None:
            parts.append(f"lat{self.observer_lat:+.6f}")
        else:
            parts.append("latnone")
        
        # 3. Longitude
        if self.observer_lon is not None:
            parts.append(f"lon{self.observer_lon:+.6f}")
        else:
            parts.append("lonnone")
        
        # 4. View orientation
        parts.append(f"view{self.orientation_mode}")
        
        # 5. Current camera parameters (at the time of screenshot) - encoded as base64
        if self.rt is not None:
            try:
                cam = self.rt.get_camera("cam1")
                if cam is not None:
                    eye = cam["Eye"]
                    target = cam["Target"]
                    up = cam["Up"]
                    # Get FOV using the internal method (more reliable than dictionary lookup)
                    fov = self.rt._optix.get_camera_fov(0)
                    
                    # Encode camera params into compact base64 string
                    cam_encoded = encode_camera_params(eye, target, up, fov)
                    parts.append(f"cam{cam_encoded}")
                else:
                    parts.append("nocam")
            except Exception as e:
                print(f"Error getting camera: {e}")
                parts.append("nocam")
        else:
            parts.append("nocam")
        
        return "_".join(parts)
    
    def save_image_dialog(self):
        """
        Open a save dialog with a custom default filename.
        """
        if self.rt is None:
            return
        
        default_name = self.get_default_filename()
        
        filename = filedialog.asksaveasfilename(
            initialdir=".",
            title="Save output as image",
            initialfile=f"{default_name}.jpg",
            defaultextension=".jpg",
            filetypes=(
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("TIFF 8-bit files", "*.tif"),
                ("TIFF 16-bit files", "*.tiff")
            )
        )
        if filename:
            fname, fext = os.path.splitext(filename)
            if fext.lower() == ".tiff":
                self.rt.save_image(filename, bps="Bps16")
            else:
                self.rt.save_image(filename, bps="Bps8")
            print(f"Saved: {filename}")
    
    def search_feature_dialog(self):
        """
        Open a search dialog to find Moon features by name.
        """
        if self.rt is None:
            return
        
        # Set flag to prevent main window key handling
        self.search_dialog_open = True
        
        # Create search window
        search_win = tk.Toplevel(self.rt._root)
        search_win.title("Search Moon Feature")
        search_win.geometry("400x300")
        search_win.transient(self.rt._root)
        search_win.grab_set()
        
        def on_close():
            self.search_dialog_open = False
            search_win.destroy()
        
        search_win.protocol("WM_DELETE_WINDOW", on_close)
        
        # Search entry
        frame = tk.Frame(search_win)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(frame, text="Search:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        entry = tk.Entry(frame, textvariable=search_var, width=40)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        entry.focus_set()
        
        # Results listbox with scrollbar
        list_frame = tk.Frame(search_win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Store matching features
        matching_features = []
        
        def update_results(*args):
            nonlocal matching_features
            query = search_var.get().lower().strip()
            listbox.delete(0, tk.END)
            matching_features.clear()
            
            if not query:
                return
            
            for feature in self.moon_features:
                if query in feature.name.lower():
                    matching_features.append(feature)
                    size_km = feature.size_km
                    listbox.insert(tk.END, f"{feature.name} ({size_km:.1f} km)")
        
        def on_select(event=None):
            selection = listbox.curselection()
            if selection and matching_features:
                feature = matching_features[selection[0]]
                self.center_on_feature(feature)
                on_close()
        
        def on_key(event):
            if event.keysym == 'Return':
                # If listbox has selection, use it; otherwise select first
                if not listbox.curselection() and listbox.size() > 0:
                    listbox.selection_set(0)
                on_select()
            elif event.keysym == 'Escape':
                on_close()
            elif event.keysym == 'Down':
                if listbox.size() > 0:
                    listbox.focus_set()
                    if not listbox.curselection():
                        listbox.selection_set(0)
        
        search_var.trace_add('write', update_results)
        entry.bind('<Key>', on_key)
        listbox.bind('<Double-Button-1>', on_select)
        listbox.bind('<Return>', on_select)
        
        # Center the window
        search_win.update_idletasks()
        x = self.rt._root.winfo_x() + (self.rt._root.winfo_width() - search_win.winfo_width()) // 2
        y = self.rt._root.winfo_y() + (self.rt._root.winfo_height() - search_win.winfo_height()) // 2
        search_win.geometry(f"+{x}+{y}")
    
    def open_datetime_dialog(self):
        """
        Open a dialog to set date, time, and timezone.
        The dialog stays open and syncs with Q/W key time changes.
        """
        if self.rt is None:
            return
        
        # If already open, just bring it to front
        if self.datetime_dialog is not None and self.datetime_dialog.winfo_exists():
            self.datetime_dialog.lift()
            self.datetime_dialog.focus_set()
            return
        
        # Create datetime window (non-modal, stays open)
        dt_win = tk.Toplevel(self.rt._root)
        dt_win.title("Date/Time")
        dt_win.geometry("360x130")
        dt_win.transient(self.rt._root)
        dt_win.resizable(False, False)
        
        self.datetime_dialog = dt_win
        
        def on_close():
            self.datetime_dialog = None
            self.datetime_dialog_focused = False
            dt_win.destroy()
        
        def on_focus_in(event):
            self.datetime_dialog_focused = True
        
        def on_focus_out(event):
            self.datetime_dialog_focused = False
        
        dt_win.protocol("WM_DELETE_WINDOW", on_close)
        dt_win.bind("<FocusIn>", on_focus_in)
        dt_win.bind("<FocusOut>", on_focus_out)
        
        # Main frame with padding
        main_frame = tk.Frame(dt_win, padx=15, pady=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Get current local time and its timezone for later use
        current_dt_local = self.dt_local
        local_tz = current_dt_local.tzinfo
        
        # Date and Time rows using grid for proper alignment
        grid_frame = tk.Frame(main_frame)
        grid_frame.pack(fill=tk.X, pady=3)
        
        # Format timezone offset as +HH:MM or -HH:MM
        offset = current_dt_local.strftime('%z')  # e.g., +0100
        offset_formatted = f"{offset[:3]}:{offset[3:]}" if offset else ""  # e.g., +01:00
        
        # Date row
        tk.Label(grid_frame, text="Date:", anchor='w').grid(row=0, column=0, sticky='e', pady=2)
        date_var = tk.StringVar(value=current_dt_local.strftime('%Y-%m-%d'))
        date_entry = tk.Entry(grid_frame, textvariable=date_var, width=15)
        date_entry.grid(row=0, column=1, padx=5, pady=2)
        tk.Label(grid_frame, text="(YYYY-MM-DD)", fg='gray').grid(row=0, column=2, sticky='w', pady=2)
        
        # Time row
        tk.Label(grid_frame, text=f"Local Time (UTC{offset_formatted}):", anchor='e').grid(row=1, column=0, sticky='w', pady=2)
        time_var = tk.StringVar(value=current_dt_local.strftime('%H:%M:%S'))
        time_entry = tk.Entry(grid_frame, textvariable=time_var, width=15)
        time_entry.grid(row=1, column=1, padx=5, pady=2)
        tk.Label(grid_frame, text="(HH:MM:SS)", fg='gray').grid(row=1, column=2, sticky='w', pady=2)
        
        # Error label
        error_var = tk.StringVar()
        error_label = tk.Label(main_frame, textvariable=error_var, fg='red')
        error_label.pack(fill=tk.X, pady=2)
        
        # Button frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        def go_to_time():
            """Apply the selected date/time in local timezone."""
            try:
                date_str = date_var.get().strip()
                time_str = time_var.get().strip()
                
                # Parse date and time
                dt_str = f"{date_str} {time_str}"
                try:
                    new_dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Try without seconds
                    new_dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
                
                # Apply the fixed local timezone
                new_dt_local = new_dt_naive.replace(tzinfo=local_tz)
                
                # Update the view
                self.update_moon_for_time(new_dt_local, self.observer_lat, self.observer_lon)
                
                # Regenerate grid and labels with new orientation
                if self.moon_grid_visible:
                    self.update_moon_grid_orientation()
                if self.standard_labels_visible:
                    self.update_standard_labels_orientation()
                if self.spot_labels_visible:
                    self.update_spot_labels_orientation()
                
                # Update pins positions
                self.update_pins_orientation()
                
                # Update status bar
                self._update_all_status_panels()
                
                error_var.set("")
                
            except Exception as e:
                error_var.set(f"Error: {str(e)}")
        
        def set_now():
            """Set to current local time."""
            now_local = datetime.now(local_tz)
            date_var.set(now_local.strftime('%Y-%m-%d'))
            time_var.set(now_local.strftime('%H:%M:%S'))
        
        def sync_from_renderer():
            """Sync dialog fields with current renderer time."""
            current_dt_local = self.dt_local
            date_var.set(current_dt_local.strftime('%Y-%m-%d'))
            time_var.set(current_dt_local.strftime('%H:%M:%S'))
        
        tk.Button(btn_frame, text="Now", command=set_now, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Sync with Moon", command=sync_from_renderer, width=16).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Set", command=go_to_time, width=10).pack(side=tk.RIGHT, padx=5)
        
        # Position near the top-right of the main window
        dt_win.update_idletasks()
        x = self.rt._root.winfo_x() + self.rt._root.winfo_width() - dt_win.winfo_width() - 50
        y = self.rt._root.winfo_y() + 100
        dt_win.geometry(f"+{x}+{y}")
        
        # Focus on time entry for quick editing
        time_entry.focus_set()
        time_entry.select_range(0, tk.END)

    def center_on_feature(self, feature: MoonFeature):
        """
        Center the view on a Moon feature and zoom to show it.
        
        Parameters
        ----------
        feature : MoonFeature
            The feature to center on
        """
        if self.rt is None or self.moon_rotation is None:
            return
        
        # Convert selenographic coordinates to 3D position
        lat_rad = np.radians(feature.lat)
        lon_rad = np.radians(feature.lon)
        
        # In original Moon coordinates:
        # - +Z is north pole
        # - -Y is prime meridian (lon=0)
        # - +X is east (lon=90)
        x = self.moon_radius * np.cos(lat_rad) * np.sin(lon_rad)
        y = -self.moon_radius * np.cos(lat_rad) * np.cos(lon_rad)
        z = self.moon_radius * np.sin(lat_rad)
        
        # Apply Moon rotation to get scene coordinates
        original_pos = np.array([x, y, z])
        scene_pos = self.moon_rotation @ original_pos
        
        # Get current camera
        cam = self.rt.get_camera("cam1")
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        
        # Calculate new camera distance based on feature size
        # Aim to have feature fill about 30% of the view
        feature_radius_scene = feature.angular_radius * (self.moon_radius / 90)  # Rough conversion
        current_fov = self.rt._optix.get_camera_fov(0)
        
        # Calculate distance to make feature appear at desired size
        desired_angular_size = current_fov * 0.3  # 30% of FOV
        new_distance = feature_radius_scene / np.tan(np.radians(desired_angular_size / 2))
        
        # Clamp distance to reasonable range
        min_dist = self.moon_radius * 1.1
        max_dist = self.moon_radius * 15
        new_distance = np.clip(new_distance, min_dist, max_dist)
        
        # Direction from target to eye
        direction = eye - target
        direction = direction / np.linalg.norm(direction)
        
        # New eye position
        new_target = scene_pos
        new_eye = new_target + direction * new_distance
        
        # Update camera
        self.rt.setup_camera("cam1", eye=new_eye.tolist(), target=new_target.tolist())
            
    def get_info(self) -> str:

        if self.moon_ephem is None:
            return "No view set"
        
        eph = self.moon_ephem
        return (f"Moon topocentric ephemeris:\n"
                f"  Altitude: {eph.alt:.2f}° {'(below horizon)' if eph.alt < 0 else ''}\n"
                f"  Azimuth: {eph.az:.2f}°\n"
                f"  RA: {eph.ra:.2f}°\n"
                f"  DEC: {eph.dec:.2f}°\n"
                f"  Distance: {eph.distance:.0f} km\n"
                f"  Phase: {eph.phase:.2f}°\n"
                f"  Illumination: {eph.illum:.2f}%\n"
                f"  Libration: L={eph.libr_long:+.2f}° B={eph.libr_lat:+.2f}°")
    
    def setup_moon_grid(self, lat_step: float = 15.0, lon_step: float = 15.0):
        """
        Create selenographic coordinate grid.
        
        Parameters
        ----------
        lat_step : float
            Spacing between latitude lines in degrees
        lon_step : float
            Spacing between longitude lines in degrees
        """
        if self.rt is None:
            print("Renderer not initialized")
            return
            
        # Generate grid data
        self.moon_grid = create_moon_grid(
            moon_radius=self.moon_radius,
            lat_step=lat_step,
            lon_step=lon_step,
            points_per_line=100,
            offset=0.0
        )
        
        # Create an emissive material for the grid lines (so they glow and are visible in shadow)
        m_grid = m_flat.copy()
        self.rt.update_material("grid_material", m_grid)
        
        # Add latitude lines
        for i, points in enumerate(self.moon_grid.lat_lines):
            name = f"grid_lat_{i}"
            self.rt.set_data(name, pos=points, r=GRID_LINE_RADIUS, 
                            c=GRID_COLOR, geom="BezierChain", mat="grid_material")
        
        # Add longitude lines
        for i, points in enumerate(self.moon_grid.lon_lines):
            name = f"grid_lon_{i}"
            self.rt.set_data(name, pos=points, r=GRID_LINE_RADIUS,
                            c=GRID_COLOR, geom="BezierChain", mat="grid_material")
        
        # Add latitude labels
        for i, segments in enumerate(self.moon_grid.lat_labels):
            for j, seg in enumerate(segments):
                name = f"grid_lat_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=GRID_LABEL_RADIUS,
                                c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        # Add longitude labels
        for i, segments in enumerate(self.moon_grid.lon_labels):
            for j, seg in enumerate(segments):
                name = f"grid_lon_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=GRID_LABEL_RADIUS,
                                c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        # Add north pole "N" label
        for j, seg in enumerate(self.moon_grid.N):
            name = f"grid_north_label_{j}"
            self.rt.set_data(name, pos=seg, r=GRID_LABEL_RADIUS,
                            c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        self.moon_grid_visible = True
        
        # Update labels for current view orientation if not default
        if self.orientation_mode != ORIENTATION_NSWE:
            self.update_grid_labels_for_orientation()
        
        self.update_moon_grid_orientation()
    
    def show_moon_grid(self, visible: bool = True):
        """
        Show or hide the selenographic grid.
        
        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return
        
        if self.moon_grid is None:
            if visible:
                self.setup_moon_grid()
            return
        
        # Toggle visibility by setting zero radius (hide) or restoring (show)
        line_radius = GRID_LINE_RADIUS if visible else 0.0
        
        for i in range(len(self.moon_grid.lat_lines)):
            name = f"grid_lat_{i}"
            try:
                self.rt.update_data(name, r=line_radius)
            except:
                pass
        
        for i in range(len(self.moon_grid.lon_lines)):
            name = f"grid_lon_{i}"
            try:
                self.rt.update_data(name, r=line_radius)
            except:
                pass
        
        # Toggle label visibility
        label_radius = GRID_LABEL_RADIUS if visible else 0.0
        
        for i, segments in enumerate(self.moon_grid.lat_labels):
            for j in range(len(segments)):
                name = f"grid_lat_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        for i, segments in enumerate(self.moon_grid.lon_labels):
            for j in range(len(segments)):
                name = f"grid_lon_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        # Toggle north pole label visibility
        for j in range(len(self.moon_grid.N)):
            name = f"grid_north_label_{j}"
            try:
                self.rt.update_data(name, r=label_radius)
            except:
                pass
        
        self.moon_grid_visible = visible
        
        # When showing the grid, update its orientation to match current view and Moon position
        # This is needed in case view orientation or time changed while the grid was hidden
        if visible:
            self.update_grid_labels_for_orientation()  # View orientation for labels
            self.update_moon_grid_orientation()  # Moon rotation for grid lines
    
    def toggle_grid(self):
        """Toggle the selenographic grid visibility."""
        self.show_moon_grid(not self.moon_grid_visible)

    def toggle_info_panel(self):
        """Toggle the Moon info panel visibility."""
        self.show_info_panel = not self.show_info_panel
        if self._info_frame is not None:
            if self.show_info_panel:
                self._info_frame.place(relx=0.0, rely=1.0, anchor='sw', x=6, y=-6)
            else:
                self._info_frame.place_forget()

    def setup_standard_labels(self):
        """
        Create standard feature labels for Moon features with standard_label=true.
        """
        if self.rt is None:
            print("Renderer not initialized")
            return

        # Determine flip flags based on current orientation
        flip_horizontal = self.orientation_mode in (ORIENTATION_NSEW, ORIENTATION_SNEW)
        flip_vertical = self.orientation_mode in (ORIENTATION_SNEW, ORIENTATION_SNWE)

        # Get ALL features with standard_label=True (illumination checked during rendering)
        self.standard_label_features = [f for f in self.moon_features if f.standard_label]
        self.standard_labels = create_standard_labels(
            self.standard_label_features,
            moon_radius=self.moon_radius,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        
        # Create an emissive material for the labels (so they glow and are visible in shadow)
        m_label = m_flat.copy()
        self.rt.update_material("standard_label_material", m_label)
        
        # Line thickness for labels
        label_radius = STANDARD_LABEL_RADIUS
        
        # Color for standard labels (white/light gray for visibility)
        label_color = [0.85, 0.85, 0.85]
        
        # Get Moon rotation matrix
        R = self.moon_rotation
        
        # Add each label's segments with rotation and inversion applied
        for i, label in enumerate(self.standard_labels):
            # Check if this label's feature is illuminated
            feature = self.standard_label_features[i]
            current_radius = label_radius if self._is_feature_illuminated(feature) else 0.0
            for j, seg in enumerate(label.segments):
                name = f"standard_label_{i}_{j}"
                # Apply Moon rotation
                if R is not None:
                    seg = (R @ seg.T).T
                self.rt.set_data(name, pos=seg, r=current_radius,
                                c=label_color, geom="SegmentChain", mat="standard_label_material")
        
        self.standard_labels_visible = True
    
    def show_standard_labels(self, visible: bool = True):
        """
        Show or hide the standard feature labels.
        
        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return
        
        if self.standard_labels is None:
            if visible:
                self.setup_standard_labels()
            return
        
        # Toggle visibility by setting zero radius (hide) or restoring (show)
        label_radius = STANDARD_LABEL_RADIUS if visible else 0.0
        
        for i, label in enumerate(self.standard_labels):
            for j in range(len(label.segments)):
                name = f"standard_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        self.standard_labels_visible = visible
        
        # When showing labels, update their orientation to match current Moon position and view
        # This is needed in case time or view orientation changed while labels were hidden
        if visible:
            self.update_standard_labels_for_view_orientation()
    
    def toggle_standard_labels(self):
        """Toggle the feature standard labels visibility."""
        self.show_standard_labels(not self.standard_labels_visible)
    
    def setup_spot_labels(self):
        """
        Create spot labels for Moon features with spot_label=true.
        """
        if self.rt is None:
            print("Renderer not initialized")
            return

        # Determine flip flags based on current orientation
        flip_horizontal = self.orientation_mode in (ORIENTATION_NSEW, ORIENTATION_SNEW)
        flip_vertical = self.orientation_mode in (ORIENTATION_SNEW, ORIENTATION_SNWE)

        # Get ALL features with spot_label=True (illumination checked during rendering)
        self.spot_label_features = [f for f in self.moon_features if f.spot_label]
        self.spot_labels = create_spot_labels(
            self.spot_label_features,
            moon_radius=self.moon_radius,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        
        # Create an emissive material for the labels (so they glow and are visible in shadow)
        m_label = m_flat.copy()
        self.rt.update_material("spot_label_material", m_label)
        
        # Line thickness for labels
        label_radius = SPOT_LABEL_RADIUS
        
        # Color for spot labels (yellow/gold for visibility)
        label_color = [1.0, 0.9, 0.3]
        
        # Get Moon rotation matrix
        R = self.moon_rotation
        
        # Add each label's segments with rotation and inversion applied
        for i, label in enumerate(self.spot_labels):
            # Check if this label's feature is illuminated
            feature = self.spot_label_features[i]
            current_radius = label_radius if self._is_feature_illuminated(feature) else 0.0
            for j, seg in enumerate(label.segments):
                name = f"spot_label_{i}_{j}"
                # Apply Moon rotation
                if R is not None:
                    seg = (R @ seg.T).T
                self.rt.set_data(name, pos=seg, r=current_radius,
                                c=label_color, geom="SegmentChain", mat="spot_label_material")
        
        self.spot_labels_visible = True
    
    def show_spot_labels(self, visible: bool = True):
        """
        Show or hide the spot feature labels.
        
        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return
        
        if self.spot_labels is None:
            if visible:
                self.setup_spot_labels()
            return
        
        # Toggle visibility by setting zero radius (hide) or restoring (show)
        label_radius = SPOT_LABEL_RADIUS if visible else 0.0
        
        for i, label in enumerate(self.spot_labels):
            for j in range(len(label.segments)):
                name = f"spot_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        self.spot_labels_visible = visible
        
        # When showing labels, update their orientation to match current Moon position and view
        # This is needed in case time or view orientation changed while labels were hidden
        if visible:
            self.update_spot_labels_for_view_orientation()
    
    def toggle_spot_labels(self):
        """Toggle the spot labels visibility."""
        self.show_spot_labels(not self.spot_labels_visible)
    
    def update_spot_labels_orientation(self):
        """
        Update spot labels to match current Moon orientation.
        
        This should be called after update_view() to rotate the labels
        along with the Moon surface.
        """
        if self.rt is None or self.spot_labels is None:
            return
        
        R = self.moon_rotation

        if R is None:
            return
        
        # Update spot labels
        for i, label in enumerate(self.spot_labels):
            # Check if this label's feature is illuminated
            feature = self.spot_label_features[i]
            label_radius = SPOT_LABEL_RADIUS if self._is_feature_illuminated(feature) else 0.0
            for j, orig_seg in enumerate(label.segments):
                name = f"spot_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated, r=label_radius)
                except:
                    pass
    
    def update_standard_labels_orientation(self):
        """
        Update standard labels to match current Moon orientation.
        
        This should be called after update_view() to rotate the labels
        along with the Moon surface.
        """
        if self.rt is None or self.standard_labels is None:
            return
        
        R = self.moon_rotation

        if R is None:
            return
        
        # Update standard labels
        for i, label in enumerate(self.standard_labels):
            # Check if this label's feature is illuminated
            feature = self.standard_label_features[i]
            label_radius = STANDARD_LABEL_RADIUS if self._is_feature_illuminated(feature) else 0.0
            for j, orig_seg in enumerate(label.segments):
                name = f"standard_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated, r=label_radius)
                except:
                    pass
    
    def create_pin(self, digit: int, lat: float, lon: float):
        """
        Create a pin with the given digit at the specified selenographic coordinates.
        
        Parameters
        ----------
        digit : int
            The digit (1-9) for the pin
        lat : float
            Selenographic latitude in degrees
        lon : float
            Selenographic longitude in degrees
        """
        if self.rt is None:
            return
        
        # Determine flip flags based on current orientation
        flip_horizontal = self.orientation_mode in (ORIENTATION_NSEW, ORIENTATION_SNEW)
        flip_vertical = self.orientation_mode in (ORIENTATION_SNEW, ORIENTATION_SNWE)
        
        # Generate pin digit segments (left-bottom corner at cursor position)
        pin_segments = create_single_digit_on_sphere(
            digit=digit,
            lat=lat,
            lon=lon,
            moon_radius=self.moon_radius,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        
        # Store pin data (original segments for rotation updates)
        self.pins[digit] = pin_segments
        
        # Create material for pins if not already created
        m_pin = m_flat.copy()
        self.rt.update_material("pin_material", m_pin)
        
        # Line thickness for pins
        pin_radius = PIN_LABEL_RADIUS
        
        # Apply Moon rotation to segments and add to renderer
        R = self.moon_rotation
        
        for j, seg in enumerate(pin_segments):
            name = f"pin_{digit}_{j}"
            if R is not None:
                rotated = (R @ seg.T).T
            else:
                rotated = seg
            self.rt.set_data(name, pos=rotated, r=pin_radius,
                            c=PIN_COLOR, geom="SegmentChain", mat="pin_material")
    
    def remove_pin(self, digit: int):
        """
        Remove a pin with the given digit.
        
        Parameters
        ----------
        digit : int
            The digit (1-9) of the pin to remove
        """
        if self.rt is None or digit not in self.pins:
            return
        
        pin_segments = self.pins[digit]
        
        # Remove all segments from renderer
        for j in range(len(pin_segments)):
            name = f"pin_{digit}_{j}"
            try:
                self.rt.delete_geometry(name)
            except:
                pass
        
        del self.pins[digit]
    
    def toggle_pin_at_cursor(self, event, digit: int):
        """
        Toggle a pin at the cursor position.
        
        If pins are not visible, do nothing.
        If a pin with this digit exists, remove it.
        Otherwise, create a new pin at the cursor position.
        
        Parameters
        ----------
        event : tk.Event
            The keyboard event containing mouse position
        digit : int
            The digit (1-9) for the pin
        """
        if self.rt is None:
            return
        
        # Do nothing if pins are not visible
        if not self.pins_visible:
            return
        
        # If pin already exists, remove it
        if digit in self.pins:
            self.remove_pin(digit)
            return
        
        # Get mouse position in image coordinates
        x, y = self.rt._get_image_xy(event.x, event.y)
        
        # Get hit position at mouse location
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)
        
        # Check if we hit something (distance > 0 means valid hit)
        if hd <= 0:
            return
        
        # Convert hit position to selenographic coordinates
        lat, lon = self.hit_to_selenographic(hx, hy, hz)
        
        if lat is None or lon is None:
            return
        
        # Create the pin
        self.create_pin(digit, lat, lon)
    
    def show_pins(self, visible: bool = True):
        """
        Show or hide all pins.
        
        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return
        
        # Toggle visibility by setting zero radius (hide) or restoring (show)
        pin_radius = PIN_LABEL_RADIUS if visible else 0.0
        
        for digit, pin_segments in self.pins.items():
            for j in range(len(pin_segments)):
                name = f"pin_{digit}_{j}"
                self.rt.update_data(name, r=pin_radius)
        
        self.pins_visible = visible
        
        # When showing pins, update their orientation to match current Moon position
        # This is needed in case time changed while pins were hidden
        if visible:
            self.update_pins_orientation()
        
        self._update_status_pins()
    
    def toggle_pins(self):
        """Toggle the pins visibility."""
        self.show_pins(not self.pins_visible)
    
    def update_pins_orientation(self):
        """
        Update pins to match current Moon orientation.
        
        This should be called after update_view() to rotate the pins
        along with the Moon surface.
        """
        if self.rt is None or not self.pins or not self.pins_visible:
            return
        
        R = self.moon_rotation
        
        if R is None:
            return
        
        for digit, pin_segments in self.pins.items():
            for j, orig_seg in enumerate(pin_segments):
                name = f"pin_{digit}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
    
    def find_feature_for_status_bar(self, lat: float, lon: float) -> Optional[MoonFeature]:
        """
        Find a Moon feature by the given selenographic coordinates to be displayed on status bar.
        
        When multiple features overlap at the given position, returns the
        feature with the smallest angular size (most specific feature).
        
        Since moon_features is sorted by angular_radius (smallest first),
        the first match is guaranteed to be the smallest feature.
        
        Parameters
        ----------
        lat : float
            Selenographic latitude in degrees
        lon : float
            Selenographic longitude in degrees
            
        Returns
        -------
        MoonFeature
            Moon feature if found, None otherwise
        """
        for moon_feature in self.moon_features:
            if not moon_feature.status_bar:
                continue
            # Calculate angular distance from feature center
            dlat = lat - moon_feature.lat
            dlon = lon - moon_feature.lon
            # Simple approximation for small angles
            # Use cos_lat correction for longitude
            angular_dist2 = dlat**2 + (dlon * moon_feature.cos_lat)**2
            
            # Check if within feature's angular radius (half of angular diameter)
            # First match is smallest due to sorted order - early exit
            if angular_dist2 <= moon_feature.angular_radius**2:
                return moon_feature
        
        return None
    
    def reset_camera_position(self):
        """
        Reset the camera to its initial position.
        
        Restores the camera to the position it had when the view was first set up,
        undoing any mouse rotation/panning performed by the user.
        Also resets the time back to the initial time.
        """

        cp = self.initial_camera_params

        if self.rt is None or cp is None:
            return
        
        # Reset orientation mode to initial
        if self.orientation_mode != self.initial_orientation_mode:
            self.set_orientation(self.initial_orientation_mode)
            self.rt._view_orientation = self.initial_orientation_mode
        
        # Reset time back to initial time if it was changed
        if self.initial_dt_local is not None and self.dt_local != self.initial_dt_local:
            # Reset to initial time - this will restore Moon orientation and lighting
            self.update_moon_for_time(self.initial_dt_local, self.observer_lat, self.observer_lon)
            
            # Update grid and labels with restored orientation
            if self.moon_grid_visible:
                self.update_moon_grid_orientation()
            if self.standard_labels_visible:
                self.update_standard_labels_orientation()
            if self.spot_labels_visible:
                self.update_spot_labels_orientation()
            
            # Update pins positions
            self.update_pins_orientation()
        
        # Restore initial camera parameters
        up = cp.up[:]

        self.rt.setup_camera("cam1", eye=cp.eye, target=cp.target, up=up, fov=cp.fov)
        
        # Update status bar
        self._update_all_status_panels()
    
    def reset_to_default_view(self):
        """
        Reset the camera to the default view calculated from ephemeris.
        
        This is the view that would be shown when starting the renderer
        without any --init-view parameter. Use this to get back to the
        standard Moon-centered view after using --init-view.
        """
        cp = self.default_camera_params

        if self.rt is None or cp is None:
            return
        
        # Reset orientation mode to initial
        if self.orientation_mode != self.initial_orientation_mode:
            self.set_orientation(self.initial_orientation_mode)
            self.rt._view_orientation = self.initial_orientation_mode
        
        # Restore default camera parameters
        up = cp.up[:]

        self.rt.setup_camera("cam1", eye=cp.eye, target=cp.target, up=up, fov=cp.fov)
    
    def center_view_on_cursor(self, event):
        """
        Center the view on the point under the mouse cursor.
        
        This method gets the 3D hit position at the current mouse location
        and moves the camera to look at that point, maintaining the current
        camera distance.
        
        Parameters
        ----------
        event : tk.Event
            The keyboard event containing mouse position (event.x, event.y)
        """
        if self.rt is None:
            return
        
        # Get mouse position in image coordinates
        x, y = self.rt._get_image_xy(event.x, event.y)
        
        # Get hit position at mouse location
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)
        
        # Check if we hit something (distance > 0 means valid hit)
        if hd <= 0:
            return
        
        # Get current camera parameters using PlotOptix internal state
        cam = self.rt.get_camera("cam1")
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        
        # Calculate current camera distance from target
        current_distance = np.linalg.norm(eye - target)
        
        # New target is the hit position
        new_target = np.array([hx, hy, hz])
        
        # Calculate direction from new target to current eye position
        direction = eye - target
        direction = direction / np.linalg.norm(direction)
        
        # New eye position: same distance from new target, same direction
        new_eye = new_target + direction * current_distance
        
        # Update camera with new eye and target
        self.rt.setup_camera("cam1", eye=new_eye.tolist(), target=new_target.tolist())
    
    def navigate_view(self, direction: str, step_factor: float = 0.05):
        """
        Navigate the view using arrow keys.
        
        Rotates the camera around the Moon while keeping the same distance.
        
        Parameters
        ----------
        direction : str
            One of 'Left', 'Right', 'Up', 'Down'
        step_factor : float
            Fraction of the view to move per key press (default 5%)
        """
        if self.rt is None:
            return
        
        # Get current camera parameters
        cam = self.rt.get_camera("cam1")
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        up = np.array(cam["Up"])
        
        # Calculate view direction and distance
        view_dir = target - eye
        distance = np.linalg.norm(view_dir)
        view_dir = view_dir / distance
        
        # Calculate right vector (perpendicular to view and up)
        right = np.cross(view_dir, up)
        right = right / np.linalg.norm(right)
        
        # Calculate actual up vector (perpendicular to view and right)
        actual_up = np.cross(right, view_dir)
        actual_up = actual_up / np.linalg.norm(actual_up)
        
        # Calculate rotation angle based on current FOV
        fov = self.rt._optix.get_camera_fov(0)
        angle = np.radians(fov * step_factor)
        
        # Determine rotation axis and direction based on arrow key
        if direction == 'Left':
            axis = actual_up
            angle = angle
        elif direction == 'Right':
            axis = actual_up
            angle = -angle
        elif direction == 'Up':
            axis = right
            angle = angle
        elif direction == 'Down':
            axis = right
            angle = -angle
        else:
            return
        
        # Rodrigues' rotation formula to rotate eye around target
        eye_rel = eye - target
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        new_eye_rel = (eye_rel * cos_a + 
                       np.cross(axis, eye_rel) * sin_a + 
                       axis * np.dot(axis, eye_rel) * (1 - cos_a))
        new_eye = target + new_eye_rel
        
        # Also rotate the up vector for up/down navigation
        if direction in ('Up', 'Down'):
            new_up = (up * cos_a + 
                      np.cross(axis, up) * sin_a + 
                      axis * np.dot(axis, up) * (1 - cos_a))
            self.rt.setup_camera("cam1", eye=new_eye.tolist(), up=new_up.tolist())
        else:
            self.rt.setup_camera("cam1", eye=new_eye.tolist())
    
    def rotate_around_moon_axis(self, direction: str, step_deg: float = 1.0):
        """
        Rotate the camera around the Moon's polar or equatorial axis.
        
        Left/Right rotates around the Moon's north-south (polar) axis.
        Up/Down rotates around the Moon's east-west (equatorial) axis.
        
        Parameters
        ----------
        direction : str
            'Left', 'Right', 'Up', or 'Down'
        step_deg : float
            Rotation step in degrees (default 1°)
        """
        if self.rt is None or self.moon_rotation is None:
            return
        
        # Get the Moon's axes in scene coordinates
        # Polar axis: direction of the Moon's north pole
        # Equatorial axis: perpendicular to polar axis, pointing east
        moon_polar_axis = self.moon_rotation @ np.array([0.0, 0.0, 1.0])
        moon_polar_axis = moon_polar_axis / np.linalg.norm(moon_polar_axis)
        
        moon_equatorial_axis = self.moon_rotation @ np.array([1.0, 0.0, 0.0])
        moon_equatorial_axis = moon_equatorial_axis / np.linalg.norm(moon_equatorial_axis)
        
        # Determine rotation axis and angle based on direction
        if direction == 'Left':
            axis = moon_polar_axis
            angle = np.radians(step_deg)
        elif direction == 'Right':
            axis = moon_polar_axis
            angle = np.radians(-step_deg)
        elif direction == 'Up':
            axis = moon_equatorial_axis
            angle = np.radians(step_deg)
        elif direction == 'Down':
            axis = moon_equatorial_axis
            angle = np.radians(-step_deg)
        else:
            return
        
        # Get current camera parameters
        cam = self.rt.get_camera("cam1")
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        up = np.array(cam["Up"])
        
        # Rodrigues' rotation formula to rotate eye around target
        eye_rel = eye - target
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        new_eye_rel = (eye_rel * cos_a + 
                       np.cross(axis, eye_rel) * sin_a + 
                       axis * np.dot(axis, eye_rel) * (1 - cos_a))
        new_eye = target + new_eye_rel
        
        # Also rotate the up vector to maintain proper orientation
        new_up = (up * cos_a + 
                  np.cross(axis, up) * sin_a + 
                  axis * np.dot(axis, up) * (1 - cos_a))
        
        self.rt.setup_camera("cam1", eye=new_eye.tolist(), up=new_up.tolist())
    
    def hit_to_selenographic(self, hx: float, hy: float, hz: float) -> tuple:
        """
        Convert 3D hit position to selenographic coordinates.
        
        Parameters
        ----------
        hx, hy, hz : float
            3D hit position in scene coordinates
            
        Returns
        -------
        tuple
            (latitude, longitude) in degrees, or (None, None) if not on Moon
        """
        if self.moon_rotation_inv is None:
            return None, None
        
        # The hit position is on the rotated Moon surface
        # We need to transform it back to the original Moon coordinates
        # where north pole is +Z and prime meridian faces -Y
        hit_pos = np.array([hx, hy, hz])
        
        # Check if hit is on the Moon surface
        # Moon radius is 10, with displacement it can vary slightly
        # Accept hits within a reasonable range around the Moon radius
        r = np.linalg.norm(hit_pos)
        if r < self.moon_radius * 0.9 or r > self.moon_radius * 1.15:
            # Hit is not on Moon surface (too close to origin or too far - e.g., background)
            return None, None
        
        hit_normalized = hit_pos / r
        
        # Transform back to original Moon coordinates
        original_pos = self.moon_rotation_inv @ hit_normalized
        
        # Convert to selenographic coordinates
        # In original coordinates:
        # - +Z is north pole
        # - -Y is prime meridian (lon=0)
        # - +X is east (lon=90)
        x, y, z = original_pos
        
        # Latitude: angle from equator (XY plane) to the point
        # sin(lat) = z / r, where r=1 for unit sphere
        lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
        
        # Longitude: angle in XY plane from -Y axis (prime meridian)
        # atan2(x, -y) gives angle from -Y toward +X (east positive)
        lon = np.degrees(np.arctan2(x, -y))
        
        return lat, lon

    def _feature_scene_position(self, feature: MoonFeature, radius: Optional[float] = None) -> np.ndarray:
        """
        Get the 3D scene coordinates of a feature given its selenographic coords.
        Applies Moon rotation so result is in scene coordinates.
        """
        if radius is None:
            r = self.moon_radius
        else:
            r = radius

        lat_rad = np.radians(feature.lat)
        lon_rad = np.radians(feature.lon)
        x = r * np.cos(lat_rad) * np.sin(lon_rad)
        y = -r * np.cos(lat_rad) * np.cos(lon_rad)
        z = r * np.sin(lat_rad)
        original_pos = np.array([x, y, z])
        if self.moon_rotation is None:
            return original_pos
        return (self.moon_rotation @ original_pos)

    def _is_feature_illuminated(self, feature: MoonFeature) -> bool:
        """
        Determine whether a feature is on the illuminated hemisphere given
        the cached light position and current Moon rotation.

        Returns True if the surface normal at the feature has positive dot
        product with the light direction (i.e., lit by the Sun in scene coords).
        """
        if self.light_pos is None or self.moon_rotation is None:
            # If we don't have light or rotation info, assume visible to avoid hiding labels
            return True

        # Position of feature on unit sphere in scene coordinates
        pos = self._feature_scene_position(feature, radius=self.moon_radius)
        norm = np.linalg.norm(pos)
        if norm == 0:
            return True
        pos_unit = pos / norm

        light = np.array(self.light_pos)
        light_norm = np.linalg.norm(light)
        if light_norm == 0:
            return True
        light_unit = light / light_norm

        # If dot > 0 => angle < 90° between normal and light direction => illuminated
        return float(np.dot(pos_unit, light_unit)) > 0.0
    
    def _update_grid_lines(self, rt: TkOptiX, R: NDArray, lines, prefix: str):
        """Update grid lines to match current Moon orientation.
        
        Parameters
        ----------
        rt : TkOptiX
            Renderer instance
        R : NDArray
            Rotation matrix
        lines : list
            List of line point arrays
        prefix : str
            Name prefix for the geometry objects
        """
        for i, orig_points in enumerate(lines):
            name = f"{prefix}_{i}"
            rotated = (R @ orig_points.T).T
            try:
                rt.update_data(name, pos=rotated)
            except:
                pass
    
    def _update_grid_nested_segments(self, rt: TkOptiX, R: NDArray, segments_list, prefix: str):
        """Update nested grid segments to match current Moon orientation.
        
        Parameters
        ----------
        rt : TkOptiX
            Renderer instance
        R : NDArray
            Rotation matrix
        segments_list : list
            List of segment lists
        prefix : str
            Name prefix for the geometry objects
        """
        for i, segments in enumerate(segments_list):
            for j, orig_seg in enumerate(segments):
                name = f"{prefix}_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    rt.update_data(name, pos=rotated)
                except:
                    pass
    
    def update_moon_grid_orientation(self):
        """
        Update grid lines to match current Moon orientation.
        
        This should be called after update_view() to rotate the grid
        along with the Moon surface.
        """
        if self.rt is None or self.moon_grid is None or not self.moon_grid_visible:
            return
        
        R = self.moon_rotation

        if R is None:
            return
        
        self._update_grid_lines(self.rt, R, self.moon_grid.lat_lines, "grid_lat")
        self._update_grid_lines(self.rt, R, self.moon_grid.lon_lines, "grid_lon")
        self._update_grid_nested_segments(self.rt, R, self.moon_grid.lat_labels, "grid_lat_label")
        self._update_grid_nested_segments(self.rt, R, self.moon_grid.lon_labels, "grid_lon_label")
        self._update_grid_lines(self.rt, R, self.moon_grid.N, "grid_north_label")
    
    def zoom_with_wheel(self, event):
        """
        Zoom in/out using mouse wheel.
        
        Parameters
        ----------
        event : tk.Event
            Mouse wheel event. event.delta indicates scroll direction:
            positive = scroll up = zoom in, negative = scroll down = zoom out
        """
        if self.rt is None:
            return
        
        # Get current FOV
        current_fov = self.rt._optix.get_camera_fov(0)  # 0 is current camera
        
        # Calculate zoom factor based on wheel delta
        # On Windows, delta is typically ±120 per notch
        # Positive delta = scroll up = zoom in = decrease FOV
        # Negative delta = scroll down = zoom out = increase FOV
        zoom_factor = 1 - (event.delta / 120) * 0.05  # 5% per notch
        
        # Apply zoom by changing FOV
        new_fov = current_fov * zoom_factor
        
        # Clamp FOV to reasonable range
        new_fov = max(1, min(90, new_fov))
        
        self.rt._optix.set_camera_fov(new_fov)

    # ==================== Distance Measurement Methods ====================
    
    MOON_RADIUS_KM = 1737.4  # Real Moon radius in kilometers
    
    def calculate_great_circle_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on the Moon surface.
        
        Uses the Haversine formula to calculate the central angle, then
        multiplies by the Moon's radius to get the arc length.
        
        Parameters
        ----------
        lat1, lon1 : float
            First point's selenographic coordinates in degrees
        lat2, lon2 : float
            Second point's selenographic coordinates in degrees
            
        Returns
        -------
        float
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        # Distance = central angle * radius
        distance_km = c * self.MOON_RADIUS_KM
        
        return distance_km
    
    def _seleno_to_scene_position(self, lat: float, lon: float, radius: float = None) -> np.ndarray:
        """
        Convert selenographic coordinates to 3D scene position.
        
        Parameters
        ----------
        lat : float
            Selenographic latitude in degrees
        lon : float
            Selenographic longitude in degrees
        radius : float, optional
            Radius at which to place the point. If None, uses moon_radius.
            
        Returns
        -------
        np.ndarray
            3D position in scene coordinates
        """
        if radius is None:
            r = self.moon_radius
        else:
            r = radius
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Convert to Cartesian in original Moon coordinates
        # +Z is north pole, -Y is prime meridian (lon=0), +X is east (lon=90)
        x = r * np.cos(lat_rad) * np.sin(lon_rad)
        y = -r * np.cos(lat_rad) * np.cos(lon_rad)
        z = r * np.sin(lat_rad)
        original_pos = np.array([x, y, z])
        
        # Apply Moon rotation to get scene coordinates
        if self.moon_rotation is None:
            return original_pos
        return self.moon_rotation @ original_pos
    
    def start_measurement(self, event):
        """
        Start distance measurement on Ctrl+B1 press.
        
        Captures the start position on the canvas and the selenographic
        coordinates at that point.
        
        Parameters
        ----------
        event : tk.Event
            Mouse button press event
        """
        if self.rt is None:
            return
        
        # Get image coordinates
        x, y = self.rt._get_image_xy(event.x, event.y)
        
        # Get hit position
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)
        
        # Check if we hit the Moon surface
        if hd <= 0:
            self.measuring = False
            return
        
        # Convert to selenographic coordinates
        lat, lon = self.hit_to_selenographic(hx, hy, hz)
        
        if lat is None or lon is None:
            self.measuring = False
            return
        
        # Store start position and coordinates
        self.measuring = True
        self.measure_start_canvas = (event.x, event.y)
        self.measure_start_coords = (lat, lon)
        
        # Create leading line on canvas (will be updated during drag)
        if hasattr(self.rt, '_canvas'):
            self.leading_line_id = self.rt._canvas.create_line(
                event.x, event.y, event.x, event.y,
                fill='yellow', width=2, dash=(4, 4)
            )
    
    def update_leading_line(self, event):
        """
        Update the leading line during drag.
        
        Parameters
        ----------
        event : tk.Event
            Mouse motion event
        """
        if not self.measuring or self.leading_line_id is None:
            return
        
        if not hasattr(self.rt, '_canvas'):
            return
        
        # Update the leading line endpoint
        start_x, start_y = self.measure_start_canvas
        self.rt._canvas.coords(
            self.leading_line_id,
            start_x, start_y, event.x, event.y
        )
    
    def finish_measurement(self, event):
        """
        Finish distance measurement on B1 release.
        
        Calculates the great circle distance, creates a display line,
        and updates the status bar.
        
        Parameters
        ----------
        event : tk.Event
            Mouse button release event
        """
        if not self.measuring:
            return
        
        # Remove the leading line from canvas
        if self.leading_line_id is not None and hasattr(self.rt, '_canvas'):
            self.rt._canvas.delete(self.leading_line_id)
            self.leading_line_id = None
        
        self.measuring = False
        
        if self.rt is None or self.measure_start_coords is None:
            return
        
        # Get end position
        x, y = self.rt._get_image_xy(event.x, event.y)
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)
        
        if hd <= 0:
            return
        
        lat2, lon2 = self.hit_to_selenographic(hx, hy, hz)
        
        if lat2 is None or lon2 is None:
            return
        
        lat1, lon1 = self.measure_start_coords
        
        # Calculate distance
        distance_km = self.calculate_great_circle_distance(lat1, lon1, lat2, lon2)
        
        # Store measured distance for status bar
        self.measured_distance = distance_km
        
        # Update status bar
        self._update_status_measured()
