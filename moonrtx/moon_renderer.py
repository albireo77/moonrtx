import numpy as np
import os
import struct
import base64
import tkinter as tk
from tkinter import filedialog
from typing import NamedTuple
from numpy.typing import NDArray

from typing import Optional
from datetime import datetime
from datetime import timezone

from moonrtx.shared_types import MoonEphemeris
from moonrtx.shared_types import MoonFeature
from moonrtx.shared_types import CameraParams
from moonrtx.astro import calculate_moon_ephemeris
from moonrtx.data_loader import load_moon_features, load_elevation_data, load_color_data, load_starmap
from moonrtx.moon_grid import create_moon_grid, create_standard_labels, create_spot_labels, create_single_digit_on_sphere

from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse
from plotoptix.materials import m_flat

GRID_COLOR = [0.50, 0.50, 0.50]
MOON_FILL_FRACTION = 0.9    # Moon fills 90% of window height (5% margins top/bottom)
SUN_RADIUS = 10             # affects Moon surface illumination
PIN_COLOR = [1.0, 0.0, 0.0]

class Scene(NamedTuple):
    eye: NDArray
    target: NDArray
    up: NDArray
    light_pos: NDArray


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
                 light_intensity: int,
                 app_name: str,
                 init_camera_params: Optional[CameraParams] = None) -> TkOptiX:
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
    light_intensity : int
        Light intensity 
    app_name : str
        Application name
    init_camera_params : CameraParams, optional
        Initial camera parameters to restore a specific view
    Returns
    -------
    TkOptiX
        The renderer instance
    """

    moon_renderer = MoonRenderer(
        app_name=app_name,
        elevation_file=elevation_file,
        color_file=color_file,
        starmap_file=starmap_file,
        downscale=downscale,
        features_file=features_file
    )
    
    # Setup renderer
    moon_renderer.setup_renderer()
    
    # Set view
    moon_renderer.update_view(dt_local=dt_local, lat=lat, lon=lon, light_intensity=light_intensity)
    
    # Apply custom camera parameters if provided (to restore a saved view)
    if init_camera_params is not None:
        moon_renderer.apply_camera_params(init_camera_params)
    
    # Print info
    print("\n" + moon_renderer.get_info())
    print("\nKeys and mouse:")
    print("  G - Toggle selenographic grid")
    print("  L - Toggle standard labels")
    print("  S - Toggle spot labels")
    print("  I - Toggle upside down view")
    print("  P - Toggle pins ON/OFF")
    print("  1-9 - Create/Remove pin (when pins are ON)")
    print("  R - Reset view to initial state")
    print("  V - Reset view to that based on ephemeris (useful after starting with --init-view parameter)")
    print("  C - Center view on point under cursor")
    print("  F - Search for Moon features (craters, mounts etc.)")
    print("  Arrows - Navigate view")
    print("  F12 - Save image")
    print("  Hold and drag left mouse button - Rotate the eye around Moon")
    print("  Hold shift + left mouse button and drag up/down - Zoom out/in")
    print("  Hold and drag right mouse button - Rotate Moon around the eye")
    print("  Hold shift + right mouse button and drag up/down - Move eye backward/forward")
    
    original_key_handler = moon_renderer.rt._gui_key_pressed
    def custom_key_handler(event):
        # Ignore key events when search dialog is open
        if moon_renderer.search_dialog_open:
            return
        if event.keysym.lower() == 'g':
            moon_renderer.toggle_grid()
        elif event.keysym.lower() == 'l':
            moon_renderer.toggle_standard_labels()
        elif event.keysym.lower() == 's':
            moon_renderer.toggle_spot_labels()
        elif event.keysym.lower() == 'i':
            moon_renderer.toggle_invert()
        elif event.keysym.lower() == 'r':
            moon_renderer.reset_camera_position()
        elif event.keysym.lower() == 'c':
            moon_renderer.center_view_on_cursor(event)
        elif event.keysym == 'F12':
            moon_renderer.save_image_dialog()
        elif event.keysym.lower() == 'f':
            moon_renderer.search_feature_dialog()
        elif event.keysym in ('Left', 'Right', 'Up', 'Down'):
            moon_renderer.navigate_view(event.keysym)
        elif event.keysym.lower() == 'v':
            moon_renderer.reset_to_default_view()
        elif event.keysym.lower() == 'p':
            moon_renderer.toggle_pins()
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
            
            coord_column = ""
            feature_column = ""
            # Check if we hit something (distance > 0 means valid hit)
            if hd > 0:
                lat, lon = moon_renderer.hit_to_selenographic(hx, hy, hz)
                if lat is not None and lon is not None:
                    # Check if hovering over a named feature
                    feature = moon_renderer.find_feature_at(lat, lon)
                    feature_name = feature.name if feature is not None and feature.status_bar else ""
                    feature_column = f"{feature_name} (size = {feature.angle * 30.34:.1f} km)" if feature_name else ""
                    lat_dir = 'N' if lat >= 0 else 'S'
                    lon_dir = 'E' if lon >= 0 else 'W'
                    coord_column = f"Lat: {abs(lat):5.2f}° {lat_dir}  Lon: {abs(lon):6.2f}° {lon_dir}"
            
            # Build status: coordinates first (fixed width), then feature name (fixed width), then pin mode
            pin_mode = "ON" if moon_renderer.pins_visible else "OFF"
            status_text = f"{coord_column:<32}{feature_column:<42.42}[Pins {pin_mode}]"
            
            moon_renderer.rt._status_action_text.set(status_text)
    
    moon_renderer.rt._gui_motion = custom_motion_handler
    moon_renderer.start()
    return moon_renderer.rt

def calculate_rotation(z: float, x: float, y: float):
    def rot_x(angle_deg):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def rot_y(angle_deg):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def rot_z(angle_deg):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot_y(y) @ rot_x(x) @ rot_z(z)

def calculate_camera_and_light(moon_ephem: MoonEphemeris, zoom: float = 1000) -> dict:
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
        
    Returns
    -------
    dict
        Camera eye position, target, up vector, and light position
    """
    moon_radius = 10
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
    
    light_x = -np.sin(bright_limb_angle) * np.sin(phase) * light_distance
    light_z = np.cos(bright_limb_angle) * np.sin(phase) * light_distance
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
                 starmap_file: Optional[str] = None,
                 downscale: int = 3,
                 width: int = 1400,
                 height: int = 900):
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
        starmap_file : str, optional
            Path to star map TIFF for background
        downscale : int
            Elevation downscale factor
        width, height : int
            Render window size
        """
        self.width = width
        self.height = height
        self.downscale = downscale
        self.gamma = 2.2
        
        # Load data
        self.elevation = load_elevation_data(elevation_file, downscale)
        self.color_data = load_color_data(color_file, self.gamma)
        self.moon_features = load_moon_features(features_file)
        self.star_map = load_starmap(starmap_file) if starmap_file else None

        self.app_name = app_name
        
        # Renderer
        self.rt = None
        self.moon_ephem = None
        
        # Grid settings
        self.moon_grid_visible = False
        self.moon_grid = None
        self.moon_radius = 10.0  # Same as in set_data("moon", ...)
        
        # View inversion (upside down)
        self.inverted = False
        
        # Initial camera parameters (for reset with R key)
        self.initial_camera_params = None
        
        # Default camera parameters calculated from ephemeris (for reset with V key)
        # This is the view without any --init-view override
        self.default_camera_params = None
        
        # Moon rotation matrix and its inverse (for selenographic coord conversion)
        self.moon_rotation_matrix = None
        self.moon_rotation_matrix_inv = None
        
        # Flag to track if window has been maximized
        self._window_maximized = False
        
        # Standard labels settings
        self.standard_labels_visible = False
        self.standard_labels = None
        
        # Spot labels settings
        self.spot_labels_visible = False
        self.spot_labels = None
        
        # Store view parameters for filename generation
        self.dt_local = None
        self.observer_lat = None
        self.observer_lon = None
        
        # Flag to track if search dialog is open
        self.search_dialog_open = False
        
        # Pins settings
        self.pins_visible = True  # Pins visible by default
        self.pins = {}  # dict mapping digit (1-9) to pin segments
        
    def _on_launch_finished(self, rt):
        """Callback to maximize window and set title on first launch."""
        if not self._window_maximized:
            self._window_maximized = True
            # Schedule maximize and title change on the main thread
            def init_window():
                rt._root.state('zoomed')
                rt._root.title(self.app_name)
                # Set monospace font for status bar to prevent text shifting
                # and increase width to fill available space
                if hasattr(rt, '_status_action'):
                    rt._status_action.configure(font=("Consolas", 9), width=85)
                # Hide FPS panel from status bar
                if hasattr(rt, '_status_fps'):
                    rt._status_fps.grid_remove()
            rt._root.after_idle(init_window)
        
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

    def update_view(self, dt_local: datetime, lat: float, lon: float, zoom: float = 1000, light_intensity: int = 180):
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
        light_intensity : float
            Light intensity
        """

        dt_utc = dt_local.astimezone(timezone.utc)
        self.moon_ephem = calculate_moon_ephemeris(dt_utc, lat, lon)
        
        # Store view parameters for filename generation
        self.dt_local = dt_local
        self.observer_lat = lat
        self.observer_lon = lon
        
        if self.moon_ephem.alt < 0:
            print(f"Warning: Moon is below horizon (altitude: {self.moon_ephem.alt:.1f}°)")
        
        scene = calculate_camera_and_light(self.moon_ephem, zoom)

        R = self.calculate_moon_rotation()
        
        # Store rotation matrix and its inverse for selenographic coordinate conversion
        self.moon_rotation_matrix = R
        self.moon_rotation_matrix_inv = R.T  # For orthogonal matrices, inverse = transpose

        # The u,v vectors define how the texture is mapped onto the sphere
        # u = north pole direction
        # v = longitude 0° direction (orthogonalized to u)
        # Base orientation (no libration, looking from -Y toward Moon at origin):
        # - Moon's north pole points to +Z (up in view)
        # - Moon's prime meridian (lon=0) faces toward -Y (toward camera)
        # - u = [0, 0, 1] (north pole)
        # - v = [0, -1, 0] (prime meridian faces -Y)
        u_new = R @ np.array([0.0, 0.0, 1.0])
        v_new = R @ np.array([0.0, -1.0, 0.0])

        # Update Moon geometry with new orientation
        self.rt.update_data("moon", u=u_new.tolist(), v=v_new.tolist())
        
        # Update camera
        # Calculate FOV so moon fills MOON_FILL_FRACTION (90%) of window height
        # Moon diameter = 2 * moon_radius, visible_height = diameter / fill_fraction
        # FOV = 2 * atan(visible_height / (2 * camera_distance))
        moon_radius = 10  # Same as in set_data("moon", ...)
        camera_distance = moon_radius * (zoom / 100)
        moon_diameter = 2 * moon_radius
        visible_height = moon_diameter / MOON_FILL_FRACTION
        fov = np.degrees(2 * np.arctan(visible_height / (2 * camera_distance)))
        fov = max(1, min(90, fov))  # Clamp to valid range
        
        # Invert up vector for upside down view
        camera_up = scene.up if not self.inverted else -scene.up
        
        self.rt.setup_camera("cam1",
                             cam_type="Pinhole",
                             eye=scene.eye.tolist(),
                             target=scene.target.tolist(),
                             up=camera_up.tolist(),
                             aperture_radius=0.01,
                             aperture_fract=0.2,
                             focal_scale=0.7,
                             fov=fov)
        
        # Store initial camera parameters for reset functionality
        if self.initial_camera_params is None:
            self.initial_camera_params = CameraParams(eye=scene.eye.tolist(), target=scene.target.tolist(), up=camera_up.tolist(), fov=fov)
        
        # Always store default camera params (the view calculated from ephemeris)
        self.default_camera_params = CameraParams(eye=scene.eye.tolist(), target=scene.target.tolist(), up=camera_up.tolist(), fov=fov)
        
        # Light intensity based on phase - full moon is brighter
        # light_intensity = 40 + 20 * np.cos(np.radians(self.moon_ephem.phase))
        
        self.rt.setup_light("sun", pos=scene.light_pos.tolist(), color=light_intensity, radius=SUN_RADIUS)
        
        # Update grid orientation if visible
        if self.moon_grid_visible:
            self.update_moon_grid_orientation()
        
        # Update standard labels orientation if visible
        if self.standard_labels_visible:
            self.update_standard_labels_orientation()

    def calculate_moon_rotation(self):
        if self.moon_ephem is None:
            return None
        # Apply rotations in order: first -longitude (Z), then latitude (X), finally pa_view (Y)
        # Matrix multiplication is right-to-left, so rightmost is applied first
        return calculate_rotation(
            -self.moon_ephem.libr_long,
            self.moon_ephem.libr_lat,
            self.moon_ephem.pa_axis_view
        )
        
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
        
        Format: datetime_lat+XX.XXXXXX_lon+XX.XXXXXX_cam<base64>
        
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
        
        # 4. Current camera parameters (at the time of screenshot) - encoded as base64
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
                    size_km = feature.angle * 30.34
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
    
    def center_on_feature(self, feature: MoonFeature):
        """
        Center the view on a Moon feature and zoom to show it.
        
        Parameters
        ----------
        feature : MoonFeature
            The feature to center on
        """
        if self.rt is None or self.moon_rotation_matrix is None:
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
        scene_pos = self.moon_rotation_matrix @ original_pos
        
        # Get current camera
        cam = self.rt.get_camera("cam1")
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        
        # Calculate new camera distance based on feature size
        # Aim to have feature fill about 30% of the view
        feature_radius_scene = feature.angle / 2 * (self.moon_radius / 90)  # Rough conversion
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
                f"  Altitude: {eph.alt:.2f}°\n"
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
        
        # Line thickness (thin lines)
        line_radius = 0.006
        
        # Add latitude lines
        for i, points in enumerate(self.moon_grid.lat_lines):
            name = f"grid_lat_{i}"
            self.rt.set_data(name, pos=points, r=line_radius, 
                            c=GRID_COLOR, geom="BezierChain", mat="grid_material")
        
        # Add longitude lines
        for i, points in enumerate(self.moon_grid.lon_lines):
            name = f"grid_lon_{i}"
            self.rt.set_data(name, pos=points, r=line_radius,
                            c=GRID_COLOR, geom="BezierChain", mat="grid_material")
        
        # Add latitude labels
        label_radius = 0.012
        for i, segments in enumerate(self.moon_grid.lat_labels):
            for j, seg in enumerate(segments):
                name = f"grid_lat_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=label_radius,
                                c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        # Add longitude labels
        for i, segments in enumerate(self.moon_grid.lon_labels):
            for j, seg in enumerate(segments):
                name = f"grid_lon_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=label_radius,
                                c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        # Add north pole "N" label
        for j, seg in enumerate(self.moon_grid.N):
            name = f"grid_north_label_{j}"
            self.rt.set_data(name, pos=seg, r=label_radius,
                            c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        self.moon_grid_visible = True
        
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
        line_radius = 0.015 if visible else 0.0
        
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
        label_radius = 0.012 if visible else 0.0
        
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
    
    def toggle_grid(self):
        """Toggle the selenographic grid visibility."""
        self.show_moon_grid(not self.moon_grid_visible)
    
    def setup_standard_labels(self):
        """
        Create standard feature labels for Moon features with standard_label=true.
        """
        if self.rt is None:
            print("Renderer not initialized")
            return
            
        self.standard_labels = create_standard_labels(
            self.moon_features,
            moon_radius=self.moon_radius,
            offset=0.0
        )
        
        # Create an emissive material for the labels (so they glow and are visible in shadow)
        m_label = m_flat.copy()
        self.rt.update_material("standard_label_material", m_label)
        
        # Line thickness for labels
        label_radius = 0.008
        
        # Color for standard labels (white/light gray for visibility)
        label_color = [0.85, 0.85, 0.85]
        
        # Add each label's segments
        for i, standard_label in enumerate(self.standard_labels):
            for j, seg in enumerate(standard_label):
                name = f"standard_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=label_radius,
                                c=label_color, geom="SegmentChain", mat="standard_label_material")
        
        self.standard_labels_visible = True
        
        # Apply current Moon orientation to the labels
        self.update_standard_labels_orientation()
    
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
        label_radius = 0.008 if visible else 0.0
        
        for i, standard_label in enumerate(self.standard_labels):
            for j in range(len(standard_label)):
                name = f"standard_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        self.standard_labels_visible = visible
    
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
            
        # Generate spot labels data
        self.spot_labels = create_spot_labels(
            self.moon_features,
            moon_radius=self.moon_radius,
            offset=0.0
        )
        
        # Create an emissive material for the labels (so they glow and are visible in shadow)
        m_label = m_flat.copy()
        self.rt.update_material("spot_label_material", m_label)
        
        # Line thickness for labels
        label_radius = 0.008
        
        # Color for spot labels (yellow/gold for visibility)
        label_color = [1.0, 0.9, 0.3]
        
        # Add each label's segments
        for i, spot_label in enumerate(self.spot_labels):
            for j, seg in enumerate(spot_label):
                name = f"spot_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=label_radius,
                                c=label_color, geom="SegmentChain", mat="spot_label_material")
        
        self.spot_labels_visible = True
        
        # Apply current Moon orientation to the labels
        self.update_spot_labels_orientation()
    
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
        label_radius = 0.008 if visible else 0.0
        
        for i, spot_label in enumerate(self.spot_labels):
            for j in range(len(spot_label)):
                name = f"spot_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        self.spot_labels_visible = visible
    
    def toggle_spot_labels(self):
        """Toggle the spot labels visibility."""
        self.show_spot_labels(not self.spot_labels_visible)
    
    def update_spot_labels_orientation(self):
        """
        Update spot labels to match current Moon orientation.
        
        This should be called after update_view() to rotate the labels
        along with the Moon surface.
        """
        if self.rt is None or self.spot_labels is None or not self.spot_labels_visible:
            return
        
        R = self.calculate_moon_rotation()

        if R is None:
            return
        
        # Update spot labels
        for i, spot_label in enumerate(self.spot_labels):
            for j, orig_seg in enumerate(spot_label):
                name = f"spot_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
    
    def update_standard_labels_orientation(self):
        """
        Update standard labels to match current Moon orientation.
        
        This should be called after update_view() to rotate the labels
        along with the Moon surface.
        """
        if self.rt is None or self.standard_labels is None or not self.standard_labels_visible:
            return
        
        R = self.calculate_moon_rotation()

        if R is None:
            return
        
        # Update standard labels
        for i, standard_label in enumerate(self.standard_labels):
            for j, orig_seg in enumerate(standard_label):
                name = f"standard_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
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
        
        # Generate pin digit segments (left-bottom corner at cursor position)
        pin_segments = create_single_digit_on_sphere(
            digit=digit,
            lat=lat,
            lon=lon,
            moon_radius=self.moon_radius,
            offset=0.0
        )
        
        # Store pin data (original segments for rotation updates)
        self.pins[digit] = pin_segments
        
        # Create material for pins if not already created
        m_pin = m_flat.copy()
        self.rt.update_material("pin_material", m_pin)
        
        # Line thickness for pins
        pin_radius = 0.012
        
        # Apply Moon rotation to segments and add to renderer
        R = self.calculate_moon_rotation()
        
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
        pin_radius = 0.012 if visible else 0.0
        
        for digit, pin_segments in self.pins.items():
            for j in range(len(pin_segments)):
                name = f"pin_{digit}_{j}"
                try:
                    self.rt.update_data(name, r=pin_radius)
                except:
                    pass
        
        self.pins_visible = visible
        
        # Update status bar to reflect pin mode change
        if self.rt is not None:
            current_status = self.rt._status_action_text.get()
            # Replace the pin mode indicator at the end
            if "[Pins ON]" in current_status:
                new_status = current_status.replace("[Pins ON]", "[Pins OFF]" if not visible else "[Pins ON]")
            elif "[Pins OFF]" in current_status:
                new_status = current_status.replace("[Pins OFF]", "[Pins ON]" if visible else "[Pins OFF]")
            else:
                # Append pin mode if not present
                pin_mode = "[Pins ON]" if visible else "[Pins OFF]"
                new_status = f"{current_status:76}{pin_mode}"
            self.rt._status_action_text.set(new_status)
    
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
        
        R = self.calculate_moon_rotation()
        
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
    
    def toggle_invert(self):
        """Invert (flip) the view upside down"""
        self.inverted = not self.inverted
        
        if self.rt is None:
            return
        
        # Flip the camera up vector to invert the view
        # Use setup_camera to get current camera and flip its up vector
        new_up = [0, 0, -1] if self.inverted else [0, 0, 1]
        self.rt.setup_camera("cam1", up=new_up)
    
    def find_feature_at(self, lat: float, lon: float) -> Optional[MoonFeature]:
        """
        Find a Moon feature at the given selenographic coordinates.
        
        When multiple features overlap at the given position, returns the
        feature with the smallest angular size (most specific feature).
        
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
        matching_features = []
        
        for moon_feature in self.moon_features:
            # Calculate angular distance from feature center
            dlat = lat - moon_feature.lat
            dlon = lon - moon_feature.lon
            # Simple approximation for small angles
            # Use cos(lat) correction for longitude
            cos_lat = np.cos(np.radians(moon_feature.lat))
            angular_dist = np.sqrt(dlat**2 + (dlon * cos_lat)**2)
            
            # Check if within feature's angular radius (half of angular size)
            if angular_dist <= moon_feature.angle / 2:
                matching_features.append(moon_feature)
        
        if not matching_features:
            return None
        
        # Return the feature with the smallest angular size
        smallest_feature = min(matching_features, key=lambda f: f.angle)
        return smallest_feature
    
    def reset_camera_position(self):
        """
        Reset the camera to its initial position.
        
        Restores the camera to the position it had when the view was first set up,
        undoing any mouse rotation/panning performed by the user.
        """

        cp = self.initial_camera_params

        if self.rt is None or cp is None:
            return
        
        # Restore initial camera parameters
        # Adjust up vector based on current inversion state
        up = cp.up[:]
        if self.inverted:
            up = [u * -1 for u in up]
        
        self.rt.setup_camera("cam1", eye=cp.eye, target=cp.target, up=up, fov=cp.fov)
    
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
        
        # Restore default camera parameters
        # Adjust up vector based on current inversion state
        up = cp.up[:]
        if self.inverted:
            up = [u * -1 for u in up]
        
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
        if self.moon_rotation_matrix_inv is None:
            return None, None
        
        # The hit position is on the rotated Moon surface
        # We need to transform it back to the original Moon coordinates
        # where north pole is +Z and prime meridian faces -Y
        hit_pos = np.array([hx, hy, hz])
        
        # Normalize to unit sphere (Moon surface)
        r = np.linalg.norm(hit_pos)
        if r < 0.1:  # Too close to origin, not a valid hit
            return None, None
        
        hit_normalized = hit_pos / r
        
        # Transform back to original Moon coordinates
        original_pos = self.moon_rotation_matrix_inv @ hit_normalized
        
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
    
    def update_moon_grid_orientation(self):
        """
        Update grid lines to match current Moon orientation.
        
        This should be called after update_view() to rotate the grid
        along with the Moon surface.
        """
        if self.rt is None or self.moon_grid is None or not self.moon_grid_visible:
            return
        
        R = self.calculate_moon_rotation()

        if R is None:
            return
        
        def update_lines(lines, prefix):
            for i, orig_points in enumerate(lines):
                name = f"{prefix}_{i}"
                rotated = (R @ orig_points.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
        
        def update_nested_segments(segments_list, prefix):
            for i, segments in enumerate(segments_list):
                for j, orig_seg in enumerate(segments):
                    name = f"{prefix}_{i}_{j}"
                    rotated = (R @ orig_seg.T).T
                    try:
                        self.rt.update_data(name, pos=rotated)
                    except:
                        pass
        
        update_lines(self.moon_grid.lat_lines, "grid_lat")
        update_lines(self.moon_grid.lon_lines, "grid_lon")
        update_nested_segments(self.moon_grid.lat_labels, "grid_lat_label")
        update_nested_segments(self.moon_grid.lon_labels, "grid_lon_label")
        update_lines(self.moon_grid.N, "grid_north_label")
