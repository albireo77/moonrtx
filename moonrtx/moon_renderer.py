import numpy as np
import cv2
import os
from typing import Optional
from datetime import datetime
from datetime import timezone

from moonrtx.types import MoonEphemeris
from moonrtx.types import MoonFeature
from moonrtx.astro import calculate_moon_ephemeris

from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse
from plotoptix.materials import m_flat
from plotoptix.utils import read_image, make_color_2d

GRID_COLOR = [0.50, 0.50, 0.50]
MOON_FILL_FRACTION = 0.9  # Moon fills 90% of window height (5% margins top/bottom)
APP_NAME = "MoonRTX"

def run_renderer(dt_local: datetime,
                 lat: float,
                 lon: float,
                 elevation_file: str,
                 color_file: str,
                 starmap_file: str,
                 downscale: int,
                 light_intensity: int,
                 moon_features: list) -> TkOptiX:
    """
    Quick function to render the Moon for a specific time and location.
    
    Parameters
    ----------
    dt_local : datetime
        Local time
    lat, lon : float
        Observer latitude and longitude in degrees
    elevation_file, color_file, starmap_file : str
        Paths to data files
    downscale : int
        Elevation downscale factor
    light_intensity : int
        Light intensity  
    moon_features : list
        Moon features (craters, mounts etc.)   
    Returns
    -------
    TkOptiX
        The renderer instance
    """

    moon_renderer = MoonRenderer(
        elevation_file=elevation_file,
        color_file=color_file,
        starmap_file=starmap_file,
        downscale=downscale,
        moon_features=moon_features
    )
    
    # Setup renderer
    moon_renderer.setup_renderer()
    
    # Set view
    moon_renderer.update_view(dt_local=dt_local, lat=lat, lon=lon, light_intensity=light_intensity)
    
    # Print info
    print("\n" + moon_renderer.get_info())
    print("\nKeys and mouse:")
    print("  G - Toggle selenographic grid")
    print("  L - Toggle standard labels")
    print("  S - Toggle spot labels")
    print("  I - Upside down view")
    print("  C - Reset scene to initial state")
    print("  F12 - Save image")
    print("  Hold and drag left mouse button - Rotate the eye around Moon")
    print("  Hold shift + left mouse button and drag up/down - Zoom out/in")
    print("  Hold and drag right mouse button - Rotate Moon around the eye")
    print("  Hold shift + right mouse button and drag up/down - Move eye backward/forward")
    print("  Hold ctrl + left mouse button and drag up/down - Change focus distance (sharpens Moon surface details)")
    
    original_key_handler = moon_renderer.rt._gui_key_pressed
    def custom_key_handler(event):
        if event.keysym.lower() == 'g':
            moon_renderer.toggle_grid()
        elif event.keysym.lower() == 'l':
            moon_renderer.toggle_standard_labels()
        elif event.keysym.lower() == 's':
            moon_renderer.toggle_spot_labels()
        elif event.keysym.lower() == 'i':
            moon_renderer.toggle_invert()
        elif event.keysym.lower() == 'c':
            moon_renderer.reset_camera_position()
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
            
            # Build status text with fixed-width columns
            # Column 1: Coordinates (fixed width)
            # Column 2: Feature name (if hovering over one)
            
            coord_column = ""
            feature_data = ""
            # Check if we hit something (distance > 0 means valid hit)
            if hd > 0:
                lat, lon = moon_renderer.hit_to_selenographic(hx, hy, hz)
                if lat is not None and lon is not None:
                    # Check if hovering over a named feature
                    feature = moon_renderer.find_feature_at(lat, lon)
                    feature_name = feature.name if feature is not None and feature.status_bar else ""
                    if feature_name:
                        feature_size_km = feature.angle * 30.34
                        feature_data = f"{feature_name} (Size = {feature_size_km:.1f} km)"
                    else:
                        feature_data = ""
                    # Format: "Lat: XX.XX° N/S  Lon: XXX.XX° E/W"
                    lat_dir = 'N' if lat >= 0 else 'S'
                    lon_dir = 'E' if lon >= 0 else 'W'
                    coord_column = f"Lat: {abs(lat):5.2f}° {lat_dir}  Lon: {abs(lon):6.2f}° {lon_dir}"
            
            # Build status: coordinates first (fixed width), then feature name
            status_text = f"{coord_column:36}{feature_data}"
            
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
    
    return {
        'eye': camera_eye,
        'target': camera_target,
        'up': camera_up,
        'light_pos': light_pos
    }


def load_elevation_data(filepath: str, downscale: int) -> np.ndarray:
    """
    Load and process the Moon elevation data.
    
    Parameters
    ----------
    filepath : str
        Path to the elevation TIFF file
    downscale : int
        Downscale factor (2-3 recommended for most GPUs)
        
    Returns
    -------
    np.ndarray
        Processed elevation data normalized for displacement mapping
    """
    print(f"Loading elevation data from {filepath}...")
    elev_src = read_image(filepath)
    
    if elev_src is None:
        raise ValueError(f"Failed to read elevation file: {filepath}")
    
    print(f"  Original dimensions: {elev_src.shape}")
    print(f"  Size: {elev_src.nbytes / (1024**3):.2f} GB")
    
    # Convert to signed 16-bit and normalize
    elev_src.dtype = np.int16
    scale = 1. / np.iinfo(np.int16).max
    
    h = elev_src.shape[0] // downscale
    w = elev_src.shape[1] // downscale
    
    # Downscale by averaging
    elevation = elev_src.reshape(1, h, downscale, w, downscale).mean(
        4, dtype=np.float32).mean(2, dtype=np.float32).reshape(h, w)
    elevation *= scale
    
    # Release source memory
    elev_src = None
    
    print(f"  Downscaled dimensions: {elevation.shape}")
    print(f"  Downscaled size: {elevation.nbytes / (1024**3):.2f} GB")
    
    # Normalize for displacement mapping
    # Real Moon: radius ~1737 km, max relief ~20 km = ~1.15% of radius
    # Increase this value for more dramatic terrain, decrease for flatter appearance
    displacement_range = 0.0115  # 1.15% of radius
    
    rmin = np.min(elevation)
    rmax = np.max(elevation)
    rv = rmax - rmin
    
    elevation += rmin
    elevation *= displacement_range / rv
    elevation += (1.0 - displacement_range)
    
    return elevation


def load_color_data(filepath: str, gamma: float = 2.2) -> np.ndarray:
    """
    Load and process the Moon color/albedo data.
    
    Parameters
    ----------
    filepath : str
        Path to the color TIFF file
    gamma : float
        Gamma correction value
        
    Returns
    -------
    np.ndarray
        Processed color data ready for texturing
    """
    print(f"Loading color data from {filepath}...")
    color_src = cv2.imread(filepath)
    
    if color_src is None:
        raise ValueError(f"Failed to read color file: {filepath}")
    
    # Convert BGR to RGB and normalize
    color_src = color_src[..., ::-1].astype(np.float32)
    color_src = 0.2 + (0.75 / 255) * color_src
    
    print(f"  Dimensions: {color_src.shape}")
    print(f"  Size: {color_src.nbytes / (1024**3):.2f} GB")
    
    # Prepare for texture
    color_data = make_color_2d(color_src, gamma=gamma, channel_order="RGBA")
    color_data *= 255
    
    return color_data.astype(np.uint8)


def load_starmap(filepath: str, target_width: int = 10240) -> Optional[np.ndarray]:
    """
    Load and process the star map for background.
    
    Parameters
    ----------
    filepath : str
        Path to the star map TIFF file
    target_width : int
        Target width for downscaling (to save memory)
        
    Returns
    -------
    np.ndarray or None
        Processed star map, or None if file not found
    """
    if not os.path.isfile(filepath):
        print(f"Star map not found: {filepath}")
        return None
    
    print(f"Loading star map from {filepath}...")
    star_src = cv2.imread(filepath)
    
    if star_src is None:
        print(f"Failed to read star map: {filepath}")
        return None
    
    # Convert BGR to RGB and normalize
    star_src = star_src[..., ::-1].astype(np.float32)
    star_src *= 1 / 255
    
    # Downscale if needed
    if target_width < star_src.shape[1]:
        target_height = int(star_src.shape[0] * target_width / star_src.shape[1])
        star_map = cv2.resize(star_src, (target_width, target_height), 
                             interpolation=cv2.INTER_CUBIC)
        np.clip(star_map, 0, 1, out=star_map)
    else:
        star_map = star_src
    
    print(f"  Dimensions: {star_map.shape}")
    
    return star_map


def create_digit_segments(digit: str, scale: float = 0.1) -> list:
    """
    Create line segments for a digit (7-segment style display).
    
    Returns list of (start, end) tuples in local 2D coordinates,
    where the digit is centered at origin, width ~0.6*scale, height ~1.0*scale.
    """
    # 7-segment layout:
    #  _a_
    # |   |
    # f   b
    # |_g_|
    # |   |
    # e   c
    # |_d_|
    
    w = 0.3 * scale  # half width
    h = 0.5 * scale  # half height
    
    # Segment endpoints (centered at origin)
    segments = {
        'a': ((-w, h), (w, h)),           # top
        'b': ((w, h), (w, 0)),            # upper right
        'c': ((w, 0), (w, -h)),           # lower right
        'd': ((-w, -h), (w, -h)),         # bottom
        'e': ((-w, 0), (-w, -h)),         # lower left
        'f': ((-w, h), (-w, 0)),          # upper left
        'g': ((-w, 0), (w, 0)),           # middle
    }
    
    # Which segments are on for each digit/letter
    digit_segments = {
        '0': 'abcdef',
        '1': 'bc',
        '2': 'abged',
        '3': 'abgcd',
        '4': 'fgbc',
        '5': 'afgcd',
        '6': 'afgedc',
        '7': 'abc',
        '8': 'abcdefg',
        '9': 'abcdfg',
        '-': 'g',
        'N': 'N',  # Special case handled below
    }
    
    # Handle letter N specially (diagonal stroke)
    if digit == 'N':
        return [
            ((-w, -h), (-w, h)),   # left vertical
            ((-w, h), (w, -h)),    # diagonal
            ((w, -h), (w, h)),     # right vertical
        ]
    
    # Handle other letters with custom segment definitions
    letter_definitions = {
        'A': [((-w, -h), (-w, h*0.3)), ((-w, h*0.3), (0, h)), ((0, h), (w, h*0.3)), ((w, h*0.3), (w, -h)), ((-w, 0), (w, 0))],
        'B': [((-w, -h), (-w, h)), ((-w, h), (w*0.6, h)), ((w*0.6, h), (w*0.6, h*0.1)), ((w*0.6, h*0.1), (-w, 0)), ((-w, 0), (w*0.6, 0)), ((w*0.6, 0), (w*0.6, -h*0.9)), ((w*0.6, -h*0.9), (-w, -h))],
        'C': [((w, h), (-w*0.3, h)), ((-w*0.3, h), (-w, h*0.5)), ((-w, h*0.5), (-w, -h*0.5)), ((-w, -h*0.5), (-w*0.3, -h)), ((-w*0.3, -h), (w, -h))],
        'D': [((-w, -h), (-w, h)), ((-w, h), (w*0.3, h)), ((w*0.3, h), (w, h*0.5)), ((w, h*0.5), (w, -h*0.5)), ((w, -h*0.5), (w*0.3, -h)), ((w*0.3, -h), (-w, -h))],
        'E': [((-w, -h), (-w, h)), ((-w, h), (w, h)), ((-w, 0), (w*0.6, 0)), ((-w, -h), (w, -h))],
        'F': [((-w, -h), (-w, h)), ((-w, h), (w, h)), ((-w, 0), (w*0.6, 0))],
        'G': [((w, h*0.6), (w*0.3, h)), ((w*0.3, h), (-w, h*0.5)), ((-w, h*0.5), (-w, -h*0.5)), ((-w, -h*0.5), (w*0.3, -h)), ((w*0.3, -h), (w, -h*0.5)), ((w, -h*0.5), (w, 0)), ((w, 0), (0, 0))],
        'H': [((-w, -h), (-w, h)), ((w, -h), (w, h)), ((-w, 0), (w, 0))],
        'I': [((-w*0.5, h), (w*0.5, h)), ((0, h), (0, -h)), ((-w*0.5, -h), (w*0.5, -h))],
        'J': [((w*0.5, h), (w*0.5, -h*0.5)), ((w*0.5, -h*0.5), (0, -h)), ((0, -h), (-w*0.5, -h*0.5))],
        'K': [((-w, -h), (-w, h)), ((-w, 0), (w, h)), ((-w, 0), (w, -h))],
        'L': [((-w, h), (-w, -h)), ((-w, -h), (w, -h))],
        'M': [((-w, -h), (-w, h)), ((-w, h), (0, 0)), ((0, 0), (w, h)), ((w, h), (w, -h))],
        'O': [((-w, h*0.5), (-w, -h*0.5)), ((-w, -h*0.5), (-w*0.3, -h)), ((-w*0.3, -h), (w*0.3, -h)), ((w*0.3, -h), (w, -h*0.5)), ((w, -h*0.5), (w, h*0.5)), ((w, h*0.5), (w*0.3, h)), ((w*0.3, h), (-w*0.3, h)), ((-w*0.3, h), (-w, h*0.5))],
        'P': [((-w, -h), (-w, h)), ((-w, h), (w*0.6, h)), ((w*0.6, h), (w*0.6, h*0.1)), ((w*0.6, h*0.1), (-w, 0))],
        'Q': [((-w, h*0.5), (-w, -h*0.5)), ((-w, -h*0.5), (-w*0.3, -h)), ((-w*0.3, -h), (w*0.3, -h)), ((w*0.3, -h), (w, -h*0.5)), ((w, -h*0.5), (w, h*0.5)), ((w, h*0.5), (w*0.3, h)), ((w*0.3, h), (-w*0.3, h)), ((-w*0.3, h), (-w, h*0.5)), ((w*0.3, -h*0.3), (w*0.8, -h*0.9))],
        'R': [((-w, -h), (-w, h)), ((-w, h), (w*0.6, h)), ((w*0.6, h), (w*0.6, h*0.1)), ((w*0.6, h*0.1), (-w, 0)), ((-w*0.2, 0), (w, -h))],
        'S': [((w, h*0.7), (w*0.3, h)), ((w*0.3, h), (-w*0.3, h)), ((-w*0.3, h), (-w, h*0.5)), ((-w, h*0.5), (-w, h*0.2)), ((-w, h*0.2), (w, -h*0.2)), ((w, -h*0.2), (w, -h*0.5)), ((w, -h*0.5), (w*0.3, -h)), ((w*0.3, -h), (-w*0.3, -h)), ((-w*0.3, -h), (-w, -h*0.7))],
        'T': [((-w, h), (w, h)), ((0, h), (0, -h))],
        'U': [((-w, h), (-w, -h*0.5)), ((-w, -h*0.5), (-w*0.3, -h)), ((-w*0.3, -h), (w*0.3, -h)), ((w*0.3, -h), (w, -h*0.5)), ((w, -h*0.5), (w, h))],
        'V': [((-w, h), (0, -h)), ((0, -h), (w, h))],
        'W': [((-w, h), (-w*0.5, -h)), ((-w*0.5, -h), (0, h*0.3)), ((0, h*0.3), (w*0.5, -h)), ((w*0.5, -h), (w, h))],
        'X': [((-w, h), (w, -h)), ((-w, -h), (w, h))],
        'Y': [((-w, h), (0, 0)), ((w, h), (0, 0)), ((0, 0), (0, -h))],
        'Z': [((-w, h), (w, h)), ((w, h), (-w, -h)), ((-w, -h), (w, -h))],
        ' ': [],  # Space - no segments
        "'": [((0, h), (0, h*0.5))],  # Apostrophe
        '>': [((-w, h*0.4), (w, 0)), ((w, 0), (-w, -h*0.4))],  # Right arrow
        '<': [((w, h*0.4), (-w, 0)), ((-w, 0), (w, -h*0.4))],  # Left arrow
    }
    
    if digit in letter_definitions:
        return letter_definitions[digit]
    
    if digit not in digit_segments:
        return []
    
    return [segments[s] for s in digit_segments[digit]]


def create_number_on_sphere(number: int, 
                            lat: float, lon: float,
                            moon_radius: float,
                            offset: float,
                            digit_scale: float = 0.3,
                            spacing: float = 0.25) -> list:
    """
    Create 3D line segments for a number positioned on the Moon sphere.
    
    Parameters
    ----------
    number : int
        The number to display (can be negative)
    lat, lon : float
        Selenographic coordinates in degrees
    moon_radius : float
        Radius of the Moon
    offset : float
        Height above surface (fraction of radius)
    digit_scale : float
        Size of digits
    spacing : float
        Spacing between digits (as fraction of scale)
        
    Returns
    -------
    list
        List of numpy arrays, each containing points for one line segment
    """
    r = moon_radius * (1 + offset + 0.005)  # Slightly above grid lines
    
    # Convert number to string
    num_str = str(number)
    
    # Calculate total width for centering
    num_digits = len(num_str)
    total_width = num_digits * digit_scale * (1 + spacing) - digit_scale * spacing
    
    all_segments = []
    
    # Position for each digit
    for i, digit in enumerate(num_str):
        # Local x offset for this digit (centered)
        local_x = -total_width/2 + i * digit_scale * (1 + spacing) + digit_scale * 0.5
        
        # Get segments for this digit
        digit_segs = create_digit_segments(digit, digit_scale)
        
        for (p1_local, p2_local) in digit_segs:
            # Transform local 2D to 3D on sphere surface
            # Local coordinates: x = along latitude, z = up (along meridian)
            points_3d = []
            for p_local in [p1_local, p2_local]:
                lx, lz = p_local
                lx += local_x  # Apply digit offset
                
                # Convert local offset to lat/lon offset
                # Approximate: at this latitude, 1 unit of local x = some degrees of longitude
                # and 1 unit of local z = some degrees of latitude
                lat_offset = np.degrees(lz / r)
                lon_offset = np.degrees(lx / (r * np.cos(np.radians(lat)))) if abs(lat) < 89 else 0
                
                new_lat = lat + lat_offset
                new_lon = lon + lon_offset
                
                # Convert to 3D
                lat_rad = np.radians(new_lat)
                lon_rad = np.radians(new_lon)
                
                x = r * np.cos(lat_rad) * np.sin(lon_rad)
                y = -r * np.cos(lat_rad) * np.cos(lon_rad)
                z = r * np.sin(lat_rad)
                
                points_3d.append([x, y, z])
            
            all_segments.append(np.array(points_3d))
    
    return all_segments

def create_text_on_sphere(text: str, 
                          lat: float, lon: float,
                          moon_radius: float,
                          offset: float,
                          char_scale: float = 0.15,
                          spacing: float = 0.15) -> list:
    """
    Create 3D line segments for text positioned on the Moon sphere.
    Text starts horizontally at the given lon (not centered).
    """

    r = moon_radius * (1 + offset + 0.005)
    all_segments = []

    char_width = char_scale * (1 + spacing)

    for i, char in enumerate(text.upper()):
        # Local x offset: text starts at lon and grows eastward
        local_x = i * char_width

        char_segs = create_digit_segments(char, char_scale)

        for seg in char_segs:
            if len(seg) != 2:
                continue

            p1_local, p2_local = seg
            points_3d = []

            for p_local in (p1_local, p2_local):
                lx, lz = p_local
                lx += local_x

                # Convert local offsets to lat/lon offsets
                lat_offset = np.degrees(lz / r)
                lon_offset = (
                    np.degrees(lx / (r * np.cos(np.radians(lat))))
                    if abs(lat) < 89 else 0
                )

                new_lat = lat + lat_offset
                new_lon = lon + lon_offset

                lat_rad = np.radians(new_lat)
                lon_rad = np.radians(new_lon)

                x = r * np.cos(lat_rad) * np.sin(lon_rad)
                y = -r * np.cos(lat_rad) * np.cos(lon_rad)
                z = r * np.sin(lat_rad)

                points_3d.append([x, y, z])

            all_segments.append(np.array(points_3d))

    return all_segments

def create_centered_text_on_sphere(text: str, 
                          lat: float, lon: float,
                          moon_radius: float,
                          offset: float,
                          char_scale: float = 0.15,
                          spacing: float = 0.15) -> list:
    """
    Create 3D line segments for text positioned on the Moon sphere.
    
    Parameters
    ----------
    text : str
        The text to display
    lat, lon : float
        Selenographic coordinates in degrees (center position of text)
    moon_radius : float
        Radius of the Moon
    offset : float
        Height above surface (fraction of radius)
    char_scale : float
        Size of characters
    spacing : float
        Spacing between characters (as fraction of scale)
        
    Returns
    -------
    list
        List of numpy arrays, each containing points for one line segment
    """
    r = moon_radius * (1 + offset + 0.005)  # Slightly above grid lines
    
    all_segments = []
    
    # Calculate total width for centering
    char_width = char_scale * (1 + spacing)
    num_chars = len(text)
    total_width = num_chars * char_width - char_scale * spacing  # subtract last spacing
    
    for i, char in enumerate(text.upper()):
        # Local x offset for this character (centered around origin)
        local_x = i * char_width - total_width / 2 + char_width / 2
        
        # Get segments for this character
        char_segs = create_digit_segments(char, char_scale)
        
        for seg in char_segs:
            if len(seg) != 2:
                continue
            p1_local, p2_local = seg
            # Transform local 2D to 3D on sphere surface
            # Local coordinates: x = along latitude (positive = east), z = up (along meridian)
            points_3d = []
            for p_local in [p1_local, p2_local]:
                lx, lz = p_local
                lx += local_x  # Apply character offset (already centered)
                
                # Convert local offset to lat/lon offset
                lat_offset = np.degrees(lz / r)
                lon_offset = np.degrees(lx / (r * np.cos(np.radians(lat)))) if abs(lat) < 89 else 0
                
                new_lat = lat + lat_offset
                new_lon = lon + lon_offset
                
                # Convert to 3D
                lat_rad = np.radians(new_lat)
                lon_rad = np.radians(new_lon)
                
                x = r * np.cos(lat_rad) * np.sin(lon_rad)
                y = -r * np.cos(lat_rad) * np.cos(lon_rad)
                z = r * np.sin(lat_rad)
                
                points_3d.append([x, y, z])
            
            all_segments.append(np.array(points_3d))
    
    return all_segments


def create_standard_labels(moon_features: list, moon_radius: float = 10.0, offset: float = 0.0) -> list:
    """
    Create labels for Moon features marked with standard label
    
    The label is centered at the feature's (latitude, longitude) position.
    
    Parameters
    ----------
    moon_features : list
        List of Moon features
    moon_radius : float
        Radius of the Moon sphere
    offset : float
        Height offset above surface
        
    Returns
    -------
    list
        List of standard labels
    """
    standard_labels = []
    
    for moon_feature in moon_features:

        if not moon_feature.standard_label:
            continue
        
        name = moon_feature.name
        feat_lat = moon_feature.lat
        feat_lon = moon_feature.lon
        
        # Label centered at feature position
        label_lat = feat_lat
        label_lon = feat_lon
        
        # Create text segments for this label
        # Use small character scale for readability
        char_scale = 0.12  # Small but readable
        label_segments = create_centered_text_on_sphere(
            text=name,
            lat=label_lat, 
            lon=label_lon,
            moon_radius=moon_radius,
            offset=offset,
            char_scale=char_scale,
            spacing=0.1
        )
        standard_labels.append(label_segments)
    
    return standard_labels

def create_spot_labels(moon_features: list, moon_radius: float = 10.0, offset: float = 0.0) -> list:
    """
    Create labels for Moon features marked as spot label
    
    Parameters
    ----------
    moon_features : list
        List of Moon features
    moon_radius : float
        Radius of the Moon sphere
    offset : float
        Height offset above surface
        
    Returns
    -------
    list
        List of spot labels.
    """
    spot_labels = []
    
    for moon_feature in moon_features:

        if not moon_feature.spot_label:
            continue
        
        name = moon_feature.name
        feat_lat = moon_feature.lat
        feat_lon = moon_feature.lon
        angular_size = moon_feature.angle
        
        label_text = "< " + name
        label_lon = feat_lon + angular_size
        label_lat = feat_lat
        
        # Create text segments for this label
        char_scale = 0.10  # Small but readable
        label_segments = create_text_on_sphere(
            label_text, 
            lat=label_lat, 
            lon=label_lon,
            moon_radius=moon_radius,
            offset=offset,
            char_scale=char_scale,
            spacing=0.1
        )
        spot_labels.append(label_segments)
    
    return spot_labels


def create_selenographic_grid(moon_radius: float = 10.0,
                               lat_step: float = 15.0,
                               lon_step: float = 15.0,
                               points_per_line: int = 100,
                               offset: float = 0.02) -> dict:
    """
    Create selenographic coordinate grid lines for the Moon.
    
    Generates latitude and longitude lines as 3D points on a sphere
    slightly above the Moon's surface (to avoid z-fighting).
    
    Parameters
    ----------
    moon_radius : float
        Radius of the Moon sphere
    lat_step : float
        Spacing between latitude lines in degrees
    lon_step : float
        Spacing between longitude lines in degrees
    points_per_line : int
        Number of points per line (more = smoother)
    offset : float
        Offset above surface (fraction of radius)
        
    Returns
    -------
    dict
        Dictionary with 'lat_lines', 'lon_lines' containing lists of point arrays
    """
    r = moon_radius * (1 + offset)  # Slightly above surface
    
    lat_lines = []
    lon_lines = []
    
    # Latitude lines (circles at constant latitude)
    # From -60° to +60° (skip poles where circles become very small)
    for lat in np.arange(-60, 61, lat_step):
        if lat == 90 or lat == -90:
            continue
        lat_rad = np.radians(lat)
        cos_lat = np.cos(lat_rad)
        z = r * np.sin(lat_rad)
        r_circle = r * cos_lat
        
        # Full circle at this latitude
        points = []
        for lon in np.linspace(0, 360, points_per_line, endpoint=True):
            lon_rad = np.radians(lon)
            x = r_circle * np.sin(lon_rad)  # lon=0 faces -Y
            y = -r_circle * np.cos(lon_rad)  # -Y is toward camera
            points.append([x, y, z])
        
        lat_lines.append(np.array(points))
    
    # Longitude lines (great circles at constant longitude)
    # Full 360° but only draw visible portion (front half approximately)
    for lon in np.arange(0, 360, lon_step):
        lon_rad = np.radians(lon)
        
        points = []
        # From south pole to north pole
        for lat in np.linspace(-90, 90, points_per_line):
            lat_rad = np.radians(lat)
            cos_lat = np.cos(lat_rad)
            z = r * np.sin(lat_rad)
            
            x = r * cos_lat * np.sin(lon_rad)
            y = -r * cos_lat * np.cos(lon_rad)  # -Y is toward camera
            points.append([x, y, z])
        
        lon_lines.append(np.array(points))
    
    # Create labels for latitude lines
    # Labels are placed at longitudes 0, 90, 180, and -90 (270) degrees
    lat_labels = []
    lat_label_values = []
    label_longitudes = [0, 90, 180, -90]  # Longitudes where latitude labels are placed
    for label_lon in label_longitudes:
        for lat in np.arange(-60, 61, lat_step):
            if lat == 90 or lat == -90:
                continue
            # Place label slightly offset from the meridian
            segments = create_number_on_sphere(
                int(lat), lat=lat+1, lon=label_lon + lat_step/2-1,
                moon_radius=moon_radius, offset=offset,
                digit_scale=0.125
            )
            lat_labels.append(segments)
            lat_label_values.append(int(lat))
    
    # Create labels for longitude lines
    # Labels are placed on the right side of the meridian (positive latitude offset)
    lon_labels = []
    lon_label_values = []    
    for lon in np.arange(0, 360, lon_step):
        # Normalize longitude to -180 to 180 for display
        display_lon = lon if lon <= 180 else lon - 360
        # Place label on the right side of the meridian
        # For negative values, add extra offset to account for the minus sign width
        lon_offset = 2 if display_lon < 0 else 1
        segments = create_number_on_sphere(
            int(display_lon), lat=lat_step/2-1, lon=display_lon+lon_offset,
            moon_radius=moon_radius, offset=offset,
            digit_scale=0.125
        )
        lon_labels.append(segments)
        lon_label_values.append(int(display_lon))
    
    # Create north pole label "N" - vertically oriented above the pole
    # The "N" will be positioned above the north pole, standing upright
    # facing the camera (which looks along +Y toward the Moon)
    n_scale = 0.50 * moon_radius / 10.0
    north_pole_label = create_digit_segments('N', scale=n_scale)
    
    # Position the "N" above the north pole
    # The letter will be in the XZ plane (facing -Y toward camera)
    # with its center at (0, y_offset, z_base) where z_base is above the pole
    r_label = moon_radius * (1 + offset + 0.005)
    z_base = r_label + n_scale * 0.6  # Position base of "N" just above the pole
    y_offset = -0.01  # Slight offset toward camera so it's visible
    
    north_label_segments = []
    for (p1_local, p2_local) in north_pole_label:
        points_3d = []
        for lx, lz in [p1_local, p2_local]:
            # lx is horizontal (maps to X in 3D)
            # lz is vertical in the letter (maps to Z in 3D, going up)
            x = lx
            y = y_offset
            z = z_base + lz
            points_3d.append([x, y, z])
        north_label_segments.append(np.array(points_3d))
    
    return {
        'lat_lines': lat_lines,
        'lon_lines': lon_lines,
        'lat_labels': lat_labels,
        'lat_label_values': lat_label_values,
        'lon_labels': lon_labels,
        'lon_label_values': lon_label_values,
        'north_pole_label': north_label_segments
    }


class MoonRenderer:
    """
    MoonRTX Application
    
    Renders the Moon surface as seen from a specific location on Earth
    at a specific time, with accurate solar illumination.
    """
    
    def __init__(self, 
                 elevation_file: str,
                 color_file: str,
                 starmap_file: Optional[str] = None,
                 downscale: int = 3,
                 width: int = 1400,
                 height: int = 900,
                 moon_features: list = []):
        """
        Initialize the planetarium.
        
        Parameters
        ----------
        elevation_file : str
            Path to Moon elevation data TIFF
        color_file : str
            Path to Moon color data TIFF
        starmap_file : str, optional
            Path to star map TIFF for background
        downscale : int
            Elevation downscale factor
        width, height : int
            Render window size
        moon_features : list
            Moon features (craters, mounts etc.)
        """
        self.width = width
        self.height = height
        self.downscale = downscale
        self.gamma = 2.2
        
        # Load data
        self.elevation = load_elevation_data(elevation_file, downscale)
        self.color_data = load_color_data(color_file, self.gamma)
        self.star_map = load_starmap(starmap_file) if starmap_file else None
        
        # Renderer
        self.rt = None
        self.moon_ephem = None
        
        # Grid settings
        self.grid_visible = False
        self.grid_data = None
        self.moon_radius = 10.0  # Same as in set_data("moon", ...)
        
        # View inversion (upside down)
        self.inverted = False
        
        # Initial camera parameters (for reset)
        self.initial_camera_params = None
        
        # Moon rotation matrix and its inverse (for selenographic coord conversion)
        self.moon_rotation_matrix = None
        self.moon_rotation_matrix_inv = None
        
        # Flag to track if window has been maximized
        self._window_maximized = False
        
        # Moon features for hover display
        self.moon_features = moon_features
        
        # Standard labels settings
        self.standard_labels_visible = False
        self.standard_labels = None
        
        # Spot labels settings
        self.spot_labels_visible = False
        self.spot_labels = None
        
    def _on_launch_finished(self, rt):
        """Callback to maximize window and set title on first launch."""
        if not self._window_maximized:
            self._window_maximized = True
            # Schedule maximize and title change on the main thread
            def init_window():
                rt._root.state('zoomed')
                rt._root.title(APP_NAME)
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
        camera_up = scene['up'] if not self.inverted else -scene['up']
        
        self.rt.setup_camera("cam1", cam_type="DoF",
                            eye=scene['eye'].tolist(),
                            target=scene['target'].tolist(),
                            up=camera_up.tolist(),
                            aperture_radius=0.01,
                            aperture_fract=0.2,
                            focal_scale=0.7,
                            fov=fov)
        
        # Store initial camera parameters for reset functionality
        if self.initial_camera_params is None:
            self.initial_camera_params = {
                'eye': scene['eye'].tolist(),
                'target': scene['target'].tolist(),
                'up': camera_up.tolist(),
                'fov': fov
            }
        
        # Light intensity based on phase - full moon is brighter
        # light_intensity = 40 + 20 * np.cos(np.radians(self.moon_ephem.phase))
        
        self.rt.setup_light("sun", 
                           pos=scene['light_pos'].tolist(),
                           color=light_intensity,
                           radius=8)
        
        # Update grid orientation if visible
        if self.grid_visible:
            self.update_grid_orientation()
        
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
            
    def get_info(self) -> str:
        """Get information about current view."""

        if self.moon_ephem is None:
            return "No view set"
        
        return (f"Moon topocentric ephemeris:\n"
                f"  Altitude: {self.moon_ephem.alt:.2f}°\n"
                f"  Azimuth: {self.moon_ephem.az:.2f}°\n"
                f"  RA: {self.moon_ephem.ra:.2f}°\n"
                f"  DEC: {self.moon_ephem.dec:.2f}°\n"
                f"  Distance: {self.moon_ephem.distance:.0f} km\n"
                f"  Phase: {self.moon_ephem.phase:.2f}°\n"
                f"  Illumination: {self.moon_ephem.illum:.2f}%\n"
                f"  Libration: L={self.moon_ephem.libr_long:+.2f}° B={self.moon_ephem.libr_lat:+.2f}°")
    
    def setup_grid(self, lat_step: float = 15.0, lon_step: float = 15.0):
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
        self.grid_data = create_selenographic_grid(
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
        for i, points in enumerate(self.grid_data['lat_lines']):
            name = f"grid_lat_{i}"
            self.rt.set_data(name, pos=points, r=line_radius, 
                            c=GRID_COLOR, geom="BezierChain", mat="grid_material")
        
        # Add longitude lines
        for i, points in enumerate(self.grid_data['lon_lines']):
            name = f"grid_lon_{i}"
            self.rt.set_data(name, pos=points, r=line_radius,
                            c=GRID_COLOR, geom="BezierChain", mat="grid_material")
        
        # Add latitude labels
        label_radius = 0.012
        for i, segments in enumerate(self.grid_data['lat_labels']):
            for j, seg in enumerate(segments):
                name = f"grid_lat_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=label_radius,
                                c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        # Add longitude labels
        for i, segments in enumerate(self.grid_data['lon_labels']):
            for j, seg in enumerate(segments):
                name = f"grid_lon_label_{i}_{j}"
                self.rt.set_data(name, pos=seg, r=label_radius,
                                c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        # Add north pole "N" label
        for j, seg in enumerate(self.grid_data['north_pole_label']):
            name = f"grid_north_label_{j}"
            self.rt.set_data(name, pos=seg, r=label_radius,
                            c=GRID_COLOR, geom="SegmentChain", mat="grid_material")
        
        self.grid_visible = True
        
        self.update_grid_orientation()
    
    def show_grid(self, visible: bool = True):
        """
        Show or hide the selenographic grid.
        
        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return
        
        if self.grid_data is None:
            if visible:
                self.setup_grid()
            return
        
        # Toggle visibility by setting zero radius (hide) or restoring (show)
        line_radius = 0.015 if visible else 0.0
        
        for i in range(len(self.grid_data['lat_lines'])):
            name = f"grid_lat_{i}"
            try:
                self.rt.update_data(name, r=line_radius)
            except:
                pass
        
        for i in range(len(self.grid_data['lon_lines'])):
            name = f"grid_lon_{i}"
            try:
                self.rt.update_data(name, r=line_radius)
            except:
                pass
        
        # Toggle label visibility
        label_radius = 0.012 if visible else 0.0
        
        for i, segments in enumerate(self.grid_data['lat_labels']):
            for j in range(len(segments)):
                name = f"grid_lat_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        for i, segments in enumerate(self.grid_data['lon_labels']):
            for j in range(len(segments)):
                name = f"grid_lon_label_{i}_{j}"
                try:
                    self.rt.update_data(name, r=label_radius)
                except:
                    pass
        
        # Toggle north pole label visibility
        for j in range(len(self.grid_data['north_pole_label'])):
            name = f"grid_north_label_{j}"
            try:
                self.rt.update_data(name, r=label_radius)
            except:
                pass
        
        self.grid_visible = visible
    
    def toggle_grid(self):
        """Toggle the selenographic grid visibility."""
        self.show_grid(not self.grid_visible)
    
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
        if self.rt is None or self.initial_camera_params is None:
            return
        
        print("Camera reset to initial position")
        
        # Restore initial camera parameters
        # Adjust up vector based on current inversion state
        up = self.initial_camera_params['up'][:]
        if self.inverted:
            up = [u * -1 for u in up]
        
        self.rt.setup_camera("cam1",
                            eye=self.initial_camera_params['eye'],
                            target=self.initial_camera_params['target'],
                            up=up,
                            fov=self.initial_camera_params['fov'])
    
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
    
    def update_grid_orientation(self):
        """
        Update grid lines to match current Moon orientation.
        
        This should be called after update_view() to rotate the grid
        along with the Moon surface.
        """
        if self.rt is None or self.grid_data is None or not self.grid_visible:
            return
        
        R = self.calculate_moon_rotation()

        if R is None:
            return
        
        # Update latitude lines
        for i, orig_points in enumerate(self.grid_data['lat_lines']):
            name = f"grid_lat_{i}"
            rotated = (R @ orig_points.T).T
            try:
                self.rt.update_data(name, pos=rotated)
            except:
                pass
        
        # Update longitude lines
        for i, orig_points in enumerate(self.grid_data['lon_lines']):
            name = f"grid_lon_{i}"
            rotated = (R @ orig_points.T).T
            try:
                self.rt.update_data(name, pos=rotated)
            except:
                pass
        
        # Update latitude labels
        for i, segments in enumerate(self.grid_data['lat_labels']):
            for j, orig_seg in enumerate(segments):
                name = f"grid_lat_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
        
        # Update longitude labels
        for i, segments in enumerate(self.grid_data['lon_labels']):
            for j, orig_seg in enumerate(segments):
                name = f"grid_lon_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
        
        # Update north pole label
        for j, orig_seg in enumerate(self.grid_data['north_pole_label']):
            name = f"grid_north_label_{j}"
            rotated = (R @ orig_seg.T).T
            try:
                self.rt.update_data(name, pos=rotated)
            except:
                pass
