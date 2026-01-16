import numpy as np

from typing import Optional
from datetime import datetime
from datetime import timezone

from moonrtx.types import MoonEphemeris
from moonrtx.types import MoonFeature
from moonrtx.types import CameraParams
from moonrtx.types import Scene
from moonrtx.astro import calculate_moon_ephemeris
from moonrtx.data_loader import load_moon_features, load_elevation_data, load_color_data, load_starmap
from moonrtx.moon_grid import create_moon_grid, create_standard_labels, create_spot_labels

from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse
from plotoptix.materials import m_flat

GRID_COLOR = [0.50, 0.50, 0.50]
MOON_FILL_FRACTION = 0.9  # Moon fills 90% of window height (5% margins top/bottom)

def run_renderer(dt_local: datetime,
                 lat: float,
                 lon: float,
                 elevation_file: str,
                 color_file: str,
                 starmap_file: str,
                 features_file: str,
                 downscale: int,
                 light_intensity: int,
                 app_name: str) -> TkOptiX:
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
        
        # Initial camera parameters (for reset)
        self.initial_camera_params = None
        
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
        
    def _on_launch_finished(self, rt):
        """Callback to maximize window and set title on first launch."""
        if not self._window_maximized:
            self._window_maximized = True
            # Schedule maximize and title change on the main thread
            def init_window():
                rt._root.state('zoomed')
                rt._root.title(self.app_name)
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
        
        # Light intensity based on phase - full moon is brighter
        # light_intensity = 40 + 20 * np.cos(np.radians(self.moon_ephem.phase))
        
        self.rt.setup_light("sun", pos=scene.light_pos.tolist(), color=light_intensity, radius=10)
        
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
            
    def get_info(self) -> str:
        """Get information about current view."""

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
        
        print("Camera reset to initial position")
        
        # Restore initial camera parameters
        # Adjust up vector based on current inversion state
        up = cp.up[:]
        if self.inverted:
            up = [u * -1 for u in up]
        
        self.rt.setup_camera("cam1", eye=cp.eye, target=cp.target, up=up, fov=cp.fov)
    
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
        
        # Update latitude lines
        for i, orig_points in enumerate(self.moon_grid.lat_lines):
            name = f"grid_lat_{i}"
            rotated = (R @ orig_points.T).T
            try:
                self.rt.update_data(name, pos=rotated)
            except:
                pass
        
        # Update longitude lines
        for i, orig_points in enumerate(self.moon_grid.lon_lines):
            name = f"grid_lon_{i}"
            rotated = (R @ orig_points.T).T
            try:
                self.rt.update_data(name, pos=rotated)
            except:
                pass
        
        # Update latitude labels
        for i, segments in enumerate(self.moon_grid.lat_labels):
            for j, orig_seg in enumerate(segments):
                name = f"grid_lat_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
        
        # Update longitude labels
        for i, segments in enumerate(self.moon_grid.lon_labels):
            for j, orig_seg in enumerate(segments):
                name = f"grid_lon_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
        
        # Update north pole label
        for j, orig_seg in enumerate(self.moon_grid.N):
            name = f"grid_north_label_{j}"
            rotated = (R @ orig_seg.T).T
            try:
                self.rt.update_data(name, pos=rotated)
            except:
                pass
