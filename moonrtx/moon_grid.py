import numpy as np
from typing import NamedTuple

from moonrtx.shared_types import MoonFeature, MoonLabel

LABEL_CHAR_SCALE = 0.12
PIN_DIGIT_SCALE = 0.4

class MoonGrid(NamedTuple):
    lat_lines: list
    lon_lines: list
    lat_labels: list
    lat_label_values: list
    lon_labels: list
    lon_label_values: list
    N: list

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


def create_single_digit_on_sphere(digit: int,
                                   lat: float, lon: float,
                                   moon_radius: float,
                                   offset: float = 0.0,
                                   digit_scale: float = PIN_DIGIT_SCALE,
                                   flip_horizontal: bool = False,
                                   flip_vertical: bool = False) -> list:
    """
    Create 3D line segments for a single digit (1-9) positioned on the Moon sphere.
    
    Parameters
    ----------
    digit : int
        The digit to display (1-9)
    lat, lon : float
        Selenographic coordinates in degrees
    moon_radius : float
        Radius of the Moon
    offset : float
        Height above surface (fraction of radius)
    digit_scale : float
        Size of digit
    flip_horizontal : bool
        If True, mirror digit horizontally
    flip_vertical : bool
        If True, mirror digit vertically
        
    Returns
    -------
    list
        List of numpy arrays, each containing points for one line segment
    """
    r = moon_radius * (1 + offset + 0.005)
    
    # Calculate offsets to position the left-bottom corner at the given lat/lon
    # In create_digit_segments: w = 0.3 * scale (half width), h = 0.5 * scale (half height)
    # Digit ranges from -w to +w horizontally and -h to +h vertically
    # To put left-bottom corner at origin, shift by +w (right) and +h (up)
    w = 0.3 * digit_scale  # half width
    h = 0.5 * digit_scale  # half height
    
    all_segments = []
    digit_segs = create_digit_segments(str(digit), digit_scale)
    
    for (p1_local, p2_local) in digit_segs:
        points_3d = []
        for p_local in [p1_local, p2_local]:
            lx, lz = p_local
            
            # Apply flips in local 2D space before positioning
            if flip_horizontal:
                lx = -lx  # Mirror around center
            if flip_vertical:
                lz = -lz  # Mirror around center
            
            # Shift so left-bottom corner is at origin (or right-bottom if flipped)
            lx += w
            lz += h
            
            lat_offset = np.degrees(lz / r)
            lon_offset = np.degrees(lx / (r * np.cos(np.radians(lat)))) if abs(lat) < 89 else 0
            
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


def create_number_on_sphere(number: int, 
                            lat: float, lon: float,
                            moon_radius: float,
                            offset: float,
                            digit_scale: float = 0.3,
                            spacing: float = 0.25,
                            flip_horizontal: bool = False,
                            flip_vertical: bool = False) -> list:
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
    flip_horizontal : bool
        If True, flip digits horizontally (mirror left-right)
    flip_vertical : bool
        If True, flip digits vertically (upside down)
        
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
    # If horizontally flipped, reverse the digit order
    digit_indices = range(num_digits)
    if flip_horizontal:
        digit_indices = list(reversed(digit_indices))
    
    for i, digit_idx in enumerate(digit_indices):
        digit = num_str[digit_idx]
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
                
                # Apply flipping to the digit shape in local 2D coords
                if flip_horizontal:
                    lx = -lx
                if flip_vertical:
                    lz = -lz
                
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
                          spacing: float = 0.15,
                          flip_horizontal: bool = False,
                          flip_vertical: bool = False) -> list:
    """
    Create 3D line segments for text positioned on the Moon sphere.
    Text starts horizontally at the given lon (not centered).
    
    Parameters
    ----------
    flip_horizontal : bool
        If True, mirror text horizontally (for NSEW, SNEW orientations)
    flip_vertical : bool
        If True, mirror text vertically (for SNEW, SNWE orientations)
    """

    r = moon_radius * (1 + offset + 0.005)
    all_segments = []

    char_width = char_scale * (1 + spacing)
    
    # Get text to process (reverse if flipping horizontally)
    display_text = text.upper()
    if flip_horizontal:
        display_text = display_text[::-1]

    for i, char in enumerate(display_text):
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
                
                # Apply flips in local 2D space before projection
                # Characters are centered at origin, so mirror by negation
                if flip_horizontal:
                    lx = -lx
                if flip_vertical:
                    lz = -lz
                
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
                                   spacing: float = 0.15,
                                   flip_horizontal: bool = False,
                                   flip_vertical: bool = False) -> list:
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
    flip_horizontal : bool
        If True, mirror text horizontally (for NSEW, SNEW orientations)
    flip_vertical : bool
        If True, mirror text vertically (for SNEW, SNWE orientations)
        
    Returns
    -------
    list
        List of numpy arrays, each containing points for one line segment
    """
    r = moon_radius * (1 + offset + 0.005)  # Slightly above grid lines
    
    all_segments = []
    
    # Calculate total width for centering
    char_width = char_scale * (1 + spacing)
    
    # Get text to process (reverse if flipping horizontally)
    display_text = text.upper()
    if flip_horizontal:
        display_text = display_text[::-1]
    
    num_chars = len(display_text)
    total_width = num_chars * char_width - char_scale * spacing  # subtract last spacing
    
    for i, char in enumerate(display_text):
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
                
                # Apply flips in local 2D space before projection
                # Characters are centered at origin, so mirror by negation
                if flip_horizontal:
                    lx = -lx
                if flip_vertical:
                    lz = -lz
                
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


def create_standard_labels(standard_label_features: list[MoonFeature], moon_radius: float = 10.0, offset: float = 0.0,
                           flip_horizontal: bool = False, flip_vertical: bool = False) -> list[MoonLabel]:
    """
    Create standard labels
    
    The label is centered at the feature's (latitude, longitude) position.
    
    Parameters
    ----------
    standard_label_features : list[MoonFeature]
        List of standard label features
    moon_radius : float
        Radius of the Moon sphere
    offset : float
        Height offset above surface
    flip_horizontal : bool
        If True, mirror text horizontally
    flip_vertical : bool
        If True, mirror text vertically
        
    Returns
    -------
    list
        List of MoonLabel objects.
    """
    standard_labels = []
    
    for standard_label_feature in standard_label_features:
        
        label_text = standard_label_feature.name
        label_lat = standard_label_feature.lat
        label_lon = standard_label_feature.lon
        standard_label_segments = create_centered_text_on_sphere(
            text=label_text,
            lat=label_lat, 
            lon=label_lon,
            moon_radius=moon_radius,
            offset=offset,
            char_scale=LABEL_CHAR_SCALE,
            spacing=0.1,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        standard_label = MoonLabel(segments=standard_label_segments, anchor_point=(label_lat, label_lon))
        standard_labels.append(standard_label)
    
    return standard_labels

def create_spot_labels(spot_label_features: list[MoonFeature], moon_radius: float = 10.0, offset: float = 0.0,
                       flip_horizontal: bool = False, flip_vertical: bool = False) -> list[MoonLabel]:
    """
    Create spot labels
    
    Parameters
    ----------
    spot_label_features : list[MoonFeature]
        List of spot label features
    moon_radius : float
        Radius of the Moon sphere
    offset : float
        Height offset above surface
    flip_horizontal : bool
        If True, mirror text horizontally
    flip_vertical : bool
        If True, mirror text vertically
        
    Returns
    -------
    list[MoonLabel]
        List of MoonLabel objects.
    """
    spot_labels = []
    
    for spot_label_feature in spot_label_features:
        
        # For spot labels, the arrow points to the feature
        # When flipped horizontally, arrow should be on the right side
        if flip_horizontal:
            label_text = spot_label_feature.name + " >"
            label_lon = spot_label_feature.lon - spot_label_feature.angular_radius * 2
        else:
            label_text = "< " + spot_label_feature.name
            label_lon = spot_label_feature.lon + spot_label_feature.angular_radius * 2
        label_lat = spot_label_feature.lat
        
        spot_label_segments = create_text_on_sphere(
            label_text, 
            lat=label_lat, 
            lon=label_lon,
            moon_radius=moon_radius,
            offset=offset,
            char_scale=LABEL_CHAR_SCALE,
            spacing=0.1,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        spot_label = MoonLabel(segments=spot_label_segments, anchor_point=(label_lat, label_lon))
        spot_labels.append(spot_label)
    
    return spot_labels


def create_moon_grid(moon_radius: float = 10.0,
                     lat_step: float = 15.0,
                     lon_step: float = 15.0,
                     points_per_line: int = 100,
                     offset: float = 0.02) -> MoonGrid:
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
    MoonGrid
        MoonGrid tuple containing lists of point arrays
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
    
    N = []
    for (p1_local, p2_local) in north_pole_label:
        points_3d = []
        for lx, lz in [p1_local, p2_local]:
            # lx is horizontal (maps to X in 3D)
            # lz is vertical in the letter (maps to Z in 3D, going up)
            x = lx
            y = y_offset
            z = z_base + lz
            points_3d.append([x, y, z])
        N.append(np.array(points_3d))

    return MoonGrid(
        lat_lines=lat_lines,
        lon_lines=lon_lines,
        lat_labels=lat_labels,
        lat_label_values=lat_label_values,
        lon_labels=lon_labels,
        lon_label_values=lon_label_values,
        N=N
    )


def create_grid_labels_for_orientation(moon_radius: float,
                                       lat_step: float,
                                       lon_step: float,
                                       offset: float,
                                       flip_horizontal: bool,
                                       flip_vertical: bool) -> tuple:
    """
    Create grid labels (lat and lon numbers) with specified orientation.
    
    Used to regenerate labels when view orientation changes (F5-F8 keys).
    
    Parameters
    ----------
    moon_radius : float
        Radius of the Moon sphere
    lat_step : float
        Spacing between latitude lines in degrees
    lon_step : float
        Spacing between longitude lines in degrees
    offset : float
        Offset above surface (fraction of radius)
    flip_horizontal : bool
        If True, flip digits horizontally (for NSEW, SNEW orientations)
    flip_vertical : bool
        If True, flip digits vertically (for SNEW, SNWE orientations)
        
    Returns
    -------
    tuple
        (lat_labels, lat_label_values, lon_labels, lon_label_values)
    """
    # Create labels for latitude lines
    lat_labels = []
    lat_label_values = []
    label_longitudes = [0, 90, 180, -90]
    for label_lon in label_longitudes:
        for lat in np.arange(-60, 61, lat_step):
            if lat == 90 or lat == -90:
                continue
            segments = create_number_on_sphere(
                int(lat), lat=lat+1, lon=label_lon + lat_step/2-1,
                moon_radius=moon_radius, offset=offset,
                digit_scale=0.125,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical
            )
            lat_labels.append(segments)
            lat_label_values.append(int(lat))
    
    # Create labels for longitude lines
    lon_labels = []
    lon_label_values = []    
    for lon in np.arange(0, 360, lon_step):
        display_lon = lon if lon <= 180 else lon - 360
        lon_offset = 2 if display_lon < 0 else 1
        segments = create_number_on_sphere(
            int(display_lon), lat=lat_step/2-1, lon=display_lon+lon_offset,
            moon_radius=moon_radius, offset=offset,
            digit_scale=0.125,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        lon_labels.append(segments)
        lon_label_values.append(int(display_lon))
    
    return lat_labels, lat_label_values, lon_labels, lon_label_values