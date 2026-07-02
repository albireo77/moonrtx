import math

import numpy as np
from typing import NamedTuple

from moonrtx.shared_types import MoonFeature, MoonLabel


LABEL_CHAR_SCALE = 0.12
PIN_DIGIT_SCALE = 0.2


class MoonGrid(NamedTuple):
    lat_lines: list[np.ndarray]
    lon_lines: list[np.ndarray]
    lat_labels: list[list[np.ndarray]]
    lat_label_values: list[int]
    lon_labels: list[list[np.ndarray]]
    lon_label_values: list[int]
    N: list[np.ndarray]


# Letter shapes normalized to scale=1 (w=0.3, h=0.5). Scaled on access in create_digit_segments.
_LETTER_SEGMENTS_NORMALIZED: dict[str, list[tuple]] = {
    'A': [((-0.3, -0.5), (-0.3,  0.15)), ((-0.3,  0.15), ( 0.0,  0.5)), (( 0.0,  0.5), ( 0.3,  0.15)), (( 0.3,  0.15), ( 0.3, -0.5)), ((-0.3,  0.0), ( 0.3,  0.0))],
    'B': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.5), ( 0.18,  0.5)), (( 0.18,  0.5), ( 0.18,  0.05)), (( 0.18,  0.05), (-0.3,  0.0)), ((-0.3,  0.0), ( 0.18,  0.0)), (( 0.18,  0.0), ( 0.18, -0.45)), (( 0.18, -0.45), (-0.3, -0.5))],
    'C': [(( 0.3,  0.5), (-0.09,  0.5)), ((-0.09,  0.5), (-0.3,  0.25)), ((-0.3,  0.25), (-0.3, -0.25)), ((-0.3, -0.25), (-0.09, -0.5)), ((-0.09, -0.5), ( 0.3, -0.5))],
    'D': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.5), ( 0.09,  0.5)), (( 0.09,  0.5), ( 0.3,  0.25)), (( 0.3,  0.25), ( 0.3, -0.25)), (( 0.3, -0.25), ( 0.09, -0.5)), (( 0.09, -0.5), (-0.3, -0.5))],
    'E': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.5), ( 0.3,  0.5)), ((-0.3,  0.0), ( 0.18,  0.0)), ((-0.3, -0.5), ( 0.3, -0.5))],
    'F': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.5), ( 0.3,  0.5)), ((-0.3,  0.0), ( 0.18,  0.0))],
    'G': [(( 0.3,  0.3), ( 0.09,  0.5)), (( 0.09,  0.5), (-0.3,  0.25)), ((-0.3,  0.25), (-0.3, -0.25)), ((-0.3, -0.25), ( 0.09, -0.5)), (( 0.09, -0.5), ( 0.3, -0.25)), (( 0.3, -0.25), ( 0.3,  0.0)), (( 0.3,  0.0), ( 0.0,  0.0))],
    'H': [((-0.3, -0.5), (-0.3,  0.5)), (( 0.3, -0.5), ( 0.3,  0.5)), ((-0.3,  0.0), ( 0.3,  0.0))],
    'I': [((-0.15,  0.5), ( 0.15,  0.5)), (( 0.0,  0.5), ( 0.0, -0.5)), ((-0.15, -0.5), ( 0.15, -0.5))],
    'J': [(( 0.15,  0.5), ( 0.15, -0.25)), (( 0.15, -0.25), ( 0.0, -0.5)), (( 0.0, -0.5), (-0.15, -0.25))],
    'K': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.0), ( 0.3,  0.5)), ((-0.3,  0.0), ( 0.3, -0.5))],
    'L': [((-0.3,  0.5), (-0.3, -0.5)), ((-0.3, -0.5), ( 0.3, -0.5))],
    'M': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.5), ( 0.0,  0.0)), (( 0.0,  0.0), ( 0.3,  0.5)), (( 0.3,  0.5), ( 0.3, -0.5))],
    'O': [((-0.3,  0.25), (-0.3, -0.25)), ((-0.3, -0.25), (-0.09, -0.5)), ((-0.09, -0.5), ( 0.09, -0.5)), (( 0.09, -0.5), ( 0.3, -0.25)), (( 0.3, -0.25), ( 0.3,  0.25)), (( 0.3,  0.25), ( 0.09,  0.5)), (( 0.09,  0.5), (-0.09,  0.5)), ((-0.09,  0.5), (-0.3,  0.25))],
    'P': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.5), ( 0.18,  0.5)), (( 0.18,  0.5), ( 0.18,  0.05)), (( 0.18,  0.05), (-0.3,  0.0))],
    'Q': [((-0.3,  0.25), (-0.3, -0.25)), ((-0.3, -0.25), (-0.09, -0.5)), ((-0.09, -0.5), ( 0.09, -0.5)), (( 0.09, -0.5), ( 0.3, -0.25)), (( 0.3, -0.25), ( 0.3,  0.25)), (( 0.3,  0.25), ( 0.09,  0.5)), (( 0.09,  0.5), (-0.09,  0.5)), ((-0.09,  0.5), (-0.3,  0.25)), (( 0.09, -0.15), ( 0.24, -0.45))],
    'R': [((-0.3, -0.5), (-0.3,  0.5)), ((-0.3,  0.5), ( 0.18,  0.5)), (( 0.18,  0.5), ( 0.18,  0.05)), (( 0.18,  0.05), (-0.3,  0.0)), ((-0.06,  0.0), ( 0.3, -0.5))],
    'S': [(( 0.3,  0.35), ( 0.09,  0.5)), (( 0.09,  0.5), (-0.09,  0.5)), ((-0.09,  0.5), (-0.3,  0.25)), ((-0.3,  0.25), (-0.3,  0.1)), ((-0.3,  0.1), ( 0.3, -0.1)), (( 0.3, -0.1), ( 0.3, -0.25)), (( 0.3, -0.25), ( 0.09, -0.5)), (( 0.09, -0.5), (-0.09, -0.5)), ((-0.09, -0.5), (-0.3, -0.35))],
    'T': [((-0.3,  0.5), ( 0.3,  0.5)), (( 0.0,  0.5), ( 0.0, -0.5))],
    'U': [((-0.3,  0.5), (-0.3, -0.25)), ((-0.3, -0.25), (-0.09, -0.5)), ((-0.09, -0.5), ( 0.09, -0.5)), (( 0.09, -0.5), ( 0.3, -0.25)), (( 0.3, -0.25), ( 0.3,  0.5))],
    'V': [((-0.3,  0.5), ( 0.0, -0.5)), (( 0.0, -0.5), ( 0.3,  0.5))],
    'W': [((-0.3,  0.5), (-0.15, -0.5)), ((-0.15, -0.5), ( 0.0,  0.15)), (( 0.0,  0.15), ( 0.15, -0.5)), (( 0.15, -0.5), ( 0.3,  0.5))],
    'X': [((-0.3,  0.5), ( 0.3, -0.5)), ((-0.3, -0.5), ( 0.3,  0.5))],
    'Y': [((-0.3,  0.5), ( 0.0,  0.0)), (( 0.3,  0.5), ( 0.0,  0.0)), (( 0.0,  0.0), ( 0.0, -0.5))],
    'Z': [((-0.3,  0.5), ( 0.3,  0.5)), (( 0.3,  0.5), (-0.3, -0.5)), ((-0.3, -0.5), ( 0.3, -0.5))],
    ' ': [],
    "'": [(( 0.0,  0.5), ( 0.0,  0.25))],
    '>': [((-0.3,  0.2), ( 0.3,  0.0)), (( 0.3,  0.0), (-0.3, -0.2))],
    '<': [(( 0.3,  0.2), (-0.3,  0.0)), ((-0.3,  0.0), ( 0.3, -0.2))],
}


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

    # Which segments are on for each digit
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
    }

    # Handle letter N specially (diagonal stroke)
    if digit == 'N':
        return [
            ((-w, -h), (-w, h)),   # left vertical
            ((-w, h), (w, -h)),    # diagonal
            ((w, -h), (w, h)),     # right vertical
        ]

    if digit in _LETTER_SEGMENTS_NORMALIZED:
        segs = _LETTER_SEGMENTS_NORMALIZED[digit]
        return [((x1 * scale, y1 * scale), (x2 * scale, y2 * scale)) for (x1, y1), (x2, y2) in segs]

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

    lon_scale = (r * math.cos(math.radians(lat))) if abs(lat) < 89 else None

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

            lat_offset = math.degrees(lz / r)
            lon_offset = math.degrees(lx / lon_scale) if lon_scale is not None else 0.0

            new_lat = lat + lat_offset
            new_lon = lon + lon_offset

            lat_rad = math.radians(new_lat)
            lon_rad = math.radians(new_lon)
            cos_new_lat = math.cos(lat_rad)

            x = r * cos_new_lat * math.sin(lon_rad)
            y = -r * cos_new_lat * math.cos(lon_rad)
            z = r * math.sin(lat_rad)

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

    num_str = str(number)
    num_digits = len(num_str)
    total_width = num_digits * digit_scale * (1 + spacing) - digit_scale * spacing

    lon_scale = (r * math.cos(math.radians(lat))) if abs(lat) < 89 else None

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
                lat_offset = math.degrees(lz / r)
                lon_offset = math.degrees(lx / lon_scale) if lon_scale is not None else 0.0

                new_lat = lat + lat_offset
                new_lon = lon + lon_offset

                # Convert to 3D
                lat_rad = math.radians(new_lat)
                lon_rad = math.radians(new_lon)
                cos_new_lat = math.cos(lat_rad)

                x = r * cos_new_lat * math.sin(lon_rad)
                y = -r * cos_new_lat * math.cos(lon_rad)
                z = r * math.sin(lat_rad)

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
    lon_scale = (r * math.cos(math.radians(lat))) if abs(lat) < 89 else None

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
                lat_offset = math.degrees(lz / r)
                lon_offset = math.degrees(lx / lon_scale) if lon_scale is not None else 0.0

                new_lat = lat + lat_offset
                new_lon = lon + lon_offset

                lat_rad = math.radians(new_lat)
                lon_rad = math.radians(new_lon)
                cos_new_lat = math.cos(lat_rad)

                x = r * cos_new_lat * math.sin(lon_rad)
                y = -r * cos_new_lat * math.cos(lon_rad)
                z = r * math.sin(lat_rad)

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
    lon_scale = (r * math.cos(math.radians(lat))) if abs(lat) < 89 else None

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
                lat_offset = math.degrees(lz / r)
                lon_offset = math.degrees(lx / lon_scale) if lon_scale is not None else 0.0

                new_lat = lat + lat_offset
                new_lon = lon + lon_offset

                # Convert to 3D
                lat_rad = math.radians(new_lat)
                lon_rad = math.radians(new_lon)
                cos_new_lat = math.cos(lat_rad)

                x = r * cos_new_lat * math.sin(lon_rad)
                y = -r * cos_new_lat * math.cos(lon_rad)
                z = r * math.sin(lat_rad)

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


def create_grid_labels_for_orientation(
        moon_radius: float,
        lat_step: float,
        lon_step: float,
        offset: float,
        flip_horizontal: bool,
        flip_vertical: bool,
) -> tuple[list[list[np.ndarray]], list[int], list[list[np.ndarray]], list[int]]:
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
        lat_rad = np.radians(lat)
        z = r * np.sin(lat_rad)
        r_circle = r * np.cos(lat_rad)

        lons_rad = np.radians(np.linspace(0, 360, points_per_line, endpoint=True))
        xs = r_circle * np.sin(lons_rad)   # lon=0 faces -Y
        ys = -r_circle * np.cos(lons_rad)  # -Y is toward camera
        lat_lines.append(np.column_stack([xs, ys, np.full(points_per_line, z)]))

    # Longitude lines (great circles at constant longitude)
    for lon in np.arange(0, 360, lon_step):
        lon_rad = float(np.radians(lon))
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        lats_rad = np.radians(np.linspace(-90, 90, points_per_line))
        cos_lats = np.cos(lats_rad)
        sin_lats = np.sin(lats_rad)

        xs = r * cos_lats * sin_lon
        ys = -r * cos_lats * cos_lon  # -Y is toward camera
        lon_lines.append(np.column_stack([xs, ys, r * sin_lats]))

    lat_labels, lat_label_values, lon_labels, lon_label_values = create_grid_labels_for_orientation(
        moon_radius=moon_radius,
        lat_step=lat_step,
        lon_step=lon_step,
        offset=offset,
        flip_horizontal=False,
        flip_vertical=False,
    )

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
