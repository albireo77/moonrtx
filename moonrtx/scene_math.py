"""
Scene math utilities: rotation matrices, camera/light calculation, camera encoding.
"""

import struct
import base64
import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from moonrtx.shared_types import MoonEphemeris


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


def calculate_camera_and_light(moon_ephem: MoonEphemeris, zoom: float, moon_radius: float) -> Scene:
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
    Scene
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
    # - Only affects when phase < 6.4° (very close to full moon)
    # - At 6° phase, illumination = (1 + cos(6.4°))/2 ≈ 99.6% (vs 100% at true full)
    # - A ~0.3% sliver at the Moon's edge would be in shadow, visually imperceptible
    # - Most of the lunar cycle (phase > 6.4°) is completely unaffected
    min_phase_offset = np.radians(6.4)
    # Only apply minimum offset near full moon (phase < 6.4°), not near new moon
    effective_sin_phase = np.sin(min_phase_offset if phase < min_phase_offset else phase)
    
    light_x = -np.sin(bright_limb_angle) * effective_sin_phase * light_distance
    light_z = np.cos(bright_limb_angle) * effective_sin_phase * light_distance
    light_y = -np.cos(phase) * light_distance
    
    light_pos = np.array([light_x, light_y, light_z])

    return Scene(eye=camera_eye, target=camera_target, up=camera_up, light_pos=light_pos)
