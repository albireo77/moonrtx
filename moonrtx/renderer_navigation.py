"""
NavigationMixin: camera navigation, zoom, coordinate conversion,
distance measurement, and feature lookup for MoonRenderer.
"""

import math

import tkinter as tk
import numpy as np
from typing import Optional

from moonrtx.shared_types import MoonFeature


class NavigationMixin:
    """Mixin providing camera navigation and measurement methods for MoonRenderer."""

    # The measurement line. Sizes are for a 96-dpi screen.
    MEASURE_LINE_COLOR = 'yellow'
    MEASURE_LINE_WIDTH = 2
    MEASURE_LINE_DASH = 4           # the length of each dash and of each gap
    # The circle at each end of the line, by which that end is dragged to
    # another place, and how far outside it a press still takes it - a circle
    # is small, and a hand on a mouse is not. The start's is filled and the
    # end's a ring, which says which way the line runs: which end the height
    # difference is measured to, and which way its profile reads
    MEASURE_HOOK_RADIUS = 5
    MEASURE_HOOK_GRAB = 4
    # How often an open profile window is drawn again while a line's end is
    # being dragged. Drawing it takes up to some 15 ms of sampling for the
    # longest lines and the drawing besides, and the mouse moves far more
    # often than that, so it is drawn at this pace - for the line as it stands
    # then - rather than on every move, and the drag never falls behind
    MEASURE_PROFILE_LIVE_MS = 50
    # The line is drawn along its ground (see _place_measure_line) as a run of
    # straight pieces this long on the screen - short enough for the bends of
    # a rim seen at a slant to show - and no more of them than this
    MEASURE_PATH_STEP_PX = 4
    MEASURE_PATH_MAX_POINTS = 600
    # How long the clock has to stand still, with a kept line's profile open,
    # before the profile's light is drawn in full rather than the quick way
    # (see _follow_measure_line): past the gap between the steps of a held Q
    # or W, short of making a single step wait for it
    MEASURE_PROFILE_SETTLE_MS = 400

    @staticmethod
    def _rotated(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        A vector turned about an axis through the origin (Rodrigues' formula).

        Every camera move here is one of these - the eye about the target, the
        view direction about the eye, the up vector about the line of sight -
        and each of them used to write the three terms out again. Written once,
        a reader checks the arithmetic once and reads the rest as what it does.

        Parameters
        ----------
        vector : np.ndarray
            The vector to turn
        axis : np.ndarray
            Unit vector of the axis, which the rotation is anticlockwise about
            seen from its far end
        angle : float
            How far to turn, in radians

        Returns
        -------
        np.ndarray
            The turned vector
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return (vector * cos_a + np.cross(axis, vector) * sin_a
                + axis * np.dot(axis, vector) * (1.0 - cos_a))

    def center_on_feature(self, feature: MoonFeature):
        """
        Center the view on a Moon feature.

        Parameters
        ----------
        feature : MoonFeature
            The feature to center on
        """
        if feature is not None:
            self.center_on_lat_lon(feature.lat, feature.lon)

    def center_on_lat_lon(self, lat: float, lon: float):
        """
        Center the view on a selenographic position.

        Keeps the camera's current distance and FOV, like center_view_on_cursor
        (C key) does; zooming in on it is left to the user (mouse wheel /
        Shift+drag). Unlike center_view_on_cursor, the position given here is
        not necessarily one already on screen - a planning dialog can jump the
        clock by weeks in the same action that recentres, moving what
        libration presents by a long way round the disk. center_view_on_cursor
        can safely carry the camera's current facing over to the new point
        because a cursor hit is always already close to it; carrying it over
        for an arbitrary point risks placing the eye beside or inside the
        sphere once the point is no longer nearby. The Moon is a static sphere
        at the scene origin (only its texture rotates - see update_view), so
        the point's own direction from that origin is always a valid outward
        normal to approach it from, however far the jump.

        Parameters
        ----------
        lat, lon : float
            Selenographic coordinates in degrees
        """
        if self.rt is None or self.moon_rotation is None:
            return

        # Convert selenographic coordinates to 3D position
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # In original Moon coordinates:
        # - +Z is north pole
        # - -Y is prime meridian (lon=0)
        # - +X is east (lon=90)
        x = self.MOON_RADIUS * np.cos(lat_rad) * np.sin(lon_rad)
        y = -self.MOON_RADIUS * np.cos(lat_rad) * np.cos(lon_rad)
        z = self.MOON_RADIUS * np.sin(lat_rad)

        # Apply Moon rotation to get scene coordinates
        original_pos = np.array([x, y, z])
        scene_pos = self.moon_rotation @ original_pos

        # Get current camera
        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        distance = np.linalg.norm(eye - target)

        # Approach from directly outside the sphere at the new point, not from
        # whatever direction the camera used to face - see the note above
        new_target = scene_pos
        direction = new_target / np.linalg.norm(new_target)
        new_eye = new_target + direction * distance

        # Update camera
        self.rt.update_camera(self.CAMERA_NAME, eye=new_eye.tolist(), target=new_target.tolist())

    # How far the pointer may move between pressing the left button and letting
    # it go for that still to be a click, which names the feature under it
    # (name_feature_at), rather than the start of a drag that turns the view
    CLICK_SLOP_PX = 4

    def _init_feature_lookup(self):
        """
        Build vectorized lookup arrays for find_feature_for_status_bar and
        name_feature_at.

        Called once after moon_features is loaded and sorted; the per-feature
        Python loop was too slow for a per-mouse-motion lookup with thousands
        of features (~1.8 ms scan vs ~0.02 ms vectorized).
        """
        self._sb_lookup = self._feature_lookup(
            [f for f in self.moon_features if f.status_bar])
        # What a click can name: what the status bar shows, and the features
        # the table marks for a spot label besides
        self._click_lookup = self._feature_lookup(
            [f for f in self.moon_features if f.status_bar or f.spot_label])

    @staticmethod
    def _feature_lookup(features: list) -> tuple:
        """The arrays _feature_in_lookup searches, for these features in this order."""
        lat = np.array([f.lat for f in features])
        return (features, lat, np.array([f.lon for f in features]),
                np.cos(np.radians(lat)),
                np.array([f.angular_radius for f in features]) ** 2)

    @staticmethod
    def _feature_in_lookup(lookup: tuple, lat: float, lon: float) -> Optional[MoonFeature]:
        """
        The first of a lookup's features whose disk takes in this position,
        which is the smallest of them, moon_features being sorted smallest first.
        """
        features, f_lat, f_lon, cos_lat, radius2 = lookup
        # Squared angular distance from every feature center; small-angle
        # approximation with cos_lat correction for longitude.
        dlon = (lon - f_lon + 180.0) % 360.0 - 180.0
        dist2 = (lat - f_lat) ** 2 + (dlon * cos_lat) ** 2
        hits = np.flatnonzero(dist2 <= radius2)
        return features[hits[0]] if hits.size else None

    def name_feature_at(self, event):
        """
        Put the name of the feature under a click on the Moon, as the Find
        window does for the one chosen there: it stays until Delete (see
        CatalogueMixin), or until the feature is clicked again, which takes
        the name off - whether a click, Find or a graph put it there. A click
        off the Moon, or on ground no feature the status bar or the spot labels
        name covers, does nothing.
        """
        place = self._selenographic_at(event)
        if place is None:
            return
        feature = self._feature_in_lookup(self._click_lookup, *place)
        if feature is None:
            return
        if self.is_catalogue_pinned(feature):
            self.unpin_catalogue_feature(feature)
        else:
            self.pin_catalogue_feature(feature)

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
        return self._feature_in_lookup(self._sb_lookup, lat, lon)

    def _reset_view_orientation_if_needed(self):
        if self.view_orientation != self.initial_view_orientation:
            self.set_view_orientation(self.initial_view_orientation)
            self.rt._view_orientation = self.initial_view_orientation

    def reset_camera_position(self):
        """
        Reset the camera to its initial position.

        Restores the camera to the position it had when the view was first set up,
        undoing any mouse rotation/panning performed by the user.
        Also resets the time back to the initial time.
        """
        cp = self.initial_camera

        if self.rt is None or cp is None:
            return

        self._reset_view_orientation_if_needed()

        # Reset time back to initial time if it was changed
        if self.initial_dt_local is not None and self.dt_local != self.initial_dt_local:
            # Reset to initial time - this will restore Moon orientation and lighting
            self.update_view(self.initial_dt_local)
            # Start the auto-advance interval over from this jump, so it does
            # not immediately step away from the time just restored
            if self._auto_advance_var and self._auto_advance_var.get():
                self._auto_advance_elapsed = 0

        # Restore initial camera parameters
        up = cp.up[:]

        self.rt.update_camera(self.CAMERA_NAME, eye=cp.eye, target=cp.target, up=up, fov=cp.fov)

        # Update status bar
        self._update_all_status_panels()

    def toggle_parallactic_mode(self):
        """
        Switch between holding celestial north up and following the zenith.

        The switch is a roll of the Moon about the line of sight - the parallactic
        angle enters the rotation as nothing else (see astro._rotation_matrix) -
        and that is the whole of what it should do: turn the picture, not move it.
        An eye still on that line is not touched by such a roll, so from the
        default view it needs no help. One swung off the line by the rotation keys
        would otherwise find a different patch of surface under it afterwards, the
        globe having turned beneath it, so it is carried round by the same roll.
        The eye keeps its place over the surface and the picture alone changes.
        """
        if self.rt is None:
            return

        before = self.moon_rotation
        self.parallactic_mode = not self.parallactic_mode
        self.update_view()
        self._update_status_parallactic()

        if before is None or self.moon_rotation is None:
            return

        # Nothing but the parallactic angle has changed, so what the Moon has just
        # done is a roll about the line of sight, which the scene has along +Y
        roll = self.moon_rotation @ before.T
        angle = math.atan2(roll[0, 2], roll[0, 0])
        if abs(angle) < 1e-12:
            return

        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye, target = roll @ np.array(cam["Eye"]), roll @ np.array(cam["Target"])
        up = roll @ np.array(cam["Up"])

        # Carried whole, the camera would see exactly the picture it saw before and
        # the toggle would show nothing; rolling up back about the view direction
        # turns the picture by the angle the Moon turned, and leaves it square to
        # that direction - which simply holding up still would not: swing the view
        # to the limb and a large parallactic angle would bring the two into line
        # and leave the camera with no picture at all.
        view = target - eye
        norm = np.linalg.norm(view)
        if norm > 0.0:
            up = self._rotated(up, view / norm, -angle)

        if (np.allclose(eye, cam["Eye"]) and np.allclose(target, cam["Target"])
                and np.allclose(up, cam["Up"])):
            return                      # on the line of sight, where all of that cancels

        self.rt.update_camera(self.CAMERA_NAME, eye=eye.tolist(), target=target.tolist(),
                              up=up.tolist())

    def reset_to_default_view(self):
        """
        Reset the camera to the default view calculated from ephemeris.

        This is the view that would be shown when starting the renderer
        without any --init-view parameter. Use this to get back to the
        standard Moon-centered view after using --init-view.
        """
        cp = self.default_camera

        if self.rt is None or cp is None:
            return

        self._reset_view_orientation_if_needed()

        # Restore default camera parameters
        up = cp.up[:]

        self.rt.update_camera(self.CAMERA_NAME, eye=cp.eye, target=cp.target, up=up, fov=cp.fov)

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

        # Ignore hits on non-Moon geometry (e.g. the visible Sun disk)
        if np.linalg.norm([hx, hy, hz]) > self.MOON_RADIUS * 1.15:
            return

        # Get current camera parameters using PlotOptix internal state
        cam = self.rt.get_camera(self.CAMERA_NAME)
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
        self.rt.update_camera(self.CAMERA_NAME, eye=new_eye.tolist(), target=new_target.tolist())

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
        cam = self.rt.get_camera(self.CAMERA_NAME)
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
        
        # Which axis each arrow key turns the eye about, and which way round it
        turns = {'Left': (actual_up, 1), 'Right': (actual_up, -1),
                 'Up': (right, 1), 'Down': (right, -1)}
        if direction not in turns:
            return
        axis, sense = turns[direction]
        angle *= sense

        new_eye = target + self._rotated(eye - target, axis, angle)

        # Also rotate the up vector for up/down navigation
        if direction in ('Up', 'Down'):
            new_up = self._rotated(up, axis, angle)
            self.rt.update_camera(self.CAMERA_NAME, eye=new_eye.tolist(), up=new_up.tolist())
        else:
            self.rt.update_camera(self.CAMERA_NAME, eye=new_eye.tolist())

    def pan_tilt_view(self, dx_px: float, dy_px: float):
        """
        Pan/tilt the view (rotate the camera view direction around the eye).

        Replaces the built-in right-mouse-drag handler, which rotates by fixed
        angles per pixel and is therefore far too sensitive for narrow FOVs.
        Here the rotation is scaled to the current FOV so the image content
        follows the mouse cursor 1:1 at any zoom level.

        Parameters
        ----------
        dx_px, dy_px : float
            Mouse move in image pixels
        """
        if self.rt is None:
            return

        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        up = np.array(cam["Up"])

        view = target - eye
        distance = np.linalg.norm(view)
        view_dir = view / distance

        right = np.cross(view_dir, up)
        right = right / np.linalg.norm(right)
        actual_up = np.cross(right, view_dir)
        actual_up = actual_up / np.linalg.norm(actual_up)

        # FOV is vertical; one pixel of drag = one pixel of content movement
        fov = self.rt._optix.get_camera_fov(0)
        angle_per_px = np.radians(fov) / self.rt._height

        # Drag right -> look right (content moves left), drag down -> look down
        yaw = -dx_px * angle_per_px
        pitch = -dy_px * angle_per_px

        # Yaw: rotate the view direction around the view-up axis
        view_dir = self._rotated(view_dir, actual_up, yaw)

        # Pitch: rotate the view direction and up around the right axis (no roll)
        new_up = self._rotated(up, right, pitch)
        view_dir = self._rotated(view_dir, right, pitch)

        new_target = eye + view_dir * distance
        self.rt.update_camera(self.CAMERA_NAME, target=new_target.tolist(), up=new_up.tolist())

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
        moon_polar_axis = self.moon_rotation @ np.array([0.0, 0.0, 1.0])
        moon_equatorial_axis = self.moon_rotation @ np.array([1.0, 0.0, 0.0])
        
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
        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        up = np.array(cam["Up"])
        
        new_eye = target + self._rotated(eye - target, axis, angle)

        # Also rotate the up vector to maintain proper orientation
        new_up = self._rotated(up, axis, angle)
        
        self.rt.update_camera(self.CAMERA_NAME, eye=new_eye.tolist(), up=new_up.tolist())

    def rotate_around_view_direction(self, direction: str, step_deg: float = 1.0):
        """
        Rotate (roll) the camera around the current view direction.

        Parameters
        ----------
        direction : str
            'cw' for clockwise, 'ccw' for counter-clockwise
        step_deg : float
            Rotation step in degrees (default 1°)
        """
        if self.rt is None:
            return

        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"])
        target = np.array(cam["Target"])
        up = np.array(cam["Up"])

        # View direction is the roll axis
        view_dir = target - eye
        axis = view_dir / np.linalg.norm(view_dir)

        if direction == 'cw':
            angle = np.radians(-step_deg)
        else:
            angle = np.radians(step_deg)

        # Only the up vector turns; the eye and the target stay where they are
        new_up = self._rotated(up, axis, angle)

        self.rt.update_camera(self.CAMERA_NAME, up=new_up.tolist())

    def _ground_behind(self, through: np.ndarray) -> Optional[np.ndarray]:
        """
        Where the line from the eye through a point of the scene meets the
        ground, or None if it passes the Moon by.

        The ground's radius depends on where it is met, so the meeting is found
        on a sphere of the mean radius first and then on one of the height of
        the ground found there, a few times over: the line is nearly square to
        the ground on any of the disk worth pointing at, and it settles at once.
        """
        if self.rt is None or self.moon_rotation_inv is None:
            return None
        eye = np.asarray(self.rt.get_camera(self.CAMERA_NAME)["Eye"], dtype=float)
        sight = through - eye
        sight /= np.linalg.norm(sight)
        radius = self.MOON_RADIUS / getattr(self, "elevation_radius_scale", 1.0)
        point = None
        for _ in range(4):
            b = eye @ sight
            disc = b * b - (eye @ eye - radius * radius)
            if disc < 0.0:
                return None
            point = eye + (-b - np.sqrt(disc)) * sight
            x, y, z = self.moon_rotation_inv @ (point / np.linalg.norm(point))
            radius = self.MOON_RADIUS * float(self._ground_factor(
                np.degrees(np.arcsin(np.clip(z, -1, 1))), np.degrees(np.arctan2(x, -y))))
        return point

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
        hit_pos = np.array([hx, hy, hz])
        
        # Check if hit is on the Moon surface
        r = np.linalg.norm(hit_pos)
        if r < self.MOON_RADIUS * 0.9 or r > self.MOON_RADIUS * 1.15:
            return None, None

        # Above the highest peak is no ground: what was hit is lettering - a
        # name, a pin, the place mark, the grid - which stands off the surface
        # and, aimed and stood up to face the eye, off the line from the centre
        # to the ground it names (see LabelsMixin._aimed_at_ground). The point
        # meant is the ground seen through it, so the line of sight is followed
        # on to the ground behind; past the limb there is none
        if r > self.MOON_RADIUS * (1.0 + 1e-6):
            hit_pos = self._ground_behind(hit_pos)
            if hit_pos is None:
                return None, None
            r = np.linalg.norm(hit_pos)

        hit_normalized = hit_pos / r
        
        # Transform back to original Moon coordinates
        original_pos = self.moon_rotation_inv @ hit_normalized
        
        # Convert to selenographic coordinates
        x, y, z = original_pos
        
        # Latitude: angle from equator (XY plane) to the point
        lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
        
        # Longitude: angle in XY plane from -Y axis (prime meridian)
        lon = np.degrees(np.arctan2(x, -y))
        
        return lat, lon

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
        current_fov = self.rt._optix.get_camera_fov(0)

        # Calculate zoom factor based on wheel delta
        zoom_factor = 1 - (event.delta / 120) * 0.05  # 5% per notch

        # Apply zoom by changing FOV
        new_fov = current_fov * zoom_factor

        # Clamp FOV to reasonable range. The lower bound keeps the maximum zoom-in
        # (relative to the ~4.2 degree default FOV of the camera at 30 Moon radii)
        # similar to the previous 10-radii/1-degree setup.
        new_fov = max(0.3, min(90, new_fov))

        self.rt._optix.set_camera_fov(new_fov)

    # ==================== Distance Measurement Methods ====================

    def _selenographic_at(self, event) -> Optional[tuple]:
        """(lat, lon) of the Moon under a pointer event, or None off the Moon."""
        if self.rt is None:
            return None
        x, y = self.rt._get_image_xy(event.x, event.y)
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)
        if hd <= 0:
            return None
        lat, lon = self.hit_to_selenographic(hx, hy, hz)
        if lat is None or lon is None:
            return None
        return lat, lon

    def _measure_between(self, start: tuple, end: tuple):
        """Set the measured distance, angle and height difference, and show them."""
        (lat1, lon1), (lat2, lon2) = start, end
        self.measured_distance = self.calculate_great_circle_distance(lat1, lon1, lat2, lon2)
        self.measured_angle = self.apparent_separation(lat1, lon1, lat2, lon2)
        self.measured_height_diff = self.get_elevation_m(lat2, lon2) - self.get_elevation_m(lat1, lon1)
        self._update_status_measured()

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
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance_km = c * self.MOON_RADIUS_KM
        return distance_km

    def apparent_separation(self, lat1: float, lon1: float,
                            lat2: float, lon2: float) -> Optional[float]:
        """
        The angle between two places on the Moon as the observer sees them, in
        radians - or None before there is an ephemeris.

        The chord between them, less its part along the line of sight, over
        the Moon's distance: foreshortening included, so a line running
        towards the limb spans the small angle it looks to.
        """
        if self.moon_ephem is None:
            return None
        towards = self._sub_point_direction(self.moon_ephem.libr_lat_topo,
                                            self.moon_ephem.libr_long_topo)
        chord = (self._sub_point_direction(lat2, lon2)
                 - self._sub_point_direction(lat1, lon1)) * self.MOON_RADIUS_KM
        across = chord - (chord @ towards) * towards
        return float(np.linalg.norm(across)) / self.moon_ephem.distance

    def _ground_factor(self, lat, lon) -> np.ndarray:
        """
        The radius of the ground at selenographic coordinates, as the stored
        displacement factor: a share of the scene's Moon radius, the highest
        peak being 1.0. Vectorized over arrays of places, for aiming labels at
        their ground (see LabelsMixin._aimed_at_ground); looked up as
        get_elevation_m does, which is left to its own arithmetic so that the
        heights a measurement shows stay exactly what they were.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        h, w = self.elevation.shape

        row = (90.0 - lat) / 180.0 * h - 0.5
        col = ((lon + 180.0) / 360.0 * w - 0.5) % w

        r0 = np.clip(np.floor(row), 0, h - 2).astype(np.int64)
        fr = np.clip(row - r0, 0.0, 1.0)
        c0 = np.floor(col).astype(np.int64)
        c1 = (c0 + 1) % w   # longitude wraps at the +/-180 seam
        fc = col - c0

        return (self.elevation[r0, c0] * (1.0 - fr) * (1.0 - fc)
                + self.elevation[r0 + 1, c0] * fr * (1.0 - fc)
                + self.elevation[r0, c1] * (1.0 - fr) * fc
                + self.elevation[r0 + 1, c1] * fr * fc)

    def get_elevation_m(self, lat: float, lon: float) -> float:
        """
        Look up the elevation at a selenographic coordinate relative to the mean Moon radius.
        
        Parameters
        ----------
        lat : float
            Selenographic latitude in degrees (-90 to +90)
        lon : float
            Selenographic longitude in degrees (-180 to +180)
            
        Returns
        -------
        float
            Elevation in meters relative to Moon reference radius
            (positive = above, negative = below)
        """
        h, w = self.elevation.shape

        # Equirectangular projection: row 0 = +90° lat, last row = -90° lat.
        # Bilinear interpolation between the four surrounding texel centers
        # (-0.5 offset); a texel spans ~354 m of ground at downscale 3, so
        # nearest-texel lookup would quantize measured height differences.
        row = (90.0 - lat) / 180.0 * h - 0.5
        col = ((lon + 180.0) / 360.0 * w - 0.5) % w

        r0 = int(np.clip(np.floor(row), 0, h - 2))
        fr = min(max(row - r0, 0.0), 1.0)
        c0 = int(np.floor(col))
        c1 = (c0 + 1) % w   # longitude wraps at the +/-180 seam
        fc = col - c0

        val = (self.elevation[r0, c0] * (1.0 - fr) * (1.0 - fc)
               + self.elevation[r0 + 1, c0] * fr * (1.0 - fc)
               + self.elevation[r0, c1] * (1.0 - fr) * fc
               + self.elevation[r0 + 1, c1] * fr * fc)

        # Stored elevation is a displacement factor normalized so that the highest
        # peak is 1.0; elevation_radius_scale converts it back to a factor of the
        # Moon reference radius
        displacement = val * self.elevation_radius_scale
        return (displacement - 1.0) * self.MOON_RADIUS_KM * 1000.0

    def start_measurement(self, event, with_profile: bool = False):
        """
        Start distance measurement on Ctrl+B1 press.

        Parameters
        ----------
        event : tk.Event
            Mouse button press event
        with_profile : bool
            Draw the elevation profile of the line when it is finished, as the
            right Ctrl asks (see ProfileMixin)
        """
        if self.rt is None:
            return

        self.measure_with_profile = with_profile
        start = self._selenographic_at(event)
        if start is None:
            self.measuring = False
            return

        self.drop_measure_line(close_profile=not with_profile)
        self.measuring = True
        self.measure_start_canvas = (event.x, event.y)
        self.measure_start_coords = start

        if hasattr(self.rt, '_canvas'):
            # The line and the circles on its ends share a tag of their own,
            # which is what the rest of the measurement holds on to, so they
            # are moved, kept and deleted as one (see _place_measure_line)
            self._measure_serial += 1
            tag = f"measure{self._measure_serial}"
            canvas = self.rt._canvas
            width = max(1, round(self._overlay_px(self.MEASURE_LINE_WIDTH)))
            dash = max(1, round(self._overlay_px(self.MEASURE_LINE_DASH)))
            canvas.create_line(event.x, event.y, event.x, event.y, tags=tag,
                               fill=self.MEASURE_LINE_COLOR, width=width, dash=(dash, dash))
            # A circle at each end, the start's first and filled (see
            # MEASURE_HOOK_RADIUS)
            for fill in (self.MEASURE_LINE_COLOR, ''):
                canvas.create_oval(event.x, event.y, event.x, event.y, tags=tag,
                                   outline=self.MEASURE_LINE_COLOR, fill=fill, width=width)
            self.leading_line_id = tag

    def _place_measure_line(self, tag, x0: float, y0: float, x1: float, y1: float,
                            ground: Optional[tuple] = None):
        """
        Lay the measurement line from (x0, y0) to (x1, y1), with a circle on
        each end: filled at the start, a ring at the end.

        Given the (lat, lon) of its two ends as `ground`, the line is drawn
        along the ground between them - the great circle its profile samples,
        each point at its own height where the picture shows it (see
        ProfileMixin._canvas_points) - so it bends over rims and floors seen at
        a slant as a line painted on them would, and the profile's mark for
        any point of it sits on it. Straight otherwise: an end off the Moon,
        or a point of the way the eye cannot see.
        """
        canvas = self.rt._canvas
        line, start_hook, end_hook = canvas.find_withtag(tag)    # in the order made
        path = None if ground is None else self._measure_line_path(*ground, (x0, y0), (x1, y1))
        canvas.coords(line, *(path if path is not None else (x0, y0, x1, y1)))
        radius = self._overlay_px(self.MEASURE_HOOK_RADIUS)
        for hook, (x, y) in ((start_hook, (x0, y0)), (end_hook, (x1, y1))):
            canvas.coords(hook, x - radius, y - radius, x + radius, y + radius)

    def _measure_line_path(self, start: tuple, end: tuple, start_xy: tuple,
                           end_xy: tuple) -> Optional[list]:
        """
        The measurement line's way over the ground as flat canvas coordinates,
        a point every MEASURE_PATH_STEP_PX of it, from start_xy to end_xy -
        the ends where they were placed, the way between where the ground is.
        None for two ends on one place, or a way the eye cannot see all of.
        """
        if start == end:
            return None
        length = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
        count = int(np.clip(length / self._overlay_px(self.MEASURE_PATH_STEP_PX),
                            1, self.MEASURE_PATH_MAX_POINTS))
        lats, lons = self._along(start, end, np.linspace(0.0, 1.0, count + 1))
        points = self._canvas_points(lats, lons)
        if points is None:
            return None
        points[0], points[-1] = start_xy, end_xy
        return points.ravel().tolist()

    def _init_measure_line(self):
        """Reset the kept-line state; called from MoonRenderer.__init__."""
        self._measure_line = None           # the kept measurement line (its tag), or None
        self._measure_line_view = None      # the view it was drawn in
        self._measure_line_time = None      # and the moment shown then
        self._measure_line_watch = None     # the poll that notices the view move
        # The profile drawn in full once the clock has stood still a while,
        # after being drawn the quick way as it moved (see _follow_measure_line)
        self._measure_profile_settle = None
        # Where the kept line ends, in the window and on the Moon, as the start
        # is kept in measure_start_canvas and measure_start_coords
        self.measure_end_canvas = None
        self.measure_end_coords = None
        # Whether the kept line is the one the profile window shows
        self._measure_line_graphed = False
        # Which end is being dragged by its ring - "start" or "end" - or None
        self._measure_hook = None
        # The profile drawing booked while a drag goes on (see _follow_profile)
        self._measure_profile_due = None

    def _keep_measure_line(self, line_id):
        """
        Leave a measurement's line on the picture after the button is released,
        so the numbers (and the profile window, if one was drawn) say what the
        line is of.

        It is drawn in the window's pixels, as it was during the drag, and
        whenever the view changes - the clock moved, the view panned, zoomed or
        turned, the window resized - its ends are carried to where their ground
        now is on the window (see _follow_measure_line), so it stays on the
        ground it measures. It goes only when an end has left the side of the
        Moon the eye looks at.

        Until then either end can be dragged by the ring on it to another place,
        the measurement following it (see grab_measure_hook).
        """
        self._measure_line = line_id
        self._measure_line_graphed = self.measure_with_profile
        self._measure_line_view = self._measure_line_view_now()
        self._measure_line_time = self.dt_local
        self._measure_line_watch = self._schedule_overlay(self._watch_measure_line)

    def _measure_line_view_now(self):
        """
        The view as the kept line depends on it: what every canvas overlay
        depends on (see _overlay_view_state), and the zoom, which the wheel
        changes without moving the camera.
        """
        if self.rt is None:
            return None
        return self._overlay_view_state(extra=(self.rt._optix.get_camera_fov(0),))

    def _watch_measure_line(self):
        """One beat of the poll: see to the kept line once the view has moved."""
        self._measure_line_watch = None
        if self._measure_line is None:
            return
        self.check_measure_line()
        if self._measure_line is not None:
            self._measure_line_watch = self._schedule_overlay(self._watch_measure_line)

    def check_measure_line(self):
        """
        Carry the kept line to where its ground now is if the view has changed
        since it was drawn, or take it away if an end cannot be seen any more;
        nothing if the view stands as it was. From the poll, and straight after
        whatever moves the view from the window - a key, a drag, the wheel, a
        step of the clock (see update_view) - so that the line does not trail a
        beat of the poll behind the Moon.
        """
        if self._measure_line is None or self._measure_line_view_now() == self._measure_line_view:
            return
        if not self._follow_measure_line():
            self.drop_measure_line()
            self.clear_measurement()

    def _follow_measure_line(self) -> bool:
        """
        Draw the kept line between where its two ends' ground now is on the
        window, and say whether it could be: an end turned away to the far
        side of the globe, or behind the eye, has nowhere to go.

        The ground under the line is what it was, so its length and height
        difference are too. When the clock has moved, though, the angle it
        spans as seen from Earth has followed the Moon's distance, and is
        measured again; and its profile is the same ground in another light,
        drawn again the quick way as the clock moves and in full once it has
        stood still for MEASURE_PROFILE_SETTLE_MS.
        """
        ends = []
        for place in (self.measure_start_coords, self.measure_end_coords):
            at = self._canvas_xy_of(*place) if self._faces_eye(*place) else None
            if at is None:
                return False
            ends.append(at)
        self.measure_start_canvas, self.measure_end_canvas = ends
        self._place_measure_line(self._measure_line, *ends[0], *ends[1],
                                 ground=(self.measure_start_coords, self.measure_end_coords))
        self._measure_line_view = self._measure_line_view_now()
        if self.dt_local == self._measure_line_time:
            return True
        self._measure_line_time = self.dt_local
        self._measure_between(self.measure_start_coords, self.measure_end_coords)

        if self._measure_line_graphed and getattr(self, "_profile", None) is not None:
            # The light alone: the ground under the line has not changed
            self.show_profile(self.measure_start_coords, self.measure_end_coords,
                              live=True, light_only=True)
            self._cancel_overlay(self._measure_profile_settle)
            self._measure_profile_settle = self._schedule_overlay(
                self._settle_measure_profile, self.MEASURE_PROFILE_SETTLE_MS)
        return True

    def _faces_eye(self, lat: float, lon: float) -> bool:
        """
        Whether a place on the Moon is on the side of the globe the eye looks
        at - the test LocatorMixin._locator_on_screen makes, for a point at
        the surface.
        """
        if self.rt is None or self.moon_rotation is None:
            return False
        eye = np.asarray(self.rt.get_camera(self.CAMERA_NAME)["Eye"], dtype=float)
        la, lo = math.radians(lat), math.radians(lon)
        body = np.array([math.cos(la) * math.sin(lo), -math.cos(la) * math.cos(lo), math.sin(la)])
        return float((self.moon_rotation @ body) @ eye) * self.MOON_RADIUS > self.MOON_RADIUS ** 2

    def _settle_measure_profile(self):
        """The kept line's profile drawn in full, the clock having stood still."""
        self._measure_profile_settle = None
        if (self._measure_line is not None and self._measure_line_graphed
                and getattr(self, "_profile", None) is not None):
            self.show_profile(self.measure_start_coords, self.measure_end_coords, light_only=True)

    def drop_measure_line(self, close_profile: bool = True):
        """
        Take the kept measurement line off the picture, if there is one, and
        the profile window with it if the window shows that line's profile -
        unless `close_profile` is False, as it is when a new profile line is
        taking its place and will be drawn into the same window.
        """
        if self._measure_line is not None and self._measure_line_graphed and close_profile:
            self.close_profile()
        self._measure_line_graphed = False
        self._measure_line_watch = self._cancel_overlay(self._measure_line_watch)
        self._measure_line_view = None
        if self._measure_line is not None and self.rt is not None:
            try:
                self.rt._canvas.delete(self._measure_line)
            except tk.TclError:         # the main window went first
                pass
        self._measure_line = None
        self._measure_hook = None
        self._measure_profile_due = self._cancel_overlay(self._measure_profile_due)
        self._measure_profile_settle = self._cancel_overlay(self._measure_profile_settle)
        self.measure_end_canvas = None
        self.measure_end_coords = None
        if self.rt is not None:
            try:
                self.rt._canvas.configure(cursor='')
            except tk.TclError:
                pass

    def update_leading_line(self, event):
        """
        Update the leading line during drag and show live distance.

        Parameters
        ----------
        event : tk.Event
            Mouse motion event
        """
        if not self.measuring or self.leading_line_id is None:
            return

        # Along the ground while the pointer is on the Moon, straight to it off it
        start_x, start_y = self.measure_start_canvas
        end = self._selenographic_at(event)
        self._place_measure_line(self.leading_line_id, start_x, start_y, event.x, event.y,
                                 ground=None if end is None else (self.measure_start_coords, end))

        # Live distance update during drag
        if end is not None:
            self._measure_between(self.measure_start_coords, end)
            self._live_end = end
            self._follow_profile(lambda: (self.measure_start_coords, self._live_end))

    def finish_measurement(self, event):
        """
        Finish distance measurement on B1 release.

        The line drawn during the drag stays on the picture, on its ground
        whatever the view does, until Backspace takes it off or a new
        measurement starts (see _keep_measure_line). A click
        that did not drag, or a release off the Moon, measures nothing and
        leaves nothing.

        Parameters
        ----------
        event : tk.Event
            Mouse button release event
        """
        if not self.measuring:
            return
        self.measuring = False
        self._measure_profile_due = self._cancel_overlay(self._measure_profile_due)

        line_id, self.leading_line_id = self.leading_line_id, None
        start_x, start_y = self.measure_start_canvas
        end = self._selenographic_at(event)
        moved = math.hypot(event.x - start_x, event.y - start_y) > self.CLICK_SLOP_PX

        if end is None or not moved:
            if line_id is not None:
                self.rt._canvas.delete(line_id)
            self.clear_measurement()
            # The line a profile window may still be showing went when this
            # one began, and this one leaves none in its place
            if self.measure_with_profile:
                self.close_profile()
            return

        self._measure_between(self.measure_start_coords, end)
        self.measure_end_canvas = (event.x, event.y)
        self.measure_end_coords = end
        # The ground in between, which the two numbers say nothing about
        if self.measure_with_profile:
            self.show_profile(self.measure_start_coords, end)
        if line_id is not None:
            self._place_measure_line(line_id, start_x, start_y, event.x, event.y,
                                     ground=(self.measure_start_coords, end))
            self._keep_measure_line(line_id)

    # ---- moving the ends of a kept line ----

    def _measure_hook_at(self, event) -> Optional[str]:
        """Which end of the kept line has its ring under the pointer, if either."""
        if (self._measure_line is None or self.measuring
                or self.measure_end_canvas is None or self.measure_start_canvas is None):
            return None
        reach = self._overlay_px(self.MEASURE_HOOK_RADIUS + self.MEASURE_HOOK_GRAB)
        # The end first: a line too short to tell its rings apart is most
        # likely being lengthened, and that is the end that was drawn last
        for which, (x, y) in (("end", self.measure_end_canvas),
                              ("start", self.measure_start_canvas)):
            if math.hypot(event.x - x, event.y - y) <= reach:
                return which
        return None

    def grab_measure_hook(self, event) -> bool:
        """
        Take hold of an end of the kept line if the left button went down on
        its ring, and say whether it did - so the press goes no further: it is
        neither a drag of the view nor a click naming a feature.
        """
        self._measure_hook = self._measure_hook_at(event)
        return self._measure_hook is not None

    @property
    def dragging_measure_hook(self) -> bool:
        return self._measure_hook is not None

    def drag_measure_hook(self, event):
        """
        Move the end being held to the pointer, and measure again from the end
        left where it was. Off the Moon there is nothing to measure to, so the
        end waits at the last place on it until the pointer comes back.
        """
        if self._measure_hook is None:
            return
        place = self._selenographic_at(event)
        if place is None:
            return
        if self._measure_hook == "start":
            self.measure_start_canvas, self.measure_start_coords = (event.x, event.y), place
        else:
            self.measure_end_canvas, self.measure_end_coords = (event.x, event.y), place
        self._place_measure_line(self._measure_line, *self.measure_start_canvas,
                                 *self.measure_end_canvas,
                                 ground=(self.measure_start_coords, self.measure_end_coords))
        self._measure_between(self.measure_start_coords, self.measure_end_coords)
        self._follow_profile(lambda: (self.measure_start_coords, self.measure_end_coords))

    def _follow_profile(self, ends):
        """
        Draw the open profile window again for the line being dragged - at most
        once every MEASURE_PROFILE_LIVE_MS, and for the line as it stands when
        the time comes, which `ends` is asked for then. Only a profile line, and
        only while its window is up: a window that is not open is opened when
        the drag ends, as it always was, not sprung on the user halfway through.
        """
        if (not self.measure_with_profile or getattr(self, "_profile", None) is None
                or self._measure_profile_due is not None):
            return

        def draw():
            self._measure_profile_due = None
            start, end = ends()
            if start is not None and end is not None and getattr(self, "_profile", None) is not None:
                self.show_profile(start, end, live=True)

        self._measure_profile_due = self.rt._root.after(self.MEASURE_PROFILE_LIVE_MS, draw)

    def release_measure_hook(self, event):
        """
        Let go of the end being held, where it stands, and draw the profile
        afresh if the line has one, for where it ends up (see _follow_profile
        for the drawing during the drag).
        """
        if self._measure_hook is None:
            return
        self.drag_measure_hook(event)
        self._measure_hook = None
        self._measure_profile_due = self._cancel_overlay(self._measure_profile_due)
        if self.measure_with_profile:
            self.show_profile(self.measure_start_coords, self.measure_end_coords)

    def hover_measure_hook(self, event):
        """Show a hand over a ring that can be dragged, and the usual pointer elsewhere."""
        if self.rt is None:
            return
        cursor = 'hand2' if self._measure_hook_at(event) is not None else ''
        if self.rt._canvas.cget('cursor') != cursor:
            self.rt._canvas.configure(cursor=cursor)

    def clear_measurement(self):
        """
        Forget the last measurement, and take its reading off the status bar and
        the full-screen panel alike.

        Called from the motion handler when the cursor leaves the Moon (see
        run_renderer). A distance and a height difference are between two places
        on the surface; once the cursor is off the surface they answer a
        question nobody is asking any more, and left standing they read as
        though they still applied to wherever the cursor has gone.

        Nothing is written when there was nothing to forget: the handler runs on
        every movement of the mouse, and the sky is most of the window.
        """
        if self.measured_distance is None and self.measured_height_diff is None:
            return
        self.measured_distance = None
        self.measured_angle = None
        self.measured_height_diff = None
        self._update_status_measured()
