"""
NavigationMixin: camera navigation, zoom, coordinate conversion,
distance measurement, and feature lookup for MoonRenderer.
"""

import numpy as np
from typing import Optional

from moonrtx.shared_types import MoonFeature, CameraParams


class NavigationMixin:
    """Mixin providing camera navigation and measurement methods for MoonRenderer."""

    # Real Moon radius in kilometers
    MOON_RADIUS_KM = 1737.4

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
        hit_pos = np.array([hx, hy, hz])
        
        # Check if hit is on the Moon surface
        r = np.linalg.norm(hit_pos)
        if r < self.moon_radius * 0.9 or r > self.moon_radius * 1.15:
            return None, None
        
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
        
        # Clamp FOV to reasonable range
        new_fov = max(1, min(90, new_fov))
        
        self.rt._optix.set_camera_fov(new_fov)

    # ==================== Distance Measurement Methods ====================

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
        
        x = r * np.cos(lat_rad) * np.sin(lon_rad)
        y = -r * np.cos(lat_rad) * np.cos(lon_rad)
        z = r * np.sin(lat_rad)
        original_pos = np.array([x, y, z])
        
        if self.moon_rotation is None:
            return original_pos
        return self.moon_rotation @ original_pos

    def start_measurement(self, event):
        """
        Start distance measurement on Ctrl+B1 press.
        
        Parameters
        ----------
        event : tk.Event
            Mouse button press event
        """
        if self.rt is None:
            return
        
        x, y = self.rt._get_image_xy(event.x, event.y)
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)
        
        if hd <= 0:
            self.measuring = False
            return
        
        lat, lon = self.hit_to_selenographic(hx, hy, hz)
        
        if lat is None or lon is None:
            self.measuring = False
            return
        
        self.measuring = True
        self.measure_start_canvas = (event.x, event.y)
        self.measure_start_coords = (lat, lon)
        
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
        
        start_x, start_y = self.measure_start_canvas
        self.rt._canvas.coords(
            self.leading_line_id,
            start_x, start_y, event.x, event.y
        )

    def finish_measurement(self, event):
        """
        Finish distance measurement on B1 release.
        
        Parameters
        ----------
        event : tk.Event
            Mouse button release event
        """
        if not self.measuring:
            return
        
        if self.leading_line_id is not None and hasattr(self.rt, '_canvas'):
            self.rt._canvas.delete(self.leading_line_id)
            self.leading_line_id = None
        
        self.measuring = False
        
        if self.rt is None or self.measure_start_coords is None:
            return
        
        x, y = self.rt._get_image_xy(event.x, event.y)
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)
        
        if hd <= 0:
            return
        
        lat2, lon2 = self.hit_to_selenographic(hx, hy, hz)
        
        if lat2 is None or lon2 is None:
            return
        
        lat1, lon1 = self.measure_start_coords
        
        distance_km = self.calculate_great_circle_distance(lat1, lon1, lat2, lon2)
        
        self.measured_distance = distance_km
        self._update_status_measured()
