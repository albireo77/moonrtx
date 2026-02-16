"""
LabelsMixin: grid, standard labels, spot labels, and illumination logic for MoonRenderer.
"""

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from plotoptix import TkOptiX
from plotoptix.materials import m_flat

from moonrtx.constants import (
    GRID_COLOR, GRID_LINE_RADIUS, GRID_LABEL_RADIUS,
    STANDARD_LABEL_RADIUS, SPOT_LABEL_RADIUS,
    ORIENTATION_NSWE, ORIENTATION_NSEW, ORIENTATION_SNEW, ORIENTATION_SNWE,
)
from moonrtx.shared_types import MoonFeature
from moonrtx.moon_grid import (
    create_moon_grid, create_standard_labels, create_spot_labels,
    create_grid_labels_for_orientation,
)


class LabelsMixin:
    """Mixin providing grid, label, and illumination methods for MoonRenderer."""

    # ---- orientation helpers ----

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

    # ---- grid setup / show / hide ----

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

    # ---- standard labels ----

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

    # ---- spot labels ----

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

    # ---- orientation updates (after time change) ----

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
        
        for i, label in enumerate(self.spot_labels):
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
        
        for i, label in enumerate(self.standard_labels):
            feature = self.standard_label_features[i]
            label_radius = STANDARD_LABEL_RADIUS if self._is_feature_illuminated(feature) else 0.0
            for j, orig_seg in enumerate(label.segments):
                name = f"standard_label_{i}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated, r=label_radius)
                except:
                    pass

    # ---- grid orientation helpers ----

    def _update_grid_lines(self, rt: TkOptiX, R: NDArray, lines, prefix: str):
        """Update grid lines to match current Moon orientation."""
        for i, orig_points in enumerate(lines):
            name = f"{prefix}_{i}"
            rotated = (R @ orig_points.T).T
            try:
                rt.update_data(name, pos=rotated)
            except:
                pass

    def _update_grid_nested_segments(self, rt: TkOptiX, R: NDArray, segments_list, prefix: str):
        """Update nested grid segments to match current Moon orientation."""
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

    # ---- illumination helpers ----

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
