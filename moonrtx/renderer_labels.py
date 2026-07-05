"""
LabelsMixin: grid, standard labels, spot labels, and illumination logic for MoonRenderer.

Each overlay group (grid lines, grid labels, standard labels, spot labels) is
merged into a single PlotOptiX graph geometry, so updating an overlay after a
time change is one update_graph call instead of one update_data call per line
segment (hundreds to thousands of calls before the merge).
"""

import numpy as np

from plotoptix.materials import m_flat

from moonrtx.shared_types import MoonFeature, MoonLabel
from moonrtx.view_orientation import FLIP_HORIZONTAL_VIEW_ORIENTATIONS, FLIP_VERTICAL_VIEW_ORIENTATIONS, VIEW_ORIENTATIONS
from moonrtx.moon_grid import (
    create_moon_grid, create_standard_labels, create_spot_labels, create_grid_labels_for_orientation,
    merge_segments_to_graph
)

class LabelsMixin:
    """Mixin providing grid, label, and illumination methods for MoonRenderer."""

    GRID_LINE_RADIUS = 0.006    # Thin lines for grid
    GRID_LABEL_RADIUS = 0.012   # Slightly thicker lines for grid labels
    STANDARD_LABEL_RADIUS = 0.008  # Standard feature label thickness
    SPOT_LABEL_RADIUS = 0.008   # Spot feature label thickness
    GRID_COLOR = [0.50, 0.50, 0.50]
    STANDARD_LABEL_COLOR = [0.85, 0.85, 0.85]
    SPOT_LABEL_COLOR = [1.0, 0.9, 0.3]

    GRID_LINES_GEOM = "grid_lines"
    GRID_LABELS_GEOM = "grid_labels"
    STANDARD_LABELS_GEOM = "standard_labels_graph"
    SPOT_LABELS_GEOM = "spot_labels_graph"

    # ---- merged-graph helpers ----

    def _rotate_to_scene(self, pos: np.ndarray) -> np.ndarray:
        """Rotate body-frame vertices to scene coordinates with the current Moon rotation."""
        R = self.moon_rotation
        return pos if R is None else pos @ R.T

    def _view_orientation_flips(self) -> tuple[bool, bool]:
        # NSWE (default): N up, W left - no flips
        # NSEW: N up, E left - horizontal flip
        # SNEW: S up, E left - both flips (180° rotation)
        # SNWE: S up, W left - vertical flip
        return (self.view_orientation in FLIP_HORIZONTAL_VIEW_ORIENTATIONS,
                self.view_orientation in FLIP_VERTICAL_VIEW_ORIENTATIONS)

    @staticmethod
    def _label_graph_arrays(labels: list[MoonLabel]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge all label segments into one graph.

        Returns
        -------
        tuple
            (pos, edges, counts): graph vertices and edges, plus the number of
            vertices belonging to each label (for per-label radii).
        """
        segments = [seg for label in labels for seg in label.segments]
        pos, edges = merge_segments_to_graph(segments)
        counts = np.array([sum(seg.shape[0] for seg in label.segments) for label in labels],
                          dtype=np.int64)
        return pos, edges, counts

    @staticmethod
    def _features_unit_vectors(features: list[MoonFeature]) -> np.ndarray:
        """Body-frame unit position vectors of features, shape (n, 3)."""
        lat = np.radians([f.lat for f in features])
        lon = np.radians([f.lon for f in features])
        cos_lat = np.cos(lat)
        return np.column_stack((cos_lat * np.sin(lon), -cos_lat * np.cos(lon), np.sin(lat)))

    def _lit_mask(self, units: np.ndarray) -> np.ndarray:
        """
        Boolean mask of features on the illuminated hemisphere, given the cached
        light position and current Moon rotation (vectorized over all features).
        """
        if self.light_pos is None or self.moon_rotation is None:
            # If we don't have light or rotation info, assume visible to avoid hiding labels
            return np.ones(units.shape[0], dtype=bool)
        light = np.asarray(self.light_pos, dtype=float)
        light_norm = np.linalg.norm(light)
        if light_norm == 0:
            return np.ones(units.shape[0], dtype=bool)
        # dot > 0 => angle < 90° between surface normal and light direction => illuminated
        return units @ (self.moon_rotation.T @ (light / light_norm)) > 0.0

    def _label_radii(self, units: np.ndarray, counts: np.ndarray, radius: float) -> np.ndarray:
        """Per-vertex radii hiding labels of features on the night side."""
        return np.repeat(np.where(self._lit_mask(units), radius, 0.0), counts).astype(np.float32)

    @staticmethod
    def _no_shadow_flat_material() -> dict:
        # Flat material with shadow rays passing through (base_color alpha 0
        # + transparent occlusion program), so overlays cast no shadow on the surface
        m = m_flat.copy()
        m["OcclusionProgram"] = "chit7_occlusion_transp.ptx::__closesthit__occlusion_transparency"
        m["VarFloat4"] = {"base_color": [1.0, 1.0, 1.0, 0.0]}
        return m

    # ---- orientation helpers ----

    def set_view_orientation(self, view_orientation: str):
        """
        Set the view orientation mode and update the status bar.

        Called when F5-F8 keys are pressed to match plotoptix internal orientation change.

        Parameters
        ----------
        view_orientation : str
        """
        self.view_orientation = view_orientation

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

    def _rebuild_grid_labels_arrays(self):
        """Rebuild the merged vertex/edge arrays for all grid number labels and the N label."""
        segments = [seg for segs in self.moon_grid.lat_labels for seg in segs]
        segments += [seg for segs in self.moon_grid.lon_labels for seg in segs]
        segments += list(self.moon_grid.N)
        self._grid_labels_pos, self._grid_labels_edges = merge_segments_to_graph(segments)

    def update_grid_labels_for_orientation(self):
        """
        Update grid number labels to match current view orientation.

        Regenerates latitude and longitude number labels so they are
        always readable (not upside down) in the current view orientation.
        """
        if self.rt is None or self.moon_grid is None:
            return

        flip_horizontal, flip_vertical = self._view_orientation_flips()

        # Generate new labels with proper orientation
        lat_labels, lat_label_values, lon_labels, lon_label_values = create_grid_labels_for_orientation(
            moon_radius=self.MOON_RADIUS,
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

        # Flipping mirrors coordinates but keeps the segment structure, so the
        # edge indices stay valid and only vertex positions need an update
        self._rebuild_grid_labels_arrays()
        self.rt.update_graph(self.GRID_LABELS_GEOM, pos=self._rotate_to_scene(self._grid_labels_pos))

    def update_standard_labels_for_view_orientation(self):
        """
        Update standard labels to match current view orientation.

        Regenerates standard labels so they are always readable
        (not upside down) in the current view orientation.
        """
        if self.rt is None or self.standard_labels is None or self.standard_label_features is None:
            return

        flip_horizontal, flip_vertical = self._view_orientation_flips()

        # Regenerate labels with proper orientation
        self.standard_labels = create_standard_labels(
            self.standard_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )

        self._standard_labels_pos, self._standard_labels_edges, self._standard_labels_counts = \
            self._label_graph_arrays(self.standard_labels)
        self.rt.update_graph(
            self.STANDARD_LABELS_GEOM,
            pos=self._rotate_to_scene(self._standard_labels_pos),
            r=self._label_radii(self._standard_units, self._standard_labels_counts, self.STANDARD_LABEL_RADIUS))

    def update_spot_labels_for_view_orientation(self):
        """
        Update spot labels to match current view orientation.

        Regenerates spot labels so they are always readable
        (not upside down) in the current view orientation.
        """
        if self.rt is None or self.spot_labels is None or self.spot_label_features is None:
            return

        flip_horizontal, flip_vertical = self._view_orientation_flips()

        # Regenerate labels with proper orientation
        self.spot_labels = create_spot_labels(
            self.spot_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )

        self._spot_labels_pos, self._spot_labels_edges, self._spot_labels_counts = \
            self._label_graph_arrays(self.spot_labels)
        self.rt.update_graph(
            self.SPOT_LABELS_GEOM,
            pos=self._rotate_to_scene(self._spot_labels_pos),
            r=self._label_radii(self._spot_units, self._spot_labels_counts, self.SPOT_LABEL_RADIUS))

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
            moon_radius=self.MOON_RADIUS,
            lat_step=lat_step,
            lon_step=lon_step,
            points_per_line=100,
            offset=0.0
        )

        self.rt.update_material("grid_material", self._no_shadow_flat_material())

        # All grid lines in one graph geometry, all number labels in another
        self._grid_lines_pos, self._grid_lines_edges = merge_segments_to_graph(
            self.moon_grid.lat_lines + self.moon_grid.lon_lines)
        self._rebuild_grid_labels_arrays()

        self.rt.set_graph(self.GRID_LINES_GEOM,
                          pos=self._grid_lines_pos, edges=self._grid_lines_edges,
                          r=self.GRID_LINE_RADIUS, c=self.GRID_COLOR, mat="grid_material")
        self.rt.set_graph(self.GRID_LABELS_GEOM,
                          pos=self._grid_labels_pos, edges=self._grid_labels_edges,
                          r=self.GRID_LABEL_RADIUS, c=self.GRID_COLOR, mat="grid_material")

        self.moon_grid_visible = True

        # Update labels for current view orientation if not default
        if self.view_orientation != VIEW_ORIENTATIONS[0]:
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
        self.rt.update_graph(self.GRID_LINES_GEOM, r=self.GRID_LINE_RADIUS if visible else 0.0)
        self.rt.update_graph(self.GRID_LABELS_GEOM, r=self.GRID_LABEL_RADIUS if visible else 0.0)

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

        flip_horizontal, flip_vertical = self._view_orientation_flips()

        # Get ALL features with standard_label=True (illumination checked during rendering)
        self.standard_label_features = [f for f in self.moon_features if f.standard_label]
        if not self.standard_label_features:
            return
        self.standard_labels = create_standard_labels(
            self.standard_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )

        self.rt.update_material("standard_label_material", self._no_shadow_flat_material())

        self._standard_units = self._features_unit_vectors(self.standard_label_features)
        self._standard_labels_pos, self._standard_labels_edges, self._standard_labels_counts = \
            self._label_graph_arrays(self.standard_labels)

        # All labels in one graph geometry; night-side labels hidden via zero vertex radii
        self.rt.set_graph(
            self.STANDARD_LABELS_GEOM,
            pos=self._rotate_to_scene(self._standard_labels_pos),
            edges=self._standard_labels_edges,
            r=self._label_radii(self._standard_units, self._standard_labels_counts, self.STANDARD_LABEL_RADIUS),
            c=self.STANDARD_LABEL_COLOR,
            mat="standard_label_material")

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

        self.standard_labels_visible = visible

        if visible:
            # Restores orientation and per-label illumination radii, in case
            # time or view orientation changed while labels were hidden
            self.update_standard_labels_for_view_orientation()
        else:
            self.rt.update_graph(self.STANDARD_LABELS_GEOM, r=0.0)

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

        flip_horizontal, flip_vertical = self._view_orientation_flips()

        # Get ALL features with spot_label=True (illumination checked during rendering)
        self.spot_label_features = [f for f in self.moon_features if f.spot_label]
        if not self.spot_label_features:
            return
        self.spot_labels = create_spot_labels(
            self.spot_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )

        self.rt.update_material("spot_label_material", self._no_shadow_flat_material())

        self._spot_units = self._features_unit_vectors(self.spot_label_features)
        self._spot_labels_pos, self._spot_labels_edges, self._spot_labels_counts = \
            self._label_graph_arrays(self.spot_labels)

        # All labels in one graph geometry; night-side labels hidden via zero vertex radii
        self.rt.set_graph(
            self.SPOT_LABELS_GEOM,
            pos=self._rotate_to_scene(self._spot_labels_pos),
            edges=self._spot_labels_edges,
            r=self._label_radii(self._spot_units, self._spot_labels_counts, self.SPOT_LABEL_RADIUS),
            c=self.SPOT_LABEL_COLOR,
            mat="spot_label_material")

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

        self.spot_labels_visible = visible

        if visible:
            # Restores orientation and per-label illumination radii, in case
            # time or view orientation changed while labels were hidden
            self.update_spot_labels_for_view_orientation()
        else:
            self.rt.update_graph(self.SPOT_LABELS_GEOM, r=0.0)

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

        if self.moon_rotation is None:
            return

        self.rt.update_graph(
            self.SPOT_LABELS_GEOM,
            pos=self._rotate_to_scene(self._spot_labels_pos),
            r=self._label_radii(self._spot_units, self._spot_labels_counts, self.SPOT_LABEL_RADIUS))

    def update_standard_labels_orientation(self):
        """
        Update standard labels to match current Moon orientation.

        This should be called after update_view() to rotate the labels
        along with the Moon surface.
        """
        if self.rt is None or self.standard_labels is None:
            return

        if self.moon_rotation is None:
            return

        self.rt.update_graph(
            self.STANDARD_LABELS_GEOM,
            pos=self._rotate_to_scene(self._standard_labels_pos),
            r=self._label_radii(self._standard_units, self._standard_labels_counts, self.STANDARD_LABEL_RADIUS))

    def update_moon_grid_orientation(self):
        """
        Update grid lines to match current Moon orientation.

        This should be called after update_view() to rotate the grid
        along with the Moon surface.
        """
        if self.rt is None or self.moon_grid is None or not self.moon_grid_visible:
            return

        if self.moon_rotation is None:
            return

        self.rt.update_graph(self.GRID_LINES_GEOM, pos=self._rotate_to_scene(self._grid_lines_pos))
        self.rt.update_graph(self.GRID_LABELS_GEOM, pos=self._rotate_to_scene(self._grid_labels_pos))
