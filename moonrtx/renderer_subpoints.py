"""
SubPointsMixin: the two points the Sun and Earth stand overhead, marked on the
surface for MoonRenderer.

The sub-Earth point is where the Moon is turned exactly towards us, so it is the
one place with no foreshortening and the centre libration moves about; the
sub-solar point is where the Sun is at the zenith, the middle of the lit face
and the far end of the terminator's noon line. Both wander with time, so unlike
a pin - planted once and only rotated afterwards - their markers are rebuilt on
every view update. That costs a handful of line segments.
"""

import numpy as np

from moonrtx.moon_grid import create_centered_text_on_sphere, merge_segments_to_graph


class SubPointsMixin:
    """Mixin drawing the sub-solar and sub-Earth markers."""

    SUB_POINT_GEOM = {"sun": "subsolar_marker", "earth": "subearth_marker"}
    SUB_POINT_COLOR = {"sun": [1.0, 0.85, 0.25], "earth": [0.45, 0.75, 1.0]}
    SUB_POINT_STROKE = 0.012        # line thickness, as the pins use
    SUB_POINT_ARM_DEG = 4.0         # half-length of the cross arms
    SUB_POINT_LABEL_SCALE = 0.10
    SUB_POINT_LABEL_DROP_DEG = 6.0  # how far below the cross the name is written,
                                    # measured from its centre: the arms reach 4

    @staticmethod
    def _sub_point_direction(lat: float, lon: float) -> np.ndarray:
        """
        Body-frame unit vector of a selenographic position.

        The same frame the labels and pins are placed in (see
        renderer_labels._features_unit_vectors): +Z north, longitude 0 towards -Y.
        """
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        cos_lat = np.cos(lat_rad)
        return np.array([cos_lat * np.sin(lon_rad), -cos_lat * np.cos(lon_rad), np.sin(lat_rad)])

    def _sub_point_graph(self, lat: float, lon: float, name: str):
        """A cross at the point with its name written under it, as one graph."""
        radius = self.MOON_RADIUS
        arm = self.SUB_POINT_ARM_DEG
        # Latitude arm shortened near the poles, longitude arm widened: a degree
        # of longitude covers less ground the further from the equator it is
        north = min(lat + arm, 90.0)
        south = max(lat - arm, -90.0)
        east_west = arm / max(np.cos(np.radians(lat)), 0.2)

        segments = [
            np.array([self._sub_point_direction(south, lon) * radius,
                      self._sub_point_direction(north, lon) * radius]),
            np.array([self._sub_point_direction(lat, lon - east_west) * radius,
                      self._sub_point_direction(lat, lon + east_west) * radius]),
        ]

        flip_horizontal, flip_vertical = self._glyph_flips(lat, lon)
        label_lat = lat - self.SUB_POINT_LABEL_DROP_DEG
        if label_lat < -85.0:                       # write it above instead
            label_lat = lat + self.SUB_POINT_LABEL_DROP_DEG
        segments += create_centered_text_on_sphere(
            name, lat=label_lat, lon=lon, moon_radius=self.MOON_RADIUS,
            offset=0.0, char_scale=self.SUB_POINT_LABEL_SCALE,
            spacing=self.SUB_POINT_LABEL_SCALE,
            flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)
        return merge_segments_to_graph(segments)

    def update_sub_points(self):
        """
        Put both markers where the Sun and Earth stand now.

        The sub-solar point comes straight from the ephemeris; the sub-Earth
        point is the libration, which is defined as exactly that - how far the
        centre of the disk has wandered from (0, 0).
        """
        if self.rt is None or not self.sub_points_visible or self.moon_ephem is None:
            return

        places = {
            "sun": (self.moon_ephem.subsolar_lat, self.moon_ephem.subsolar_lon, "SUN"),
            "earth": (self.moon_ephem.libr_lat_topo, self.moon_ephem.libr_long_topo, "EARTH"),
        }
        self.rt.update_material("sub_point_material", self._no_shadow_flat_material())
        for key, (lat, lon, name) in places.items():
            pos, edges = self._sub_point_graph(lat, lon, name)
            self.rt.set_graph(self.SUB_POINT_GEOM[key], pos=self._rotate_to_scene(pos),
                              edges=edges, r=self.SUB_POINT_STROKE,
                              c=self.SUB_POINT_COLOR[key], mat="sub_point_material")

    def show_sub_points(self, visible: bool = True):
        """Show or hide both markers."""
        if self.rt is None:
            return
        self.sub_points_visible = visible
        if visible:
            self.update_sub_points()
            return
        for name in self.SUB_POINT_GEOM.values():
            try:
                self.rt.delete_geometry(name)
            except Exception:       # never drawn yet, so there is nothing to take away
                pass

    def toggle_sub_points(self):
        """Toggle the sub-solar and sub-Earth markers."""
        self.show_sub_points(not self.sub_points_visible)
