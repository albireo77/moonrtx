"""
PinsMixin: pin creation, removal, toggle, and orientation for MoonRenderer.

Each pin digit is a single graph geometry (all strokes merged), so rotating
pins after a time change is one update_graph call per pin.
"""

import numpy as np

from moonrtx.moon_grid import (PIN_DIGIT_SCALE, create_single_digit_on_sphere,
                              merge_segments_to_graph)

class PinsMixin:
    """Mixin providing pin management methods for MoonRenderer."""

    PIN_LABEL_RADIUS = 0.012    # Pin digit label thickness
    PIN_COLOR = [1.0, 0.0, 0.0]

    def create_pin(self, digit: int, lat: float, lon: float):
        """
        Create a pin with the given digit at the specified selenographic coordinates.

        Parameters
        ----------
        digit : int
            The digit (1-9) for the pin
        lat : float
            Selenographic latitude in degrees
        lon : float
            Selenographic longitude in degrees
        """
        if self.rt is None:
            return

        # Mirror the digit so it reads the right way round where it is planted
        flip_horizontal, flip_vertical = self._glyph_flips(lat, lon)

        # Generate pin digit segments (left-bottom corner at cursor position)
        pin_segments = create_single_digit_on_sphere(
            digit=digit,
            lat=lat,
            lon=lon,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            digit_scale=PIN_DIGIT_SCALE * self.label_scale(),
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )

        # All strokes of the digit merged into one graph geometry; the place it
        # was planted and its body-frame vertices are kept, the first to rebuild
        # the digit when the view orientation changes, the second to rotate it
        pos, edges = merge_segments_to_graph(pin_segments)
        self.pins[digit] = (lat, lon, pos)

        self.rt.update_material("pin_material", self._no_shadow_flat_material())

        self.rt.set_graph(f"pin_{digit}", pos=self._rotate_to_scene(pos), edges=edges,
                          r=self.PIN_LABEL_RADIUS * self.label_scale(), c=self.PIN_COLOR,
                          mat="pin_material")

    def remove_pin(self, digit: int):
        """
        Remove a pin with the given digit.

        Parameters
        ----------
        digit : int
            The digit (1-9) of the pin to remove
        """
        if self.rt is None or digit not in self.pins:
            return

        self.rt.delete_geometry(f"pin_{digit}")
        del self.pins[digit]

    def toggle_pin_at_cursor(self, event, digit: int):
        """
        Toggle a pin at the cursor position.

        If pins are not visible, do nothing.
        If a pin with this digit exists, remove it.
        Otherwise, create a new pin at the cursor position.

        Parameters
        ----------
        event : tk.Event
            The keyboard event containing mouse position
        digit : int
            The digit (1-9) for the pin
        """
        if self.rt is None:
            return

        # Do nothing if pins are not visible
        if not self.pins_visible:
            return

        # If pin already exists, remove it
        if digit in self.pins:
            self.remove_pin(digit)
            return

        # Get mouse position in image coordinates
        x, y = self.rt._get_image_xy(event.x, event.y)

        # Get hit position at mouse location
        hx, hy, hz, hd = self.rt._get_hit_at(x, y)

        # Check if we hit something (distance > 0 means valid hit)
        if hd <= 0:
            return

        # Convert hit position to selenographic coordinates
        lat, lon = self.hit_to_selenographic(hx, hy, hz)

        if lat is None or lon is None:
            return

        # Create the pin
        self.create_pin(digit, lat, lon)

    def show_pins(self, visible: bool = True):
        """
        Show or hide all pins.

        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return

        # Toggle visibility by setting zero radius (hide) or restoring (show) -
        # restoring the width create_pin gave it, at the lettering size of the zoom
        pin_radius = self.PIN_LABEL_RADIUS * self.label_scale() if visible else 0.0

        for digit in self.pins:
            self.rt.update_graph(f"pin_{digit}", r=pin_radius)

        self.pins_visible = visible

        # When showing pins, update their orientation to match current Moon position
        # This is needed in case time changed while pins were hidden
        if visible:
            self.update_pins_orientation()

        self._update_status_pins()

    def toggle_pins(self):
        """Toggle the pins visibility."""
        self.show_pins(not self.pins_visible)

    def update_pins_orientation(self):
        """
        Update pins to match current Moon orientation.

        This should be called after update_view() to rotate the pins
        along with the Moon surface.
        """
        if self.rt is None or not self.pins or not self.pins_visible:
            return

        if self.moon_rotation is None:
            return

        for digit, (_, _, pos) in self.pins.items():
            self.rt.update_graph(f"pin_{digit}", pos=self._rotate_to_scene(pos))

    # ---- the place the Find window went to ----

    # A cross with its middle left open, so the point itself stays in sight:
    # each arm runs this many degrees out from the place at the full lettering
    # size, and shrinks with the lettering as the view is magnified, so the
    # cross keeps about the same size on screen
    PLACE_MARK_GEOM = "place_mark"
    PLACE_MARK_ARM_DEG = (0.5, 2.0)
    PLACE_MARK_ARM_POINTS = 6           # along each arm, so it lies on the sphere

    def mark_place(self, lat: float, lon: float):
        """
        Mark a place on the surface - where the Find window was told to go,
        which has no name to label it with. One place at a time; it stays until
        Delete takes it off with the labels Find leaves (see hide_pinned_labels).
        """
        self._place_mark = (lat, lon)
        self._draw_place_mark()

    def clear_place_mark(self):
        """Take the place's mark off, if there is one."""
        self._place_mark = None
        self._place_mark_pos = None
        if self.rt is not None and self.PLACE_MARK_GEOM in self.rt.geometry_data:
            self.rt.delete_geometry(self.PLACE_MARK_GEOM)

    def _draw_place_mark(self):
        """
        Build the mark for the place, at the lettering size of the moment.

        Each arm is a stretch of great circle from the place towards north,
        south, east or west, so the cross lies on the globe and is square to
        the meridian there wherever the place is, poles included.
        """
        if self.rt is None or self._place_mark is None:
            return
        lat, lon = self._place_mark
        la, lo = np.radians(lat), np.radians(lon)
        # The place and the two directions along the surface from it, in the
        # frame the pins and labels are placed in (see _sub_point_direction)
        place = self._sub_point_direction(lat, lon)
        north = np.array([-np.sin(la) * np.sin(lo), np.sin(la) * np.cos(lo), np.cos(la)])
        east = np.array([np.cos(lo), np.sin(lo), 0.0])
        inner, outer = (np.radians(a) * self.label_scale() for a in self.PLACE_MARK_ARM_DEG)
        angles = np.linspace(inner, outer, self.PLACE_MARK_ARM_POINTS)
        arms = [(np.outer(np.cos(angles), place) + np.outer(np.sin(angles), way)) * self.MOON_RADIUS
                for way in (north, -north, east, -east)]

        pos, edges = merge_segments_to_graph(arms)
        self._place_mark_pos = pos
        self.rt.update_material("pin_material", self._no_shadow_flat_material())
        self.rt.set_graph(self.PLACE_MARK_GEOM, pos=self._rotate_to_scene(pos), edges=edges,
                          r=self.PIN_LABEL_RADIUS * self.label_scale(), c=self.PIN_COLOR,
                          mat="pin_material")

    def update_place_mark_orientation(self):
        """Turn the mark with the Moon after a step in time, as the pins are."""
        if self.rt is None or self._place_mark_pos is None:
            return
        self.rt.update_graph(self.PLACE_MARK_GEOM, pos=self._rotate_to_scene(self._place_mark_pos))

    def update_pins_for_view_orientation(self):
        """
        Rebuild pin digits so they read the right way round in the current view.

        Called when the view orientation changes (F5-F8 keys); the pins stay
        where they were planted, only their digits are mirrored to suit.
        """
        if self.rt is None or not self.pins:
            return

        for digit, (lat, lon, _) in list(self.pins.items()):
            self.create_pin(digit, lat, lon)

        if not self.pins_visible:
            self.show_pins(False)
