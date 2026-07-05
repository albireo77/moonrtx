"""
PinsMixin: pin creation, removal, toggle, and orientation for MoonRenderer.

Each pin digit is a single graph geometry (all strokes merged), so rotating
pins after a time change is one update_graph call per pin.
"""

from moonrtx.view_orientation import FLIP_HORIZONTAL_VIEW_ORIENTATIONS, FLIP_VERTICAL_VIEW_ORIENTATIONS
from moonrtx.moon_grid import create_single_digit_on_sphere, merge_segments_to_graph

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

        # Determine flip flags based on current orientation
        flip_horizontal = self.view_orientation in FLIP_HORIZONTAL_VIEW_ORIENTATIONS
        flip_vertical = self.view_orientation in FLIP_VERTICAL_VIEW_ORIENTATIONS

        # Generate pin digit segments (left-bottom corner at cursor position)
        pin_segments = create_single_digit_on_sphere(
            digit=digit,
            lat=lat,
            lon=lon,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )

        # All strokes of the digit merged into one graph geometry;
        # body-frame vertices are kept for rotation updates
        pos, edges = merge_segments_to_graph(pin_segments)
        self.pins[digit] = pos

        self.rt.update_material("pin_material", self._no_shadow_flat_material())

        self.rt.set_graph(f"pin_{digit}", pos=self._rotate_to_scene(pos), edges=edges,
                          r=self.PIN_LABEL_RADIUS, c=self.PIN_COLOR, mat="pin_material")

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

        # Toggle visibility by setting zero radius (hide) or restoring (show)
        pin_radius = self.PIN_LABEL_RADIUS if visible else 0.0

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

        for digit, pos in self.pins.items():
            self.rt.update_graph(f"pin_{digit}", pos=self._rotate_to_scene(pos))
