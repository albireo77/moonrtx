"""
PinsMixin: pin creation, removal, toggle, and orientation for MoonRenderer.
"""

import numpy as np

from plotoptix.materials import m_flat

from moonrtx.constants import (
    PIN_COLOR, PIN_LABEL_RADIUS,
    ORIENTATION_NSEW, ORIENTATION_SNEW, ORIENTATION_SNWE,
)
from moonrtx.moon_grid import create_single_digit_on_sphere


class PinsMixin:
    """Mixin providing pin management methods for MoonRenderer."""

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
        flip_horizontal = self.orientation_mode in (ORIENTATION_NSEW, ORIENTATION_SNEW)
        flip_vertical = self.orientation_mode in (ORIENTATION_SNEW, ORIENTATION_SNWE)
        
        # Generate pin digit segments (left-bottom corner at cursor position)
        pin_segments = create_single_digit_on_sphere(
            digit=digit,
            lat=lat,
            lon=lon,
            moon_radius=self.moon_radius,
            offset=0.0,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical
        )
        
        # Store pin data (original segments for rotation updates)
        self.pins[digit] = pin_segments
        
        # Create material for pins if not already created
        m_pin = m_flat.copy()
        self.rt.update_material("pin_material", m_pin)
        
        # Line thickness for pins
        pin_radius = PIN_LABEL_RADIUS
        
        # Apply Moon rotation to segments and add to renderer
        R = self.moon_rotation
        
        for j, seg in enumerate(pin_segments):
            name = f"pin_{digit}_{j}"
            if R is not None:
                rotated = (R @ seg.T).T
            else:
                rotated = seg
            self.rt.set_data(name, pos=rotated, r=pin_radius,
                            c=PIN_COLOR, geom="SegmentChain", mat="pin_material")

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
        
        pin_segments = self.pins[digit]
        
        # Remove all segments from renderer
        for j in range(len(pin_segments)):
            name = f"pin_{digit}_{j}"
            try:
                self.rt.delete_geometry(name)
            except:
                pass
        
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
        pin_radius = PIN_LABEL_RADIUS if visible else 0.0
        
        for digit, pin_segments in self.pins.items():
            for j in range(len(pin_segments)):
                name = f"pin_{digit}_{j}"
                self.rt.update_data(name, r=pin_radius)
        
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
        
        R = self.moon_rotation
        
        if R is None:
            return
        
        for digit, pin_segments in self.pins.items():
            for j, orig_seg in enumerate(pin_segments):
                name = f"pin_{digit}_{j}"
                rotated = (R @ orig_seg.T).T
                try:
                    self.rt.update_data(name, pos=rotated)
                except:
                    pass
