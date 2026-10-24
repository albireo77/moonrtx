"""
AngleScaleBarMixin: a bar along the bottom of the picture saying how wide a
stretch of it is as an angle, for MoonRenderer. A companion of ScaleBarMixin,
which says the same in kilometres.

The angle is that of a scale laid across the picture at the middle of the
view: the angle the stretch would span seen from the Earth if it faced the
Earth, the way an observer with a telescope would measure it. It is a measure
of the picture, not of the ground along a path over it, so it takes no account
of slopes or of the curve of the surface. The whole disk of the Moon is about
31' across, however close the virtual camera is, and close to the ground the
bar reads in arcseconds.

The bar is the kilometre bar's drawing with other numbers on it: marked at its
quarters, numbered as a ruler (0 at the left end, a round angle at the middle
mark, twice that at the right end with the unit), the middle value being 1, 2
or 5 times a power of ten in the unit chosen (degrees, arcminutes or
arcseconds). Where it stands, how it is drawn and when it is drawn again come
from ScaleBarMixin; only its scale, its unit and its place over the kilometre
bar are its own.

Like the compass, the locator and the kilometre bar it is drawn on the canvas
over the render, and so needs drawing into a saved image or a video frame (see
CanvasOverlayMixin.overlay_image). When the kilometre bar is showing too, this
one stands above it.
"""

import math
from typing import Optional

import numpy as np


class AngleScaleBarMixin:
    """Mixin drawing the angular scale bar."""

    ANGLE_BAR_STACK_PX = 44             # how far above the kilometre bar this one stands

    # unit symbol, degrees in one unit
    ANGLE_UNITS = (("°", 1.0), ("′", 1.0 / 60), ("″", 1.0 / 3600))
    # The middle values a bar in arcminutes or arcseconds takes. Sixty of
    # either make the next unit up, and the right end reads twice the middle,
    # so the middle stops at 30: with 1, 2 and 5 times powers of ten it reached
    # 50 and the right end read 100' for what is 1 deg 40'
    ANGLE_SEXAGESIMAL_STEPS = (30, 15, 10, 5, 2, 1)

    def _init_angle_bar(self):
        """Reset the overlay state; called from MoonRenderer.__init__."""
        self.angle_bar_visible = False
        self._angle_bar_items = []
        self._angle_bar_refresh_id = None
        self._angle_bar_last_view = None

    # ---- the numbers ----

    def _angle_bar_deg_per_px(self, height: int) -> Optional[float]:
        """
        Degrees of angle to a pixel at the middle of the view, across the line
        of sight, as a surface square to the line of sight there would be seen
        from the Earth - or None when the view gives no scale (see
        ScaleBarMixin._bar_ground).

        The kilometre bar's kilometres to a pixel, so the two bars always agree,
        over the observer's distance from the ground they are measured at - not
        from the Moon's centre, which is one Moon radius further than the middle
        of the disk and made the angle there some 0.45% too small. The observer
        stands on the scene's -Y axis, the Moon's distance out: the scene is
        turned so (see astro._rotation_matrix), which is why the default view,
        from -Y, shows the disk as the observer sees it.
        """
        ground = self._bar_ground(height)
        if ground is None or not ground[0] or self.moon_ephem is None:
            return None
        km_per_px, point = ground
        observer = np.array([0.0, -self.moon_ephem.distance, 0.0])
        return math.degrees(km_per_px / float(np.linalg.norm(point - observer)))

    @classmethod
    def _angle_bar_length(cls, deg_per_px: float, max_px: float) -> tuple:
        """
        The roundest angle whose bar, twice its length, is no longer than
        max_px: returns (value in the unit, unit symbol, whole bar's length in px).
        The unit is the largest of degrees, arcminutes, arcseconds in which the
        half bar is at least 1. In arcminutes and arcseconds the value is one
        of ANGLE_SEXAGESIMAL_STEPS, so the right end never reads past 60; in
        degrees, and below a second, 1, 2 or 5 times a power of ten.
        """
        longest_deg = deg_per_px * max_px / 2
        symbol, size = cls.ANGLE_UNITS[-1]
        for sym, deg in cls.ANGLE_UNITS:
            if longest_deg >= deg:
                symbol, size = sym, deg
                break
        longest = longest_deg / size
        if size < 1.0 and longest >= 1.0:
            value = next(m for m in cls.ANGLE_SEXAGESIMAL_STEPS if m <= longest)
        else:
            value = cls._round_bar_value(longest)
        return value, symbol, 2 * value * size / deg_per_px

    # ---- drawing ----

    def _clear_angle_bar_items(self):
        self._angle_bar_items = self._clear_overlay(self._angle_bar_items)

    def _draw_angle_bar(self):
        """Draw the bar for the view as it stands, above the kilometre bar if that is up."""
        self._clear_angle_bar_items()

        canvas = self._overlay_canvas()
        if canvas is None or not self.angle_bar_visible:
            return
        width, height = self._bar_canvas_size(canvas)
        deg_per_px = self._angle_bar_deg_per_px(height)
        if not deg_per_px:
            return
        value, unit, length = self._angle_bar_length(
            deg_per_px, self._overlay_px(self.SCALE_BAR_MAX_PX))
        raised = self._overlay_px(self.ANGLE_BAR_STACK_PX) if self.scale_bar_visible else 0.0
        self._angle_bar_items = self._draw_bar(canvas, width, height, length,
                                               f"{value:g}", f"{2 * value:g}", unit, raised)

    # ---- keeping up with the view ----

    def _angle_bar_refresh_tick(self):
        self._angle_bar_refresh_id = None
        if not self.angle_bar_visible:
            return
        # Whether the kilometre bar is showing too, which decides where this one stands
        state = self._bar_view_state(self.scale_bar_visible)
        if state != self._angle_bar_last_view:
            self._angle_bar_last_view = state
            self._draw_angle_bar()
        self._angle_bar_refresh_id = self._schedule_overlay(self._angle_bar_refresh_tick)

    def show_angle_bar(self, visible: bool = True):
        """Show or hide the angular scale bar."""
        if self.rt is None:
            return

        self.angle_bar_visible = visible
        self._angle_bar_refresh_id = self._cancel_overlay(self._angle_bar_refresh_id)

        if visible:
            self._angle_bar_last_view = self._bar_view_state(self.scale_bar_visible)
            self._draw_angle_bar()
            self._angle_bar_refresh_id = self._schedule_overlay(self._angle_bar_refresh_tick)
        else:
            self._clear_angle_bar_items()

    def toggle_angle_bar(self):
        """Toggle the angular scale bar."""
        self.show_angle_bar(not self.angle_bar_visible)
