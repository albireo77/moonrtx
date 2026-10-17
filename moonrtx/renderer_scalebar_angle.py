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

The bar is drawn like the kilometre one: marked at its quarters, numbered as a
ruler (0 at the left end, a round angle at the middle mark, twice that at the
right end with the unit), the middle value being 1, 2 or 5 times a power of ten
in the unit chosen (degrees, arcminutes or arcseconds).

Like the compass, the locator and the kilometre bar it is drawn on the canvas
over the render, and so needs drawing into a saved image or a video frame (see
CanvasOverlayMixin.overlay_image). When the kilometre bar is showing too, this
one stands above it.
"""

import math
from typing import Optional

import numpy as np
import tkinter as tk
import tkinter.font as tkfont


class AngleScaleBarMixin:
    """Mixin drawing the angular scale bar."""

    ANGLE_BAR_FONT = ("Consolas", 10, "bold")
    ANGLE_BAR_MAX_PX = 320
    ANGLE_BAR_LINE_WIDTH = 2
    ANGLE_BAR_TICK_PX = 6
    ANGLE_BAR_MINOR_TICK_PX = 4
    ANGLE_BAR_MINOR_AT = (0.25, 0.5, 0.75)
    ANGLE_BAR_RIM_PX = 1
    ANGLE_BAR_MARGIN_PX = 16
    ANGLE_BAR_PANEL_GAP_PX = 16
    ANGLE_BAR_LABEL_GAP_PX = 2
    ANGLE_BAR_STACK_PX = 44             # how far above the kilometre bar this one stands

    # unit symbol, degrees in one unit
    ANGLE_UNITS = (("\u00b0", 1.0), ("\u2032", 1.0 / 60), ("\u2033", 1.0 / 3600))

    def _init_angle_bar(self):
        """Reset the overlay state; called from MoonRenderer.__init__."""
        self.angle_bar_visible = False
        self._angle_bar_items = []
        self._angle_bar_refresh_id = None
        self._angle_bar_last_view = None
        # How much of the bottom-right corner of a picture is taken already
        self._angle_bar_floor = 0.0

    def _angle_bar_colour(self) -> str:
        """The yellow of the labels on the Moon (LabelsMixin.SPOT_LABEL_COLOR)."""
        return "#%02x%02x%02x" % tuple(round(255 * c) for c in self.SPOT_LABEL_COLOR)

    # ---- the numbers ----

    def _angle_bar_distance(self, eye, forward) -> Optional[float]:
        """
        Distance from the camera to the ground at the middle of the view, or None
        when the view looks past the Moon. The ground is the one the kilometre bar
        measures to (see ScaleBarMixin._scale_bar_ground_radius).
        """
        radius = self._scale_bar_ground_radius()
        ahead = -float(eye @ forward)
        if ahead <= 0.0:
            return None
        miss2 = float(eye @ eye) - ahead ** 2
        return ahead - math.sqrt(radius ** 2 - miss2) if miss2 < radius ** 2 else ahead

    def _angle_bar_deg_per_px(self, height: int) -> Optional[float]:
        """
        Degrees of angle to a pixel at the middle of the view, across the line
        of sight, as a surface square to the line of sight there would be seen
        from the Earth - or None when the view looks away from the Moon.
        """
        if self.rt is None or height <= 0:
            return None
        fov = self.rt._optix.get_camera_fov(0)
        if fov <= 0.0:
            return None

        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"], dtype=float)
        forward = np.array(cam["Target"], dtype=float) - eye
        norm = np.linalg.norm(forward)
        if norm == 0.0:
            return None
        forward /= norm

        distance = self._angle_bar_distance(eye, forward)
        if distance is None:
            return None

        scene_per_px = 2 * distance * math.tan(math.radians(fov) / 2) / height
        km_per_px = scene_per_px * self.MOON_RADIUS_KM / self.reference_radius
        return math.degrees(km_per_px / self.moon_ephem.distance)

    @classmethod
    def _angle_bar_length(cls, deg_per_px: float, max_px: float) -> tuple:
        """
        The roundest angle whose bar, twice its length, is no longer than
        max_px: returns (value in the unit, unit symbol, whole bar's length in px).
        The unit is the largest of degrees, arcminutes, arcseconds in which the
        half bar is at least 1.
        """
        longest_deg = deg_per_px * max_px / 2
        symbol, size = cls.ANGLE_UNITS[-1]
        for sym, deg in cls.ANGLE_UNITS:
            if longest_deg >= deg:
                symbol, size = sym, deg
                break
        longest = longest_deg / size
        power = 10.0 ** math.floor(math.log10(longest))
        value = next(m * power for m in (5, 2, 1) if m * power <= longest)
        return value, symbol, 2 * value * size / deg_per_px

    @staticmethod
    def _angle_bar_numbers(value: float) -> tuple:
        """The numbers over the left end, the middle mark and the right end."""
        return "0", f"{value:g}", f"{2 * value:g}"

    def _angle_bar_measure(self, canvas, text: str) -> float:
        """How wide text is in the bar's lettering, in the pixels of the canvas."""
        if self._overlay_surface is not None:
            return canvas.measure(text, self.ANGLE_BAR_FONT)
        return tkfont.Font(root=canvas, font=self.ANGLE_BAR_FONT).measure(text)

    # ---- drawing ----

    def _angle_bar_place(self, width: int, height: int) -> tuple:
        """
        Where the bar's right end and its line stand: in the bottom-right
        corner, beside the status panel on the window when that is up, and
        above the kilometre bar when that is showing.
        """
        margin = self._overlay_px(self.ANGLE_BAR_MARGIN_PX)
        right, bottom = width - margin, height - margin - self._angle_bar_floor
        panel = getattr(self, "_status_panel_frame", None)
        if self._overlay_surface is None and panel is not None and self.show_status_panel:
            try:
                if panel.winfo_ismapped():
                    right = panel.winfo_x() - self._overlay_px(self.ANGLE_BAR_PANEL_GAP_PX)
                    bottom = panel.winfo_y() + panel.winfo_height()
            except tk.TclError:
                pass
        if getattr(self, "scale_bar_visible", False):
            bottom -= self._overlay_px(self.ANGLE_BAR_STACK_PX)
        line = self._overlay_px(self.ANGLE_BAR_LINE_WIDTH + 2 * self.ANGLE_BAR_RIM_PX)
        return right, bottom - line / 2

    def _clear_angle_bar_items(self):
        self._angle_bar_items = self._clear_overlay(self._angle_bar_items)

    def _draw_angle_bar(self):
        """Draw the bar for the view as it stands."""
        self._clear_angle_bar_items()

        canvas = self._overlay_canvas()
        if canvas is None or not self.angle_bar_visible:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:                       # not laid out yet
            width, height = self.rt._width, self.rt._height

        deg_per_px = self._angle_bar_deg_per_px(height)
        if not deg_per_px:
            return
        value, unit, length = self._angle_bar_length(
            deg_per_px, self._overlay_px(self.ANGLE_BAR_MAX_PX))
        start, middle, end = self._angle_bar_numbers(value)

        # Each number stands centred over its mark, the unit running on after
        # the last; the bar is moved in by what that reaches past its end
        end_half = self._angle_bar_measure(canvas, end) / 2
        right, y = self._angle_bar_place(width, height)
        right -= end_half + self._angle_bar_measure(canvas, unit)
        left = right - length
        tick = self._overlay_px(self.ANGLE_BAR_TICK_PX)
        minor = self._overlay_px(self.ANGLE_BAR_MINOR_TICK_PX)
        # The bar with its ends turned up, and the quarters marked on it
        shapes = [(left, y - tick, left, y, right, y, right, y - tick)]
        shapes += [(left + share * length, y, left + share * length, y - minor)
                   for share in self.ANGLE_BAR_MINOR_AT]

        # A dark rim under the yellow; every rim goes down before any yellow
        line = self._overlay_px(self.ANGLE_BAR_LINE_WIDTH)
        rim = self._overlay_px(self.ANGLE_BAR_RIM_PX)
        colour = self._angle_bar_colour()
        for fill, stroke in ((self.OVERLAY_HALO_COLOR, line + 2 * rim), (colour, line)):
            for shape in shapes:
                self._angle_bar_items.append(canvas.create_line(
                    *shape, fill=fill, width=stroke,
                    capstyle=tk.PROJECTING, joinstyle=tk.MITER))
        above = y - tick - self._overlay_px(self.ANGLE_BAR_LABEL_GAP_PX)
        for x, text, anchor in ((left, start, "s"), ((left + right) / 2, middle, "s"),
                                (right - end_half, f"{end}{unit}", "sw")):
            self._angle_bar_items.extend(self._rimmed_text(
                canvas, x, above, text, colour, self.ANGLE_BAR_FONT, anchor=anchor))

    # ---- keeping up with the view ----

    def _angle_bar_view_state(self):
        """
        What the bar is drawn from: the camera and the window, the zoom, where
        the status panel begins, and whether the kilometre bar is showing.
        """
        if self.rt is None:
            return None
        panel = getattr(self, "_status_panel_frame", None)
        edge = None
        if panel is not None and self.show_status_panel:
            try:
                edge = (panel.winfo_ismapped(), panel.winfo_x(),
                        panel.winfo_y() + panel.winfo_height())
            except tk.TclError:
                pass
        km_bar = getattr(self, "scale_bar_visible", False)
        return self._overlay_view_state(
            extra=(self.rt._optix.get_camera_fov(0), edge, km_bar))

    def _angle_bar_refresh_tick(self):
        self._angle_bar_refresh_id = None
        if not self.angle_bar_visible:
            return
        state = self._angle_bar_view_state()
        if state != self._angle_bar_last_view:
            self._angle_bar_last_view = state
            self._draw_angle_bar()
        self._schedule_angle_bar_refresh()

    def _schedule_angle_bar_refresh(self):
        self._angle_bar_refresh_id = self._schedule_overlay(self._angle_bar_refresh_tick)

    def show_angle_bar(self, visible: bool = True):
        """Show or hide the angular scale bar."""
        if self.rt is None:
            return

        self.angle_bar_visible = visible
        self._angle_bar_refresh_id = self._cancel_overlay(self._angle_bar_refresh_id)

        if visible:
            self._angle_bar_last_view = self._angle_bar_view_state()
            self._draw_angle_bar()
            self._schedule_angle_bar_refresh()
        else:
            self._clear_angle_bar_items()

    def toggle_angle_bar(self):
        """Toggle the angular scale bar."""
        self.show_angle_bar(not self.angle_bar_visible)