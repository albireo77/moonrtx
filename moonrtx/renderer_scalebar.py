"""
ScaleBarMixin: a bar along the bottom of the picture saying how far a stretch
of it is on the ground, for MoonRenderer.

The picture gives no sense of size on its own. A crater filling the window may
be two kilometres across or two hundred, and the zoom, the eye moved in with
Shift and the right button, and the Moon's own distance all change which. The
bar is drawn as long as it is on the surface and marked at its quarters, and
numbered as a ruler is: 0 at its left end, the distance at its middle mark, and
twice that at its right end with the unit - each number over the place it
measures to, so none can be read as the length of something else. The middle
one is a round distance, 1, 2 or 5 times a power of ten, chosen afresh as the
view changes, so the bar stays a comfortable length and reads at a glance.

It is true at the middle of the view, where the line of sight meets the ground,
and across the line of sight. Towards the limb the ground is foreshortened
along the radius of the disk, and a bar laid across the picture cannot say so;
the middle is where the eye is looking, and what the bar is for.

Like the compass and the locator it is drawn on the canvas over the render,
and so needs drawing into a saved image or a video frame (see
CanvasOverlayMixin.overlay_image). On the window it stands in the bottom-right
corner, beside the status panel when F3 has that up; in a picture, which the
panel is not part of, in the corner itself.
"""

import math
from typing import Optional

import numpy as np
import tkinter as tk
import tkinter.font as tkfont


class ScaleBarMixin:
    """Mixin drawing the scale bar."""

    SCALE_BAR_FONT = ("Consolas", 10, "bold")
    SCALE_BAR_MAX_PX = 320          # the longest the bar is drawn, at 96 dpi
    SCALE_BAR_LINE_WIDTH = 2
    SCALE_BAR_TICK_PX = 6           # the ends, turned up
    SCALE_BAR_MINOR_TICK_PX = 4     # the quarters between them, shorter
    SCALE_BAR_MINOR_AT = (0.25, 0.5, 0.75)
    SCALE_BAR_RIM_PX = 1            # the dark edge round the line, as round the lettering
    SCALE_BAR_MARGIN_PX = 16        # from the edge of a picture, as the compass's
    SCALE_BAR_PANEL_GAP_PX = 16     # from the status panel on the window
    SCALE_BAR_LABEL_GAP_PX = 2      # between the ticks and the distance over them

    def _init_scale_bar(self):
        """Reset the overlay state; called from MoonRenderer.__init__."""
        self.scale_bar_visible = False
        self._scale_bar_items = []
        self._scale_bar_refresh_id = None
        self._scale_bar_last_view = None
        # How much of the bottom-right corner of a picture is taken already -
        # by a video's own label there - for the bar to stand above it
        self._scale_bar_floor = 0.0

    def _scale_bar_colour(self) -> str:
        """The yellow of the labels on the Moon (LabelsMixin.SPOT_LABEL_COLOR)."""
        return "#%02x%02x%02x" % tuple(round(255 * c) for c in self.SPOT_LABEL_COLOR)

    # ---- the numbers ----

    def _scale_bar_km_per_px(self, height: int) -> Optional[float]:
        """
        Kilometres of ground to a pixel at the middle of the view, across the
        line of sight - or None when the view looks away from the Moon.

        Measured where the line of sight meets the reference sphere, which is
        the ground to within the relief; one passing beside the Moon is
        measured at the plane through its centre, which is where the limb it
        passes is.

        The reference sphere is not the scene's MOON_RADIUS but some 0.6%
        inside it, and a scene unit that much more ground (see
        MoonRenderer.reference_radius).
        """
        if self.rt is None or height <= 0:
            return None
        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"], dtype=float)
        forward = np.array(cam["Target"], dtype=float) - eye
        norm = np.linalg.norm(forward)
        fov = self.rt._optix.get_camera_fov(0)
        if norm == 0.0 or fov <= 0.0:
            return None
        forward /= norm

        ahead = -float(eye @ forward)               # the Moon's centre, along the view
        if ahead <= 0.0:
            return None
        miss2 = float(eye @ eye) - ahead ** 2       # how far the view passes from it, squared
        radius = self.reference_radius
        distance = ahead - math.sqrt(radius ** 2 - miss2) if miss2 < radius ** 2 else ahead

        # The field is the height of the picture (see pan_tilt_view)
        scene_per_px = 2 * distance * math.tan(math.radians(fov) / 2) / height
        return scene_per_px * self.MOON_RADIUS_KM / radius

    @staticmethod
    def _scale_bar_length(km_per_px: float, max_px: float) -> tuple:
        """
        The roundest distance whose bar, twice its length, is no longer than
        max_px - the distance being that of half the bar, the number over the
        middle mark - and the whole bar's length.
        """
        longest = km_per_px * max_px / 2
        power = 10.0 ** math.floor(math.log10(longest))
        km = next(m * power for m in (5, 2, 1) if m * power <= longest)
        return km, 2 * km / km_per_px

    @staticmethod
    def _scale_bar_numbers(km: float) -> tuple:
        """
        The numbers over the left end, the middle mark and the right end, and
        the unit the last one carries - metres while the middle one is under a
        kilometre, so they are all whole.
        """
        value, unit = (km, "km") if km >= 1.0 else (km * 1000, "m")
        return "0", f"{value:g}", f"{2 * value:g}", unit

    def _scale_bar_measure(self, canvas, text: str) -> float:
        """How wide text is in the bar's lettering, in the pixels of the canvas."""
        if self._overlay_surface is not None:
            return canvas.measure(text, self.SCALE_BAR_FONT)
        return tkfont.Font(root=canvas, font=self.SCALE_BAR_FONT).measure(text)

    # ---- drawing ----

    def _scale_bar_place(self, width: int, height: int) -> tuple:
        """
        Where the bar's right end and its line stand: in the bottom-right
        corner, and on the window beside the status panel when that is up,
        along its bottom.
        """
        margin = self._overlay_px(self.SCALE_BAR_MARGIN_PX)
        right, bottom = width - margin, height - margin - self._scale_bar_floor
        panel = getattr(self, "_status_panel_frame", None)
        if self._overlay_surface is None and panel is not None and self.show_status_panel:
            try:
                if panel.winfo_ismapped():
                    right = panel.winfo_x() - self._overlay_px(self.SCALE_BAR_PANEL_GAP_PX)
                    bottom = panel.winfo_y() + panel.winfo_height()
            except tk.TclError:
                pass
        line = self._overlay_px(self.SCALE_BAR_LINE_WIDTH + 2 * self.SCALE_BAR_RIM_PX)
        return right, bottom - line / 2

    def _clear_scale_bar_items(self):
        self._scale_bar_items = self._clear_overlay(self._scale_bar_items)

    def _draw_scale_bar(self):
        """Draw the bar for the view as it stands."""
        self._clear_scale_bar_items()

        canvas = self._overlay_canvas()
        if canvas is None or not self.scale_bar_visible:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:                       # not laid out yet
            width, height = self.rt._width, self.rt._height

        km_per_px = self._scale_bar_km_per_px(height)
        if not km_per_px:
            return
        km, length = self._scale_bar_length(km_per_px, self._overlay_px(self.SCALE_BAR_MAX_PX))
        start, middle, end, unit = self._scale_bar_numbers(km)
        # Each number stands centred over its mark, the unit running on after
        # the last; the bar is moved in by what that reaches past its end, so
        # the lettering keeps the margin the bar would have
        end_half = self._scale_bar_measure(canvas, end) / 2
        right, y = self._scale_bar_place(width, height)
        right -= end_half + self._scale_bar_measure(canvas, f" {unit}")
        left = right - length
        tick = self._overlay_px(self.SCALE_BAR_TICK_PX)
        minor = self._overlay_px(self.SCALE_BAR_MINOR_TICK_PX)
        # The bar with its ends turned up, and the quarters marked on it
        shapes = [(left, y - tick, left, y, right, y, right, y - tick)]
        shapes += [(left + share * length, y, left + share * length, y - minor)
                   for share in self.SCALE_BAR_MINOR_AT]

        # A dark rim under the yellow, as the lettering has, so the bar reads
        # over black sky and lit highland alike. Every rim goes down before any
        # yellow, so no rim is drawn across the bar's own line
        line = self._overlay_px(self.SCALE_BAR_LINE_WIDTH)
        rim = self._overlay_px(self.SCALE_BAR_RIM_PX)
        colour = self._scale_bar_colour()
        for fill, stroke in ((self.OVERLAY_HALO_COLOR, line + 2 * rim), (colour, line)):
            for shape in shapes:
                self._scale_bar_items.append(canvas.create_line(
                    *shape, fill=fill, width=stroke,
                    capstyle=tk.PROJECTING, joinstyle=tk.MITER))
        above = y - tick - self._overlay_px(self.SCALE_BAR_LABEL_GAP_PX)
        for x, text, anchor in ((left, start, "s"), ((left + right) / 2, middle, "s"),
                                (right - end_half, f"{end} {unit}", "sw")):
            self._scale_bar_items.extend(self._rimmed_text(
                canvas, x, above, text, colour, self.SCALE_BAR_FONT, anchor=anchor))

    # ---- keeping up with the view ----

    def _scale_bar_view_state(self):
        """
        What the bar is drawn from: the camera and the window, which every
        overlay follows, and besides them the zoom, which the wheel changes
        without moving the camera, and where the status panel begins - F3 puts
        it up and takes the corner, and puts it away and gives the corner back.
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
        return self._overlay_view_state(extra=(self.rt._optix.get_camera_fov(0), edge))

    def _scale_bar_refresh_tick(self):
        self._scale_bar_refresh_id = None
        if not self.scale_bar_visible:
            return
        state = self._scale_bar_view_state()
        if state != self._scale_bar_last_view:
            self._scale_bar_last_view = state
            self._draw_scale_bar()
        self._schedule_scale_bar_refresh()

    def _schedule_scale_bar_refresh(self):
        self._scale_bar_refresh_id = self._schedule_overlay(self._scale_bar_refresh_tick)

    def show_scale_bar(self, visible: bool = True):
        """Show or hide the scale bar."""
        if self.rt is None:
            return

        self.scale_bar_visible = visible

        self._scale_bar_refresh_id = self._cancel_overlay(self._scale_bar_refresh_id)

        if visible:
            self._scale_bar_last_view = self._scale_bar_view_state()
            self._draw_scale_bar()
            self._schedule_scale_bar_refresh()
        else:
            self._clear_scale_bar_items()

    def toggle_scale_bar(self):
        """Toggle the scale bar."""
        self.show_scale_bar(not self.scale_bar_visible)
