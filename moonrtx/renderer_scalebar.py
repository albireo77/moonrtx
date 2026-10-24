"""
ScaleBarMixin: a bar along the bottom of the picture saying how far a stretch
of it is on the ground, for MoonRenderer - and the drawing it shares with its
companion, the angular bar (see renderer_scalebar_angle).

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

The two bars are one drawing with different numbers on it: where a bar stands,
how it is marked and lettered, and when it is drawn again are written here once
(_bar_place, _draw_bar, _bar_view_state) for both, each bar bringing only its
own scale and unit. Kept apart, the two copies had drifted: the room a video
keeps in the corner for its own label was given to one bar and not the other.
"""

import math
from typing import Optional

import numpy as np
import tkinter as tk
import tkinter.font as tkfont


class ScaleBarMixin:
    """Mixin drawing the scale bar, and the drawing both scale bars share."""

    # Sizes for both bars, at 96 dpi
    SCALE_BAR_FONT = ("Consolas", 10, "bold")
    SCALE_BAR_MAX_PX = 320          # the longest the bar is drawn
    SCALE_BAR_LINE_WIDTH = 2
    SCALE_BAR_TICK_PX = 6           # the ends, turned up
    SCALE_BAR_MINOR_TICK_PX = 4     # the quarters between them, shorter
    SCALE_BAR_MINOR_AT = (0.25, 0.5, 0.75)
    SCALE_BAR_RIM_PX = 1            # the dark edge round the line, as round the lettering
    SCALE_BAR_MARGIN_PX = 16        # from the edge of a picture, as the compass's
    SCALE_BAR_PANEL_GAP_PX = 16     # from the status panel on the window
    SCALE_BAR_LABEL_GAP_PX = 2      # between the ticks and the distance over them
    SCALE_BAR_HIT_TOLERANCE = 0.02  # share of reference_radius a terrain hit may stray from it

    def _init_scale_bar(self):
        """Reset the overlay state; called from MoonRenderer.__init__."""
        self.scale_bar_visible = False
        self._scale_bar_items = []
        self._scale_bar_refresh_id = None
        self._scale_bar_last_view = None
        # How much of the bottom-right corner of a picture is taken already -
        # by a video's own label there - for both bars to stand above it
        self._scale_bar_floor = 0.0

    def _scale_bar_colour(self) -> str:
        """The yellow of the labels on the Moon (LabelsMixin.SPOT_LABEL_COLOR)."""
        return "#%02x%02x%02x" % tuple(round(255 * c) for c in self.SPOT_LABEL_COLOR)

    # ---- the numbers ----

    def _scale_bar_ground_radius(self) -> float:
        """
        The radius of the ground at the middle of the view: the reference radius,
        or the radius of the relief the render shows there when it can tell it
        (see rt._get_hit_at).
        """
        radius = self.reference_radius

        # Not while a video export runs: it draws from the render thread, before
        # the frame exists, so the hit buffer would be the previous frame's
        if self._video_export is None:
            rt = self.rt
            hx, hy, hz, hd = rt._get_hit_at(rt._width // 2, rt._height // 2)
            if hd > 0:
                ground = float(np.linalg.norm([hx, hy, hz]))
                if abs(ground - radius) <= self.SCALE_BAR_HIT_TOLERANCE * radius:
                    return ground
        return radius

    def _scale_bar_km_per_px(self, height: int) -> Optional[float]:
        """
        Kilometres of ground to a pixel at the middle of the view, across the
        line of sight - or None when the Moon is behind the camera, or the view
        gives no scale (see _bar_ground).
        """
        ground = self._bar_ground(height)
        return None if ground is None else ground[0]

    def _bar_ground(self, height: int) -> Optional[tuple]:
        """
        The scale at the middle of the view, for both bars: (kilometres of
        ground to a pixel there, across the line of sight; the point of the
        ground the line of sight meets, in kilometres from the Moon's centre
        along the scene's axes) - or None when the Moon is behind the camera, or
        the view gives no scale (no camera field, no direction). The angular bar
        is the first over the observer's distance from the second, so the two
        always agree.

        Measured where the line of sight meets the ground: the reference sphere, raised
        or lowered to the relief under the middle of the view when the render shows it;
        one passing beside the Moon is measured at the plane through its centre, which
        is where the limb it passes is.

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
        ground = self._scale_bar_ground_radius()
        distance = ahead - math.sqrt(ground ** 2 - miss2) if miss2 < ground ** 2 else ahead

        # The field is the height of the picture (see pan_tilt_view)
        scene_per_px = 2 * distance * math.tan(math.radians(fov) / 2) / height
        km_per_scene = self.MOON_RADIUS_KM / self.reference_radius
        return scene_per_px * km_per_scene, (eye + distance * forward) * km_per_scene

    @staticmethod
    def _round_bar_value(longest: float) -> float:
        """The roundest value - 1, 2 or 5 times a power of ten - no larger than longest."""
        power = 10.0 ** math.floor(math.log10(longest))
        return next(m * power for m in (5, 2, 1) if m * power <= longest)

    @classmethod
    def _scale_bar_length(cls, km_per_px: float, max_px: float) -> tuple:
        """
        The roundest distance whose bar, twice its length, is no longer than
        max_px - the distance being that of half the bar, the number over the
        middle mark - and the whole bar's length.
        """
        km = cls._round_bar_value(km_per_px * max_px / 2)
        return km, 2 * km / km_per_px

    @staticmethod
    def _scale_bar_numbers(km: float) -> tuple:
        """
        The numbers over the middle mark and the right end, and the unit the
        last one carries - metres while the middle one is under a kilometre, so
        they are all whole.
        """
        value, unit = (km, "km") if km >= 1.0 else (km * 1000, "m")
        return f"{value:g}", f"{2 * value:g}", unit

    def _scale_bar_measure(self, canvas, text: str) -> float:
        """How wide text is in the bars' lettering, in the pixels of the canvas."""
        if self._overlay_surface is not None:
            return canvas.measure(text, self.SCALE_BAR_FONT)
        return tkfont.Font(root=canvas, font=self.SCALE_BAR_FONT).measure(text)

    # ---- drawing, for both bars ----

    def _bar_place(self, width: int, height: int, raised: float = 0.0) -> tuple:
        """
        Where a bar's right end and its line stand: in the bottom-right corner,
        above the room a video keeps there for its own label, and on the window
        beside the status panel when that is up, along its bottom - and raised
        by `raised` above that, which is how the angular bar stands over the
        kilometre one.
        """
        margin = self._overlay_px(self.SCALE_BAR_MARGIN_PX)
        right, bottom = width - margin, height - margin - self._scale_bar_floor
        panel = getattr(self, "_status_panel_frame", None)
        # Only while the panel is in its corner: dragged elsewhere, it no
        # longer takes the corner the bar stands in
        if (self._overlay_surface is None and panel is not None and self.show_status_panel
                and self._status_panel_at is None):
            try:
                if panel.winfo_ismapped():
                    right = panel.winfo_x() - self._overlay_px(self.SCALE_BAR_PANEL_GAP_PX)
                    bottom = panel.winfo_y() + panel.winfo_height()
            except tk.TclError:
                pass
        line = self._overlay_px(self.SCALE_BAR_LINE_WIDTH + 2 * self.SCALE_BAR_RIM_PX)
        return right, bottom - raised - line / 2

    def _bar_canvas_size(self, canvas) -> tuple:
        """The size a bar is drawn for: the canvas's, or the render's before it is laid out."""
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:                       # not laid out yet
            width, height = self.rt._width, self.rt._height
        return width, height

    def _draw_bar(self, canvas, width: int, height: int, length: float,
                  middle: str, end: str, unit: str, raised: float = 0.0) -> list:
        """
        Draw a bar `length` pixels long in the corner, marked at its quarters
        and numbered as a ruler: 0 over its left end, `middle` over its middle
        mark, and `end` over its right end with `unit` run on after it - unit
        carrying whatever space it wants before it. Returns what was drawn.

        Each number stands centred over its mark, so the bar is moved in by
        what the last one and its unit reach past its end, and the lettering
        keeps the margin the bar would have.
        """
        end_half = self._scale_bar_measure(canvas, end) / 2
        right, y = self._bar_place(width, height, raised)
        right -= end_half + self._scale_bar_measure(canvas, unit)
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
        items = []
        line = self._overlay_px(self.SCALE_BAR_LINE_WIDTH)
        rim = self._overlay_px(self.SCALE_BAR_RIM_PX)
        colour = self._scale_bar_colour()
        for fill, stroke in ((self.OVERLAY_HALO_COLOR, line + 2 * rim), (colour, line)):
            for shape in shapes:
                items.append(canvas.create_line(
                    *shape, fill=fill, width=stroke,
                    capstyle=tk.PROJECTING, joinstyle=tk.MITER))
        above = y - tick - self._overlay_px(self.SCALE_BAR_LABEL_GAP_PX)
        for x, text, anchor in ((left, "0", "s"), ((left + right) / 2, middle, "s"),
                                (right - end_half, f"{end}{unit}", "sw")):
            items.extend(self._rimmed_text(
                canvas, x, above, text, colour, self.SCALE_BAR_FONT, anchor=anchor))
        return items

    def _bar_view_state(self, *extra):
        """
        What a bar is drawn from: the camera and the window, which every
        overlay follows, and besides them the zoom, which the wheel changes
        without moving the camera, and where the status panel begins - F3 puts
        it up and takes the corner, and puts it away and gives the corner back.
        A bar depending on more passes it in `extra`.
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
        return self._overlay_view_state(extra=(self.rt._optix.get_camera_fov(0), edge) + extra)

    # ---- the kilometre bar ----

    def _clear_scale_bar_items(self):
        self._scale_bar_items = self._clear_overlay(self._scale_bar_items)

    def _draw_scale_bar(self):
        """Draw the bar for the view as it stands."""
        self._clear_scale_bar_items()

        canvas = self._overlay_canvas()
        if canvas is None or not self.scale_bar_visible:
            return
        width, height = self._bar_canvas_size(canvas)
        km_per_px = self._scale_bar_km_per_px(height)
        if not km_per_px:
            return
        km, length = self._scale_bar_length(km_per_px, self._overlay_px(self.SCALE_BAR_MAX_PX))
        middle, end, unit = self._scale_bar_numbers(km)
        self._scale_bar_items = self._draw_bar(canvas, width, height, length,
                                               middle, end, f" {unit}")

    def _scale_bar_refresh_tick(self):
        self._scale_bar_refresh_id = None
        if not self.scale_bar_visible:
            return
        state = self._bar_view_state()
        if state != self._scale_bar_last_view:
            self._scale_bar_last_view = state
            self._draw_scale_bar()
        self._scale_bar_refresh_id = self._schedule_overlay(self._scale_bar_refresh_tick)

    def show_scale_bar(self, visible: bool = True):
        """Show or hide the scale bar."""
        if self.rt is None:
            return

        self.scale_bar_visible = visible
        self._scale_bar_refresh_id = self._cancel_overlay(self._scale_bar_refresh_id)

        if visible:
            self._scale_bar_last_view = self._bar_view_state()
            self._draw_scale_bar()
            self._scale_bar_refresh_id = self._schedule_overlay(self._scale_bar_refresh_tick)
        else:
            self._clear_scale_bar_items()

    def toggle_scale_bar(self):
        """Toggle the scale bar."""
        self.show_scale_bar(not self.scale_bar_visible)
