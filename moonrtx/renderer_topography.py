"""
TopographyMixin: a rectangle of the Moon drawn as a block of ground in 3-D,
coloured by height, lit as a relief map is and with lines of equal height on
it, turned with the mouse, for MoonRenderer.

The rectangle is drawn on the picture with Ctrl and the right button, and the
area is the ground under it and nothing more: a grid stretched between its four
corners' ground, which the view's turn and slant make a four-sided patch of its
own shape. Heights are sampled on that grid from the elevation map the surface
is drawn with, for the block's shape - and sampled again, as finely as the map
itself where TOPO_DETAIL_POINTS allows, for what is painted on it: the colours,
the light and the lines of equal height, so the block shows the map's own
detail rather than its grid's.

The block is painted into an image: the grid's cells from the furthest to the
nearest, so nearer ground hides what lies behind it, and the fine samples laid
over each pixel by where it falls in its cell. It is shown in a window of its
own, in the profile window's lettering. Its heights are exaggerated, the
relief of the Moon being slight against its extent - a crater some four
kilometres deep is a hundred across - by a round factor the window states,
chosen so the relief is seen at all. Dragging with the left button turns it;
a double click returns it to the view it opened in.
"""

import math
import re
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageTk

from moonrtx.display import ToolTip


class TopographyMixin:
    """Mixin drawing a rectangle of the Moon as a block of ground in 3-D."""

    TOPO_WINDOW = "topography"              # its name in DialogsMixin._window_memory
    # No side longer than this: past it the area stops being a piece of ground
    # and becomes a stretch of globe, with the curvature taken out
    TOPO_MAX_SIDE_KM = 800.0
    # Samples along the longer side: the map's own spacing where there is room,
    # never fewer than the first nor more than the second, which keeps the
    # painting quick enough to follow the mouse
    TOPO_MIN_POINTS = 16
    TOPO_MAX_POINTS = 240
    # And for what is painted on it - its colours, its light and its lines -
    # the map's own spacing up to this many along the longer side: the
    # painting's time goes by the window's pixels, not by these
    TOPO_DETAIL_POINTS = 1600
    # Up to this many levels from the lowest ground to the highest, on the
    # roundest step that keeps to it - 200 m across Copernicus' 4.4 km - a
    # colour for each, and every fourth or fifth, on a rounder one, a line
    TOPO_LEVELS = 24
    # The exaggerated relief comes to about this share of the area's longer side
    # - no more: a fifth stood the hills so tall as to mislead about them
    TOPO_RELIEF_SHARE = 0.1
    # The slider's factors run in the same round steps from 1, true scale, to
    # this many times the chosen one
    TOPO_EXAGGERATION_RANGE = 10.0
    # The view the block opens in and goes back to: turned as the rectangle
    # lies on the picture (see show_topography), and looked at from this far
    # above level
    TOPO_START_ELEVATION = 35.0
    TOPO_ELEVATION_LIMITS = (5.0, 90.0)     # from nearly side-on to straight down
    TOPO_DEGREES_PER_PX = (0.5, 0.3)        # a drag across, and a drag up or down
    # The wheel: this much larger or smaller a notch, around the middle of the
    # block, from this many times smaller than it opens - filling the drawing
    # as it is first seen, it may stand past the edges once turned - to this
    # many times larger
    TOPO_ZOOM_STEP = 1.1
    TOPO_ZOOM_MIN = 0.5
    TOPO_ZOOM_MAX = 20.0
    TOPO_WIDTH_FRACTION = 0.35              # of the screen, the first time it opens
    TOPO_BASE_COLOUR = "#808080"
    # The corner marked on the picture and on the block's grey outline alike,
    # by its turn round the area from the top left as drawn: the bottom right
    TOPO_MARKED_CORNER = 2
    # The colours: the bands between the levels from the lowest to the highest
    # along this ramp, blue, teal, green, orange, red, as shares of the way up
    TOPO_RAMP = ((0.0, (40, 60, 160)), (0.25, (30, 150, 170)), (0.5, (110, 180, 60)),
                 (0.75, (230, 170, 50)), (1.0, (200, 50, 40)))
    TOPO_INDEX_COLOUR = (40, 30, 20)        # the lines over them
    # The colours lit, as a shaded relief map is: each slope by how squarely it
    # faces a light this far round from north and this high, fixed to the
    # ground so that it turns with the block - and a slope turned away from it
    # keeps this share of its colour
    TOPO_LIGHT = (315.0, 40.0)
    TOPO_SHADE = 0.35
    # Painted this many times larger and then reduced, for smooth edges - or
    # fewer, down to once, where that would come to more than this many pixels
    # to work out one by one: a window that large is on a screen whose own
    # pixels are fine enough not to need it
    TOPO_SUPERSAMPLE = 3
    TOPO_PAINT_PIXELS = 1_600_000
    # The place under the pointer in the window, marked on the Moon in the
    # main window as the profile's is (see ProfileMixin._mark_profile_place)
    TOPO_MARK_TAG = "topo_mark"
    # The names of the features inside the area, from the program's own list,
    # each at its place on the block: all those at least this share of the
    # area's longer side across; a satellite crater's by its letter alone, as
    # "Stadius M" is written "M"
    TOPO_NAME_MIN_SHARE = 0.01
    TOPO_SATELLITE_NAME = re.compile(r".+ ([A-Z]{1,2})$")
    # Each in a box of the window's own ground, readable over any colour, a
    # dot of this colour at its place
    TOPO_NAME_COLOUR = "#303030"
    _topo_names = True                      # whether they are shown; kept between runs

    def _init_topography(self):
        """Reset the state; called from MoonRenderer.__init__."""
        self._topography = None             # the open window's drawing, or None
        # The area on the picture - while it is drawn and after (see
        # _keep_area): its canvas tag, its corners' ground and where they are
        # in the window, top left first and round as drawn; None for none
        self._area = None
        self._area_serial = 0
        self._area_drawing_from = None      # (x, y) the rectangle is drawn from, while it is
        self._area_corner = None            # the corner held by its ring, while one is
        self._area_view = None              # the view the kept area was placed in
        self._area_watch = None             # the poll that notices the view move

    # ---- the area on the picture ----

    @property
    def selecting_area(self) -> bool:
        return self._area_drawing_from is not None

    @staticmethod
    def _screen_corners(x0: float, y0: float, x1: float, y1: float) -> list:
        """A rectangle's corners on the picture: top left, top right, bottom right, bottom left, as drawn."""
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def start_area_selection(self, event):
        """
        Ctrl and the right button down: a new rectangle from there, in place
        of the one kept - whose window stays open, to show the new one's
        ground (see finish_area_selection).
        """
        if self.rt is None:
            return
        self.drop_area(close_window=False)
        canvas = self.rt._canvas
        width = max(1, round(self._overlay_px(self.MEASURE_LINE_WIDTH)))
        dash = max(1, round(self._overlay_px(self.MEASURE_LINE_DASH)))
        # Drawn as the measurement line is, being a measuring of its own: the
        # outline, and an empty ring on each corner, all under one tag
        self._area_serial += 1
        tag = f"area{self._area_serial}"
        canvas.create_line(event.x, event.y, event.x, event.y, tags=tag, width=width,
                           fill=self.MEASURE_LINE_COLOR, dash=(dash, dash))
        # The bottom right one filled, as the measurement line's start is, so
        # the window's block can be told the same way round (see
        # TOPO_MARKED_CORNER)
        for k in range(4):
            canvas.create_oval(event.x, event.y, event.x, event.y, tags=tag, width=width,
                               outline=self.MEASURE_LINE_COLOR,
                               fill=self.MEASURE_LINE_COLOR if k == self.TOPO_MARKED_CORNER else '')
        corners = self._screen_corners(event.x, event.y, event.x, event.y)
        self._area = {"tag": tag, "ground": [None] * 4, "canvas": corners}
        self._area_drawing_from = (event.x, event.y)

    def _place_area(self):
        """
        The outline and its rings where the area's corners now are on the
        window, laid on the ground as the measurement line is: each side the
        great circle between its two corners' ground - the very edge of the
        area the window draws (see _topography_grid) - each point of it at its
        own height where the picture shows it, so it bends over rims and floors
        seen at a slant. Straight on the picture where a corner is off the
        Moon, or a side cannot be seen.
        """
        canvas = self.rt._canvas
        outline, *rings = canvas.find_withtag(self._area["tag"])     # in the order made
        ground, on_screen = self._area["ground"], self._area["canvas"]
        points = list(on_screen[0])
        for a in range(4):
            b = (a + 1) % 4
            path = None
            if ground[a] is not None and ground[b] is not None:
                path = self._measure_line_path(ground[a], ground[b], on_screen[a], on_screen[b])
            # Each side from where the last one ended
            points += path[2:] if path is not None else list(on_screen[b])
        canvas.coords(outline, *points)
        radius = self._overlay_px(self.MEASURE_HOOK_RADIUS)
        for ring, (x, y) in zip(rings, on_screen):
            canvas.coords(ring, x - radius, y - radius, x + radius, y + radius)

    def update_area_selection(self, event):
        """The rectangle drawn on to where the pointer is."""
        if self._area_drawing_from is None:
            return
        on_screen = self._screen_corners(*self._area_drawing_from, event.x, event.y)
        self._area["canvas"] = on_screen
        self._area["ground"] = [self._selenographic_at(SimpleNamespace(x=cx, y=cy))
                                for cx, cy in on_screen]
        self._place_area()

    def finish_area_selection(self, event):
        """
        The button let go: the area kept on the picture and its ground drawn -
        unless it was no rectangle at all, or a corner of it is off the Moon,
        where there is no ground to draw; then it goes, and the window showing
        the one it replaced with it.
        """
        if self._area_drawing_from is None:
            return
        x, y = self._area_drawing_from
        self.update_area_selection(event)       # to where it ends
        self._area_drawing_from = None
        slop = self._overlay_px(self.CLICK_SLOP_PX)
        if abs(event.x - x) <= slop or abs(event.y - y) <= slop:
            self.drop_area()
            return
        if any(corner is None for corner in self._area["ground"]):
            print("Topography: every corner of the rectangle has to be on the Moon")
            self.drop_area()
            return
        self._keep_area()
        self.show_topography(self._area["ground"])

    def _keep_area(self):
        """
        Leave the area on the picture after the button is released, as the
        measurement line is left (see NavigationMixin._keep_measure_line):
        carried with its ground whenever the view moves, until Backspace takes
        it off, a new one is drawn, or a corner leaves the side of the Moon the
        eye looks at. Each corner can be dragged by its ring meanwhile.
        """
        self._area_view = self._measure_line_view_now()
        self._area_watch = self._schedule_overlay(self._watch_area)

    def _watch_area(self):
        """One beat of the poll: see to the kept area once the view has moved."""
        self._area_watch = None
        if self._area is None:
            return
        self.check_area()
        if self._area is not None:
            self._area_watch = self._schedule_overlay(self._watch_area)

    def check_area(self):
        """
        Carry the kept area to where its ground now is if the view has changed
        since it was placed, or take it away - and its window - if a corner
        cannot be seen any more. From the poll, and straight after whatever
        moves the view from the window, as check_measure_line is.
        """
        if (self._area is None or self.selecting_area
                or self._measure_line_view_now() == self._area_view):
            return
        on_screen = []
        for place in self._area["ground"]:
            at = self._canvas_xy_of(*place) if self._faces_eye(*place) else None
            if at is None:
                self.drop_area()
                return
            on_screen.append(at)
        self._area["canvas"] = on_screen
        self._place_area()
        self._area_view = self._measure_line_view_now()

    def drop_area(self, close_window: bool = True):
        """
        Take the area off the picture, if there is one, and the topography
        window with it - unless `close_window` is False, as it is when a new
        area is taking its place and will be drawn into the same window.
        """
        if self._area is not None and close_window and self._topography is not None:
            self._topography["close"]()
        self._area_watch = self._cancel_overlay(self._area_watch)
        if self._area is not None and self.rt is not None:
            try:
                self.rt._canvas.delete(self._area["tag"])
            except tk.TclError:         # the main window went first
                pass
        self._area = None
        self._area_drawing_from = None
        self._area_corner = None
        self._area_view = None

    # ---- moving the corners of a kept area ----

    def area_corner_at(self, event) -> Optional[int]:
        """Which corner of the kept area has its ring under the pointer, if any."""
        if self._area is None or self.selecting_area or self._area_view is None:
            return None
        reach = self._overlay_px(self.MEASURE_HOOK_RADIUS + self.MEASURE_HOOK_GRAB)
        for k, (x, y) in enumerate(self._area["canvas"]):
            if math.hypot(event.x - x, event.y - y) <= reach:
                return k
        return None

    def grab_area_corner(self, event) -> bool:
        """
        Take hold of a corner of the kept area if the left button went down on
        its ring, and say whether it did - so the press goes no further, as a
        press on a measurement line's ring does not.
        """
        self._area_corner = self.area_corner_at(event)
        return self._area_corner is not None

    @property
    def dragging_area_corner(self) -> bool:
        return self._area_corner is not None

    def drag_area_corner(self, event):
        """
        Move the corner being held to the pointer, the others staying where
        they are. Off the Moon there is no ground for it, so it waits at the
        last place on it until the pointer comes back.
        """
        if self._area_corner is None:
            return
        place = self._selenographic_at(event)
        if place is None:
            return
        self._area["ground"][self._area_corner] = place
        self._area["canvas"][self._area_corner] = (event.x, event.y)
        self._place_area()

    def release_area_corner(self, event):
        """
        Let go of the corner being held, where it stands, and draw the ground
        of the area as it now is - opening the window again if it was closed.
        """
        if self._area_corner is None:
            return
        self.drag_area_corner(event)
        self._area_corner = None
        self._area_view = self._measure_line_view_now()
        # In the view the window stands in, which may have been turned since
        self.show_topography(self._area["ground"], turn=False)

    # ---- the ground ----

    def _topography_grid(self, corners: list) -> tuple:
        """
        The ground under the rectangle drawn on the picture - and nothing
        beyond it - as a grid stretched between its four corners' ground, as
        fine as the map where TOPO_MAX_POINTS allows: (east, north, heights,
        sides, spacing, ground, frame, fine), the first two the grid's places in
        kilometres east and north of its middle, the heights in metres from the
        Moon's mean radius, sides the lengths of its edges across and down and
        spacing the grid's, in kilometres, ground the places as body-frame unit
        vectors, frame the (middle, east, north) unit vectors they are laid
        flat by - and fine the same patch sampled as finely as the map where
        TOPO_DETAIL_POINTS allows, (east, north, heights) again, its corners
        those of the grid.

        The rectangle's ground is rarely a rectangle: the view is turned from
        north, and seen at a slant it is squeezed, so it is a four-sided patch
        of some other shape, and the grid takes that shape. One larger than
        TOPO_MAX_SIDE_KM is drawn in towards its middle until it is not.
        """
        radius = self.MOON_RADIUS_KM
        # Top left, top right, bottom right, bottom left, as drawn
        units = np.array([self._unit(lat, lon) for lat, lon in corners])
        middle = units.sum(axis=0)
        middle /= np.linalg.norm(middle)

        def arc(a, b):
            return radius * math.acos(max(-1.0, min(1.0, float(a @ b))))

        across = max(arc(units[0], units[1]), arc(units[3], units[2]))
        down = max(arc(units[0], units[3]), arc(units[1], units[2]))
        longest = max(across, down)
        if longest > self.TOPO_MAX_SIDE_KM:
            units = middle + (units - middle) * (self.TOPO_MAX_SIDE_KM / longest)
            units /= np.linalg.norm(units, axis=1, keepdims=True)
            across, down = (across * self.TOPO_MAX_SIDE_KM / longest,
                            down * self.TOPO_MAX_SIDE_KM / longest)

        # Laid flat on the plane touching the middle, east and north along it
        east = np.cross([0.0, 0.0, 1.0], middle)
        if np.linalg.norm(east) < 1e-9:                 # a pole: any way round will do
            east = np.array([1.0, 0.0, 0.0])
        east /= np.linalg.norm(east)
        north = np.cross(middle, east)
        longest = max(across, down)

        def sampled(points: int) -> tuple:
            """
            The patch as a grid of this many places along its longer side:
            (east, north, heights, spacing, ground), as above.
            """
            spacing = longest / (points - 1)
            u = np.linspace(0.0, 1.0, max(2, int(round(across / spacing)) + 1))
            v = np.linspace(0.0, 1.0, max(2, int(round(down / spacing)) + 1))
            gu, gv = np.meshgrid(u, v)
            # Each place a blend of the four corners by how far across and down it is
            ground = (((1 - gu) * (1 - gv))[..., None] * units[0] + (gu * (1 - gv))[..., None] * units[1]
                      + (gu * gv)[..., None] * units[2] + ((1 - gu) * gv)[..., None] * units[3])
            ground /= np.linalg.norm(ground, axis=-1, keepdims=True)
            out = ground @ middle
            x = radius * np.arctan2(ground @ east, out)
            y = radius * np.arctan2(ground @ north, out)
            lat = np.degrees(np.arcsin(np.clip(ground[..., 2], -1.0, 1.0)))
            lon = np.degrees(np.arctan2(ground[..., 0], -ground[..., 1]))
            heights = (self._ground_factor(lat, lon) * self.elevation_radius_scale - 1.0) * radius * 1000.0
            return x, y, np.asarray(heights, dtype=float), spacing, ground

        texel_km = 2 * math.pi * radius / self.elevation.shape[1]
        on_map = round(longest / texel_km) + 1
        count = int(np.clip(on_map, self.TOPO_MIN_POINTS, self.TOPO_MAX_POINTS))
        x, y, heights, spacing, ground = sampled(count)
        fine_x, fine_y, fine_heights, _spacing, _ground = sampled(
            int(np.clip(on_map, count, self.TOPO_DETAIL_POINTS)))
        return (x, y, heights, (across, down), spacing, ground, (middle, east, north),
                (fine_x, fine_y, fine_heights))

    def _topography_names(self, frame: tuple, edge: np.ndarray, longer_km: float) -> list:
        """
        All the features of the program's list whose middle lies inside the
        area and at least TOPO_NAME_MIN_SHARE of its longer side across, the
        largest first: (name, east, north, radius) each, laid flat
        as the grid is (see _topography_grid) - in kilometres east and north
        of its middle - and the feature's radius in kilometres. A satellite
        crater's name is its letter alone (see TOPO_SATELLITE_NAME).
        """
        features = [f for f in getattr(self, "moon_features", ())
                    if f.diameter_km >= longer_km * self.TOPO_NAME_MIN_SHARE]
        if not features:
            return []
        middle, east, north = frame
        radius = self.MOON_RADIUS_KM
        units = np.array([self._unit(f.lat, f.lon) for f in features])
        out = units @ middle
        x = radius * np.arctan2(units @ east, out)
        y = radius * np.arctan2(units @ north, out)
        outline = edge.astype(np.float32).reshape(-1, 1, 2)
        inside = [k for k in range(len(features))
                  if out[k] > 0.0 and cv2.pointPolygonTest(outline, (float(x[k]), float(y[k])), False) >= 0]
        inside.sort(key=lambda k: -features[k].diameter_km)
        named = []
        for k in inside:
            name = features[k].name
            satellite = self.TOPO_SATELLITE_NAME.match(name)
            named.append((satellite.group(1) if satellite else name,
                          float(x[k]), float(y[k]), features[k].diameter_km / 2.0))
        return named

    def _mark_topography_place(self, place: Optional[tuple]):
        """
        Mark the place under the pointer in the topography window on the Moon
        in the main window, as the profile marks its own (see
        ProfileMixin._mark_profile_place) - or take the mark away (None). Only
        while the area is on the picture: a mark left without it would point
        at ground nobody chose.
        """
        if self.rt is None:
            return
        canvas = self.rt._canvas
        canvas.delete(self.TOPO_MARK_TAG)
        if place is None or self._area is None:
            return
        at = self._canvas_xy_of(*place)
        if at is None:
            return
        x, y = at
        r = self._overlay_px(self.PROFILE_MARK_RADIUS_PX)
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=self.PROFILE_MARK_COLOUR,
                           outline="black", width=max(1, round(self._overlay_px(1))),
                           tags=self.TOPO_MARK_TAG)

    @staticmethod
    def _topography_exaggeration(longer_km: float, relief_m: float) -> float:
        """
        The round factor - 1, 2 or 5 times a power of ten, never below 1 - that
        brings the relief nearest to TOPO_RELIEF_SHARE of the longer side
        without passing it.
        """
        if relief_m <= 0.0:
            return 1.0
        wanted = TopographyMixin.TOPO_RELIEF_SHARE * longer_km * 1000.0 / relief_m
        if wanted <= 1.0:
            return 1.0
        power = 10.0 ** math.floor(math.log10(wanted))
        return next(m * power for m in (5, 2, 1) if m * power <= wanted)

    def show_topography(self, corners: list, turn: bool = True):
        """
        Draw the ground the corners enclose in the topography window, opening
        it if need be - and with `turn`, in the view it starts in: turned as
        the rectangle lies on the picture, the side drawn at its top the far
        one, so the block is seen the way up the Moon is. On a picture
        mirrored for a star diagonal the block cannot be mirrored with it,
        being ground as it is: there its top is still the far side, and left
        and right are swapped.
        """
        if self.rt is None or getattr(self, "elevation", None) is None:
            return
        x, y, heights, sides, spacing_km, ground, frame, fine = self._topography_grid(corners)
        # Up the picture, across the ground: from the middle of the grid's last
        # row, the rectangle's bottom side, to the middle of its first, the top
        up_x = float(x[0, :].mean() - x[-1, :].mean())
        up_y = float(y[0, :].mean() - y[-1, :].mean())
        facing = math.degrees(math.atan2(up_x, up_y)) if turn else None
        # The heights' range, and the levels, from the grid: the fine samples
        # reach a little further, to the peaks and floors it passes between,
        # and enough to push the step a size up - from 200 m to 500 m across
        # Copernicus and its neighbours - and thin the lines out. What lies
        # past the range takes the colour of the band at its end
        low, high = float(heights.min()), float(heights.max())
        step = self._nice_step(max(high - low, 1.0), self.TOPO_LEVELS)
        leading = step / 10 ** math.floor(math.log10(step))
        thick_every = 4 if round(leading) == 5 else 5
        # The edge of the patch, round its four sides
        edge = np.concatenate((np.column_stack((x[0, :], y[0, :])),
                               np.column_stack((x[:, -1], y[:, -1])),
                               np.column_stack((x[-1, ::-1], y[-1, ::-1])),
                               np.column_stack((x[::-1, 0], y[::-1, 0]))))
        if self._topography is None:
            self._open_topography_window()
        self._topography["draw"](edge, sides, low, high, step, step * thick_every,
                                 (x, y, heights, ground, spacing_km, fine), facing,
                                 self._topography_names(frame, edge, max(sides)))

    # ---- the window ----

    def _open_topography_window(self):
        memory = self._window_memory(self.TOPO_WINDOW)

        def before_close():
            memory["canvas"] = (width, height)
            self._mark_topography_place(None)
            self._topography = None

        # Not holding the renderer's keys and not modal, as the profile
        win, frame, close = self._dialog_window("Topography", takes_keys=False,
                                                before_close=before_close,
                                                remember=self.TOPO_WINDOW)
        font = ('Consolas', 8)
        metrics = tkfont.Font(font=font)
        cell_w = metrics.measure('0')
        line_h = metrics.metrics('linespace')
        pad = max(2, round(cell_w * 2 / 3))
        line_w = max(1, cell_w // 5)
        corner_r = max(3, round(cell_w * 0.6))  # the marked corner, as the profile's ends

        top_row = tk.Frame(frame)
        top_row.pack(fill=tk.X)
        summary_var = tk.StringVar()
        # Asking for no width of its own, and wrapped to the room the drawing
        # leaves it: at 300% scaling the lettering grows three times and the
        # drawing, a share of the screen, two, and the summary in one line
        # widened the window well past the drawing, which stood in it with
        # empty margins
        summary = tk.Label(top_row, textvariable=summary_var, font=font, anchor='w',
                           justify=tk.LEFT, width=1)
        summary.pack(side=tk.LEFT, fill=tk.X, expand=True)

        wrapped_at = {"width": None}

        def wrap_summary(event):
            # Only on a new width: the wrapping changes the label's height, which
            # brings another Configure of the same width
            if event.width != wrapped_at["width"]:
                wrapped_at["width"] = event.width
                summary.configure(wraplength=max(1, event.width))

        summary.bind("<Configure>", wrap_summary)
        help_hint = (
            "• The ground inside the rectangle drawn with Ctrl and the right\n"
            "   button on the Moon, coloured by height from blue at the lowest to\n"
            "   red at the highest, as the strip at the top left shows, and lit\n"
            "   from the north-west as on a relief map.\n"
            "• The dark lines join places of equal height, every height given above.\n"
            "• With Names ticked, the names of the features inside the area at least\n"
            "   1% of its longer side across, from the program's own list - a\n"
            "   satellite crater's by its letter\n"
            "   alone - each over a dot above its middle, at the height of its\n"
            "   highest ground, a crater's rim, so it is not sunk in its own floor.\n"
            "   A name with no room over its dot is put nearby, joined to it by a line.\n"
            "• The height and the place of the ground under the pointer are written\n"
            "   at the bottom left, and the place is marked on the Moon as well.\n"
            "• Drag with the left button to turn it: left and right spin it round,\n"
            "   up and down tilt it, from nearly side-on to straight down.\n"
            "• The mouse wheel zooms in and out, the block kept in the middle.\n"
            "• It opens turned as the rectangle lies on the picture, its top side\n"
            "   the far one. Double-click to return to that view, the whole area in view.\n"
            "• The grey outline is the edge of the rectangle drawn,\n"
            "   as it lies on the ground, at the area's lowest height; its filled\n"
            "   corner is the one with the filled ring on the picture.\n"
            "• Heights are exaggerated by the factor below: at true scale the\n"
            "   relief of the Moon is slight against its extent, and would look flat.\n"
            "   Each area opens at the factor that brings its relief to about a tenth\n"
            "   of its longer side. Drag the slider to change it, in round steps;\n"
            "   double-click the slider to return to that factor, which the small\n"
            "   mark under the slider shows.\n"
            "• From the elevation map the surface is drawn with, its heights\n"
            "   counted from the Moon's mean radius - the dark mark across the strip,\n"
            "   where the area reaches both above and below it.\n"
            "• Drag the window's edges to make it larger.\n"
            "• Save... writes the drawing as it stands to a PNG or JPEG file,\n"
            "   the area's figures and the heights' factor written under it.")
        help_label = tk.Label(top_row, text="Help", font=font, fg='#606060')
        help_label.pack(side=tk.RIGHT, padx=pad)
        ToolTip(help_label, help_hint)
        names_var = tk.BooleanVar(value=self._topo_names)

        def names_toggled():
            self._topo_names = names_var.get()
            if state["surface"] is not None:
                place_names()

        ttk.Checkbutton(top_row, text="Names", variable=names_var,
                        command=names_toggled).pack(side=tk.RIGHT)

        first_width = round(win.winfo_screenwidth() * self.TOPO_WIDTH_FRACTION)
        first_height = round(first_width * 0.75)
        width, height = memory.get("canvas", (first_width, first_height))
        canvas = tk.Canvas(frame, width=width, height=height, highlightthickness=0,
                           bg=win.cget('bg'))
        canvas.pack()

        factor_row = tk.Frame(frame)
        factor_row.pack(fill=tk.X)
        # At the bottom right; bound to save() once it is defined, further down
        save_button = tk.Button(factor_row, text="Save...", font=font)
        save_button.pack(side=tk.RIGHT)
        tk.Label(factor_row, text="Heights ×", font=font).pack(side=tk.LEFT)
        factor_var = tk.StringVar()
        tk.Label(factor_row, textvariable=factor_var, font=font, width=5, anchor='w').pack(
            side=tk.LEFT)
        track = tk.Frame(factor_row)
        track.pack(side=tk.LEFT, padx=pad)
        # Not taking the keyboard: the renderer's keys go on working over it.
        # Of a length of its own, in the lettering's widths so it scales with it
        slider = ttk.Scale(track, from_=0, to=1, orient=tk.HORIZONTAL, takefocus=0,
                           length=cell_w * 40)
        slider.pack(fill=tk.X)
        # Under it, a mark at the factor chosen for the area
        mark_h = max(4, line_h // 2)
        # Asking for no width either - a canvas asks for 10 cm unless told, a
        # third of the screen at 300% scaling - and stretched to the slider's
        marks = tk.Canvas(track, width=1, height=mark_h, highlightthickness=0, bg=win.cget('bg'))
        marks.pack(fill=tk.X)

        def mark_chosen(_event=None):
            """The mark under the slider's step for the chosen factor."""
            marks.delete("all")
            if state["given"] is None:
                return
            # Where the slider puts the chosen step, across its own width -
            # the mark's canvas spanning the same
            x = float(slider.tk.call(slider._w, "coords", state["chosen"])[0])
            half = max(3, cell_w // 2)
            marks.create_polygon(x, 0, x - half, mark_h, x + half, mark_h,
                                 fill=self.TOPO_BASE_COLOUR, outline="")

        # Once Tk is idle, as the slider lays out its steps only then
        slider.bind("<Configure>", lambda _event: marks.after_idle(mark_chosen), add="+")
        # The colours are painted into an image on the window's own ground
        background = tuple(v >> 8 for v in canvas.winfo_rgb(canvas.cget('bg')))
        base_rgb = tuple(v >> 8 for v in canvas.winfo_rgb(self.TOPO_BASE_COLOUR))

        # The ground, in kilometres east, north and (exaggerated) up from the
        # middle of the block, and the image it is painted into; the view;
        # the area as given, the slider's factors, which of them is set and
        # which was chosen for the area
        # Where the grid stands on the canvas as last painted, and its cells'
        # depths, for finding the ground under the pointer; and the pointer
        # How far the block reaches across and up the canvas in the view it
        # opens in, unscaled (see to_canvas), and where the room for it starts,
        # past the strip
        state = {"extent": (-1.0, 1.0, -1.0, 1.0), "room_left": 0, "pending": False,
                 "projected": None, "pointer": None, "middle": (0.0, 0.0),
                 "surface": None, "image": None, "photo": None, "picture": None,
                 "given": None, "factors": [1.0], "factor": 0, "chosen": 0,
                 "start": (0.0, self.TOPO_START_ELEVATION),
                 "azimuth": 0.0, "elevation": self.TOPO_START_ELEVATION,
                 "drag": None, "zoom": 1.0}

        def to_canvas(x, y, z) -> tuple:
            """
            Points of the block on the canvas, for the view as it stands, and
            how far each lies from the eye along the view, larger further.

            At the scale that fills the room for it - right of the strip, above
            the readout's line - with the block as it opens (see state's
            "extent"), whatever the canvas's size; the middle of the block put
            where that leaves it, and kept there as it is turned and zoomed.
            """
            a, e = math.radians(state["azimuth"]), math.radians(state["elevation"])
            across = x * math.cos(a) - y * math.sin(a)
            away = x * math.sin(a) + y * math.cos(a)
            # Inset by the marked corner's dot, which stands out past the outline
            left, right = state["room_left"] + corner_r, width - pad - corner_r
            top, bottom = pad + corner_r, height - pad - line_h - corner_r
            low_a, high_a, low_v, high_v = state["extent"]
            fit = max(1e-9, min((right - left) / max(high_a - low_a, 1e-9),
                                (bottom - top) / max(high_v - low_v, 1e-9)))
            middle_x = (left + right) / 2.0 - fit * (low_a + high_a) / 2.0
            middle_y = (top + bottom) / 2.0 + fit * (low_v + high_v) / 2.0
            scale = fit * state["zoom"]
            sx = middle_x + scale * across
            sy = middle_y - scale * (away * math.sin(e) + z * math.cos(e))
            return sx, sy, away * math.cos(e) - z * math.sin(e)

        def paint():
            """
            The ground, pixel by pixel. First which cell of the grid is seen
            at each pixel: the cells painted with their numbers, the furthest
            from the eye first, so nearer ground covers what lies behind it.
            Then where the pixel falls within its cell, and so on the fine
            samples, which are laid over it (see shape): the map's own detail,
            not the grid's. The thicker lines where the fine heights pass one
            of their levels between two pixels side by side on ground that
            runs on from one to the other - not where a ridge stands against
            ground far behind it.

            Painted up to TOPO_SUPERSAMPLE times larger (see
            TOPO_PAINT_PIXELS) and then reduced, which smooths the edges.
            """
            ground, given = state["surface"], state["given"]
            times = int(max(1, min(self.TOPO_SUPERSAMPLE,
                                   math.sqrt(self.TOPO_PAINT_PIXELS / (width * height)))))

            def on_image(x, y, z):
                sx, sy, depth = to_canvas(x, y, z)
                return sx * times, sy * times, depth

            sx, sy, depth = on_image(ground["x"], ground["y"], ground["z"])
            depth = (depth[:-1, :-1] + depth[:-1, 1:] + depth[1:, 1:] + depth[1:, :-1]) / 4.0
            state["projected"] = (sx / times, sy / times, depth)
            order = np.argsort(-depth, axis=None)
            r, c = np.unravel_index(order, depth.shape)
            cells = np.column_stack((sx[r, c], sy[r, c], sx[r, c + 1], sy[r, c + 1],
                                     sx[r + 1, c + 1], sy[r + 1, c + 1],
                                     sx[r + 1, c], sy[r + 1, c])).tolist()
            size = (width * times, height * times)
            seen = Image.new("I", size, 0)
            pen = ImageDraw.Draw(seen)
            for cell, number in zip(cells, (order + 1).tolist()):
                pen.polygon(cell, fill=number, outline=number)
            seen = np.asarray(seen)

            image = Image.new("RGB", size, background)
            pen = ImageDraw.Draw(image)
            # The edge at the lowest height first, for the ground to cover
            base_x, base_y, _ = on_image(*ground["base"].T)
            pen.line(np.column_stack((base_x, base_y)).ravel().tolist(), fill=base_rgb,
                     width=line_w * times)
            (mark_x, mark_y), = np.column_stack(on_image(*ground["corner"].T)[:2])
            r = corner_r * times
            pen.ellipse((mark_x - r, mark_y - r, mark_x + r, mark_y + r), fill=base_rgb)
            pixels = np.array(image)

            iy, ix = np.nonzero(seen)
            if iy.size:
                row, col = np.divmod(seen[iy, ix] - 1, depth.shape[1])
                px, py = ix.astype(np.float32), iy.astype(np.float32)
                sx, sy = sx.astype(np.float32), sy.astype(np.float32)
                # Each cell two triangles across its diagonal from the top
                # left corner: the pixel's, its third corner the top right or
                # the bottom left by the side of the diagonal it is on
                ax, ay = sx[row, col], sy[row, col]
                cx, cy = sx[row + 1, col + 1] - ax, sy[row + 1, col + 1] - ay
                qx, qy = px - ax, py - ay
                side = cx * qy - cy * qx
                upper = (side * (cx * (sy[row, col + 1] - ay) - cy * (sx[row, col + 1] - ax))) >= 0
                third_row = np.where(upper, row, row + 1)
                third_col = np.where(upper, col + 1, col)
                ex, ey = sx[third_row, third_col] - ax, sy[third_row, third_col] - ay
                det = cx * ey - ex * cy
                det = np.where(np.abs(det) > 1e-6, det, 1e-6)
                # Kept inside the triangle: the pixels along a cell's edge,
                # painted with it, can stand a little outside
                wc = np.clip((qx * ey - ex * qy) / det, 0.0, 1.0)
                we = np.clip((cx * qy - qx * cy) / det, 0.0, 1.0)
                total = np.maximum(wc + we, 1.0)
                wc, we = wc / total, we / total
                # Where that is on the fine samples, counted in their steps
                fine_rows, fine_cols = given["fine_heights"].shape
                along = (fine_cols - 1) / depth.shape[1]
                down = (fine_rows - 1) / depth.shape[0]
                on_fine_x = np.full(seen.shape, -1.0, dtype=np.float32)
                on_fine_y = np.full(seen.shape, -1.0, dtype=np.float32)
                on_fine_x[iy, ix] = (col + wc + we * (third_col - col)) * along
                on_fine_y[iy, ix] = (row + wc + we * (third_row - row)) * down
                on = seen > 0
                laid = cv2.remap(ground["texture"], on_fine_x, on_fine_y, cv2.INTER_LINEAR)
                pixels[on] = laid[on]

                # The lines: a pixel whose level differs from its neighbour's
                # to the right or below, both on ground that runs on between
                # them - a step or so apart on the grid
                metres = cv2.remap(given["fine_heights"], on_fine_x, on_fine_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                level = np.floor(metres / given["thick"]).astype(np.int32)
                lines = np.zeros(seen.shape, dtype=np.uint8)
                for here, there in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                                    ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
                    lines[here] |= (on[here] & on[there] & (level[here] != level[there])
                                    & (np.abs(on_fine_x[here] - on_fine_x[there]) <= 1.5 * along)
                                    & (np.abs(on_fine_y[here] - on_fine_y[there]) <= 1.5 * down))
                thick = line_w * times
                if thick > 1:
                    lines = cv2.dilate(lines, np.ones((thick, thick), np.uint8))
                pixels[(lines > 0) & on] = self.TOPO_INDEX_COLOUR
            image = Image.fromarray(pixels)
            if times > 1:
                image = image.resize((width, height), getattr(Image, "Resampling", Image).LANCZOS)
            state["picture"] = image                # for a file (see picture)
            state["photo"] = ImageTk.PhotoImage(image, master=canvas)
            canvas.itemconfigure(state["image"], image=state["photo"])

        def place():
            """Everything where the view now puts it."""
            state["pending"] = False
            if not canvas.winfo_exists() or state["surface"] is None:
                return
            paint()
            place_names()
            # The ground under a pointer standing still has moved under it
            refresh_readout()

        def on_ground(text):
            """
            A box of the window's own ground under a text, readable over the
            block wherever it is turned or zoomed to; tagged as the text is,
            to go with it.
            """
            x0, y0, x1, y1 = canvas.bbox(text)
            box = canvas.create_rectangle(x0 - pad // 2, y0, x1 + pad // 2, y1, fill=canvas.cget('bg'),
                                          outline="", tags=canvas.gettags(text))
            canvas.tag_lower(box, text)

        def place_names():
            """
            The names where the view puts their places, each above its
            feature's middle at the height of its highest ground (see draw) -
            a crater's over its rim, not down on its floor, where the rim
            would hide it from all but straight above. Every one whose place
            is on the canvas, whatever stands in front of it.

            Each name over its dot where there is room; where its box would
            cover another's, or a dot, the strip or the readout's line, the
            nearest free place round the dot instead, further out ring by
            ring, joined to its dot by a line. The larger features first, so
            it is a smaller one that is moved.
            """
            canvas.delete("names")
            given, surface = state["given"], state["surface"]
            if not self._topo_names or not given["names"]:
                return
            points = surface["names"]
            nx, ny, _ = to_canvas(points[:, 0], points[:, 1], points[:, 2])
            dot = 2 * line_w
            gap = dot + pad // 2
            shown = [(name, x, y) for (name, *_rest), x, y in zip(given["names"], nx.tolist(), ny.tolist())
                     if 0 <= x < width and 0 <= y < height]
            taken = [(x - dot, y - dot, x + dot, y + dot) for _name, x, y in shown]
            taken += [(0, 0, state["room_left"], height), (0, height - pad - line_h, width, height)]
            # The ways out from the dot, straight up first and then those
            # nearest it, and how far out, ring by ring
            turns = np.radians(np.arange(16) * 22.5)
            turns = turns[np.argsort(np.abs(np.angle(np.exp(1j * (turns + math.pi / 2)))), kind='stable')]
            cos, sin = np.cos(turns)[None, :], np.sin(turns)[None, :]
            rings = np.arange(12)[:, None]
            placed = []
            for name, x, y in shown:
                w, h = metrics.measure(name) + pad, line_h
                cx = (x + cos * (w / 2 + gap + rings * h)).ravel()
                cy = (y + sin * (h / 2 + gap + rings * h)).ravel()
                boxes = np.column_stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                t = np.array(taken, dtype=float)
                apart = ((boxes[:, None, 2] <= t[None, :, 0]) | (t[None, :, 2] <= boxes[:, None, 0])
                         | (boxes[:, None, 3] <= t[None, :, 1]) | (t[None, :, 3] <= boxes[:, None, 1])).all(axis=1)
                on_canvas = ((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
                             & (boxes[:, 2] <= width) & (boxes[:, 3] <= height))
                free = np.flatnonzero(apart & on_canvas)
                # Over its dot after all where nothing round it is free
                k = int(free[0]) if free.size else 0
                box = tuple(boxes[k].tolist())
                taken.append(box)
                placed.append((name, x, y, box, k != 0))
            # The lines under the boxes, the dots over everything
            for _name, x, y, (x0, y0, x1, y1), moved in placed:
                if moved:
                    canvas.create_line(x, y, min(max(x, x0), x1), min(max(y, y0), y1),
                                       fill=self.TOPO_NAME_COLOUR, width=line_w, tags="names")
            ground = canvas.cget('bg')
            for name, _x, _y, (x0, y0, x1, y1), _moved in placed:
                canvas.create_rectangle(x0, y0, x1, y1, fill=ground, outline="", tags="names")
                canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=name, font=font,
                                   fill=self.TOPO_NAME_COLOUR, tags="names")
            for _name, x, y, *_rest in placed:
                canvas.create_oval(x - dot, y - dot, x + dot, y + dot, fill=self.TOPO_NAME_COLOUR,
                                   outline="", tags="names")

        def place_soon():
            # Once Tk is idle: a drag moves faster than the block is drawn
            if not state["pending"]:
                state["pending"] = True
                canvas.after_idle(place)

        def ramp(share: float) -> tuple:
            """The colour a share of the way from the lowest band to the highest."""
            stops = self.TOPO_RAMP
            for (s0, c0), (s1, c1) in zip(stops, stops[1:]):
                if share <= s1:
                    f = (share - s0) / (s1 - s0)
                    return tuple(round(a + f * (b - a)) for a, b in zip(c0, c1))
            return stops[-1][1]

        def shape(exaggeration: float):
            """
            The block at an exaggeration: the heights of its ground and of
            its edge - and what is laid over it (see paint), the colours of
            the fine samples, each lit by the slope there, which steepens
            with them.
            """
            given = state["given"]

            def up(metres):
                return (metres - given["mid_z"]) / 1000.0 * exaggeration

            x, y, heights = given["x"], given["y"], given["heights"]
            z = up(heights)
            # Each fine sample lit by how squarely the ground there faces the
            # light: its slope along the rows and down the columns, as the
            # block is drawn, exaggerated
            fine = (given["fine_x"], given["fine_y"], up(given["fine_heights"]).astype(np.float32))
            along = np.stack([np.gradient(v, axis=1) for v in fine], -1)
            down = np.stack([np.gradient(v, axis=0) for v in fine], -1)
            facing = np.cross(along, down)
            facing *= np.sign(facing[..., 2:3])             # facing up, whichever way round
            facing /= np.linalg.norm(facing, axis=-1, keepdims=True)
            around, high_up = (math.radians(v) for v in self.TOPO_LIGHT)
            light = np.array([math.cos(high_up) * math.sin(around),
                              math.cos(high_up) * math.cos(around), math.sin(high_up)])
            # Level ground keeps its colour as it is; slopes towards the light are brighter
            lit = np.clip(facing @ light, 0.0, 1.0) / light[2]
            shade = self.TOPO_SHADE + (1.0 - self.TOPO_SHADE) * lit
            texture = np.clip(given["fine_colours"] * shade[..., None], 0, 255)
            flat = given["flat"]
            state["surface"] = {
                "x": x, "y": y, "z": z, "texture": texture.round().astype(np.uint8),
                "base": np.column_stack((flat, np.full(len(flat), up(given["low"])))),
                # The grid's corners run round as the area's do: top left, top
                # right, bottom right, bottom left
                "corner": np.array([[*(v[(0, 0, -1, -1)[self.TOPO_MARKED_CORNER],
                                         (0, -1, -1, 0)[self.TOPO_MARKED_CORNER]] for v in (x, y)),
                                     up(given["low"])]]),
                "names": np.array([(x, y, up(h)) for _name, x, y, h in given["names"]]
                                  ).reshape(-1, 3)}
            factor_var.set(f"{exaggeration:g}")

        def slid(value):
            """The slider moved: kept to the factors' steps, and the block reshaped."""
            at = round(float(value))
            if float(value) != at:
                slider.set(at)              # comes back here, on the step
                return
            if at != state["factor"] and state["given"] is not None:
                state["factor"] = at
                shape(state["factors"][at])
                place_soon()

        def back_to_chosen(_event):
            slider.set(state["chosen"])
            return "break"

        slider.configure(command=slid)
        slider.bind("<Double-Button-1>", back_to_chosen)

        def draw(edge, sides, low, high, step, thick_step, grid, facing=None, names=()):
            if facing is not None:
                # A new area: seen as it lies on the picture, whole
                state["start"] = (facing, self.TOPO_START_ELEVATION)
                state["azimuth"], state["elevation"] = state["start"]
                state["zoom"] = 1.0
            canvas.delete("all")
            # Turned about the middle of the patch, as it lies on the ground
            mid_x = (edge[:, 0].min() + edge[:, 0].max()) / 2.0
            mid_y = (edge[:, 1].min() + edge[:, 1].max()) / 2.0
            mid_z = (low + high) / 2.0
            flat = edge - [mid_x, mid_y]
            # The factors, round steps from 1 past the one chosen for the area
            chosen = self._topography_exaggeration(max(sides), high - low)
            top_factor = chosen * self.TOPO_EXAGGERATION_RANGE
            factors = [m * 10.0 ** k for k in range(int(math.log10(top_factor)) + 1)
                       for m in (1, 2, 5) if m * 10.0 ** k <= top_factor]
            x, y, heights, ground, spacing_km, (fine_x, fine_y, fine_heights) = grid
            # The block's size on the canvas: filling the room for it as it is
            # first seen, at the factor chosen - and kept whatever the slider
            # is set to, other factors standing the hills taller or lower
            # rather than shrinking or growing the whole
            a, e = (math.radians(v) for v in state["start"])
            ground_x = np.concatenate(((x - mid_x).ravel(), flat[:, 0]))
            ground_y = np.concatenate(((y - mid_y).ravel(), flat[:, 1]))
            ground_z = np.concatenate(((heights.ravel() - mid_z) / 1000.0 * chosen,
                                       np.full(len(flat), (low - mid_z) / 1000.0 * chosen)))
            across = ground_x * math.cos(a) - ground_y * math.sin(a)
            upward = (ground_x * math.sin(a) + ground_y * math.cos(a)) * math.sin(e) + ground_z * math.cos(e)
            state["extent"] = (float(across.min()), float(across.max()),
                               float(upward.min()), float(upward.max()))

            # The colours: a band between every two neighbouring levels, each
            # fine sample in the band of its height
            first = math.floor(low / step)
            bands = math.floor(high / step) - first + 1
            palette = [ramp(b / max(1, bands - 1)) for b in range(bands)]
            band = np.clip(np.floor(fine_heights / step) - first, 0, bands - 1).astype(int)
            # Each name at the height of the highest ground within its
            # feature - a crater's rim, a mountain's top - or, for one smaller
            # than the grid's steps, of the ground nearest its middle
            raised = []
            for name, nx, ny, radius_km in names:
                apart = np.hypot(x - nx, y - ny)
                within = apart <= max(radius_km, spacing_km)
                top = heights[within].max() if within.any() else heights.flat[apart.argmin()]
                raised.append((name, nx - mid_x, ny - mid_y, float(top)))
            state["given"] = {
                "x": x - mid_x, "y": y - mid_y, "heights": heights, "low": low, "mid_z": mid_z,
                "flat": flat, "ground": ground, "spacing": spacing_km, "names": raised,
                "fine_x": (fine_x - mid_x).astype(np.float32), "fine_y": (fine_y - mid_y).astype(np.float32),
                "fine_heights": fine_heights.astype(np.float32),
                "fine_colours": np.array(palette, dtype=np.float32)[band], "thick": thick_step}
            # Where the area is, to name a file after (see save)
            state["middle"] = lat_lon(ground[ground.shape[0] // 2, ground.shape[1] // 2])
            state["image"] = canvas.create_image(0, 0, anchor='nw')
            # The strip, at the top left: the lowest band at its foot, and room
            # above it for the half of the highest height's figure over its top
            strip_w, band_h = round(cell_w * 1.5), max(1, round(line_h * 10 / bands))
            top = pad + line_h // 2
            foot = top + bands * band_h
            for b, rgb in enumerate(palette):
                canvas.create_rectangle(pad, foot - (b + 1) * band_h, pad + strip_w, foot - b * band_h,
                                        fill='#%02x%02x%02x' % rgb, width=0)
            # Its ends are the outer edges of its outer bands, not quite the
            # lowest and highest ground, which lie within them
            figures = [(top, (first + bands) * step), (foot, first * step)]
            if first < 0 < first + bands:
                # Zero, where it falls inside the strip, always between two
                # bands: marked across it, and given a figure where that
                # leaves room for the end ones
                zero = foot + first * band_h
                canvas.create_line(pad - line_w, zero, pad + strip_w + pad // 2, zero,
                                   fill='#%02x%02x%02x' % self.TOPO_INDEX_COLOUR, width=line_w)
                if top + line_h <= zero <= foot - line_h:
                    figures.append((zero, 0))
            for at, metres in figures:
                text = canvas.create_text(pad + strip_w + pad, at, text=f"{metres:g} m", font=font,
                                          anchor='w', fill=self.TOPO_BASE_COLOUR)
                on_ground(text)
            # The block's room starts past the strip and its figures
            state["room_left"] = (pad + strip_w + pad + 2 * pad
                                  + max(metrics.measure(f"{metres:g} m") for _, metres in figures))
            summary_var.set(
                f"{sides[0]:.0f} × {sides[1]:.0f} km, a colour every {step:g} m, "
                f"lines every {thick_step:g} m, relief {round(high - low)} m "
                f"({round(low)} to {round(high)} m)")

            # The slider at the chosen factor
            state["factors"] = factors
            state["chosen"] = state["factor"] = factors.index(chosen)
            slider.configure(to=len(factors) - 1)
            slider.set(state["chosen"])
            # Once the slider has taken its new range
            marks.after_idle(mark_chosen)
            shape(chosen)
            place()

        # ---- the ground under the pointer ----

        def nearest_cell(px: float, py: float) -> Optional[tuple]:
            """
            The cell painted at a point of the canvas, or None off the block:
            (depth, row, col, weights) - of the cells the point falls in, each
            taken as two triangles, the nearest of them to the eye, the one
            painted over the others, with its depth as paint() took it and the
            point's place within it, as the triangle's corners with their
            weights.
            """
            if state["projected"] is None:
                return None
            sx, sy, depth = state["projected"]
            corners = [(sx[:-1, :-1], sy[:-1, :-1]), (sx[:-1, 1:], sy[:-1, 1:]),
                       (sx[1:, 1:], sy[1:, 1:]), (sx[1:, :-1], sy[1:, :-1])]
            xs = np.stack([c[0] for c in corners])
            ys = np.stack([c[1] for c in corners])
            near = ((xs.min(axis=0) <= px) & (px <= xs.max(axis=0))
                    & (ys.min(axis=0) <= py) & (py <= ys.max(axis=0)))
            rows, cols = np.nonzero(near)
            if rows.size == 0:
                return None
            # Top left, top right, bottom right; then top left, bottom right,
            # bottom left - the corners as grid steps from the cell's top left
            found = []
            for a, b, c in (((0, 0), (0, 1), (1, 1)), ((0, 0), (1, 1), (1, 0))):
                (ax, ay), (bx, by), (cx, cy) = ((sx[rows + r, cols + k], sy[rows + r, cols + k])
                                                for r, k in (a, b, c))
                across = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
                safe = np.where(np.abs(across) > 1e-12, across, 1.0)
                wb = ((px - ax) * (cy - ay) - (cx - ax) * (py - ay)) / safe
                wc = ((bx - ax) * (py - ay) - (px - ax) * (by - ay)) / safe
                wa = 1.0 - wb - wc
                inside = (np.abs(across) > 1e-12) & (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
                for k in np.flatnonzero(inside):
                    found.append((float(depth[rows[k], cols[k]]), rows[k], cols[k],
                                  ((a, wa[k]), (b, wb[k]), (c, wc[k]))))
            return min(found, key=lambda f: f[0]) if found else None

        def lat_lon(unit) -> tuple:
            """The latitude and longitude, in degrees, of a direction from the Moon's centre."""
            unit = unit / np.linalg.norm(unit)
            lat = math.degrees(math.asin(max(-1.0, min(1.0, float(unit[2])))))
            lon = math.degrees(math.atan2(float(unit[0]), float(-unit[1])))
            return lat, lon

        def ground_at(px: float, py: float) -> Optional[tuple]:
            """
            The ground under a point of the canvas: (height in metres, lat,
            lon), or None off the block - where the cell painted there (see
            nearest_cell) puts it.
            """
            cell = nearest_cell(px, py)
            if cell is None:
                return None
            _depth, row, col, weights = cell
            given = state["given"]
            metres = sum(w * given["heights"][row + r, col + k] for (r, k), w in weights)
            unit = sum(w * given["ground"][row + r, col + k] for (r, k), w in weights)
            return (float(metres), *lat_lon(unit))

        def refresh_readout():
            """
            The readout for where the pointer is, at the bottom left of the
            drawing, written as the profile writes its own - and its mark on
            the Moon. Nothing while the block is being turned.
            """
            canvas.delete("readout")
            pointer = state["pointer"]
            found = None if pointer is None or state["drag"] is not None else ground_at(*pointer)
            if found is None:
                self._mark_topography_place(None)
                return
            metres, lat, lon = found
            self._mark_topography_place((lat, lon))
            text = canvas.create_text(pad, height - pad, anchor='sw', font=font,
                                      fill=self.PROFILE_COLOURS["readout"], tags="readout",
                                      text=(f"{metres:.0f} m  {abs(lat):.2f}°{'N' if lat >= 0 else 'S'} "
                                            f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"))
            on_ground(text)

        def hover(event):
            state["pointer"] = (event.x, event.y)
            refresh_readout()

        def pointer_left(_event):
            state["pointer"] = None
            refresh_readout()

        canvas.bind("<Motion>", hover)
        canvas.bind("<Leave>", pointer_left)

        # ---- turning it ----

        def press(event):
            state["drag"] = (event.x, event.y, state["azimuth"], state["elevation"])
            refresh_readout()

        def drag(event):
            if state["drag"] is None:
                return
            x, y, azimuth, elevation = state["drag"]
            per_x, per_y = self.TOPO_DEGREES_PER_PX
            low_e, high_e = self.TOPO_ELEVATION_LIMITS
            state["azimuth"] = (azimuth + (event.x - x) * per_x) % 360.0
            state["elevation"] = min(high_e, max(low_e, elevation + (event.y - y) * per_y))
            place_soon()

        def release(event):
            state["drag"] = None
            state["pointer"] = (event.x, event.y)
            refresh_readout()

        def reset(_event):
            state["azimuth"], state["elevation"] = state["start"]
            state["zoom"] = 1.0
            place_soon()

        def zoom(event):
            """
            The wheel: larger or smaller around the middle of the block, which
            stays where it is however it is zoomed and turned - down to half
            the size it opens at, so a block turned past the edges can be seen
            whole again. Bound on the window and
            taken only with the pointer over the drawing, as the graph's wheel
            is: Tk on Windows can hand the wheel to whichever widget has the
            keyboard.
            """
            px = event.x_root - canvas.winfo_rootx()
            py = event.y_root - canvas.winfo_rooty()
            if not (0 <= px < canvas.winfo_width() and 0 <= py < canvas.winfo_height()):
                return None
            new = min(self.TOPO_ZOOM_MAX, max(self.TOPO_ZOOM_MIN,
                                              state["zoom"] * self.TOPO_ZOOM_STEP ** (event.delta / 120)))
            if new != state["zoom"]:
                state["zoom"] = new
                place_soon()
            return "break"

        # ---- saving it ----

        def picture() -> Image.Image:
            """
            The drawing as it stands, for a file: the block as painted and what
            the canvas draws over it - the strip, its figures, the names - in
            the window's lettering; and under it what the window writes round
            it, the area's figures and the heights' factor, which the drawing
            means little without. Not the readout, which is the pointer's.
            """
            size = metrics.actual("size")
            letter_px = -size if size < 0 else size * float(canvas.tk.call("tk", "scaling"))
            try:
                face = ImageFont.truetype("consola.ttf", max(1, round(letter_px)))
            except OSError:
                face = ImageFont.load_default()

            def rgb(colour) -> tuple:
                return tuple(v >> 8 for v in canvas.winfo_rgb(colour))

            drawing = state["picture"].copy()
            pen = ImageDraw.Draw(drawing)
            # In the canvas's own order, the image of the block the lowest
            for item in canvas.find_all():
                kind, xy, fill = canvas.type(item), canvas.coords(item), ""
                if kind != "image" and "readout" not in canvas.gettags(item):
                    fill = canvas.itemcget(item, "fill")
                if not fill:
                    continue
                # Tk fills up to the far corner, PIL through it
                box = (xy[0], xy[1], max(xy[0], xy[2] - 1), max(xy[1], xy[3] - 1)) if len(xy) == 4 else None
                if kind == "rectangle":
                    pen.rectangle(box, fill=rgb(fill))
                elif kind == "oval":
                    pen.ellipse(box, fill=rgb(fill))
                elif kind == "line":
                    pen.line(xy, fill=rgb(fill), width=max(1, round(float(canvas.itemcget(item, "width")))))
                elif kind == "text":
                    x0, y0, _x1, _y1 = canvas.bbox(item)
                    pen.text((x0, y0), canvas.itemcget(item, "text"), font=face, fill=rgb(fill), anchor="la")

            # The caption, wrapped to the drawing's width
            caption = []
            for line in (summary_var.get(), f"Heights ×{factor_var.get()}"):
                current = ""
                for word in line.split(" "):
                    longer = f"{current} {word}".strip()
                    if current and pen.textlength(longer, font=face) > drawing.width - 2 * pad:
                        caption.append(current)
                        current = word
                    else:
                        current = longer
                caption.append(current)
            whole = Image.new("RGB", (drawing.width, drawing.height + len(caption) * line_h + pad),
                              background)
            whole.paste(drawing, (0, 0))
            pen = ImageDraw.Draw(whole)
            for k, line in enumerate(caption):
                pen.text((pad, drawing.height + k * line_h), line, font=face,
                         fill=rgb(summary.cget("fg")), anchor="la")
            return whole

        def save():
            if state["picture"] is None:
                return
            lat, lon = state["middle"]
            path = filedialog.asksaveasfilename(
                parent=win, initialdir=".", title="Save the topography as image",
                initialfile=(f"topography_{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"
                             f"_{abs(lon):.2f}{'E' if lon >= 0 else 'W'}.png"),
                defaultextension=".png",
                filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg")))
            if not path:
                return
            jpeg = path.lower().endswith((".jpg", ".jpeg"))
            try:
                picture().save(path, **({"quality": 95} if jpeg else {}))
            except (OSError, ValueError) as e:
                print(f"Not saved: {path} ({e})")
                messagebox.showerror("Image not saved",
                                     f"The image could not be saved to\n{path}\n\n{e}", parent=win)
                return
            print(f"Saved: {path}")

        save_button.configure(command=save)

        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", release)
        canvas.bind("<Double-ButtonPress-1>", reset)
        win.bind("<MouseWheel>", zoom)

        # Larger or smaller by either edge, never smaller than it first opens
        def fit(spare_w: int, spare_h: int):
            nonlocal width, height
            width = max(first_width, width + spare_w)
            height = max(first_height, height + spare_h)
            return width, height

        self._follow_window_size(frame, canvas, fit, place_soon)

        self._topography = {"win": win, "draw": draw, "close": close}
        self._show_tool_window(win, self.TOPO_WINDOW, resizable=(True, True),
                               shrink=(width - first_width, height - first_height),
                               focus=False)
