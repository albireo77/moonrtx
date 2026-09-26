"""
ProfileMixin: the height of the ground along a measurement, drawn as a profile.

The Ctrl-drag measurement gives two numbers - how far, and how much higher the
end stands than the start - which says nothing of what lies between: whether a
scarp rises in one step or a slope climbs the whole way, whether the line
crosses a rim and drops into a crater on the way. The profile draws the ground
along the whole line, from the same elevation map the renderer displaces the
surface with, so it shows the terrain the picture is made of.

Drawn for a measurement made with the right Ctrl; the left one measures as it
always has. One window, kept open across measurements: each one made with the
right Ctrl redraws it rather than opening another. It does not take the renderer's keys, so the
clock and the view stay in reach while it is up.
"""

import math
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Optional

import numpy as np


class ProfileMixin:
    """Mixin drawing the elevation profile along a measurement."""

    PROFILE_COLOURS = {
        "ground": "#7a4e22",      # the profile line
        "fill": "#eadfcc",        # the ground under it
        "grid": "#d0d0d0",
        "readout": "#a06010",     # the place under the pointer, as in the graph
    }
    # Wide enough to follow a long line's detail, and as a share of the screen so
    # a 4K display gets the room it has rather than a FullHD-sized window
    PROFILE_WIDTH_FRACTION = 0.5
    PROFILE_PLOT_LINES = 14                 # plot height, in lines of lettering
    # The map is sampled at twice its own spacing: enough that the bilinear
    # lookup's straight runs between texels draw as the map has them, more
    # adding nothing the map does not hold
    PROFILE_SAMPLES_PER_TEXEL = 2
    PROFILE_MIN_SAMPLES = 64
    PROFILE_MAX_SAMPLES = 2000
    # Whether the pointer's readout gives the place as well as the distance and
    # height. Off to begin with, the readout being busy enough without it;
    # changing it stores it on the instance, so it holds for the rest of the
    # session as the planner's own choices do
    _profile_show_place = False

    def _init_profile(self):
        """Reset the profile window state; called from MoonRenderer.__init__."""
        self._profile = None                # the open window and its redraw, or None
        self._profile_position = None       # where it was left, for the next one
        self._profile_line = None           # the measurement's line, kept while shown

    # ---- the numbers ----

    @staticmethod
    def _unit(lat: float, lon: float) -> np.ndarray:
        """
        The body-frame unit vector of a selenographic position - the app's own
        frame (see center_on_lat_lon): +Z north, longitude 0 towards -Y, +X east.
        """
        la, lo = math.radians(lat), math.radians(lon)
        return np.array([math.cos(la) * math.sin(lo),
                         -math.cos(la) * math.cos(lo),
                         math.sin(la)])

    def _along(self, start: tuple, end: tuple, fractions) -> tuple:
        """
        Where points that far along the great circle from start to end lie, as
        (lat, lon) arrays in degrees: 0 is start and 1 is end.

        By spherical interpolation, which spaces them evenly over the ground as
        a straight chord would not. The two ends must not be the same place.
        """
        a, b = self._unit(*start), self._unit(*end)
        angle = math.acos(max(-1.0, min(1.0, float(a @ b))))
        t = np.atleast_1d(np.asarray(fractions, dtype=float))
        points = (np.sin((1 - t) * angle)[:, None] * a
                  + np.sin(t * angle)[:, None] * b) / math.sin(angle)
        return (np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0))),
                np.degrees(np.arctan2(points[:, 0], -points[:, 1])))

    def elevation_profile(self, start: tuple, end: tuple) -> Optional[tuple]:
        """
        Heights along the great circle from start to end.

        Parameters
        ----------
        start, end : tuple
            (lat, lon) in selenographic degrees

        Returns
        -------
        tuple or None
            (distance_km, height_m) arrays, the distances measured from start
            along the surface; None when the two points are the same place
        """
        a, b = self._unit(*start), self._unit(*end)
        if math.acos(max(-1.0, min(1.0, float(a @ b)))) < 1e-9:
            return None

        length_km = self.calculate_great_circle_distance(*start, *end)
        texel_km = 2 * math.pi * self.MOON_RADIUS_KM / self.elevation.shape[1]
        count = int(np.clip(math.ceil(length_km / texel_km * self.PROFILE_SAMPLES_PER_TEXEL),
                            self.PROFILE_MIN_SAMPLES, self.PROFILE_MAX_SAMPLES))

        t = np.linspace(0.0, 1.0, count)
        lats, lons = self._along(start, end, t)
        heights = np.array([self.get_elevation_m(la, lo) for la, lo in zip(lats, lons)])
        return t * length_km, heights

    @staticmethod
    def _nice_step(span: float, lines: int) -> float:
        """A round step - 1, 2 or 5 times a power of ten - giving about that many lines."""
        if span <= 0:
            return 1.0
        raw = span / max(1, lines)
        power = 10 ** math.floor(math.log10(raw))
        for multiple in (1, 2, 5, 10):
            if raw <= multiple * power:
                return multiple * power
        return 10 * power

    # ---- the window ----

    def show_profile(self, start: tuple, end: tuple) -> bool:
        """
        Draw the profile of the line from start to end, in the profile window -
        opening it if it is not already open - and say whether it was drawn.
        """
        if self.rt is None or getattr(self, "elevation", None) is None:
            return False
        profile = self.elevation_profile(start, end)
        if profile is None:
            return False
        if self._profile is None:
            self._open_profile_window()
        self._profile["draw"](*profile, start, end)
        return True

    def _keep_profile_line(self, line_id, end_x: float, end_y: float):
        """
        Leave a measurement's line on the picture for as long as its profile is
        shown, so the window says what it is the ground of. The line of the
        profile it replaces goes; this one goes with the window.

        It is drawn in the window's pixels, as it was during the drag, so it
        marks the ground only while the view stays as it was measured.
        """
        self._drop_profile_line()
        # The line is the first of the measurement's items: where it starts is
        # read off it, and it is laid again to where the button came up
        x0, y0 = self.rt._canvas.coords(line_id)[:2]
        self._place_measure_line(line_id, x0, y0, end_x, end_y)
        self._profile_line = line_id

    def _drop_profile_line(self):
        """Take the kept measurement line off the picture, if there is one."""
        if self._profile_line is not None and self.rt is not None:
            try:
                self.rt._canvas.delete(self._profile_line)
            except tk.TclError:         # the main window went first
                pass
        self._profile_line = None

    def _open_profile_window(self):
        colours = self.PROFILE_COLOURS

        def before_close():
            # Where it was left, read the way the graph reads its own (see
            # feature_graph_dialog): the corner _show_dialog writes back
            try:
                _, _, corner = win.geometry().partition("+")
                x, _, y = corner.partition("+")
                self._profile_position = (int(x), int(y))
            except ValueError:
                pass
            self._profile = None
            self._drop_profile_line()

        # Not holding the renderer's keys and not modal: measuring again, and
        # stepping the clock, go on in the main window while the profile is up
        win, frame, _close = self._dialog_window("Elevation profile", takes_keys=False,
                                                 before_close=before_close)

        font = ('Consolas', 8)
        metrics = tkfont.Font(font=font)
        cell_w = metrics.measure('0')
        line_h = metrics.metrics('linespace')
        pad = max(2, round(cell_w * 2 / 3))
        line_w = max(1, cell_w // 5)

        # The summary on the left of the top row, the readout's choice on its right
        top_row = tk.Frame(frame)
        top_row.pack(fill=tk.X)
        summary_var = tk.StringVar()
        tk.Label(top_row, textvariable=summary_var, font=font, anchor='w').pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        place_var = tk.BooleanVar(value=self._profile_show_place)

        def place_toggled():
            self._profile_show_place = place_var.get()
            refresh_readout()

        ttk.Checkbutton(top_row, text="Show coordinates", variable=place_var,
                        command=place_toggled).pack(side=tk.RIGHT)

        label_w = metrics.measure('-10000 m') + 2 * pad
        width = round(win.winfo_screenwidth() * self.PROFILE_WIDTH_FRACTION)
        plot_x0, plot_x1 = label_w, width - pad - cell_w
        plot_y0 = pad + line_h                  # a line of room for the readout
        plot_y1 = plot_y0 + self.PROFILE_PLOT_LINES * line_h
        height = plot_y1 + pad + line_h + pad

        canvas = tk.Canvas(frame, width=width, height=height,
                           highlightthickness=0, bg=win.cget('bg'))
        canvas.pack()

        state = {"x": None, "h": None, "x_of": None, "y_of": None, "base": 0.0,
                 "pointer": None}

        def signed(metres: float) -> str:
            """A height counted from the start of the line: signed, and plain 0 at it."""
            value = round(metres)
            return f"{value:+d} m" if value else "0 m"

        def draw(distance_km: np.ndarray, height_m: np.ndarray, start: tuple, end: tuple):
            canvas.delete("all")
            state.update(x=distance_km, h=height_m, start=start, end=end)
            length = float(distance_km[-1])
            low, high = float(height_m.min()), float(height_m.max())
            # A little room above and below the ground, and a span of at least
            # ten metres so a flat mare does not draw its noise as mountains
            margin = max(5.0, 0.08 * (high - low))
            y_min, y_max = low - margin, high + margin

            def x_of(km):
                return plot_x0 + (plot_x1 - plot_x0) * km / length

            def y_of(m):
                return plot_y1 - (plot_y1 - plot_y0) * (m - y_min) / (y_max - y_min)

            state.update(x_of=x_of, y_of=y_of)

            # Grid and axis labels. Heights are read from the start of the line,
            # which stands at 0, and the lines fall on round steps of that
            # reading - so one of them always passes through the start
            h_step = self._nice_step(y_max - y_min, 6)
            base = float(height_m[0])
            state.update(base=base)
            for k in range(math.ceil((y_min - base) / h_step),
                           math.floor((y_max - base) / h_step) + 1):
                h = k * h_step
                y = y_of(base + h)
                canvas.create_line(plot_x0, y, plot_x1, y, fill=colours["grid"])
                canvas.create_text(plot_x0 - pad, y, text=signed(h),
                                   anchor='e', font=font)
            # The unit stands at the right-hand end, where the last number
            # would run into it whenever a step falls near the end of the line:
            # such a number is left out, its grid line kept
            canvas.create_text(plot_x1, plot_y1 + pad, text="km", anchor='ne', font=font)
            unit_left = plot_x1 - metrics.measure("km") - cell_w
            d_step = self._nice_step(length, 10)
            d = 0.0
            while d <= length + 1e-9:
                x = x_of(d)
                canvas.create_line(x, plot_y0, x, plot_y1, fill=colours["grid"])
                text = f"{d:.0f}" if d_step >= 1 else f"{d:.1f}"
                if x + metrics.measure(text) / 2 <= unit_left:
                    canvas.create_text(x, plot_y1 + pad, text=text, anchor='n', font=font)
                d += d_step

            # The ground: filled down to the bottom of the plot, then its edge
            xs = [x_of(float(k)) for k in distance_km]
            ys = [y_of(float(m)) for m in height_m]
            outline = [c for pair in zip(xs, ys) for c in pair]
            canvas.create_polygon([plot_x0, plot_y1] + outline + [plot_x1, plot_y1],
                                  fill=colours["fill"], outline="")
            canvas.create_line(outline, fill=colours["ground"], width=line_w)
            canvas.create_rectangle(plot_x0, plot_y0, plot_x1, plot_y1, outline="#808080")

            # How much the height axis is stretched against the distance axis:
            # a profile drawn to fill its box always looks dramatic, and this
            # is what says by how much it overstates the slopes
            m_per_px_x = length * 1000.0 / (plot_x1 - plot_x0)
            m_per_px_y = (y_max - y_min) / (plot_y1 - plot_y0)
            exaggeration = m_per_px_x / m_per_px_y
            # Counted from the start of the line, as the axis is
            rise = float(height_m[-1]) - base
            summary_var.set(
                f"{length:.1f} km, Δh {signed(rise)}, "
                f"lowest {signed(low - base)}, highest {signed(high - base)}, "
                f"vertical exaggeration ×{exaggeration:.0f}")

        def readout(event):
            state.update(pointer=event.x)
            refresh_readout()

        def pointer_left(_event):
            state.update(pointer=None)
            canvas.delete("readout")

        def refresh_readout():
            """The readout for where the pointer is, drawn afresh."""
            canvas.delete("readout")
            px = state["pointer"]
            if state["x"] is None or px is None or not (plot_x0 <= px <= plot_x1):
                return
            km = (px - plot_x0) / (plot_x1 - plot_x0) * float(state["x"][-1])
            m = float(np.interp(km, state["x"], state["h"]))
            x, y = state["x_of"](km), state["y_of"](m)
            canvas.create_line(x, plot_y0, x, plot_y1, fill=colours["readout"], tags="readout")
            canvas.create_oval(x - 2 * line_w, y - 2 * line_w, x + 2 * line_w, y + 2 * line_w,
                               outline=colours["readout"], width=line_w, tags="readout")
            anchor = 'sw' if x < (plot_x0 + plot_x1) / 2 else 'se'
            # The height read as the axis reads, from the start of the line, and
            # where on the Moon the point is when that is asked for, written as
            # the status bar writes it
            text = f" {km:.1f} km  {signed(m - state['base'])} "
            if self._profile_show_place:
                lat, lon = (float(v[0]) for v in
                            self._along(state["start"], state["end"], km / float(state["x"][-1])))
                text += (f" {abs(lat):.2f}°{'N' if lat >= 0 else 'S'} "
                         f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'} ")
            canvas.create_text(x, plot_y0 - 1, text=text,
                               anchor=anchor, fill=colours["readout"], font=font,
                               tags="readout")

        canvas.bind("<Motion>", readout)
        canvas.bind("<Leave>", pointer_left)

        self._profile = {"win": win, "draw": draw}
        self._show_dialog(win, position=self._profile_position, grab=False)
