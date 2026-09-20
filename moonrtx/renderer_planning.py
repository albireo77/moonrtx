"""
PlanningMixin: the dialogs that answer when to observe, for MoonRenderer.

Three of them, and one shape between two. The rise-and-set chart draws a month
of the observer's nights with the Moon laid over them. The clair-obscur finder
lists the light-and-shadow shapes the terminator draws, and the observation
planner lists the windows when one named feature is worth looking at - by the
Sun standing low over it, or by libration turning it into view. The last two ask
different questions of different parts of astro and put the answers in the same
window, which is built once here (_results_frame) and filled by each of them
with what is its own.

What they find is worth keeping, being the nights to put in a diary, so all
three can hand their results to the clipboard, a spreadsheet or a calendar file.

Nothing here works anything out about the sky: every one of them asks astro and
lays out what comes back.
"""

import io
import csv
import zlib
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox
from datetime import datetime, timedelta, timezone
from typing import Callable, NamedTuple, Optional

from moonrtx import astro
from moonrtx.display import ToolTip, bring_to_front
from moonrtx.shared_types import MoonFeature


class ResultsFrame(NamedTuple):
    """
    The parts of a results dialog that its caller fills in.

    Built by DialogsMixin._results_frame, which puts up the window and
    everything the two results dialogs have in common and leaves these for
    whichever of them asked: the row the filters go in, the two pieces of
    text that change with what is listed, the list itself, and the way to
    shut the window - which the buttons and the Escape key share.
    """
    win: tk.Toplevel
    frame: tk.Frame
    controls: tk.Frame
    description: tk.StringVar
    header: tk.StringVar
    listbox: tk.Listbox
    close: Callable[[], None]


class PlanningMixin:
    """Mixin providing the when-to-observe dialogs for MoonRenderer."""

    # Observation planner filter settings. In terminator mode a feature is
    # worth observing while the Sun stands 0-12 degrees above it (terrain lit,
    # shadows long); in libration mode it only has to be lit at all, since
    # what is being ranked is how far libration turns it into view. Both need
    # the Moon usefully above the observer's horizon.
    # See astro.find_terminator_windows / astro.find_libration_windows.
    PLANNER_SCAN_DAYS = 60
    PLANNER_SUN_ALT_MAX = 12.0
    PLANNER_MOON_ALT_MIN = 5.0
    PLANNER_LIBRATION_SUN_ALT_MIN = 3.0
    PLANNER_MAX_RESULTS = 20
    # How far the Sun has to be under the observer's horizon for the sky to
    # count as dark: nautical twilight, where the Sky column turns to night
    PLANNER_DARK_SUN_ALT = -12.0

    # Clair-obscur finder. The events last hours, so the scan reaches over
    # several lunations to find ones that are actually up at the observer's
    # site, at a step fine enough to place a four-hour window.
    # See astro.find_clair_obscur_events.
    # Rise and set chart. A month of rows shows the whole cycle of the Moon
    # drifting later each night and back again. See astro.find_visibility_chart.
    VISIBILITY_CHART_DAYS = 30
    VISIBILITY_ROW_LEADING = 1  # pixels of air a row has beyond its lettering
    VISIBILITY_HOUR_CELLS = 3.5  # character cells of the row font, per hour
    VISIBILITY_DAY_START = 12   # rows run midday to midday, so a night is one row
    VISIBILITY_COLOURS = {
        "day": "#cfe0f5",       # Sun up
        "twilight": "#5f7ea8",  # Sun down but less than 12 degrees under
        "night": "#101a2b",     # astronomical darkness
        "moon": "#f0c419",      # the Moon above the horizon
        "moon_edge": "#8a6d00",
        "transit": "#ffffff",
        "grid": "#46587a",
        "today": "#c02020",
        "lit": "#fdfbe8",       # the sunlit part of the phase icon
        "unlit": "#39414f",
        "limb": "#707070",
    }

    CLAIR_OBSCUR_SCAN_DAYS = 120
    CLAIR_OBSCUR_STEP_MINUTES = 30
    CLAIR_OBSCUR_ALL_EVENTS = "All events"
    # Filters last chosen in the dialog. Declared on the class so the first
    # opening has defaults; changing one stores it on the instance, so both hold
    # for the rest of the session and start over on the next run.
    _clair_obscur_filter = CLAIR_OBSCUR_ALL_EVENTS
    _clair_obscur_visible_only = True
    # The observation planner's dark-sky filter, kept the same way. The feature
    # graph reads it too, so its window strips mark what the planner lists
    _planner_dark_only = False
    # The feature graph's choices, kept the same way: what a click does to the
    # view, whether the feature's name is shown, and the span of the time axis
    _graph_view = "keep"
    _graph_show_name = True
    # Whether the Moon's altitude in the observer's sky is drawn as a curve.
    # Off to begin with - the graph is opened for the Sun and the libration,
    # and the ribbon already says when the Moon is up
    _graph_show_moon_alt = False
    _graph_span = PLANNER_SCAN_DAYS
    # The arrow keys' step in the graph, in minutes; None until first set, when
    # it starts from the renderer's own Q/W step, which it then leaves alone
    _graph_step_minutes = None
    # Where the graph was last left on the screen, as (x, y); None until one has
    # been closed, when the next opens centred on the main window as any other
    # dialog does. A graph is opened, read, closed and opened again for the next
    # feature, and it is a wide window to drag back into place each time
    _graph_position = None

    # Feature graph. Same span as the observation planner it is opened from,
    # so the two agree on what "the next while" means; a coarser step than the
    # planner's default since a 60-day line plot has no use for hour-level
    # wiggle. See astro.sample_feature_series / PlanningMixin.feature_graph_dialog.
    GRAPH_DAYS = PLANNER_SCAN_DAYS
    GRAPH_STEP_MINUTES = 120
    # Spans the time axis can be set to, the first of them the default. The
    # shorter the span the finer the sampling (GRAPH_STEP_MINUTES scaled with
    # it), so a curve stretched across a zoomed plot stays smooth
    GRAPH_SPANS = (GRAPH_DAYS, 15, 5)
    # Spans the mouse wheel zooms through, a notch at a time, longest first.
    # Whole days, as the buttons' spans are, and those among them so the
    # buttons show where the wheel has got to
    GRAPH_ZOOM_SPANS = (GRAPH_DAYS, 45, 30, 20, 15, 10, 7, 5, 3, 2, 1)
    # Share of the screen's width the plot fills. Libration cycles once a
    # lunation, so a click that lands even a few days off its intended date
    # can recentre on a very differently-presented feature; the wider the
    # plot, the smaller that miss is in real time, not just on screen.
    GRAPH_WIDTH_FRACTION = 0.9
    GRAPH_PLOT_LINES = 18       # plot height, in lines of the axis font
    # Floor of the altitude axis. Below it the Sun curve only says the feature
    # is in lunar night, however deep, and a near-side feature's libration never
    # gets there - around -10° at most, for one on the limb - so the room goes to
    # the 0-12° terminator band instead. A curve under the floor is left out
    # rather than pinned to the edge, where it would read as standing at -30°.
    GRAPH_ALT_MIN = -30
    GRAPH_COLOURS = {
        "sun_alt": "#e8a33d",     # Sun altitude over the feature
        "earth_alt": "#1c6fb0",   # libration figure of merit (Earth altitude)
        # The Moon's altitude in the observer's sky, at the shorter spans. Not
        # the ribbon's yellow, which beside the Sun curve's orange reads as the same
        "moon_alt": "#8e44ad",
        "zero": "#8a8a8a",        # the feature's own horizon / limb
        "threshold": "#c0602a",   # the terminator planner's Sun altitude cap
        "grid": "#d0d0d0",
        # The readout of the moment under the pointer: the line under the plot
        # and the time over it, one colour for the two
        "readout": "#a06010",
        "today": "#c02020",
        "day": "#cfe0f5",
        "twilight": "#5f7ea8",
        "night": "#101a2b",
        "moon": "#f0c419",
        # The observation planner's two lists, marked on the time axis. Far
        # apart in lightness as well as hue, since their windows often overlap
        # and the two strips sit directly on top of each other
        "terminator_window": "#7ddc5a",
        "libration_window": "#146b2e",
    }

    def visibility_chart_dialog(self):
        """
        Chart when the Moon is up over the coming month, a row to the day, laid
        over the darkness of the observer's own sky.

        Rise and set alone do not say whether a night is worth anything: a Moon
        that clears the horizon at noon is no use to anybody. So the row is
        painted with the Sun first - daylight, twilight, then astronomical dark
        - and the Moon's own spell laid on top of it, leaving the nights it is
        up in darkness to be picked out at a glance. Clicking anywhere in the
        chart takes the app to that moment.
        """
        if self.rt is None:
            return

        win, main_frame, on_close = self._dialog_window("Moon rise and set")

        colours = self.VISIBILITY_COLOURS
        font = ('Consolas', 8)
        head_font = ('Consolas', 8, 'bold')

        # Every measurement below is taken from the font rather than written in
        # pixels. A font asked for in points follows the display's dots per inch
        # - at 300% scaling it is three times what it is at 100% - while a number
        # of pixels follows nothing. Written in pixels the rows came out shorter
        # than the lettering standing in them and the columns narrower than the
        # times filling them, so on such a screen the chart collapsed into
        # overlapping text. Taken from the font, it holds its proportions at any
        # scaling, and at 100% it comes to within a few pixels of the pixel
        # counts these replace.
        metrics = tkfont.Font(font=font)
        head_metrics = tkfont.Font(font=head_font)
        cell_w = metrics.measure('0')       # one cell of the fixed-pitch row font
        line_h = metrics.metrics('linespace')
        pad = max(2, round(cell_w * 2 / 3))  # the air around a piece of lettering

        row_h = line_h + self.VISIBILITY_ROW_LEADING
        head_h = line_h + pad - 1
        bar_pad = max(1, row_h // 4)        # over and under the Moon-up bar
        hour_w = max(1, round(cell_w * self.VISIBILITY_HOUR_CELLS))

        date_w = metrics.measure('Fri 14 Aug') + 2 * pad + cell_w
        chart_w = 24 * hour_w
        # Title, column width, and how many characters to nudge the title by:
        # the values are right-aligned in their columns, so a title shorter or
        # longer than them does not sit over them squarely on its own. A column
        # is as wide as the wider of its title and the times it holds.
        columns = tuple(
            (title,
             max(metrics.measure('00:00'), head_metrics.measure(title)) + 2 * cell_w,
             nudge)
            for title, nudge in (("Rise", -1), ("Transit", 1), ("Set", -1), ("Max", -1)))
        chart_x = date_w
        table_x = chart_x + chart_w + 2 * cell_w
        phase_x = table_x + sum(w for _, w, _ in columns)
        phase_w = row_h + 2 * cell_w
        width = phase_x + phase_w
        rule = max(1, cell_w // 3)          # the line marking the moment on show

        canvas = tk.Canvas(main_frame, width=width,
                           height=head_h + self.VISIBILITY_CHART_DAYS * row_h + pad // 2,
                           highlightthickness=0, bg=win.cget('bg'))
        canvas.pack()

        # Where the chart currently sits. "first_date" is the night the top row
        # stands for, which the arrows move and Reset takes back to the clock;
        # "rows" is zero while nothing is drawn, as when a page falls outside
        # the dates the kernels cover
        span = {"first_date": None, "rows": 0}

        def hour_of(moment) -> float:
            """
            Where a moment falls along a row, in hours from its left edge.

            Rows run from midday to midday rather than midnight to midnight, so
            that a night belongs to one row entire. Cut at midnight instead, the
            nights split in two would be the ones with the Moon up across it -
            which is when it rides highest and is most worth having.
            """
            wall_clock = moment.hour + moment.minute / 60.0 + moment.second / 3600.0
            return (wall_clock - self.VISIBILITY_DAY_START) % 24.0

        def row_of(moment) -> int:
            """Which row a moment belongs to, the row being a night, not a date."""
            night = (moment - timedelta(hours=self.VISIBILITY_DAY_START)).date()
            return (night - span["first_date"]).days

        def midday_on(date):
            """The moment a row opens: midday on the date it is labelled with."""
            return self.from_observer_clock(
                datetime.combine(date, datetime.min.time())
                + timedelta(hours=self.VISIBILITY_DAY_START))

        def x_of(hour: float) -> float:
            return chart_x + hour * hour_w

        def y_of(row: int) -> int:
            return head_h + row * row_h

        def phase_icon(centre_y: int, lit_fraction: float, waxing: bool):
            """
            Draw the Moon as it is lit, at the given height in the phase column.

            The terminator is the edge of the lit hemisphere seen side on, so it
            projects to a half-ellipse whose width follows the phase: widest at
            new and full, closing to nothing at the quarters. Half the disc is
            painted lit and that ellipse then either eats into it, leaving a
            crescent, or fills out beside it, leaving a gibbous disc.
            """
            radius = (row_h - pad) // 2
            centre_x = phase_x + phase_w // 2
            box = (centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius)
            canvas.create_oval(*box, fill=colours["unlit"], outline="")
            # Waxing lights the western limb, which is drawn to the right here,
            # the side the Sun is on for the first half of the month
            canvas.create_arc(*box, start=-90 if waxing else 90, extent=180,
                              style=tk.PIESLICE, fill=colours["lit"], outline="")
            gibbous = lit_fraction > 0.5
            terminator = radius * abs(2.0 * lit_fraction - 1.0)
            ellipse = (centre_x - terminator, centre_y - radius,
                       centre_x + terminator, centre_y + radius)
            canvas.create_oval(*ellipse, outline="",
                               fill=colours["lit"] if gibbous else colours["unlit"])
            # Only half that ellipse divides light from dark - the other half
            # falls inside whichever the fill just extended, where there is no
            # edge to draw. It is stroked because within a few days of new or
            # full the sliver it cuts off is thinner than a pixel at this size,
            # and a run of dates would otherwise all render as flatly full
            if terminator >= 1.0:
                canvas.create_arc(*ellipse, start=90 if waxing == gibbous else -90,
                                  extent=180, style=tk.ARC, outline=colours["limb"])
            canvas.create_oval(*box, fill="", outline=colours["limb"])

        now_marker = []

        def draw_now_marker():
            """A line down the chart at the moment the app is showing."""
            for item in now_marker:
                canvas.delete(item)
            now_marker.clear()
            if not span["rows"]:
                return
            row = row_of(self.dt_local)
            if not 0 <= row < span["rows"]:
                return
            x_now = x_of(hour_of(self.dt_local))
            now_marker.append(canvas.create_line(x_now, y_of(row), x_now, y_of(row) + row_h,
                                                 fill=colours["today"], width=rule))

        def redraw():
            """Draw the chart afresh over the span it currently sits on."""
            canvas.delete('all')
            now_marker.clear()
            span["rows"] = 0

            first_date = span["first_date"]
            try:
                chart = astro.find_visibility_chart(midday_on(first_date),
                                                    self.VISIBILITY_CHART_DAYS)
            except ValueError as e:
                # This page falls outside the bundled ephemeris kernel range.
                # The date it was asked for is kept, so paging back off the end
                # returns to where it came from rather than to the clock
                canvas.config(height=3 * line_h + 2 * pad)
                canvas.create_text(pad, pad, text=str(e), anchor='nw',
                                   width=width - 2 * pad, font=font)
                return

            # The span is trimmed where it would run past the dates the kernels
            # cover, so the last rows are dropped rather than drawn empty
            rows = max(1, min(self.VISIBILITY_CHART_DAYS,
                              -((chart.start - chart.end).days)))
            span["rows"] = rows
            canvas.config(height=head_h + rows * row_h + pad // 2)

            def pieces(spells: list) -> list:
                """
                Cut spells into per-row runs of hours. A spell carrying over the
                midday a row ends at belongs to both rows it touches, so it is
                drawn as one piece on each rather than wrapping round.
                """
                out = []
                for start_utc, end_utc in spells:
                    start = self.in_observer_clock(start_utc)
                    end = self.in_observer_clock(end_utc)
                    for row in range(row_of(start), row_of(end) + 1):
                        if 0 <= row < rows:
                            from_hour = hour_of(start) if row == row_of(start) else 0.0
                            to_hour = hour_of(end) if row == row_of(end) else 24.0
                            if to_hour > from_hour:
                                out.append((row, from_hour, to_hour))
                return out

            def band(row: int, from_hour: float, to_hour: float, fill: str):
                canvas.create_rectangle(x_of(from_hour), y_of(row),
                                        x_of(to_hour), y_of(row) + row_h,
                                        fill=fill, outline="")

            # Hour scale along the top, every three hours, reading from the
            # midday a row opens at round to the midday it closes at
            for hour in range(0, 25, 3):
                clock = (self.VISIBILITY_DAY_START + hour) % 24
                canvas.create_text(x_of(hour), head_h - pad, text=f"{clock:02d}",
                                   font=font, anchor='s')

            # Night first, then the lighter spells over it: the Sun above the
            # horizon is also above the twilight depth, so day paints last
            for row in range(rows):
                band(row, 0.0, 24.0, colours["night"])
            for row, a, b in pieces(chart.sun_twilight):
                band(row, a, b, colours["twilight"])
            for row, a, b in pieces(chart.sun_up):
                band(row, a, b, colours["day"])

            for hour in range(3, 24, 3):
                canvas.create_line(x_of(hour), head_h, x_of(hour), head_h + rows * row_h,
                                   fill=colours["grid"])

            for row, a, b in pieces(chart.moon_up):
                canvas.create_rectangle(x_of(a), y_of(row) + bar_pad,
                                        x_of(b), y_of(row) + row_h - bar_pad,
                                        fill=colours["moon"], outline=colours["moon_edge"])

            # Per-row times: the rise and set of the night that row stands for,
            # which is the pair belonging together far more often than the two
            # an almanac lists against a calendar date. A spell already under
            # way when the chart opens, or still running when it closes, was
            # clipped to the span - those ends are where the chart stops rather
            # than where the Moon crossed, so are not listed
            rise_on, set_on = {}, {}
            for start_utc, end_utc in chart.moon_up:
                if start_utc > chart.start:
                    rise = self.in_observer_clock(start_utc)
                    rise_on.setdefault(row_of(rise), rise)
                if end_utc < chart.end:
                    setting = self.in_observer_clock(end_utc)
                    set_on.setdefault(row_of(setting), setting)
            transit_on = {}
            for when_utc, altitude in chart.transits:
                local = self.in_observer_clock(when_utc)
                row = row_of(local)
                transit_on.setdefault(row, (local, altitude))
                if 0 <= row < rows and altitude > 0.0:
                    canvas.create_line(x_of(hour_of(local)), y_of(row) + bar_pad,
                                       x_of(hour_of(local)), y_of(row) + row_h - bar_pad,
                                       fill=colours["transit"])

            # How much of the disc is lit through each night, and which way it
            # is going: the fraction alone cannot say waxing from waning, so the
            # night after tells, or the one before for the last row of the chart.
            # Each reading is taken at the middle of its span, which with rows
            # running midday to midday is the middle of the night itself
            lit_on = {}
            readings = chart.illumination
            for i, (_, fraction) in enumerate(readings):
                if i + 1 < len(readings):
                    waxing = readings[i + 1][1] >= fraction
                elif i > 0:
                    waxing = fraction >= readings[i - 1][1]
                else:
                    waxing = True
                lit_on[i] = (fraction, waxing)

            x = table_x
            for title, column_w, nudge in columns:
                canvas.create_text(x + column_w - pad + nudge * cell_w, head_h - pad,
                                   text=title, font=head_font, anchor='se')
                x += column_w
            canvas.create_text(phase_x + phase_w // 2, head_h - pad, text="Lit",
                               font=head_font, anchor='s')

            for row in range(rows):
                date = first_date + timedelta(days=row)
                centre = y_of(row) + row_h // 2
                canvas.create_text(date_w - 2 * pad, centre, text=f"{date:%a %d %b}",
                                   font=font, anchor='e')
                transit = transit_on.get(row)
                values = (
                    f"{rise_on[row]:%H:%M}" if row in rise_on else "-",
                    f"{transit[0]:%H:%M}" if transit else "-",
                    f"{set_on[row]:%H:%M}" if row in set_on else "-",
                    f"{transit[1]:+.0f}°" if transit else "-",
                )
                x = table_x
                for (_, column_w, _), value in zip(columns, values):
                    canvas.create_text(x + column_w - pad, centre, text=value,
                                       font=font, anchor='e')
                    x += column_w
                if row in lit_on:
                    phase_icon(centre, *lit_on[row])

            draw_now_marker()

        def go_to(event):
            """Take the app to the moment clicked in the chart."""
            row = (event.y - head_h) // row_h
            # Nothing is drawn when rows is zero, so this turns clicks away too
            if not 0 <= row < span["rows"] or not chart_x <= event.x <= chart_x + chart_w:
                return
            hours = (event.x - chart_x) / hour_w
            date = span["first_date"] + timedelta(days=int(row))
            self._go_to_moment(midday_on(date) + timedelta(hours=hours))
            draw_now_marker()

        canvas.bind('<Button-1>', go_to)

        legend = tk.Frame(main_frame)
        legend.pack(fill=tk.X, pady=(2 * pad, 0))
        for text, colour in (("Moon up", colours["moon"]), ("Daylight", colours["day"]),
                             ("Twilight", colours["twilight"]), ("Dark", colours["night"])):
            swatch = tk.Frame(legend, bg=colour, width=row_h, height=line_h - bar_pad,
                              highlightthickness=1, highlightbackground="#808080")
            swatch.pack(side=tk.LEFT)
            swatch.pack_propagate(False)
            tk.Label(legend, text=text, font=font).pack(
                side=tk.LEFT, padx=(max(1, cell_w // 2), cell_w + pad))
        tk.Label(legend, text="Click the chart to go to that moment", font=font,
                 fg='#606060').pack(side=tk.LEFT)
        def page(days: int):
            """
            Move the chart on by a span, without disturbing the Moon.

            Measured from the page on screen, not from the clock: clicking a row
            moves the clock, and paging from that instead would land the next
            page a few days along from where this one ends.
            """
            span["first_date"] += timedelta(days=days)
            redraw()

        def reset():
            """Take the chart back to the night the app is showing."""
            span["first_date"] = (self.dt_local
                                  - timedelta(hours=self.VISIBILITY_DAY_START)).date()
            redraw()

        # Close packed first so it keeps the right-hand end, then Reset, then
        # the pager - which reads left to right once packed in that order
        tk.Button(legend, text="Close", command=on_close, width=10).pack(side=tk.RIGHT)
        tk.Button(legend, text="Reset", command=reset,
                  width=10).pack(side=tk.RIGHT, padx=(0, pad + 2))
        tk.Button(legend, text="▶", width=2,
                  command=lambda: page(self.VISIBILITY_CHART_DAYS)).pack(
            side=tk.RIGHT, padx=(0, pad + 2))
        tk.Button(legend, text="◀", width=2,
                  command=lambda: page(-self.VISIBILITY_CHART_DAYS)).pack(side=tk.RIGHT)

        reset()   # the first draw sits on the night the app is showing

        self._show_dialog(win)

    # ---- taking results out of the dialogs ----

    # Planner results are worth keeping: they are the nights to put in a diary,
    # and until now they lived only in a window that closes. Copy hands them to
    # whatever the user is writing in; Save writes a spreadsheet (CSV) or a
    # calendar (ICS) that any diary application reads.
    EXPORT_FILE_TYPES = (("Calendar file", "*.ics"), ("Spreadsheet", "*.csv"))

    def _go_to_moment(self, when: datetime):
        """
        Take the view to that moment, as choosing a result does.

        Auto-advance is put back to the start of its interval so the jump is not
        followed a moment later by a step it was already part way through, and
        the panels are told because the clock moved under them.
        """
        self.update_view(when)
        if self._auto_advance_var and self._auto_advance_var.get():
            self._auto_advance_elapsed = 0
        self._update_all_status_panels()

    # ---- the frame a dialog of results is built in ----

    # The clair-obscur dialog and the observation planner ask different
    # questions and list different columns, and are the same window: a line
    # saying what is being looked for, a row of controls that narrow it, a
    # description that follows the selection, a header, the list itself, and a
    # row of buttons to go to a result or take the lot away. That window is
    # built here, empty, and each of them fills it with what is its own.
    RESULTS_FONT = ('Consolas', 9)
    RESULTS_HEADER_FONT = ('Consolas', 9, 'bold')
    RESULTS_ROWS = 16
    # Enough room for the widest header, and two characters over so the last
    # column does not sit against the scrollbar
    RESULTS_WIDTH_MARGIN = 2

    def _results_frame(self, title: str, caption: str, header_width: int):
        """
        Build a results dialog with nothing in it yet, and hand back the parts
        its caller has to fill: the controls row, the description and header
        text, the list, and the way to close it.

        The description is wrapped to the width the list asks for rather than at
        newlines written into it, so it fills the dialog whatever is in it, and
        given a fixed height so the window does not resize as it changes.
        """
        win, frame, close = self._dialog_window(title)

        tk.Label(frame, anchor='w', font=self.RESULTS_FONT,
                 text=caption).pack(fill=tk.X)

        controls = tk.Frame(frame)
        controls.pack(fill=tk.X, pady=(4, 0))

        description = tk.StringVar()
        desc_label = tk.Label(frame, textvariable=description, justify=tk.LEFT,
                              anchor='nw', font=self.RESULTS_FONT, height=3)
        desc_label.pack(fill=tk.X, pady=(4, 6))

        header = tk.StringVar()
        tk.Label(frame, textvariable=header, font=self.RESULTS_HEADER_FONT,
                 anchor='w').pack(fill=tk.X)

        list_frame = tk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                             font=self.RESULTS_FONT,
                             width=header_width + self.RESULTS_WIDTH_MARGIN,
                             height=self.RESULTS_ROWS)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        win.update_idletasks()
        desc_label.config(wraplength=listbox.winfo_reqwidth())

        return ResultsFrame(win, frame, controls, description, header, listbox,
                            close)

    def _results_actions(self, results, go_to, table, name, extra=None):
        """
        The row of buttons a results dialog ends with: go to the one selected,
        put them all on the clipboard, write them to a file, close.

        `table` answers with the columns, the rows and the calendar entries of
        whatever is listed at the time it is asked, and `name` with the stem to
        offer the save under - both of them called rather than passed in, since
        both follow the filter the dialog is left on.

        `extra`, when given, is a (label, command) pair for one more button
        placed next to "Go to selected" - the observation planner's way in to
        feature_graph_dialog, which the clair-obscur finder has no use for.
        """
        def copy():
            columns, rows, _entries = table()
            if self._copy_lines_to_clipboard(
                    ["\t".join(columns)] + ["\t".join(row) for row in rows]):
                copy_button.config(text="Copied")
                results.win.after(1200, lambda: copy_button.config(text="Copy"))

        def save():
            columns, rows, entries = table()
            self._save_results(results.win, name() + self._date_range_suffix(entries),
                               columns, rows, entries)

        row = tk.Frame(results.frame)
        row.pack(fill=tk.X, pady=(8, 0))
        tk.Button(row, text="Go to selected", command=go_to,
                  width=16).pack(side=tk.LEFT)
        if extra is not None:
            label, command = extra
            tk.Button(row, text=label, command=command, width=10).pack(
                side=tk.LEFT, padx=(6, 0))
        copy_button = tk.Button(row, text="Copy", command=copy, width=8)
        copy_button.pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(row, text="Save...", command=save,
                  width=8).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(row, text="Close", command=results.close,
                  width=10).pack(side=tk.RIGHT)

    @staticmethod
    def _sky_of(record: dict) -> str:
        """
        What the observer's own sky is doing while the event is on: the Sun up,
        or down but not far enough for the sky to be dark, or properly night.
        The threshold is nautical twilight, past which the fainter detail on the
        Moon starts to show.
        """
        if record["observer_sun_alt"] > 0.0:
            return "day"
        return "twilight" if record["observer_sun_alt"] > PlanningMixin.PLANNER_DARK_SUN_ALT else "night"

    @staticmethod
    def _lasting(windows: list) -> list:
        """
        The windows that last, without the ones that start and end at the same
        moment.

        Such a window is the scan catching its conditions on a single sample and
        losing them by the next: what it stands for is a spell shorter than the
        step, somewhere around that moment, which is nothing to plan a night
        around. Left in, it reads as an opportunity of no length at all, and
        goes into a calendar as an entry that ends when it begins.
        """
        return [w for w in windows if w["end"] > w["start"]]

    def _planner_observer_sun_alt_max(self) -> float:
        """
        The Sun altitude at the observer's site the planner's windows have to
        stay under: dark sky only when the planner's filter asks for it,
        otherwise no limit.
        """
        return self.PLANNER_DARK_SUN_ALT if self._planner_dark_only else 90.0

    def _copy_lines_to_clipboard(self, lines: list[str]) -> bool:
        """
        Put the listed rows on the clipboard, tab separated.

        Tk owns the clipboard only while it runs, which is exactly the case
        here, and the text survives to other applications as long as the app is
        open - the usual Tk caveat, and the reason this is a copy rather than a
        cut.
        """
        if self.rt is None or not lines:
            return False
        try:
            self.rt._root.clipboard_clear()
            self.rt._root.clipboard_append("\n".join(lines))
            self.rt._root.update()          # hand it over before returning
            return True
        except Exception as e:
            print(f"Could not copy to the clipboard: {e}")
            return False

    @staticmethod
    def _csv_text(columns: list[str], rows: list[list[str]]) -> str:
        """The rows as CSV, quoting whatever needs it."""
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
        return out.getvalue()

    @staticmethod
    def _ics_text(events: list[dict]) -> str:
        """
        The events as an iCalendar file.

        Times go in as UTC, which every calendar reads and which needs no
        timezone definition inside the file. Each event carries the observing
        detail in its description, so the entry is still useful months later,
        and an identifier built from its own times, so importing the same scan
        twice updates the entries rather than doubling them.
        """
        def stamp(when: datetime) -> str:
            return when.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        def escape(text: str) -> str:
            """Backslash, semicolon, comma and newline carry meaning in a property value."""
            return (text.replace("\\", "\\\\")
                        .replace(";", "\\;")
                        .replace(",", "\\,")
                        .replace("\r\n", "\n")
                        .replace("\n", "\\n"))

        def fold(line: str) -> str:
            """
            Break a long property over several lines, as RFC 5545 asks.

            A continuation starts with one space, which the reader takes back
            off again. The limit counts octets, not characters.
            """
            out, current = [], ""
            for char in line:
                limit = 75 if not out else 74      # a continuation spends one on its space
                if len(current.encode("utf-8")) + len(char.encode("utf-8")) > limit:
                    out.append(current)
                    current = ""
                current += char
            out.append(current)
            return ("\r\n ").join(out)

        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//MoonRTX//Observation planner//EN",
                 "CALSCALE:GREGORIAN"]
        now = stamp(datetime.now(timezone.utc))
        for event in events:
            # CRC-32 of the title rather than hash(): Python gives a string a
            # different hash in every run, so an identifier built from it held
            # only within one session, and the same scan exported another day
            # and imported again doubled every entry instead of updating it
            title = zlib.crc32(event['summary'].encode('utf-8'))
            uid = f"{stamp(event['start'])}-{title}@moonrtx"
            lines += ["BEGIN:VEVENT",
                      f"UID:{uid}",
                      f"DTSTAMP:{now}",
                      f"DTSTART:{stamp(event['start'])}",
                      f"DTEND:{stamp(event['end'])}",
                      f"SUMMARY:{escape(event['summary'])}",
                      f"DESCRIPTION:{escape(event['description'])}",
                      "END:VEVENT"]
        lines.append("END:VCALENDAR")
        return "\r\n".join(fold(line) for line in lines) + "\r\n"       # RFC 5545 asks for CRLF

    def _date_range_suffix(self, entries: list[dict]) -> str:
        """
        The span the results cover, for the file name.

        Dates are on the observer's clock, as the list shows them, so a file
        saved in the evening is named for the night it is about. A scan that
        lands on one day is named for that day once rather than twice.
        """
        if not entries:
            return ""
        first = min(self.in_observer_clock(e["start"]).date() for e in entries)
        last = max(self.in_observer_clock(e["end"]).date() for e in entries)
        return f"_{first:%Y-%m-%d}" if first == last else f"_{first:%Y-%m-%d}_{last:%Y-%m-%d}"

    def _save_results(self, parent, default_name: str, columns: list[str],
                      rows: list[list[str]], events: list[dict]) -> Optional[str]:
        """
        Ask where to put the results and write them, as a calendar or a sheet.

        Returns the path written, or None when the user backed out. The format
        follows the extension chosen in the dialog, so the same button serves
        both without a second question.
        """
        if not rows:
            return None
        path = filedialog.asksaveasfilename(
            parent=parent, title="Save the results", initialfile=default_name,
            defaultextension=".ics", filetypes=self.EXPORT_FILE_TYPES)
        if not path:
            return None
        text = self._csv_text(columns, rows) if path.lower().endswith(".csv") \
            else self._ics_text(events)
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        except OSError as e:
            messagebox.showerror("Error", f"Could not write {path}:\n{e}", parent=parent)
            return None
        print(f"Saved: {path}")
        return path

    def clair_obscur_dialog(self):
        """
        Show upcoming clair-obscur events - the light-and-shadow shapes that
        stand for a few hours when the terminator lights only the high ground
        of a formation - and let the user jump the app time to one of them.

        "Go to selected" moves to the peak of the pattern rather than to the
        part of it visible from the observer's site: the renderer has no sky of
        its own, so it shows the event whether or not the Moon is up outside.
        The visible column is there to plan the observation itself.
        """
        if self.rt is None:
            return

        events = []                      # results currently listed

        # The altitudes carry the status bar's notation: h(sun) over the event
        # itself, h(moon) in the observer's sky. Here the header is a single
        # Label in a single font, so the signs cannot be lowered into subscripts
        # as they are there.
        #
        # Both altitudes are left-aligned like every other column, so each label
        # sits directly over the start of its values. Two header fields are
        # deliberately not the width of the values below them: the date field is
        # 22 against the rows' 20 because the rows put two spaces after the
        # timestamp, and the h(moon) field is 7 against their 8 to absorb the
        # signs, which are wider than a Consolas cell (+3 px for the Sun, +6 for
        # the Moon). Every label then starts within 3 px - under half a
        # character - of its column.
        header = (f"{'Event':<20}{'Peak (local)':<22}{'Pattern':<16}{'Visible here':<16}"
                  f"{'h☉':<8}{'h☾':<7}  {'Sky':<8}")

        dialog = self._results_frame(
            "Clair-obscur events",
            f"Shapes drawn by the terminator, over the next "
            f"{self.CLAIR_OBSCUR_SCAN_DAYS} days",
            len(header))
        win, on_close, listbox = dialog.win, dialog.close, dialog.listbox
        desc_var = dialog.description
        dialog.header.set(header)

        all_events = self.CLAIR_OBSCUR_ALL_EVENTS
        event_names = [all_events] + [e.name for e in astro.CLAIR_OBSCUR_EVENTS]
        # Resume the filter this session was last left on, unless it names an
        # event the catalogue no longer has
        filter_var = tk.StringVar(
            value=self._clair_obscur_filter if self._clair_obscur_filter in event_names
            else all_events)

        filter_row = dialog.controls
        tk.Label(filter_row, text="Show:", anchor='w').pack(side=tk.LEFT)
        option = tk.OptionMenu(filter_row, filter_var, *event_names, command=lambda _: rescan())
        option.config(width=max(len(name) for name in event_names), anchor='w')
        option.pack(side=tk.LEFT, padx=(4, 12))

        visible_only_var = tk.BooleanVar(value=self._clair_obscur_visible_only)
        # Naming the column ties the filter to the figure it acts on
        # Themed, so the little box follows the display
        ttk.Checkbutton(filter_row, variable=visible_only_var,
                       text=f"only when the Moon altitude (h☾) is at least "
                            f"{self.PLANNER_MOON_ALT_MIN:.0f}° in my sky",
                       command=lambda: rescan()).pack(side=tk.LEFT)

        def show_description(event=None):
            selection = listbox.curselection()
            if events and selection and selection[0] < len(events):
                o = events[selection[0]]
                desc_var.set(f"{o['event']} (lat {o['lat']:.1f}°, lon {o['lon']:.1f}°): "
                             f"{o['description']}")

        def rescan():
            nonlocal events
            listbox.delete(0, tk.END)
            chosen = filter_var.get()
            self._clair_obscur_filter = chosen
            self._clair_obscur_visible_only = visible_only_var.get()
            catalogue = astro.CLAIR_OBSCUR_EVENTS if chosen == all_events else tuple(
                e for e in astro.CLAIR_OBSCUR_EVENTS if e.name == chosen)
            try:
                events = astro.find_clair_obscur_events(
                    self.dt_local, self.CLAIR_OBSCUR_SCAN_DAYS,
                    step_minutes=self.CLAIR_OBSCUR_STEP_MINUTES,
                    moon_alt_min=self.PLANNER_MOON_ALT_MIN if visible_only_var.get() else 0.0,
                    events=catalogue)
            except ValueError as e:
                # Scan start outside the bundled ephemeris kernel range
                events = []
                desc_var.set(str(e))
                return

            if not events:
                desc_var.set("")
                message = "  No events found in the scanned period."
                if visible_only_var.get():
                    message += "  Untick the filter to include the ones below your horizon."
                listbox.insert(tk.END, message)
                return

            for o in events:
                peak = self.in_observer_clock(o["peak"])
                start = self.in_observer_clock(o["start"])
                end = self.in_observer_clock(o["end"])
                pattern = f"{start:%H:%M}-{end:%H:%M}"
                if o["visible_start"] is not None:
                    vs = self.in_observer_clock(o["visible_start"])
                    ve = self.in_observer_clock(o["visible_end"])
                    visible = f"{vs:%H:%M}-{ve:%H:%M}"
                else:
                    visible = "-"
                # Left-aligned like the columns before them, so the values start
                # under their labels (see header)
                sun_alt = f"{o['sun_alt']:+.1f}°"
                moon_alt = f"{o['moon_alt']:+.0f}°"
                listbox.insert(tk.END,
                               f"{o['event']:<20}{peak:%Y-%m-%d %a %H:%M}  {pattern:<16}"
                               f"{visible:<16}{sun_alt:<8}{moon_alt:<8}  {self._sky_of(o)}")
            listbox.selection_set(0)
            show_description()

        def go_to(event=None):
            selection = listbox.curselection()
            if not events or not selection or selection[0] >= len(events):
                return
            o = events[selection[0]]
            on_close()
            self._go_to_moment(self.in_observer_clock(o["peak"]))
            self.center_on_lat_lon(o["lat"], o["lon"])

        listbox.bind('<<ListboxSelect>>', show_description)
        listbox.bind('<Double-Button-1>', go_to)
        listbox.bind('<Return>', go_to)

        def results_for_export():
            """The listed events as a table and as calendar entries."""
            columns = ["Event", "Peak", "Pattern start", "Pattern end", "Visible from here",
                       "Sun over event (deg)", "Moon altitude (deg)", "Sky"]
            rows, entries = [], []
            for o in events:
                peak = self.in_observer_clock(o["peak"])
                start = self.in_observer_clock(o["start"])
                end = self.in_observer_clock(o["end"])
                if o["visible_start"] is not None:
                    vs = self.in_observer_clock(o["visible_start"])
                    ve = self.in_observer_clock(o["visible_end"])
                    visible = f"{vs:%Y-%m-%d %H:%M} .. {ve:%H:%M}"
                else:
                    visible = "below the horizon"
                rows.append([o["event"], f"{peak:%Y-%m-%d %H:%M}", f"{start:%Y-%m-%d %H:%M}",
                             f"{end:%Y-%m-%d %H:%M}", visible, f"{o['sun_alt']:+.1f}",
                             f"{o['moon_alt']:+.0f}", self._sky_of(o)])
                entries.append({
                    "summary": f"MoonRTX: {o['event']}",
                    # The pattern itself, not the part of it above the horizon:
                    # the window to watch for is the whole of it, and the
                    # description says when it is up here
                    "start": o["start"], "end": o["end"],
                    "description": (f"Peak at {peak:%Y-%m-%d %H:%M}. Visible from here: {visible}. "
                                    f"Sun {o['sun_alt']:+.1f} deg over the formation, Moon "
                                    f"{o['moon_alt']:+.0f} deg up in a {self._sky_of(o)} sky."),
                })
            return columns, rows, entries

        self._results_actions(dialog, go_to, results_for_export,
                              lambda: "clair_obscur_events")

        rescan()

        self._show_dialog(win)

    def observation_planner_dialog(self, feature: MoonFeature):
        """
        Show upcoming windows when the given feature is worth observing and
        let the user jump the app time to one of them. Two criteria:

        - terminator: the Sun low over the feature, so it stands in relief
          with long shadows (astro.find_terminator_windows)
        - libration: the feature turned toward Earth as far as it gets, which
          is what decides whether a limb formation shows anything at all
          (astro.find_libration_windows)

        Parameters
        ----------
        feature : MoonFeature
            The feature to plan for (None is ignored, so the method can be
            called directly with the status-bar feature)
        """
        if self.rt is None or feature is None:
            return

        windows = []                     # results currently listed

        # The sky column is padded to its longest value ("twilight"), so the
        # header length already covers the widest row and the listbox sized
        # from it never clips one
        terminator_header = (f"{'Best time (local)':<22}{'Event':<9}{'Window (local)':<29}"
                             f"{'Sun@feat':>7}{'Moon alt':>10}  {'Sky':<8}")
        libration_header = (f"{'Best time (local)':<22}{'Window (local)':<29}{'Presented':>10}"
                            f"{'Libr L':>9}{'Libr B':>9}{'Sun@feat':>10}{'Moon alt':>10}  {'Sky':<8}")

        dialog = self._results_frame(
            f"Observation Planner - {feature.name}",
            f"{feature.name}  (lat {feature.lat:.2f}°, lon {feature.lon:.2f}°)"
            f"  -  next {self.PLANNER_SCAN_DAYS} days",
            max(len(terminator_header), len(libration_header)))
        win, on_close, listbox = dialog.win, dialog.close, dialog.listbox
        desc_var, header_var = dialog.description, dialog.header

        mode_var = tk.StringVar(value="terminator")
        mode_row = dialog.controls
        # Room between the choices, measured in the lettering rather than in a
        # fixed number of pixels: at 300% on a 4K screen the words are three
        # times the size, and a dozen pixels between them reads as none at all
        gap = 2 * tkfont.Font(font='TkDefaultFont').measure('0')
        tk.Label(mode_row, text="Show:", anchor='w').pack(side=tk.LEFT)
        for value, label in (("terminator", "near the terminator"),
                             ("libration", "best presented (libration)")):
            ttk.Radiobutton(mode_row, text=label, value=value, variable=mode_var,
                            command=lambda: rescan()).pack(side=tk.LEFT, padx=(0, gap))
        # Resumes where this session last left it, as the clair-obscur filter does
        dark_only_var = tk.BooleanVar(value=self._planner_dark_only)
        ttk.Checkbutton(mode_row, variable=dark_only_var,
                        text=f"only when the sky is dark (Sun {-self.PLANNER_DARK_SUN_ALT:.0f}° "
                             f"below my horizon)",
                        command=lambda: rescan()).pack(side=tk.LEFT)

        def rescan():
            nonlocal windows
            listbox.delete(0, tk.END)
            libration = mode_var.get() == "libration"
            self._planner_dark_only = dark_only_var.get()
            dark = ", while the sky is dark" if self._planner_dark_only else ""
            try:
                if libration:
                    windows = astro.find_libration_windows(
                        self.dt_local, self.PLANNER_SCAN_DAYS, feature.lat, feature.lon,
                        sun_alt_min=self.PLANNER_LIBRATION_SUN_ALT_MIN,
                        moon_alt_min=self.PLANNER_MOON_ALT_MIN,
                        max_results=self.PLANNER_MAX_RESULTS,
                        observer_sun_alt_max=self._planner_observer_sun_alt_max())
                else:
                    windows = astro.find_terminator_windows(
                        self.dt_local, self.PLANNER_SCAN_DAYS, feature.lat, feature.lon,
                        sun_alt_max=self.PLANNER_SUN_ALT_MAX,
                        moon_alt_min=self.PLANNER_MOON_ALT_MIN,
                        observer_sun_alt_max=self._planner_observer_sun_alt_max())
            except ValueError as e:
                # Scan start outside the bundled ephemeris kernel range
                windows = []
                desc_var.set(str(e))
                header_var.set("")
                return

            windows = self._lasting(windows)

            if libration:
                desc_var.set(
                    "How far inside the limb libration turns the feature, best first: 90° is the "
                    "centre of the disk, 0° exactly on the limb, and the feature is squashed by the "
                    "sine of it. Listed only while the feature is sunlit and the Moon at least "
                    f"{self.PLANNER_MOON_ALT_MIN:.0f}° up in your sky{dark}.")
                header_var.set(libration_header)
            else:
                desc_var.set(
                    f"Times when the Sun stands 0-{self.PLANNER_SUN_ALT_MAX:.0f}° above the feature, "
                    f"lighting it with long shadows, the feature is turned toward Earth, and the "
                    f"Moon is at least {self.PLANNER_MOON_ALT_MIN:.0f}° up in your sky{dark}.")
                header_var.set(terminator_header)

            if not windows:
                message = "  No opportunities found in the scanned period."
                if self._planner_dark_only:
                    message += "  Untick the dark-sky filter to include twilight and daylight."
                listbox.insert(tk.END, message)
                return

            for w in windows:
                best = self.in_observer_clock(w["best"])
                start = self.in_observer_clock(w["start"])
                end = self.in_observer_clock(w["end"])
                span = f"{start:%m-%d %H:%M} .. {end:%m-%d %H:%M}   "
                if libration:
                    listbox.insert(tk.END,
                                   f"{best:%Y-%m-%d %a %H:%M}  {span}"
                                   f"{w['earth_alt']:>9.2f}°{w['libr_long']:>+8.2f}°"
                                   f"{w['libr_lat']:>+8.2f}°{w['sun_alt']:>9.1f}°"
                                   f"{w['moon_alt']:>9.0f}°  {self._sky_of(w)}")
                else:
                    listbox.insert(tk.END,
                                   f"{best:%Y-%m-%d %a %H:%M}  {w['event']:<9}{span}"
                                   f"{w['sun_alt']:>6.1f}°{w['moon_alt']:>9.0f}°  {self._sky_of(w)}")
            listbox.selection_set(0)

        def go_to(event=None):
            selection = listbox.curselection()
            if not windows or not selection or selection[0] >= len(windows):
                return
            target = self.in_observer_clock(windows[selection[0]]["best"])
            self._go_to_moment(target)
            self.center_on_feature(feature)

        listbox.bind('<Double-Button-1>', go_to)
        listbox.bind('<Return>', go_to)

        def results_for_export():
            """
            The listed windows as a table and as calendar entries.

            Times are written on the observer's clock, as the list shows them,
            except in the calendar, where _ics_text puts them in UTC.
            """
            libration = mode_var.get() == "libration"
            columns = ["Best time", "Window start", "Window end"]
            columns += (["Presented (deg)", "Libration long", "Libration lat"] if libration
                        else ["Event"])
            columns += ["Sun over feature (deg)", "Moon altitude (deg)", "Sky"]
            rows, events = [], []
            for w in windows:
                best = self.in_observer_clock(w["best"])
                start = self.in_observer_clock(w["start"])
                end = self.in_observer_clock(w["end"])
                row = [f"{best:%Y-%m-%d %H:%M}", f"{start:%Y-%m-%d %H:%M}", f"{end:%Y-%m-%d %H:%M}"]
                if libration:
                    row += [f"{w['earth_alt']:.2f}", f"{w['libr_long']:+.2f}", f"{w['libr_lat']:+.2f}"]
                    headline = f"{feature.name} best presented ({w['earth_alt']:.0f} deg from the limb)"
                else:
                    row += [w["event"]]
                    headline = f"{feature.name} at the terminator ({w['event']})"
                row += [f"{w['sun_alt']:.1f}", f"{w['moon_alt']:.0f}", self._sky_of(w)]
                rows.append(row)
                events.append({
                    "summary": f"MoonRTX: {headline}",
                    "start": w["start"], "end": w["end"],
                    "description": (f"{feature.name} at lat {feature.lat:.2f}, lon {feature.lon:.2f}. "
                                    f"Best at {best:%Y-%m-%d %H:%M}. "
                                    f"Sun {w['sun_alt']:.1f} deg over the feature, Moon "
                                    f"{w['moon_alt']:.0f} deg up in a {self._sky_of(w)} sky."),
                })
            return columns, rows, events

        def open_graph():
            # This window closes first: Tk allows one grab at a time, so it and
            # the graph cannot both be modal and both be usable
            on_close()
            self.feature_graph_dialog(feature)

        self._results_actions(
            dialog, go_to, results_for_export,
            lambda: f"{feature.name.replace(' ', '_')}_{mode_var.get()}",
            extra=("Graph...", open_graph))

        rescan()

        self._show_dialog(win)

    def feature_graph_dialog(self, feature: MoonFeature):
        """
        Plot Sun altitude and libration presentation for a feature across the
        planner's scan span, rather than the discrete windows the planner
        reduces them to - so a trend (a shallowing terminator pass, a
        libration peak drifting later each month) shows at a glance.

        Local visibility is drawn as a Sky/Moon ribbon: the Moon's altitude
        swings through a full cycle about once a day, which over a span of
        weeks would draw as a scribble rather than a trend - see
        astro.sample_feature_series. It is drawn as a third curve as well while
        its own box is ticked, at any span: over weeks the daily arcs run
        together into a band, which is worth seeing when it is asked for.
        Clicking the plot jumps the view to that moment, as the other planning
        dialogs do.
        """
        if self.rt is None or feature is None:
            return

        def before_close():
            if label_var.get():
                self.unpin_catalogue_feature(feature)
            # Where the window was left, for the next graph to open at. Taken
            # from the geometry string rather than winfo_x and winfo_y, that
            # being what _show_dialog writes back, so the corner it is put at is
            # the corner it was read from. "WxH+X+Y", and a negative coordinate
            # keeps its sign inside its own group ("+-8+-3")
            try:
                _, _, corner = win.geometry().partition("+")
                x, _, y = corner.partition("+")
                self._graph_position = (int(x), int(y))
            except ValueError:      # never mapped, so there is nothing to keep
                pass

        win, main_frame, on_close = self._dialog_window(
            f"{feature.name} - graph", before_close=before_close)

        colours = self.GRAPH_COLOURS
        font = ('Consolas', 8)
        metrics = tkfont.Font(font=font)
        cell_w = metrics.measure('0')
        line_h = metrics.metrics('linespace')
        pad = max(2, round(cell_w * 2 / 3))
        line_w = max(1, cell_w // 5)
        rule = max(1, cell_w // 3)

        plot_h = self.GRAPH_PLOT_LINES * line_h
        ribbon_h = line_h
        label_w = max(metrics.measure('-90°'), metrics.measure('Moon')) + 2 * pad

        # The plot fills a fixed share of the screen rather than a fixed
        # number of pixels per day: at 90% of screen width every day gets as
        # much room to click as the display allows, on a small window and a
        # 4K one alike, rather than the window's total size following the
        # screen only by accident of the font metrics scaling with it.
        width = round(win.winfo_screenwidth() * self.GRAPH_WIDTH_FRACTION)
        plot_w = max(self.GRAPH_DAYS, width - label_w - pad)

        plot_x0, plot_x1 = label_w, label_w + plot_w
        # A line of room above the +90° line, for the time under the pointer
        plot_y0 = pad + line_h
        plot_y1 = plot_y0 + plot_h
        # Two thin strips straight under the plot's bottom edge for the
        # planner's windows - marks on the time axis rather than rows of their
        # own, so they take no labels
        strip_h = max(3, line_h // 3)
        term_y0, term_y1 = plot_y1, plot_y1 + strip_h
        libr_y0, libr_y1 = term_y1, term_y1 + strip_h
        sky_y0, sky_y1 = libr_y1 + pad, libr_y1 + pad + ribbon_h
        moon_y0, moon_y1 = sky_y1, sky_y1 + ribbon_h
        date_y = moon_y1 + pad
        width = plot_x1 + pad
        height = date_y + line_h + pad

        # The choices are gathered in a block of their own, which fit_rows puts
        # beside the readout or on a row above it once both have been measured
        controls_row = tk.Frame(main_frame)
        # What a click on the graph does to the camera, besides moving the clock.
        # Themed, as in the launcher, so the little circles follow the display:
        # Tk's own are drawn at much the same size whatever the screen, which on
        # a 4K one at 300% is a tenth the height of the lettering beside them
        view_var = tk.StringVar(value=self._graph_view)
        view_row = tk.Frame(controls_row)
        view_row.pack(side=tk.RIGHT)
        view_buttons = []
        for value, text in (("keep", "Keep view"), ("standard", "Standard view"),
                            ("centre", "View fixed on feature")):
            radio = ttk.Radiobutton(view_row, text=text, value=value, variable=view_var,
                                    command=lambda: apply_view())
            radio.pack(side=tk.LEFT)
            view_buttons.append(radio)
        # The Moon's altitude in the observer's sky, drawn as a curve while this
        # is ticked, at every span: over weeks its daily arcs run together into
        # a band rather than a trend, which is worth seeing if it is asked for.
        # Packed before the name's box, and so standing to the right of it: with
        # side=RIGHT the first packed is the furthest over
        moon_alt_var = tk.BooleanVar(value=self._graph_show_moon_alt)

        def toggle_moon_alt():
            self._graph_show_moon_alt = moon_alt_var.get()
            redraw()

        ttk.Checkbutton(controls_row, text="Show Moon altitude", variable=moon_alt_var,
                        command=toggle_moon_alt).pack(side=tk.RIGHT, padx=(0, 2 * cell_w))
        # The feature's name on the Moon, pinned into the catalogue while this
        # is ticked, so it is drawn the way the P key draws names and never
        # twice; unpinned again when the window closes - see CatalogueMixin
        label_var = tk.BooleanVar(value=self._graph_show_name)

        def toggle_name():
            self._graph_show_name = label_var.get()
            if label_var.get():
                self.pin_catalogue_feature(feature)
            else:
                self.unpin_catalogue_feature(feature)

        ttk.Checkbutton(controls_row, text="Show feature name", variable=label_var,
                        command=toggle_name).pack(side=tk.RIGHT, padx=(0, 2 * cell_w))
        # Left ticked last time, so named from the start this time
        if label_var.get():
            self.pin_catalogue_feature(feature)

        # The readout of the moment under the pointer, or of the red line while
        # the pointer is away, on a row between the choices and the plot; the
        # graph's own title is in the window's title bar (set on each redraw, as
        # it says the span). Indented by the plot's left margin so it starts
        # where the time over the +90° line starts at its leftmost, and with no
        # padding or border of its own, which would put the lettering further in
        readout_row = tk.Frame(main_frame)
        readout_row.pack(fill=tk.X)
        status_var = tk.StringVar()
        tk.Label(readout_row, textvariable=status_var, font=font, padx=0, borderwidth=0,
                 fg=colours["readout"], anchor='w').pack(side=tk.LEFT, padx=(plot_x0, 0))

        # Against the left edge, so the plot's left margin measures from the same
        # place as the readout's indent even when the window is wider than the canvas
        canvas = tk.Canvas(main_frame, width=width, height=height,
                           highlightthickness=0, bg=win.cget('bg'))
        canvas.pack(pady=(4, 0), anchor='w')

        # "days" is the span on show, "step" its sampling in minutes and "day_w"
        # the width a day takes at that span - see set_span
        def step_for(days: int) -> int:
            """Sampling step for a span: GRAPH_STEP_MINUTES at the full span, finer in proportion."""
            return max(1, self.GRAPH_STEP_MINUTES * days // self.GRAPH_DAYS)

        state = {"start": None, "dts": [], "now_line": None, "hover_line": None, "hover_time": None,
                 "days": self._graph_span, "step": step_for(self._graph_span),
                 "day_w": plot_w / self._graph_span}

        def x_of(moment_utc) -> float:
            return (plot_x0 + (moment_utc - state["dts"][0]).total_seconds() / 86400.0
                    * state["day_w"])

        def clip_x(x: float) -> float:
            return min(max(x, plot_x0), plot_x1)

        alt_span = 90.0 - self.GRAPH_ALT_MIN

        def y_of(deg: float) -> float:
            return plot_y0 + (90.0 - deg) / alt_span * plot_h

        def band(y0: float, y1: float, spells: list, fill: str, min_w: float = 0.0):
            for start_utc, end_utc in spells:
                x0 = clip_x(x_of(max(start_utc, state["dts"][0])))
                x1 = clip_x(x_of(min(end_utc, state["dts"][-1])))
                # A window of an hour or two comes to less than a pixel across a
                # span of weeks, and would otherwise draw nothing at all
                x1 = max(x1, min(x0 + min_w, plot_x1))
                if x1 > x0:
                    canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")

        def curve(values, colour: str):
            floor = self.GRAPH_ALT_MIN
            runs, run, prev = [], [], None
            for t_utc, deg in zip(state["dts"], values):
                x, deg = x_of(t_utc), float(deg)
                if prev is not None and (prev[1] < floor) != (deg < floor):
                    # Cut at the floor itself, so a run meets the edge of the
                    # plot rather than stopping a sample short of it
                    px, pdeg = prev
                    run += [px + (x - px) * (floor - pdeg) / (deg - pdeg), y_of(floor)]
                    if deg < floor:
                        runs.append(run)
                        run = []
                if deg >= floor:
                    run += [x, y_of(deg)]
                prev = (x, deg)
            runs.append(run)
            for run in runs:
                if len(run) >= 4:
                    canvas.create_line(*run, fill=colour, width=line_w)

        def redraw():
            canvas.delete('all')
            moon_curve = moon_alt_var.get()
            win.title(f"{feature.name}  (lat {feature.lat:.2f}°, lon {feature.lon:.2f}°)  -  "
                      f"Sun altitude and libration over {state['days']} "
                      f"{'day' if state['days'] == 1 else 'days'}")
            try:
                series = astro.sample_feature_series(
                    state["start"], state["days"], feature.lat, feature.lon,
                    step_minutes=state["step"])
                chart = astro.find_visibility_chart(state["start"], state["days"])
                # Asked with exactly the planner's own settings, so a strip
                # marks the same windows its list shows - including only the
                # best PLANNER_MAX_RESULTS in libration mode, as the list does
                term_windows = astro.find_terminator_windows(
                    state["start"], state["days"], feature.lat, feature.lon,
                    sun_alt_max=self.PLANNER_SUN_ALT_MAX,
                    moon_alt_min=self.PLANNER_MOON_ALT_MIN,
                    observer_sun_alt_max=self._planner_observer_sun_alt_max())
                libr_windows = astro.find_libration_windows(
                    state["start"], state["days"], feature.lat, feature.lon,
                    sun_alt_min=self.PLANNER_LIBRATION_SUN_ALT_MIN,
                    moon_alt_min=self.PLANNER_MOON_ALT_MIN,
                    max_results=self.PLANNER_MAX_RESULTS,
                    observer_sun_alt_max=self._planner_observer_sun_alt_max())
                # Dropped here as the planner's list drops them, the strips
                # standing for the windows that list holds (see _lasting)
                term_windows = self._lasting(term_windows)
                libr_windows = self._lasting(libr_windows)
            except ValueError as e:
                # Scan start outside the bundled ephemeris kernel range
                state["dts"] = []
                canvas.config(height=3 * line_h + 2 * pad)
                canvas.create_text(pad, pad, text=str(e), anchor='nw',
                                   width=width - 2 * pad, font=font)
                return

            state["dts"] = series["times"]
            canvas.config(height=height)

            for deg in range(self.GRAPH_ALT_MIN, 91, 30):
                y = y_of(deg)
                canvas.create_line(plot_x0, y, plot_x1, y, fill=colours["grid"])
                canvas.create_text(plot_x0 - pad, y, anchor='e', font=font,
                                   text=f"{deg:+d}°" if deg else "0°")
            canvas.create_line(plot_x0, y_of(0.0), plot_x1, y_of(0.0),
                               fill=colours["zero"], width=2)
            y_thr = y_of(self.PLANNER_SUN_ALT_MAX)
            canvas.create_line(plot_x0, y_thr, plot_x1, y_thr,
                               fill=colours["threshold"], dash=(4, 2))

            # A line and a date at local midnight - the first on or after the start
            # of the span, then every tick_days - so a date stands for the start of
            # its day rather than for whatever hour the span happened to begin at.
            # Each midnight is made from its own date on the observer's clock, so
            # it stays on 00:00 across a daylight saving change
            tick_days = max(1, state["days"] // 10)
            span_end = state["dts"][-1]
            day = self.in_observer_clock(state["dts"][0]).date()
            if self.from_observer_clock(datetime.combine(day, datetime.min.time())) < state["dts"][0]:
                day += timedelta(days=1)
            first = True
            while True:
                moment = self.from_observer_clock(datetime.combine(day, datetime.min.time()))
                if moment > span_end:
                    break
                x = x_of(moment)
                canvas.create_line(x, plot_y0, x, moon_y1, fill=colours["grid"])
                if first:
                    # The first date carries the year, and is started at its line
                    # rather than centred on it: wider than the others, it would
                    # otherwise run off the canvas with its line near the left edge
                    canvas.create_text(x, date_y, anchor='nw', font=font,
                                       text=f"{day:%d %b %Y}")
                    first = False
                else:
                    text = f"{day:%d %b}"
                    # Ended at its line instead near the right edge, for the same reason
                    anchor = 'ne' if x + metrics.measure(text) / 2 > width else 'n'
                    canvas.create_text(x, date_y, anchor=anchor, font=font, text=text)
                day += timedelta(days=tick_days)

            canvas.create_rectangle(plot_x0, sky_y0, plot_x1, sky_y1,
                                    fill=colours["night"], outline="")
            canvas.create_rectangle(plot_x0, moon_y0, plot_x1, moon_y1,
                                    fill=colours["night"], outline="")
            band(sky_y0, sky_y1, chart.sun_twilight, colours["twilight"])
            band(sky_y0, sky_y1, chart.sun_up, colours["day"])
            band(moon_y0, moon_y1, chart.moon_up, colours["moon"])
            band(term_y0, term_y1, [(w["start"], w["end"]) for w in term_windows],
                 colours["terminator_window"], min_w=1)
            band(libr_y0, libr_y1, [(w["start"], w["end"]) for w in libr_windows],
                 colours["libration_window"], min_w=1)
            canvas.create_text(plot_x0 - pad, (sky_y0 + sky_y1) / 2, text="Sky",
                               anchor='e', font=font)
            canvas.create_text(plot_x0 - pad, (moon_y0 + moon_y1) / 2, text="Moon",
                               anchor='e', font=font)
            canvas.create_rectangle(plot_x0, plot_y0, plot_x1, moon_y1, outline=colours["grid"])

            # The Moon's first, so its daily arcs pass under the two slow curves
            # rather than breaking them up
            if moon_curve:
                curve(series["moon_alt"], colours["moon_alt"])
            curve(series["sun_alt"], colours["sun_alt"])
            curve(series["earth_alt"], colours["earth_alt"])

            # Everything was deleted above, both lines and the pointer's time with them
            state["now_line"] = None
            state["hover_line"] = None
            state["hover_time"] = None
            place_now_line()
            # Put the pointer's line and time back if the pointer is still over the
            # graph - after a click, a page or a change of span - rather than
            # leaving them gone until it next moves
            if pointer["x"] is not None:
                place_hover(pointer["x"], pointer["y"])
            else:
                show_now_readout()

        def place_now_line():
            """
            Draw the red line at the moment the app is showing, moving it from
            wherever it stood, or leave it off when that moment is outside the
            span. On its own, so a step in time moves the line without the whole
            graph - and its searches - being done again.
            """
            if state["now_line"] is not None:
                canvas.delete(state["now_line"])
                state["now_line"] = None
            now_utc = self.dt_local.astimezone(timezone.utc)
            if state["dts"] and state["dts"][0] <= now_utc <= state["dts"][-1]:
                x = x_of(now_utc)
                state["now_line"] = canvas.create_line(x, plot_y0, x, moon_y1,
                                                       fill=colours["today"], width=rule)

        def apply_view():
            """
            Put the camera where the view choice says: left as it is, back on
            the standard view the End key gives, or on the feature.

            Left as it is, the camera still follows the Moon's apparent size for
            the new date, as a Q or W step does; the Moon turns under it all the
            same, by libration and, out of parallactic mode, with the hour.

            On the feature it is centred even when turned past the limb, the
            camera swinging round to face it on the far side: center_on_lat_lon
            approaches any point along its own outward normal, so that is as
            sound a view as any on the near side.
            """
            mode = view_var.get()
            self._graph_view = mode
            if mode == "standard":
                self.reset_to_default_view()
            elif mode == "centre":
                self.center_on_feature(feature)

        def go_to(event):
            # A click on a canvas does not take the keyboard focus in Tk, so after
            # a change in the step box the arrows would go on moving its cursor;
            # taken here, they step the time again
            win.focus_set()
            if not state["dts"] or not (plot_x0 <= event.x <= plot_x1) \
                    or not (plot_y0 <= event.y <= moon_y1):
                return
            target = state["dts"][0] + timedelta(days=(event.x - plot_x0) / state["day_w"])
            self._go_to_moment(self.in_observer_clock(target))
            # Only the red line follows the clock, and a click inside the span
            # keeps it inside, so the line is moved as a step moves it rather
            # than the graph and its searches being done again
            place_now_line()
            apply_view()

        canvas.bind('<Button-1>', go_to)

        pointer = {"x": None, "y": None, "pending": False}

        def place_time(x, moment_local):
            """
            The time over the +90° line, centred on x but kept clear of the
            "+90°" label on the left and of the canvas edge on the right.

            It stands over whichever line the readout is speaking for: the grey
            one under the pointer, or the red one when the pointer is away.
            """
            text = f"{moment_local:%d %b %a %H:%M}"
            half = metrics.measure(text) / 2
            cx = max(plot_x0 + half, min(width - half, x))
            if state["hover_time"] is None:
                state["hover_time"] = canvas.create_text(
                    cx, plot_y0 - 1, anchor='s', font=font, fill=colours["readout"], text=text)
            else:
                canvas.coords(state["hover_time"], cx, plot_y0 - 1)
                canvas.itemconfigure(state["hover_time"], text=text)

        def clear_time():
            if state["hover_time"] is not None:
                canvas.delete(state["hover_time"])
                state["hover_time"] = None

        def place_hover(x, y):
            """
            Draw the grey line through the pointer and the time over the +90°
            line, or take them away when the pointer is off the plot.

            On its own so a redraw can put them back where the pointer still is:
            a redraw clears the whole canvas, and they would otherwise stay gone
            until the pointer next moved.
            """
            # A vertical line through the pointer, moved at once rather than on
            # idle: shifting one canvas line is cheap, and a line lagging the
            # pointer would be the thing noticed. Below the red line for the
            # moment on show, which stays on top
            if state["dts"] and plot_x0 <= x <= plot_x1 and plot_y0 <= y <= moon_y1:
                if state["hover_line"] is None:
                    state["hover_line"] = canvas.create_line(
                        x, plot_y0, x, moon_y1, fill=colours["grid"])
                else:
                    canvas.coords(state["hover_line"], x, plot_y0, x, moon_y1)
                if state["now_line"] is not None:
                    canvas.tag_raise(state["now_line"])
                # The time at the pointer, over the +90° line on top of the grey
                # line. Only the pointer's place is needed for it, so it moves
                # with the line rather than waiting for the readout
                place_time(x, self.in_observer_clock(
                    state["dts"][0] + timedelta(days=(x - plot_x0) / state["day_w"])))
            else:
                # Only the grey line goes. The time and the line of figures are
                # left to read_out, which falls back to the red line's moment
                if state["hover_line"] is not None:
                    canvas.delete(state["hover_line"])
                    state["hover_line"] = None

        def hover(event):
            """
            Note where the pointer is, and read out that moment once Tk is idle.

            The readout is worked out exactly for the moment rather than taken
            between the graph's samples: the Moon's altitude moves fast and
            bends near the horizon, and between two-hour samples it was out by
            degrees. One exact calculation takes a few
            milliseconds, and asking for it when idle rather than on every
            motion event means a quick sweep of the pointer asks only once.
            """
            pointer["x"], pointer["y"] = event.x, event.y
            place_hover(event.x, event.y)
            if not pointer["pending"]:
                pointer["pending"] = True
                canvas.after_idle(read_out)

        def status_for(moment_utc) -> str:
            """
            The line for a moment - whether the feature can be seen then at all,
            and the three figures that answer it: the Sun over the feature, the
            libration and the Moon's altitude in the observer's sky - so a click
            can be aimed rather than guessed from the curves, with a short note
            by a value when it is the one standing in the way.

            Empty past the end of the bundled ephemeris kernels.
            """
            try:
                at = astro.sample_feature_series(moment_utc, 0, feature.lat, feature.lon)
            except ValueError:
                return ""
            # The moment's date and time are shown over the plot, by the pointer
            # A short note beside the Sun or the libration when it means the feature
            # cannot be seen: in lunar night or on the far side, where those curves
            # are below 0°. The Moon's altitude has none - its note flickered on and
            # off as the pointer crossed the thresholds. The Sun and libration parts
            # are padded to the widest they get, note and all, so in the fixed-pitch
            # status font each value after them starts in the same place
            sun = float(at["sun_alt"][0])
            libration = float(at["earth_alt"][0])
            moon = float(at["moon_alt"][0])
            sun_part = f"{sun:+5.1f}°{' (lunar night)' if sun < 0.0 else ''}"
            sun_width = len(f"{-90.0:+5.1f}° (lunar night)")
            libration_part = f"{libration:+5.1f}°{' (far side)' if libration < 0.0 else ''}"
            libration_width = len(f"{-90.0:+5.1f}° (far side)")
            # The three answers in one word, at the end of the line behind the
            # figures that decide it: the feature is seen when the Sun is up over
            # it, when the libration has it turned toward Earth, and when the
            # Moon itself is above the observer's horizon. A daylit sky is not
            # counted against it - the Moon is watched by day as well, with the
            # fainter detail washed out.
            # Right-aligned in the room "not visible" takes, so that "visible"
            # itself stands in the same place whether or not the "not" is there,
            # and the figures behind it stand still too
            seen = "visible" if sun > 0.0 and libration > 0.0 and moon > 0.0 else "not visible"
            # The Moon's figure carries no note of its own, so on its own it
            # would leave the verdict a bare separator away while the figures
            # before it stand a note's width apart. It is padded to the room the
            # libration's takes, so the gap before the verdict reads as the gaps
            # between the figures do
            moon_part = f"{moon:+5.1f}°"
            return (f"Sun over {feature.name}: "
                    f"{sun_part:<{sun_width}}          Libration: {libration_part:<{libration_width}}"
                    f"          Moon alt: {moon_part:<{libration_width}}"
                    f"          {feature.name} is {seen:>{len('not visible')}}")

        def show_now_readout():
            """
            Put the time over the plot and the figures in the line for the red
            line - the moment the renderer is showing - which is what they stand
            for whenever the pointer is away from the plot: as the window opens,
            after a step of the clock, and once the pointer has left. Both are
            cleared when that moment is outside the span on show, there being no
            red line then to speak for.
            """
            now_utc = self.dt_local.astimezone(timezone.utc)
            if not state["dts"] or not (state["dts"][0] <= now_utc <= state["dts"][-1]):
                clear_time()
                status_var.set("")
                return
            place_time(x_of(now_utc), self.in_observer_clock(now_utc))
            status_var.set(status_for(now_utc))

        def read_out():
            """
            Fill the line for the moment under the pointer, or for the red line
            when the pointer is off the plot. Called when Tk is idle - see hover.
            """
            pointer["pending"] = False
            if not canvas.winfo_exists():
                return
            x, y = pointer["x"], pointer["y"]
            if x is None or not state["dts"] or not (plot_x0 <= x <= plot_x1) \
                    or not (plot_y0 <= y <= moon_y1):
                show_now_readout()
                return
            status_var.set(status_for(
                state["dts"][0] + timedelta(days=(x - plot_x0) / state["day_w"])))

        canvas.bind('<Motion>', hover)

        def leave(event):
            # Forget the pointer, so a readout still waiting for idle does not
            # put the grey line back once the pointer has gone; the time and the
            # figures stay up, now speaking for the red line
            pointer["x"] = None
            if state["hover_line"] is not None:
                canvas.delete(state["hover_line"])
                state["hover_line"] = None
            show_now_readout()

        canvas.bind('<Leave>', leave)

        def step_time(direction: int):
            """
            Step the time back or on by the step set in this window, which the
            renderer's own Q/W step does not share - the renderer's keys being
            held while this window is open.

            Taken the same way as Q/W: nothing while a video export owns the
            clock, and the single-frame preview switched on first, so a held
            key shows each step rather than stalling on converged frames. The
            red line moves on its own; only a step past either end of the span
            pages the graph, by the span, as the arrow buttons do. The view then
            follows the choice above, as it does for a click.
            """
            # In the step box the arrows move the cursor, not the time
            try:
                if win.focus_get() is step_box:
                    return None
            except KeyError:
                pass
            if self._video_export is not None:
                return "break"
            self._begin_interactive_preview()
            self.change_time(direction * graph_step())
            now_utc = self.dt_local.astimezone(timezone.utc)
            if state["dts"] and not (state["dts"][0] <= now_utc <= state["dts"][-1]):
                state["start"] += timedelta(days=direction * state["days"])
                redraw()
            else:
                place_now_line()
                # The time and the figures follow the red line while the pointer
                # is away; under the pointer they stand for where it is
                if pointer["x"] is None:
                    show_now_readout()
            apply_view()
            # The renderer's own arrow keys move the view; not these
            return "break"

        win.bind('<Left>', lambda event: step_time(-1))
        win.bind('<Right>', lambda event: step_time(1))

        legend = tk.Frame(main_frame)
        legend.pack(fill=tk.X, pady=(2 * pad, 0))
        # Each curve carries what it means as a hint: the Sun's and the
        # libration's are altitudes above the feature's own horizon rather than
        # anything an eyepiece shows directly, and the libration one is a single
        # figure standing for what an almanac prints as two (see
        # astro.sample_feature_series).
        sun_hint = (
            "Sun altitude over the feature, which sets how long its shadows are.\n"
            "0° is sunrise or sunset there; below that the feature is in lunar\n"
            f"night. The dashed line is the {self.PLANNER_SUN_ALT_MAX:.0f}° the observation planner\n"
            "takes as the top of the terminator window.")
        libration_hint = (
            "How far inside the limb the feature lies, as seen from your site:\n"
            "the altitude of the Earth above the feature's own horizon. 90° is\n"
            "the centre of the disk, 0° exactly on the limb, and below 0° it has\n"
            "turned onto the far side, out of sight.\n"
            "The lower it is, the more the feature is squashed toward the limb:\n"
            "at 30° it looks half as wide as at the centre, at 10° about a sixth.")
        moon_alt_hint = (
            "The Moon's altitude in your sky, which swings through a cycle a\n"
            "day: at the shorter spans each day's arc shows how high it gets,\n"
            "and over weeks the arcs run together into a band.\n"
            "Drawn while Show Moon altitude is ticked.\n"
            f"The observation planner's windows need it at least {self.PLANNER_MOON_ALT_MIN:.0f}° up.")
        dash_hint = (
            "The top of the band the observation planner counts as near the\n"
            f"terminator: it lists the times the Sun stands between 0° and {self.PLANNER_SUN_ALT_MAX:.0f}°\n"
            "over the feature (see find_terminator_windows).\n"
            "Below 0° the feature is in lunar night. Between 0° and this line\n"
            "the Sun is low over it, so it is lit with long shadows and its\n"
            "relief stands out. Above the line the Sun climbs, the shadows\n"
            "shorten and the detail flattens.\n"
            f"{self.PLANNER_SUN_ALT_MAX:.0f}° is about one Earth day past sunrise there, the Sun\n"
            "crossing a lunar location at some 0.5° an hour.\n"
            "It is a mark for the Sun curve alone - the libration curve shares\n"
            "the same axis, and crosses it meaning nothing.")
        # The Sky ribbon's two lighter bands, the darkest being night itself and
        # left unlabelled: it is the ribbon's own background, not a band drawn on it
        twilight_hint = (
            "The Sky ribbon: the Sun between your horizon and "
            f"{abs(astro.ASTRONOMICAL_TWILIGHT_DEGREES):.0f}° below it.\n"
            "Darker than that is night, the ribbon's background, which is what\n"
            "the observation planner's dark-sky filter asks for.")
        daylight_hint = (
            "The Sky ribbon: the Sun above your horizon.\n"
            "The Moon can be watched by day - it is the sky's brightness, not\n"
            "the Moon's, that washes the fainter detail out.")
        # The planner's dark-sky filter is set there and cannot change while this
        # window is open, the planner being closed, so it is read once here
        dark_hint = ("\nOnly while the sky is dark, as the planner is set to."
                     if self._planner_dark_only else "")
        terminator_window_hint = (
            "Windows the Observation Planner lists under \"near the terminator\":\n"
            f"the Sun 0-{self.PLANNER_SUN_ALT_MAX:.0f}° over the feature, the feature turned toward\n"
            f"Earth and the Moon at least {self.PLANNER_MOON_ALT_MIN:.0f}° up in your sky."
            f"{dark_hint}")
        libration_window_hint = (
            "Windows the Observation Planner lists under \"best presented (libration)\":\n"
            f"the feature turned toward Earth, the Sun at least {self.PLANNER_LIBRATION_SUN_ALT_MIN:.0f}° "
            f"over it and the Moon at least {self.PLANNER_MOON_ALT_MIN:.0f}° up.\n"
            f"Only its {self.PLANNER_MAX_RESULTS} best, as in the list.{dark_hint}")
        def line_entry(text: str, colour: str, hint: str):
            """A curve's entry: a bar of its colour, then what it is."""
            swatch = tk.Frame(legend, bg=colour, width=2 * cell_w, height=line_w + 2,
                              highlightthickness=0)
            swatch.pack(side=tk.LEFT)
            swatch.pack_propagate(False)
            label = tk.Label(legend, text=text, font=font)
            label.pack(side=tk.LEFT, padx=(max(1, cell_w // 2), cell_w + pad))
            # On the lettering as well as the swatch, which is a few pixels tall
            ToolTip(swatch, hint)
            ToolTip(label, hint)

        def band_entry(text: str, colour: str, hint: str = None):
            """A ribbon's or a window strip's entry: a small filled box, then what it is."""
            swatch = tk.Frame(legend, bg=colour, width=line_w + 4, height=line_h - pad,
                              highlightthickness=1, highlightbackground="#808080")
            swatch.pack(side=tk.LEFT)
            swatch.pack_propagate(False)
            label = tk.Label(legend, text=text, font=font)
            label.pack(side=tk.LEFT, padx=(max(1, cell_w // 2), cell_w + pad))
            if hint:
                ToolTip(swatch, hint)
                ToolTip(label, hint)

        # Read left to right, working inward from the observer: the sky over the
        # site, the Moon in it - when it is up and how high - then the Sun over
        # the feature and the libration, each beside the window it decides, and
        # last the line that caps the terminator window
        band_entry("Daylight", colours["day"], daylight_hint)
        band_entry("Twilight", colours["twilight"], twilight_hint)
        band_entry("Moon up", colours["moon"])
        line_entry("Moon altitude", colours["moon_alt"], moon_alt_hint)
        line_entry("Sun over feature", colours["sun_alt"], sun_hint)
        line_entry("Libration (Earth alt)", colours["earth_alt"], libration_hint)
        band_entry("Libration window", colours["libration_window"], libration_window_hint)
        band_entry("Terminator window", colours["terminator_window"], terminator_window_hint)
        # The threshold line is dashed, which a coloured Frame cannot show, so
        # its sample is drawn the way the line itself is drawn on the plot
        dash_h = line_w + 2
        dash_swatch = tk.Canvas(legend, width=2 * cell_w, height=dash_h,
                                highlightthickness=0, bg=legend.cget('bg'))
        dash_swatch.pack(side=tk.LEFT)
        dash_swatch.create_line(0, dash_h / 2, 2 * cell_w, dash_h / 2,
                                fill=colours["threshold"], dash=(4, 2), width=line_w)
        dash_label = tk.Label(legend, font=font,
                              text=f"Terminator window top ({self.PLANNER_SUN_ALT_MAX:.0f}°)")
        dash_label.pack(side=tk.LEFT, padx=(max(1, cell_w // 2), cell_w + pad))
        ToolTip(dash_swatch, dash_hint)
        ToolTip(dash_label, dash_hint)
        # Everything this window does, under one word rather than across the
        # row: the legend has no room for a line of instructions, and what is
        # worth saying about the wheel, the keys and the buttons is longer than
        # a line anyway
        help_hint = (
            "Click the graph to go to that moment.\n"
            "← → step the time by Step (min), and page the graph by a whole\n"
            "span when a step passes either end of it.\n"
            "The mouse wheel over the graph zooms the time axis, 60 days down\n"
            "to 1, keeping the moment under the pointer where it is.\n"
            "◀ ▶ page the graph a span back or on. Reset centres it on the\n"
            "moment on show.\n"
            "Span (days) sets the axis to 60, 15 or 5 days about that moment;\n"
            "between those, the wheel's spans show no button chosen.\n"
            "Step (min) is this window's own time step - the renderer's own\n"
            "Q/W step is left as it is.\n"
            "Keep view / Standard view / View fixed on feature say what a\n"
            "click or a step does to the camera.\n"
            "Show feature name puts the name on the Moon.\n"
            "Show Moon altitude draws the Moon's altitude in your sky, whose\n"
            "daily arcs read best at the shorter spans.")
        help_label = tk.Label(legend, text="Help", font=font, fg='#606060')
        help_label.pack(side=tk.LEFT)
        ToolTip(help_label, help_hint)

        def page(days: int):
            state["start"] += timedelta(days=days)
            redraw()

        def set_span():
            """
            Stretch or squeeze the time axis to the span chosen, sampling finer
            the shorter it is. The new span is centred on the moment the app is
            showing, so a day picked at the wide span can be zoomed into and an
            hour picked in it.
            """
            days = span_var.get()
            self._graph_span = days
            state["days"] = days
            state["step"] = step_for(days)
            state["day_w"] = plot_w / days
            state["start"] = self.dt_local - timedelta(days=days / 2)
            redraw()

        def reset():
            state["start"] = self.dt_local
            redraw()

        # The span, the step and the buttons in a block of their own, which
        # fit_rows puts at the end of the legend's row where there is room for
        # it, and on a row under the legend where there is not
        legend_controls = tk.Frame(main_frame)
        tk.Button(legend_controls, text="Close", command=on_close, width=10).pack(side=tk.RIGHT)
        tk.Button(legend_controls, text="Reset", command=reset, width=10).pack(
            side=tk.RIGHT, padx=(0, pad + 2))
        tk.Button(legend_controls, text="▶", width=2,
                  command=lambda: page(state["days"])).pack(side=tk.RIGHT, padx=(0, pad + 2))
        tk.Button(legend_controls, text="◀", width=2,
                  command=lambda: page(-state["days"])).pack(side=tk.RIGHT)
        # Themed, as the other choices here, so the circles follow the display
        span_var = tk.IntVar(value=self._graph_span)
        span_row = tk.Frame(legend_controls)
        span_row.pack(side=tk.RIGHT, padx=(0, 2 * cell_w))
        tk.Label(span_row, text="Span (days):", font=font).pack(side=tk.LEFT)
        for days in self.GRAPH_SPANS:
            ttk.Radiobutton(span_row, text=str(days), value=days, variable=span_var,
                            command=set_span).pack(side=tk.LEFT)
        # The arrow keys' step for this window alone, beside the span: the
        # renderer's Q/W step (time_step_minutes) is never touched by it. Packed
        # from the right after the span, so it stands just left of it. Themed, as
        # the other controls here, so it follows the display
        step_var = tk.StringVar(value=str(self._graph_step_minutes or self.time_step_minutes))
        step_row = tk.Frame(legend_controls)
        step_row.pack(side=tk.RIGHT, padx=(0, 2 * cell_w))
        tk.Label(step_row, text="Step (min):", font=font).pack(side=tk.LEFT)
        step_box = ttk.Spinbox(step_row, from_=1, to=1440, increment=1, width=5,
                               textvariable=step_var)
        step_box.pack(side=tk.LEFT, padx=(max(1, cell_w // 2), 0))

        def graph_step() -> int:
            """
            The step in the box, in whole minutes from 1 to 1440 as the
            renderer's own step allows - put back to the last good value if what
            is there is not a number - and kept for the rest of the session.
            """
            try:
                minutes = int(step_var.get())
            except ValueError:
                minutes = self._graph_step_minutes or self.time_step_minutes
            minutes = max(1, min(1440, minutes))
            self._graph_step_minutes = minutes
            step_var.set(str(minutes))
            return minutes

        step_box.bind('<FocusOut>', lambda event: graph_step())
        # Enter settles the value and hands the arrows back to the time
        step_box.bind('<Return>', lambda event: (graph_step(), win.focus_set()))

        # The mouse wheel zooms the time axis through GRAPH_ZOOM_SPANS, keeping
        # the moment under the pointer where it is. Bound on the window rather
        # than the canvas, since Tk on Windows can hand the wheel to whichever
        # widget has the focus, and only a turn over the graph zooms. Notches
        # are gathered and the zoom done when Tk is idle, as a redraw takes a
        # tenth to a third of a second: a quick spin redraws once, not per notch
        wheel = {"turn": 0, "notches": 0, "x": plot_x0, "pending": False}

        def zoom(event):
            x = event.x_root - canvas.winfo_rootx()
            y = event.y_root - canvas.winfo_rooty()
            if not state["dts"] or not (0 <= x < canvas.winfo_width()) \
                    or not (0 <= y < canvas.winfo_height()):
                return None
            # A notch is 120; a touchpad sends it in smaller pieces
            wheel["turn"] += event.delta
            notches = int(wheel["turn"] / 120)
            wheel["turn"] -= notches * 120
            if notches:
                wheel["notches"] += notches
                wheel["x"] = clip_x(x)
                if not wheel["pending"]:
                    wheel["pending"] = True
                    canvas.after_idle(zoomed)
            return "break"

        def zoomed():
            wheel["pending"] = False
            if not canvas.winfo_exists():
                return
            notches, wheel["notches"] = wheel["notches"], 0
            spans = self.GRAPH_ZOOM_SPANS
            # Turned away from you, the wheel zooms in - toward the shorter spans
            at = min(range(len(spans)), key=lambda k: abs(spans[k] - state["days"]))
            days = spans[max(0, min(len(spans) - 1, at + notches))]
            if days == state["days"]:
                return
            share = (wheel["x"] - plot_x0) / plot_w
            moment = state["start"] + timedelta(days=share * state["days"])
            self._graph_span = days
            span_var.set(days)   # no button chosen when the span is none of theirs
            state["days"] = days
            state["step"] = step_for(days)
            state["day_w"] = plot_w / days
            state["start"] = moment - timedelta(days=share * days)
            redraw()

        win.bind('<MouseWheel>', zoom)

        def fit_rows():
            """
            Put each block of controls at the end of the row it belongs to, or on
            a row of its own against the left edge where the two together would
            be wider than the plot.

            Measured rather than settled once and for all: the plot takes a share
            of the screen while the lettering follows the display's dots per inch,
            so a row that sits comfortably at 100% on a FullHD screen runs off a
            4K one at 300%, the lettering being three times the size where the
            screen is only twice as wide.

            Called after the first draw, the readout carrying a line by then -
            and every line it carries is padded to the same width, so one of them
            measures them all (see status_for).
            """
            for block, row, above in ((controls_row, readout_row, True),
                                      (legend_controls, legend, False)):
                block.pack_forget()
                main_frame.update_idletasks()
                if row.winfo_reqwidth() + block.winfo_reqwidth() <= width:
                    block.pack(in_=row, side=tk.RIGHT)
                    # The block belongs to the frame these rows belong to, and is
                    # only laid out inside this one. Made before the row, it
                    # stands lower in the stacking order, and the row's own
                    # background is drawn straight over it - so it is lifted
                    block.lift(row)
                elif above:
                    # Set apart from each other now the row is theirs alone:
                    # packed tight to share a row with the readout, the three
                    # read as one run of lettering. Spaced after the measuring
                    # above, so the width they are judged on is the tight one
                    for radio in view_buttons[:-1]:
                        radio.pack_configure(padx=(0, 2 * cell_w))
                    # Indented as the readout is, so the first box stands over
                    # the "Sun over ..." the line below it starts with
                    block.pack(side=tk.TOP, anchor='w', before=readout_row,
                               padx=(plot_x0, 0))
                else:
                    block.pack(side=tk.TOP, anchor='w', after=legend)

        reset()   # the first draw starts at the moment the app is showing
        fit_rows()

        # Not modal, so the renderer's mouse stays in use while the graph is
        # open - the view dragged with the right button, turned with the left -
        # which a grab would stop, as it stops every button pressed outside the
        # window. The renderer's keys stay held all the same (search_dialog_open),
        # so nothing typed there moves the clock or opens a dialog over this one.
        # Brought to the front here because _show_dialog does that only for a
        # window taking the grab, and a graph opened from the planner - which
        # closes itself first - would otherwise not get the keys, Escape included
        self._show_dialog(win, position=self._graph_position, grab=False)
        win.wait_visibility()
        bring_to_front(win)
