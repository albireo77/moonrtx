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
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox
from datetime import datetime, timedelta, timezone
from typing import Callable, NamedTuple, Optional

from moonrtx import astro
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

    def _results_actions(self, results, go_to, table, name):
        """
        The row of buttons a results dialog ends with: go to the one selected,
        put them all on the clipboard, write them to a file, close.

        `table` answers with the columns, the rows and the calendar entries of
        whatever is listed at the time it is asked, and `name` with the stem to
        offer the save under - both of them called rather than passed in, since
        both follow the filter the dialog is left on.
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
        return "twilight" if record["observer_sun_alt"] > -12.0 else "night"

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
            uid = f"{stamp(event['start'])}-{abs(hash(event['summary'])) % 10**10}@moonrtx"
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
                       text=f"Only when the Moon altitude (h☾) is at least "
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
        tk.Label(mode_row, text="Show:", anchor='w').pack(side=tk.LEFT)
        for value, label in (("terminator", "near the terminator"),
                             ("libration", "best presented (libration)")):
            ttk.Radiobutton(mode_row, text=label, value=value, variable=mode_var,
                           command=lambda: rescan()).pack(side=tk.LEFT)

        def rescan():
            nonlocal windows
            listbox.delete(0, tk.END)
            libration = mode_var.get() == "libration"
            try:
                if libration:
                    windows = astro.find_libration_windows(
                        self.dt_local, self.PLANNER_SCAN_DAYS, feature.lat, feature.lon,
                        sun_alt_min=self.PLANNER_LIBRATION_SUN_ALT_MIN,
                        moon_alt_min=self.PLANNER_MOON_ALT_MIN,
                        max_results=self.PLANNER_MAX_RESULTS)
                else:
                    windows = astro.find_terminator_windows(
                        self.dt_local, self.PLANNER_SCAN_DAYS, feature.lat, feature.lon,
                        sun_alt_max=self.PLANNER_SUN_ALT_MAX,
                        moon_alt_min=self.PLANNER_MOON_ALT_MIN)
            except ValueError as e:
                # Scan start outside the bundled ephemeris kernel range
                windows = []
                desc_var.set(str(e))
                header_var.set("")
                return

            if libration:
                desc_var.set(
                    "How far inside the limb libration turns the feature, best first: 90° is the "
                    "centre of the disk, 0° exactly on the limb, and the feature is squashed by the "
                    "sine of it. Listed only while the feature is sunlit and the Moon at least "
                    f"{self.PLANNER_MOON_ALT_MIN:.0f}° up in your sky.")
                header_var.set(libration_header)
            else:
                desc_var.set(
                    f"Times when the Sun stands 0-{self.PLANNER_SUN_ALT_MAX:.0f}° above the feature, "
                    f"lighting it with long shadows, and the Moon is at least "
                    f"{self.PLANNER_MOON_ALT_MIN:.0f}° up in your sky.")
                header_var.set(terminator_header)

            if not windows:
                listbox.insert(tk.END, "  No opportunities found in the scanned period.")
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
            on_close()
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

        self._results_actions(
            dialog, go_to, results_for_export,
            lambda: f"{feature.name.replace(' ', '_')}_{mode_var.get()}")

        rescan()

        self._show_dialog(win)
