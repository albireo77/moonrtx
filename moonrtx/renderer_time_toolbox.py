"""
TimeToolboxMixin: a strip of controls for the time, laid over the bottom of the
picture, shown and hidden with T.

It took over from the date/time window: the date and the time in spinboxes,
Now, buttons standing in for Q and W - pressed once, and held down - the time
step they move by, and auto-advance, as the status bar has it.
What is typed into a field is applied with Enter, a click on its arrows at once.
It sits on the canvas rather than in a window of its own, so it goes wherever
the picture goes - full screen included - and is dragged about by its grip.

Like the panels in the corners it is a Tk widget over the canvas, not part of
the render, so it never appears in a saved image or a video frame.
"""

import calendar
import tkinter as tk
from datetime import datetime, timedelta

from moonrtx.display import ToolTip
from moonrtx.skyfield_utils import SKYFIELD_MOON_FRAME_END_UTC, SKYFIELD_MOON_FRAME_START_UTC


class TimeToolboxMixin:
    """Mixin providing the time toolbox for MoonRenderer."""

    # The ground and lettering of the info and status panels, so the toolbox
    # reads as one more of them; the fields a shade lighter, to be seen as
    # places to type
    TOOLBOX_BG = "#020202"
    TOOLBOX_FG = "#808080"
    TOOLBOX_FIELD_BG = "#202020"
    TOOLBOX_FIELD_FG = "#c0c0c0"
    TOOLBOX_ERROR_FG = "#d04040"
    TOOLBOX_DISABLED_FG = "#404040"     # a button or field that does nothing for now
    TOOLBOX_FONT = ("Consolas", 9)

    # Where it stands until it is dragged: along the foot of the canvas, its
    # bottom level with the ephemeris panel's (which keeps this much from it),
    # and just right of that panel, with this much room between the two
    TOOLBOX_MARGIN = 6
    TOOLBOX_PANEL_GAP_PX = 16

    # How often a held Q or W button steps the time: Windows' own key repeat
    # at its default speed, about thirty a second, so the button runs as the
    # key held down does
    TOOLBOX_HOLD_REPEAT_MS = 33

    def _init_toolbox(self):
        """Reset the toolbox state; called from MoonRenderer.__init__."""
        self._toolbox_frame = None
        self._toolbox_show = None       # puts a moment into the fields
        self._toolbox_show_step = None  # puts the time step into its field
        # Which hold button is down: "", "back" or "forward" (see _build_toolbox)
        self._toolbox_hold_var = None
        # Both kept between runs (see renderer_settings)
        self.toolbox_visible = False
        # Where its bottom-left corner was dragged to (see renderer_panel_handles);
        # None while it has never been dragged and stands beside the ephemeris
        # panel
        self._toolbox_at = None
        # True while one of its fields has the keyboard (see toolbox_takes_key)
        self.toolbox_focused = False

    def toggle_toolbox(self):
        """Show the toolbox, or put it away."""
        self.show_toolbox(not self.toolbox_visible)

    def show_toolbox(self, visible: bool):
        """Show the toolbox or put it away; also at start-up, if left on."""
        if self.rt is None:
            return
        if self._toolbox_frame is None:
            self._build_toolbox()
        self.toolbox_visible = visible
        if self.toolbox_visible:
            self._toolbox_show(self.dt_local)
            self._toolbox_show_step(self.time_step_minutes)
            self._place_toolbox()
        else:
            self._toolbox_frame.place_forget()
            # A held button let go of with the toolbox, so the time cannot go
            # on running with nothing in sight to stop it
            self._toolbox_hold_var.set("")
            # Typing must not go on into fields that can no longer be seen:
            # digits would land there rather than pinning features
            if self.toolbox_focused:
                self.rt._canvas.focus_set()

    def sync_toolbox(self):
        """
        Put the moment now being shown into the toolbox's fields.

        They would otherwise stand at whatever they were last set to while the
        Moon moved on without them - and Enter would then take the view back to
        a time the user had left behind.
        """
        if self._toolbox_show is not None:
            self._toolbox_show(self.dt_local)

    def release_toolbox_hold(self):
        """
        Let up whichever hold button is down; called as Q or W is pressed.

        The key then steps as it always does. Left running, the hold would add
        its steps to the key's - doubling the speed one way and leaving the
        time jittering on the spot the other - so the key takes over instead.
        """
        if self._toolbox_hold_var is not None and self._toolbox_hold_var.get():
            self._toolbox_hold_var.set("")

    def sync_toolbox_step(self):
        """Put the time step into the toolbox, N and M having changed it."""
        if self._toolbox_show_step is not None:
            self._toolbox_show_step(self.time_step_minutes)

    def _place_toolbox(self):
        """Put the toolbox where it was dragged to, or beside the ephemeris panel."""
        frame = self._toolbox_frame
        if self._place_panel_at(frame, "_toolbox_at", 'sw'):
            return
        # Measured by what the panel asks for rather than what it has, so the
        # toolbox stands in the same place whether F2 has the panel up or not
        left = self.TOOLBOX_MARGIN
        panel = getattr(self, "_info_frame", None)
        if panel is not None:
            panel.update_idletasks()
            left += panel.winfo_reqwidth() + round(self._overlay_px(self.TOOLBOX_PANEL_GAP_PX))
        frame.place(relx=0, rely=1.0, x=left, y=-self.TOOLBOX_MARGIN, anchor='sw')

    def _build_toolbox(self):
        """Build the toolbox on the canvas, hidden."""
        canvas = self.rt._canvas
        bg, fg, font = self.TOOLBOX_BG, self.TOOLBOX_FG, self.TOOLBOX_FONT
        frame = tk.Frame(canvas, bg=bg, padx=6, pady=4)
        self._toolbox_frame = frame

        # ---- the grip and the close cross (see renderer_panel_handles) ----

        self._panel_grip(frame, frame, "_toolbox_at", 'sw', "Drag to move the toolbox",
                         font, fg, bg).pack(side=tk.LEFT, padx=(0, 6))
        # At the far end, the grip holding the near one. Packed first so that
        # the error message, coming last, opens up between it and the fields
        self._panel_close(frame, lambda: self.show_toolbox(False), "Close the toolbox (T)",
                          font, fg, bg).pack(side=tk.RIGHT, padx=(8, 0))

        # ---- the clock ----

        def digits_only(proposed: str, digits: str) -> bool:
            """
            Let a spinbox hold nothing but digits, and no more than it has room
            for. Everything else typed at one is meant for the main window and
            is left alone here to reach it (see toolbox_takes_key).
            """
            return proposed == "" or (proposed.isdigit() and len(proposed) <= int(digits))

        accepts_digits = frame.register(digits_only)

        def on_focus_in(_event):
            self.toolbox_focused = True

        def on_focus_out(_event):
            self.toolbox_focused = False

        def spin_group(label: str, separator: str, parts: list, on_enter,
                       on_arrow=None, padded: bool = True, carry: bool = False) -> list:
            """
            One caption and its spinboxes, with a character between each.

            Parameters
            ----------
            label : str
                Caption before the group
            separator : str
                Character drawn between the spinboxes
            parts : list
                One (low, high, digits) tuple per spinbox
            on_enter : callable
                What Enter in any of them does
            on_arrow : callable, optional
                What a click on an arrow does, at once and on every repeat
                while it is held, given the index of the spinbox clicked;
                on_enter if not given. Typing is left to wait for Enter, a
                number half typed being no time to go to
            padded : bool
                Whether the arrows write the number with leading zeros
            carry : bool
                Whether an arrow may take a part one past either end of its
                range, for on_arrow to carry into the next part (see
                go_to_time) rather than the spinbox stopping or wrapping

            Returns
            -------
            list
                One (variable, spinbox) pair per part, in the order given
            """
            tk.Label(frame, text=label, font=font, fg=fg, bg=bg).pack(side=tk.LEFT, padx=(0, 4))
            spins = []
            for i, (low, high, digits) in enumerate(parts):
                if i:
                    tk.Label(frame, text=separator, font=font, fg=fg, bg=bg).pack(side=tk.LEFT)
                var = tk.StringVar()
                if carry:
                    low, high = low - 1, high + 1
                command = on_enter if on_arrow is None else (lambda i=i: on_arrow(i))
                spin = tk.Spinbox(frame, textvariable=var, from_=low, to=high, width=digits + 1,
                                  format=f"%0{digits}.0f" if padded else "%.0f",
                                  wrap=False, font=font, command=command,
                                  bg=self.TOOLBOX_FIELD_BG, fg=self.TOOLBOX_FIELD_FG,
                                  insertbackground=self.TOOLBOX_FIELD_FG,
                                  buttonbackground=self.TOOLBOX_FIELD_BG,
                                  disabledbackground=self.TOOLBOX_FIELD_BG,
                                  # Dimmed only to the labels' grey, not to the
                                  # dead buttons' darker one: during a hold the
                                  # fields show the running time, and in full
                                  # screen nothing else does
                                  disabledforeground=fg,
                                  relief='flat', highlightthickness=1,
                                  highlightbackground=bg, highlightcolor=fg,
                                  validate='key', validatecommand=(accepts_digits, '%P', digits))
                spin.pack(side=tk.LEFT)
                spin.bind('<FocusIn>', on_focus_in)
                spin.bind('<FocusOut>', on_focus_out)
                # Both Enters, the keypad having a key of its own
                for enter in ('<Return>', '<KP_Enter>'):
                    spin.bind(enter, lambda _event: on_enter())
                spins.append((var, spin))
            return spins

        # The year stops where the bundled Skyfield kernels do, and the day is
        # trimmed to the length of the month chosen (see below). An arrow
        # taking any part past its end carries into the rest (see go_to_time)
        date_parts = spin_group(
            "Date:", "-",
            [(SKYFIELD_MOON_FRAME_START_UTC.year, SKYFIELD_MOON_FRAME_END_UTC.year - 1, 4),
             (1, 12, 2),
             (1, 31, 2)],
            lambda: go_to_time(), lambda i: step_time(("year", "month", "day")[i]),
            carry=True)
        (year_var, _), (month_var, _), (day_var, day_spin) = date_parts

        # The offset is deliberately not shown: the date set may fall the other
        # side of a daylight saving change from the one in force now, and a
        # label read at the wrong moment would name the wrong offset. The time
        # is on the session's clock whatever the date.
        tk.Frame(frame, bg=bg, width=12).pack(side=tk.LEFT)
        time_parts = spin_group(
            "Time:", ":",
            [(0, 23, 2),
             (0, 59, 2),
             (0, 59, 2)],
            lambda: go_to_time(), lambda i: step_time(("hour", "minute", "second")[i]),
            carry=True)
        (hour_var, _), (minute_var, _), (second_var, _) = time_parts
        clock_spins = [spin for _var, spin in date_parts + time_parts]

        def clamp_day_to_month(*_):
            """Keep the day within the chosen month, so 31 Jan then Feb gives 28 or 29."""
            try:
                last = calendar.monthrange(int(year_var.get()), int(month_var.get()))[1]
            except ValueError:
                return  # half-typed year or month; the next keystroke settles it
            # One past the month's end, for an arrow to carry into the next
            day_spin.config(to=last + 1)
            if int(day_var.get() or 0) > last:
                day_var.set(f"{last:02d}")

        year_var.trace_add('write', clamp_day_to_month)
        month_var.trace_add('write', clamp_day_to_month)

        def set_now():
            if self._video_export is not None:      # the export owns the clock
                return
            self.set_time_to_now()

        def reset():
            # As Home: the time, the camera and the view orientation the
            # session started with
            if self._video_export is not None:      # the export owns the clock
                return
            self.reset_camera_position()

        tk.Frame(frame, bg=bg, width=12).pack(side=tk.LEFT)
        tk.Button(frame, text="Now", command=set_now, font=font,
                  bg=self.TOOLBOX_FIELD_BG, fg=self.TOOLBOX_FIELD_FG,
                  activebackground=fg, activeforeground=bg,
                  relief='flat', padx=6, pady=0).pack(side=tk.LEFT)
        reset_button = tk.Button(frame, text="Reset", command=reset, font=font,
                                 bg=self.TOOLBOX_FIELD_BG, fg=self.TOOLBOX_FIELD_FG,
                                 activebackground=fg, activeforeground=bg,
                                 relief='flat', padx=6, pady=0)
        reset_button.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(reset_button, "Reset camera position and time to their initial values (Home)")

        # ---- Q and W ----

        # The time out of range, as the clock fields say it; astro's own
        # sentence is too long for the toolbox
        out_of_range = (f"Only years {SKYFIELD_MOON_FRAME_START_UTC.year}"
                        f"-{SKYFIELD_MOON_FRAME_END_UTC.year - 1}")

        def step(sign: int) -> bool:
            """One press of Q (-1) or W (+1); whether the time moved."""
            if self._video_export is not None:      # the export owns the clock
                return False
            # Drawn as the keys are: a single frame, then the full render once
            # the stepping stops (see _begin_interactive_preview)
            self._begin_interactive_preview()
            try:
                self.change_time(sign * self.time_step_minutes)
            except ValueError as e:
                print(f"Time not stepped: {e}")
                error_var.set(out_of_range)
                return False
            error_var.set("")
            return True

        # Both hold buttons on the one variable, each with a value of its own:
        # pressing one sets it, and so lets the other up
        hold_var = tk.StringVar(value="")
        self._toolbox_hold_var = hold_var
        hold_id = None

        def hold_tick():
            """One repeat of the key held down, and the next one booked."""
            nonlocal hold_id
            hold_id = None
            direction = hold_var.get()
            if not direction:
                return
            if step(-1 if direction == "back" else 1):
                hold_id = frame.after(self.TOOLBOX_HOLD_REPEAT_MS, hold_tick)
            else:
                hold_var.set("")    # the end of the ephemeris, or an export

        def on_hold_change(*_):
            nonlocal hold_id
            if hold_id is not None:
                frame.after_cancel(hold_id)
                hold_id = None
            # The press buttons and the date and time fields do nothing while
            # either is held; greyed out to say so. The fields because every
            # step of the hold writes the running time over them: an arrow
            # would only nudge the run, and anything typed would be gone
            # within a step. The time step stays live, to speed the run up or
            # slow it down as it goes
            state = tk.DISABLED if hold_var.get() else tk.NORMAL
            for widget in hit_buttons + clock_spins:
                widget.config(state=state)
            # A field holding the keyboard as it goes dead would go on taking
            # the digits, meant then for the pins, to no effect
            if hold_var.get() and frame.focus_get() in clock_spins:
                canvas.focus_set()
            hold_tick()

        hold_var.trace_add('write', on_hold_change)

        def hit(sign: int):
            if not hold_var.get():
                step(sign)

        button_look = dict(font=font, bg=self.TOOLBOX_FIELD_BG, fg=self.TOOLBOX_FIELD_FG,
                           activebackground=fg, activeforeground=bg,
                           disabledforeground=self.TOOLBOX_DISABLED_FG, padx=4, pady=0)

        def hold_button(text: str, direction: str, hint: str):
            # A button that stays down, drawn lighter while it is
            button = tk.Checkbutton(frame, text=text, variable=hold_var,
                                    onvalue=direction, offvalue="", indicatoron=False,
                                    selectcolor="#505050", relief='flat', offrelief='flat',
                                    **button_look)
            button.pack(side=tk.LEFT, padx=1)
            ToolTip(button, hint)

        def hit_button(text: str, sign: int, hint: str):
            button = tk.Button(frame, text=text, command=lambda: hit(sign),
                               relief='flat', **button_look)
            button.pack(side=tk.LEFT, padx=1)
            ToolTip(button, hint)
            return button

        tk.Frame(frame, bg=bg, width=12).pack(side=tk.LEFT)
        hold_button("◀◀", "back", "Step back continuously, as Q held down (click again to stop)")
        hit_buttons = [hit_button("◀", -1, "Step back once, as Q"),
                       hit_button("▶", 1, "Step forward once, as W")]
        hold_button("▶▶", "forward", "Step forward continuously, as W held down (click again to stop)")

        # The time step Q and W move by, and auto-advance waits. Not padded:
        # a step is a count of minutes, not a reading of a clock
        tk.Frame(frame, bg=bg, width=12).pack(side=tk.LEFT)
        [(step_var, _)] = spin_group(
            "Step:", "", [(1, 1440, 4)], lambda: go_to_step(), padded=False)
        tk.Label(frame, text="min", font=font, fg=fg, bg=bg).pack(side=tk.LEFT, padx=(4, 0))

        # ---- auto-advance ----

        # On the status bar's own variable, so the two buttons are one: either
        # pressed shows both pressed, and F10 or a video export turning it on
        # or off shows on both. Made here if the toolbox comes first, T having
        # been pressed before the status bar was built (see renderer_status)
        if self._auto_advance_var is None:
            self._auto_advance_var = tk.BooleanVar(value=False)
        # Beside the step, which is what it advances by. Named rather than
        # drawn as the status bar's ▶, which here is a step forward
        auto = tk.Checkbutton(frame, text="Auto", variable=self._auto_advance_var,
                              indicatoron=False, selectcolor="#505050",
                              relief='flat', offrelief='flat',
                              command=self._on_auto_advance_toggle, **button_look)
        auto.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip(auto, "Auto-advance time (every step minutes)")

        # Empty, and so taking no room, until there is something wrong to say.
        # The toolbox is held by its left edge, so a message grows it rightwards
        # without moving the fields from under the pointer
        error_var = tk.StringVar()
        tk.Label(frame, textvariable=error_var, font=font, fg=self.TOOLBOX_ERROR_FG,
                 bg=bg).pack(side=tk.LEFT, padx=(6, 0))

        def show_datetime(dt):
            """Put a moment into the six spinboxes."""
            for var, value in ((year_var, dt.year), (month_var, dt.month), (day_var, dt.day),
                               (hour_var, dt.hour), (minute_var, dt.minute), (second_var, dt.second)):
                var.set(f"{value:04d}" if var is year_var else f"{value:02d}")
            error_var.set("")

        self._toolbox_show = show_datetime

        def show_step(minutes):
            step_var.set(str(minutes))

        self._toolbox_show_step = show_step

        def go_to_step():
            """Apply the time step in its field."""
            try:
                minutes = int(step_var.get())
            except ValueError:
                minutes = 0
            if not 1 <= minutes <= 1440:
                error_var.set("Step 1-1440 min")
                return
            # As N and M change it, so auto-advance and the status bar follow.
            # Written back even when unchanged, so "060" settles to "60"
            self.change_time_step(minutes - self.time_step_minutes)
            show_step(self.time_step_minutes)
            error_var.set("")

        def step_time(part: str):
            """
            An arrow clicked: go to the time at once. Held, the arrow repeats
            like a held Q or W, so it is drawn the way they are - a single
            frame each, and the full render once it is let go.
            """
            self._begin_interactive_preview()
            go_to_time(part)

        def carried(year, month, day, hour, minute, second, part: str) -> datetime:
            """
            The moment the parts name once a part an arrow took one past its
            range is carried into the rest, as a clock does: 04:11:59 and a
            second more is 04:12:00, and 23:59:59 on 31 December the first
            second of the next year.

            A year or a month carries into the year, and the day is then
            trimmed to the month it lands in, as when either is set by hand:
            31 January and a month more is the end of February, not 3 March.
            """
            year, month = year + (month - 1) // 12, (month - 1) % 12 + 1
            if part in ("year", "month"):
                day = min(day, calendar.monthrange(year, month)[1])
                return datetime(year, month, day, hour, minute, second)
            # The rest are lengths of time from the start of the month, so a
            # day, an hour, a minute or a second past either end runs on
            return datetime(year, month, 1) + timedelta(
                days=day - 1, hours=hour, minutes=minute, seconds=second)

        def go_to_time(part: str = ""):
            """
            Apply the date and time in the fields, on the observer's clock.
            Given the part an arrow changed, a part run past its range carries
            into the rest (see carried); typed, every part must be in range.
            """
            if self._video_export is not None:      # the export owns the clock
                return
            try:
                # A spinbox can still be typed into, so the parts are validated
                # rather than trusted; datetime rejects anything out of range
                parts = (int(year_var.get()), int(month_var.get()), int(day_var.get()),
                         int(hour_var.get()), int(minute_var.get()), int(second_var.get()))
                new_dt_naive = carried(*parts, part) if part else datetime(*parts)
            except (ValueError, OverflowError) as e:
                # Python's own wording ("invalid literal for int() with base
                # 10") is too long for the toolbox, and no clearer
                print(f"Date/time not set: {e}")
                error_var.set("Invalid date or time")
                return

            # Read on the observer's clock, so the daylight saving rules of
            # the chosen date decide the offset (see from_observer_clock)
            new_dt_local = self.from_observer_clock(new_dt_naive)

            # Refused here in a few words, rather than in astro's own long
            # sentence, which the toolbox has no room for
            if not (SKYFIELD_MOON_FRAME_START_UTC <= new_dt_local <= SKYFIELD_MOON_FRAME_END_UTC):
                error_var.set(out_of_range)
                return

            self.update_view(new_dt_local)
            # Reset auto-advance counter when time is manually set
            if self._auto_advance_var and self._auto_advance_var.get():
                self._auto_advance_elapsed = 0
            self._update_all_status_panels()
            error_var.set("")

        # The canvas never takes the keyboard on a click of its own, so without
        # this a field once clicked would keep it, and digits meant for the pins
        # would go on landing in the clock. Added to PlotOptiX's own button
        # bindings rather than bound as <ButtonPress>: Tk runs only the most
        # specific binding a widget has for an event, and those are its
        def release_keyboard(_event):
            if self.toolbox_focused:
                canvas.focus_set()

        for button in ('<Button-1>', '<Button-3>'):
            canvas.bind(button, release_keyboard, add='+')
