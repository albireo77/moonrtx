"""
ToolbarMixin: a row of buttons standing for keys, laid over the picture, shown
and hidden with Z.

Each button is its key pressed: it goes through the key handler itself (see
run_renderer), with the same guards - a video export holding the clock, a
dialog holding the keys - so it never does anything its key would not. Those
that turn something on and off stay down while it is on, whichever way it was
turned on: the key, the button, or a window of its own (the field-of-view
setup, a panel's close cross).

It stands over the time toolbox until it is dragged by its grip, and like the
toolbox it is a Tk widget over the canvas, not part of the render, so it never
appears in a saved image or a video frame.
"""

import tkinter as tk

from moonrtx.display import ToolTip
from moonrtx.view_orientation import VIEW_ORIENTATIONS


class ToolbarMixin:
    """Mixin providing the toolbar for MoonRenderer."""

    # The icons are symbols, and Segoe UI Symbol has them all; the ones it has
    # not (the magnifier, the pin, the compass, the cameras) Windows finds in
    # its emoji font, drawn in one colour as Tk draws everything. A size up from
    # the toolbox's lettering, for the symbols to be told apart
    TOOLBAR_FONT = ("Segoe UI Symbol", 11)

    # Between it and the time toolbox it stands over
    TOOLBAR_GAP_PX = 6

    # How often the buttons that stay down are matched to what they stand for,
    # for what changes it other than a key (a press of a key or a button
    # matches them at once)
    TOOLBAR_SYNC_MS = 300

    # The view orientations by their keys, F5 to F8
    TOOLBAR_ORIENTATION_KEYS = dict(zip(VIEW_ORIENTATIONS, ('F5', 'F6', 'F7', 'F8')))

    def _init_toolbar(self):
        """Reset the toolbar state; called from MoonRenderer.__init__."""
        self._toolbar_frame = None
        self._toolbar_sync = None       # matches the buttons to what they stand for
        self._toolbar_sync_id = None
        # Presses a key, given its keysym and whether Shift is down; set by
        # run_renderer, whose key handler it is
        self.toolbar_press_key = None
        # Both kept between runs (see renderer_settings)
        self.toolbar_visible = True
        # Where its bottom-left corner was dragged to (see renderer_panel_handles);
        # None while it has never been dragged and stands over the time toolbox
        self._toolbar_at = None

    def toggle_toolbar(self):
        """Show the toolbar, or put it away."""
        self.show_toolbar(not self.toolbar_visible)

    def show_toolbar(self, visible: bool):
        """Show the toolbar or put it away; also at start-up, if left on."""
        if self.rt is None:
            return
        if self._toolbar_frame is None:
            self._build_toolbar()
        self.toolbar_visible = visible
        if self.toolbar_visible:
            self.sync_toolbar()
            self._place_toolbar()
            self._schedule_toolbar_sync()
        else:
            self._toolbar_frame.place_forget()

    def sync_toolbar(self):
        """Match the buttons that stay down to what they stand for."""
        if self._toolbar_sync is not None and self.toolbar_visible:
            self._toolbar_sync()

    def _schedule_toolbar_sync(self):
        """Keep the buttons matched while the toolbar is up; stops once it is put away."""
        if self._toolbar_sync_id is not None:
            return

        def tick():
            self._toolbar_sync_id = None
            if self.toolbar_visible:
                self.sync_toolbar()
                self._schedule_toolbar_sync()

        self._toolbar_sync_id = self._toolbar_frame.after(self.TOOLBAR_SYNC_MS, tick)

    def _place_toolbar(self):
        """Put the toolbar where it was dragged to, or over the time toolbox."""
        frame = self._toolbar_frame
        if self._place_panel_at(frame, "_toolbar_at", 'sw'):
            return
        # Over the toolbox's usual place, whether T has it up or not: the
        # toolbar keeps its own place rather than going where the toolbox goes
        if self._toolbox_frame is None:
            self._build_toolbox()
        self._toolbox_frame.update_idletasks()
        above = (self.TOOLBOX_MARGIN + self._toolbox_frame.winfo_reqheight()
                 + round(self._overlay_px(self.TOOLBAR_GAP_PX)))
        frame.place(relx=0, rely=1.0, x=self._toolbox_usual_left(), y=-above, anchor='sw')

    def _build_toolbar(self):
        """Build the toolbar on the canvas, hidden."""
        canvas = self.rt._canvas
        # Its ground, lettering and buttons the time toolbox's, so the two read
        # as a pair
        bg, fg, font = self.TOOLBOX_BG, self.TOOLBOX_FG, self.TOOLBOX_FONT
        frame = tk.Frame(canvas, bg=bg, padx=6, pady=4)
        self._toolbar_frame = frame

        # ---- the grip and the close cross (see renderer_panel_handles) ----

        self._panel_grip(frame, frame, "_toolbar_at", 'sw', "Drag to move the toolbar",
                         font, fg, bg).pack(side=tk.LEFT, padx=(0, 6))
        self._panel_close(frame, lambda: self.show_toolbar(False), "Close the toolbar (X)",
                          font, fg, bg).pack(side=tk.RIGHT, padx=(8, 0))

        # ---- the buttons ----

        def press(keysym: str, shift: bool = False):
            self.toolbar_press_key(keysym, shift)
            # A button that stays down has just put itself up or down as
            # clicked; this puts it back if the key was refused
            self.sync_toolbar()

        look = dict(font=self.TOOLBAR_FONT, width=2,
                    bg=self.TOOLBOX_FIELD_BG, fg=self.TOOLBOX_FIELD_FG,
                    activebackground=fg, activeforeground=bg,
                    relief='flat', padx=2, pady=0)

        # (variable, what it shows) for every button that stays down
        states = []

        def group():
            tk.Frame(frame, bg=bg, width=12).pack(side=tk.LEFT)

        def button(icon: str, keysym: str, hint: str, shift: bool = False):
            widget = tk.Button(frame, text=icon, command=lambda: press(keysym, shift), **look)
            widget.pack(side=tk.LEFT, padx=1)
            ToolTip(widget, hint)

        def toggle(icon: str, keysym: str, hint: str, is_on, shift: bool = False):
            # A button that stays down while what it turns on is on, drawn
            # lighter while it is, as the toolbox's Auto
            var = tk.BooleanVar(value=is_on())
            widget = tk.Checkbutton(frame, text=icon, variable=var, indicatoron=False,
                                    selectcolor="#505050", offrelief='flat',
                                    command=lambda: press(keysym, shift), **look)
            widget.pack(side=tk.LEFT, padx=1)
            ToolTip(widget, hint)
            states.append((var, lambda: bool(is_on())))

        # Finding
        button("🔍", 'f', "Search for Moon features or go to lat lon (F)")
        button("◐", 'a', "Find clair-obscur events (A)")
        button("◉", 'e', "Find lunar eclipses (E)")
        button("⌓", 'u', "Chart when the Moon is up over the coming month (U)")

        # On the Moon
        group()
        toggle("▦", 'g', "Toggle selenographic grid (G)", lambda: self.moon_grid_visible)
        toggle("Aa", 'l', "Toggle standard labels (L)", lambda: self.standard_labels_visible)
        toggle("a•", 's', "Toggle spot labels (S)", lambda: self.spot_labels_visible)
        toggle("A…", 'p', "Toggle labels for all Moon features (P)", lambda: self.catalogue_visible)
        toggle("☉", 'y', "Toggle markers for sub-solar and sub-Earth points (Y)",
               lambda: self.sub_points_visible)
        toggle("📍", '0', "Toggle pins (0)", lambda: self.pins_visible)

        # Taking off what clicks and measuring left on the picture, each drawn
        # as the key it stands for is marked
        group()
        button("⌦", 'Delete', "Remove labels left by Find, Observation Planner and clicks (Delete)")
        button("⌫", 'BackSpace', "Remove measurement lines and the topography area (Backspace)")

        # Over the picture
        group()
        toggle("◯", 'b', "Toggle the field of view frame (B)", lambda: self.fov_overlay_visible)
        button("⚙", 'b', "Set up the eyepiece / camera field of view frame (Shift + B)", shift=True)
        toggle("🧭", 'c', "Toggle compass (C)", lambda: self.compass_visible)
        toggle("⌖", 'r', "Toggle locator (R)", lambda: self.locator_visible)
        toggle("⟷", 'd', "Toggle scale bar (D)", lambda: self.scale_bar_visible)
        toggle("∠", 'd', "Toggle angular scale bar (Shift + D)", lambda: self.angle_bar_visible,
               shift=True)
        # Pressed and let go as Z is, rather than clicked: a click keeps the
        # picture, and a hold shows the kept one (see BlinkMixin)
        blink = tk.Button(frame, text="⇆", **look)
        blink.pack(side=tk.LEFT, padx=1)
        blink.bind("<ButtonPress-1>", lambda _event: self.blink_press(), add="+")
        blink.bind("<ButtonRelease-1>", lambda _event: self.blink_release(), add="+")
        ToolTip(blink, "Click to keep the picture, hold to see the kept one (Z)")

        # The view
        group()
        button("⌂", 'End', "Reset camera to default position (End)")
        toggle("⭡", 'F4', "Toggle parallactic mode, celestial north up (F4)",
               lambda: self.parallactic_mode)
        orientation_var = tk.StringVar(value=self.view_orientation)
        orientation = tk.OptionMenu(
            frame, orientation_var, *VIEW_ORIENTATIONS,
            command=lambda value: press(self.TOOLBAR_ORIENTATION_KEYS[value]))
        orientation.config(font=font, bg=self.TOOLBOX_FIELD_BG, fg=self.TOOLBOX_FIELD_FG,
                           activebackground=fg, activeforeground=bg,
                           relief='flat', highlightthickness=0, padx=4, pady=0)
        orientation['menu'].config(font=font, bg=self.TOOLBOX_FIELD_BG, fg=self.TOOLBOX_FIELD_FG,
                                   activebackground=fg, activeforeground=bg)
        # As tall as the buttons beside it
        orientation.pack(side=tk.LEFT, padx=1, fill=tk.Y)
        ToolTip(orientation, "View orientation (F5 NSWE, F6 NSEW, F7 SNEW, F8 SNWE)")
        states.append((orientation_var, lambda: self.view_orientation))
        toggle("⛶", 'F11', "Toggle full screen (F11)",
               lambda: self.rt._root.attributes("-fullscreen"))

        # The panels
        group()
        toggle("◱", 'F2', "Toggle Moon ephemeris panel (F2)", lambda: self.show_info_panel)
        toggle("◲", 'F3', "Toggle status panel (F3)", lambda: self.show_status_panel)

        # Out of the program
        group()
        button("📷", 'F12', "Save image (F12)")
        button("🎥", 'v', "Export time-lapse video (V)")

        group()
        button("?", 'F1', "Help (F1)")

        def sync():
            for var, state in states:
                value = state()
                if var.get() != value:
                    var.set(value)

        self._toolbar_sync = sync
