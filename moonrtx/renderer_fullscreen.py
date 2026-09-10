"""
FullScreenMixin: the window with nothing on it but the Moon.

F11 takes away everything the desktop and the toolkit put round the picture -
the title bar, the border, the row of readouts along the bottom, and the light
edging Tk gives a canvas of its own accord - and gives it all back. Escape only
gives it back.

The row of readouts is not dealt with here. It belongs to StatusMixin, which
builds its panels into it and knows which of PlotOptiX's own labels share it, so
that mixin is asked to put the row away and to bring it out again. What is here
is the window and the canvas.
"""

from moonrtx.display import bring_to_front


class FullScreenMixin:
    """Mixin providing the full-screen window mode for MoonRenderer."""

    # What the render canvas is set to while full screen lasts. PlotOptiX builds
    # it with Tk's defaults, which give a canvas a two-pixel focus highlight in
    # the system's button-face grey and paint the same grey anywhere the rendered
    # image does not reach. Against a window frame neither is noticeable; with
    # the frame gone they are a light border round the Moon, and a light band
    # for as long as it takes the render buffer to catch up with a resize.
    FULL_SCREEN_CANVAS = {"highlightthickness": 0, "background": "black"}

    def _init_full_screen(self):
        """Reset the full-screen state; called from MoonRenderer.__init__."""
        # What the canvas looked like before full screen, read off the canvas
        # rather than written down here, so that leaving restores what was
        # actually there whatever PlotOptiX built it with
        self._windowed_canvas = {}

    def toggle_full_screen(self):
        """
        Show the Moon alone, or give the window its frame and readouts back.

        Full screen here is the picture and nothing else: no title bar, no
        border, and none of the bottom row. The canvas is the only thing in the
        window's grid with any weight, so once that row is out it has the whole
        window, and PlotOptiX resizes the render buffer to whatever the canvas
        becomes without being asked. The overlays follow it on their own poll.

        Going is done in two pieces because at start-up they cannot happen
        together - see _full_screen_window here and StatusMixin._hide_status_row.
        """
        if self.rt._root.attributes("-fullscreen"):
            self.exit_full_screen()
        else:
            self._full_screen_window()
            self._hide_status_row()

    def _full_screen_window(self):
        """
        Take the frame off the window and the light edging off the canvas.

        Kept apart from hiding the status row for the sake of start-up.
        PlotOptiX builds its window on its own thread and enters the Tk main
        loop there, and what tells the rest of the program it has started is the
        canvas's first Configure - so by the time anything here can touch that
        window it is already on the screen. The earliest moment MoonRTX is given
        is the launch-finished callback, and this runs at the top of it, before
        the status bar is built: fitting that bar's lettering runs
        update_idletasks in a loop, and every one of those repaints the window
        at its windowed size. Done afterwards instead, full screen was watched
        arriving a moment after the window had already settled.

        The row of readouts has to wait for the bottom of that callback all the
        same, because the panels are built into it and are told where it sits by
        asking the label they replace - which a widget already removed can no
        longer answer.

        Windows also leaves its taskbar over a full-screen window that is not
        the foreground one, which is why the window is brought forward here
        rather than the attribute being trusted on its own.
        """
        canvas = self.rt._canvas
        self._windowed_canvas = {name: canvas.cget(name)
                                 for name in self.FULL_SCREEN_CANVAS}
        canvas.configure(**self.FULL_SCREEN_CANVAS)
        self.rt._root.attributes("-fullscreen", True)
        bring_to_front(self.rt._root)

    def exit_full_screen(self):
        """
        Give the window back its frame and readouts, or do nothing if it never
        lost them.

        Escape is bound to this rather than to the toggle, and the two are not
        the same thing. In full screen there is no title bar and no taskbar, so
        F11 is the only way out and a reader who does not know it has nothing to
        try; Escape is what everything else on the desktop answers to. Bound to
        the toggle it would be a way in as well, which is not what anyone means
        by pressing it - so at any other time this is simply nothing happening.

        A dialog's own Escape closes the dialog and stops the press there (see
        DialogsMixin._dialog_window), so the two never both act on one key.

        The window shrinks before the row comes back, so the row is never seen
        against a window of the wrong size.
        """
        root = self.rt._root
        if not root.attributes("-fullscreen"):
            return
        root.attributes("-fullscreen", False)
        self.rt._canvas.configure(**self._windowed_canvas)
        self._windowed_canvas = {}
        self._show_status_row()
