"""
BlinkMixin: a blink comparator - a tap of Z keeps the picture, and holding Z
shows it again over the live one, to compare two moments at a glance.

The clock moved by an hour, a day or a lunation, what the light has done is
hard to tell from memory: a peak's shadow shorter by a few pixels, a crater
floor half filled with light, a rille come out at the terminator. Seen one over
the other in the same place, and flicked between, the difference stands out at
once, as on the plate blink comparators the astronomers used.

A tap of Z - let go within BLINK_HOLD_MS - keeps the picture on the screen,
and shows nothing: taking a new picture must not flash the old one. Z held past
that shows the kept one, with the moment it was kept at, until it is let go;
held with no picture kept that fits, it keeps one as a tap does. The kept
picture is the window's own, as PlotOptiX last put it on the canvas, laid over
the live one as a canvas image, not rendered: showing it costs nothing, and the
overlays drawn on the canvas - the measurement line, the compass, the frame -
stay over it. A picture kept at another size of the window no longer fits it.
"""

from moonrtx.renderer_overlay import CanvasOverlayMixin


class BlinkMixin:
    """Mixin keeping a picture to show again over the live one."""

    # How long Z is held before the kept picture is shown; let go sooner, it
    # was a tap, which keeps the picture instead. Past a quick tap, short of
    # making the wait for the picture felt
    BLINK_HOLD_MS = 250

    BLINK_LABEL_COLOR = CanvasOverlayMixin.OVERLAY_ACCENT_COLOR
    BLINK_LABEL_FONT = ("Consolas", 11, "bold")
    # From the top of the picture to the label, for a 96-dpi screen
    BLINK_LABEL_MARGIN_PX = 10

    def _init_blink(self):
        """Reset the kept picture; called from MoonRenderer.__init__."""
        self._blink_image = None    # the kept picture, a Tk PhotoImage
        self._blink_at = None       # where on the canvas it stood
        self._blink_size = None     # and the size of the render then
        self._blink_time = None     # the moment it showed
        self._blink_items = []      # its canvas items while it is shown
        self._blink_held = False    # Z, or the toolbar's button, is down
        self._blink_hold = None     # the timer telling a hold from a tap

    @property
    def showing_kept_picture(self) -> bool:
        return bool(self._blink_items)

    def keep_picture(self):
        """Keep the picture on the screen to show again."""
        rt = self.rt
        if rt is None or getattr(rt, "_tk_img", None) is None:
            return
        # PlotOptiX gives the canvas a new image with every frame rather than
        # drawing into the old one, so holding on to this one keeps it as it is
        self._blink_image = rt._tk_img
        self._blink_at = rt._image_at
        self._blink_size = (rt._width, rt._height)
        self._blink_time = self.dt_local

    def blink_press(self):
        """
        Z down. Called again and again while it is held, by the key's repeat,
        and only the first does anything: it starts the wait that tells a hold
        from a tap.
        """
        if self.rt is None or self._blink_held:
            return
        self._blink_held = True
        self._blink_hold = self.rt._root.after(self.BLINK_HOLD_MS, self._blink_held_long)

    def _blink_held_long(self):
        """Z held past a tap: show the kept picture, if there is one that fits."""
        self._blink_hold = None
        rt = self.rt
        if (self._blink_image is None or self._blink_size != (rt._width, rt._height)
                or self.showing_kept_picture):
            return
        canvas = rt._canvas
        image = canvas.create_image(*self._blink_at, image=self._blink_image, anchor='nw')
        # Over the live picture, and under everything drawn over that
        canvas.tag_raise(image, rt._img_id)
        self._blink_items = [image]
        self._blink_items.extend(self._rimmed_text(
            canvas, canvas.winfo_width() / 2, self._overlay_px(self.BLINK_LABEL_MARGIN_PX),
            self._blink_label(), self.BLINK_LABEL_COLOR, self.BLINK_LABEL_FONT, anchor='n'))

    def _blink_label(self) -> str:
        when = self._blink_time
        return "Kept picture" if when is None else \
            f"Kept picture: {when.strftime('%Y-%m-%d %H:%M:%S')}"

    def blink_release(self):
        """
        Z up: the live picture again after a hold; after a tap, or a hold that
        had nothing to show, the picture on the screen kept.
        """
        if self.rt is None or not self._blink_held:
            return
        self._blink_held = False
        if self._blink_hold is not None:
            self.rt._root.after_cancel(self._blink_hold)
            self._blink_hold = None
        if self.showing_kept_picture:
            for item in self._blink_items:
                self.rt._canvas.delete(item)
            self._blink_items = []
        else:
            self.keep_picture()
