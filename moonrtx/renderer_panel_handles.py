"""
PanelHandlesMixin: the grip and the close cross carried by what is laid over the
picture - the time toolbox, the ephemeris panel and the status panel - and the
keeping of where each was dragged to.

A place dragged to is kept as fractions of the canvas, of the corner the thing
hangs from: so it holds its place as the canvas changes size, between windowed
and full screen, and between runs on screens of other sizes (see
renderer_settings). Until it is first dragged it is None, and the owner puts
the thing in its usual place.
"""

import tkinter as tk

from moonrtx.display import ToolTip


class PanelHandlesMixin:
    """Mixin providing the grip, the close cross and the dragged-to place."""

    PANEL_GRIP = "⠿"
    PANEL_CLOSE = "✕"
    # The close cross lights up under the pointer, to the shade of the toolbox
    # fields' lettering
    PANEL_HANDLE_HOVER_FG = "#c0c0c0"

    def _panel_grip(self, parent, frame, at_attr: str, anchor: str, hint: str,
                    font, fg: str, bg: str) -> tk.Label:
        """
        A grip in parent that drags frame about the canvas.

        Parameters
        ----------
        parent : tk widget
            Where the grip is made; the caller packs it
        frame : tk.Frame
            What is dragged: placed on the canvas
        at_attr : str
            Attribute the dragged-to place is kept in
        anchor : str
            The corner frame hangs from: 'sw' or 'se'
        hint : str
            Tooltip
        """
        canvas = frame.master
        grip = tk.Label(parent, text=self.PANEL_GRIP, font=font, fg=fg, bg=bg, cursor='fleur')
        ToolTip(grip, hint)
        drag_from = None

        def start_drag(event):
            nonlocal drag_from
            drag_from = (event.x_root, event.y_root, frame.winfo_x(), frame.winfo_y())

        def drag(event):
            if drag_from is None:
                return
            x0, y0, left, top = drag_from
            width, height = canvas.winfo_width(), canvas.winfo_height()
            w, h = frame.winfo_width(), frame.winfo_height()
            # Kept wholly on the canvas, so it can never be lost off an edge
            left = min(max(0, left + event.x_root - x0), max(0, width - w))
            top = min(max(0, top + event.y_root - y0), max(0, height - h))
            corner_x = left + (w if 'e' in anchor else 0)
            corner_y = top + (h if 's' in anchor else 0)
            setattr(self, at_attr, (corner_x / width, corner_y / height))
            self._place_panel_at(frame, at_attr, anchor)

        grip.bind('<ButtonPress-1>', start_drag)
        grip.bind('<B1-Motion>', drag)
        return grip

    def _panel_close(self, parent, on_close, hint: str, font, fg: str, bg: str) -> tk.Label:
        """A close cross in parent, calling on_close when clicked; the caller packs it."""
        close = tk.Label(parent, text=self.PANEL_CLOSE, font=font, fg=fg, bg=bg, cursor='hand2')
        ToolTip(close, hint)
        close.bind('<Enter>', lambda _event: close.config(fg=self.PANEL_HANDLE_HOVER_FG), add='+')
        close.bind('<Leave>', lambda _event: close.config(fg=fg), add='+')
        # On release, so pressing and sliding off before letting go cancels it
        close.bind('<ButtonRelease-1>', lambda _event: on_close())
        return close

    def _panel_header(self, frame, at_attr: str, anchor: str, what: str, key: str,
                      on_close, font, fg: str, bg: str) -> tk.Frame:
        """
        A row across the top of a panel: the grip at its left, the close cross
        at its right. Packed into frame here, so it must be made before the
        panel's rows.
        """
        header = tk.Frame(frame, bg=bg)
        header.pack(fill=tk.X)
        self._panel_grip(header, frame, at_attr, anchor, f"Drag to move the {what}",
                         font, fg, bg).pack(side=tk.LEFT)
        self._panel_close(header, on_close, f"Close the {what} ({key})",
                          font, fg, bg).pack(side=tk.RIGHT)
        return header

    def _place_panel_at(self, frame, at_attr: str, anchor: str) -> bool:
        """
        Place frame where it was dragged to, and say whether it was ever
        dragged; if not, nothing is done and the owner places it.
        """
        at = getattr(self, at_attr)
        if at is None:
            return False
        relx, rely = at
        frame.place(relx=relx, rely=rely, x=0, y=0, anchor=anchor)
        return True
