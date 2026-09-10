"""
DialogsMixin: the dialogs that act on the view - the video export, the help, the
image save, the feature search and the clock - for MoonRenderer.

The ones that answer a question about the sky rather than acting on the view -
the rise-and-set chart, the clair-obscur finder and the observation planner -
are in renderer_planning, along with the frame the last two are built in and the
writing of results to the clipboard, a spreadsheet or a calendar.
"""

import os
import glob
import base64
import struct
import calendar
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from datetime import datetime
from typing import Optional

from moonrtx.display import screen_size
from moonrtx.shared_types import Camera
from moonrtx.skyfield_utils import SKYFIELD_MOON_FRAME_END_UTC, SKYFIELD_MOON_FRAME_START_UTC


def _ffmpeg_dlls_findable() -> bool:
    """
    Heuristic check whether the FFmpeg shared DLLs needed by the PlotOptiX
    video encoder can be found: avcodec*.dll in any PATH directory (the
    encoder itself gives no Python-queryable availability flag on Windows).
    Non-Windows platforms are assumed OK.
    """
    if os.name != 'nt':
        return True
    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            if d.strip() and glob.glob(os.path.join(d.strip(), "avcodec*.dll")):
                return True
        except OSError:
            continue
    return False

def encode_camera(camera: Camera) -> str:
    """
    Encode camera into a compact base64 string.
    
    Packs 10 floats (eye[3], target[3], up[3], fov) into binary and base64 encodes.
    Uses URL-safe base64 (- and _ instead of + and /) for filename compatibility.
    
    Parameters
    ----------
    camera : Camera
        Camera object with eye, target, up, and fov attributes
    Returns
    -------
    str
        Base64-encoded camera parameters (URL-safe, no padding)
    """
    # Pack 10 floats: eye(3) + target(3) + up(3) + fov(1)
    packed = struct.pack('<10f', 
                         camera.eye[0], camera.eye[1], camera.eye[2],
                         camera.target[0], camera.target[1], camera.target[2],
                         camera.up[0], camera.up[1], camera.up[2],
                         camera.fov)
    # URL-safe base64 without padding (= chars)
    encoded = base64.urlsafe_b64encode(packed).decode('ascii').rstrip('=')
    return encoded

class DialogsMixin:
    """Mixin providing dialog window methods for MoonRenderer."""

    # How far below the top of the screen the help window opens, and the room
    # left under it, written for a 96-dpi screen and scaled like every other
    # length here. It is put against the top rather than centred on the main
    # window: it is a tall list, at a high display scaling nearly as tall as the
    # screen, and centring something almost full height leaves it sitting low
    # with nowhere to go.
    HELP_TOP_PX = 8

    def _dialog_window(self, title: str, padding=(12, 8), takes_keys: bool = True,
                       over_main: bool = True, size: Optional[tuple] = None,
                       before_close=None):
        """
        Put up a dialog, and hand back the window, the frame its contents go
        in, and the way to shut it.

        Every dialog in the program opens the same way - withdrawn until
        _show_dialog has placed it, titled, kept above the main window, closed
        by its own button or by Escape or by the window manager, and everything
        inside one padded frame - and each of them used to write that out for
        itself. Which left differences between them that nobody had decided:
        one not kept above the main window, one that could be resized, one
        whose Escape reached the main window on the way out. They are arguments
        now, so each is a choice somebody made rather than a line somebody
        forgot.

        Parameters
        ----------
        title : str
            The window title
        padding : tuple
            Padding of the frame the contents go in, as (padx, pady)
        takes_keys : bool
            Hold the main window's key handling for as long as this dialog is
            open. PlotOptiX binds that handler with bind_all, so without this a
            dialog being typed into also drives the Moon.
        over_main : bool
            Keep the window above the main one and hide it along with it
        size : tuple, optional
            Starting size as (width, height), written for a 96-dpi screen and
            scaled to this one; the window is then free to be resized. Without
            one it is fixed at whatever it needs.
        before_close : callable, optional
            Called before the window is destroyed, for a dialog with state of
            its own to put down. Returning False stops the close - which is how
            the video export turns a close during an export into a cancel.

        Returns
        -------
        tuple
            (window, frame, close)
        """
        if takes_keys:
            self.search_dialog_open = True

        win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        win.withdraw()
        win.title(title)
        if over_main:
            win.transient(self.rt._root)
        if size is None:
            win.resizable(False, False)
        else:
            # Scaled, because everything inside is not. The lettering, the entry
            # measured in characters and the rows of the list all follow the
            # display's dots per inch, so a window pinned to raw pixels opens too
            # small to hold its own contents - at 300% scaling the search dialog
            # came up at a third of the size it wanted.
            win.geometry("%dx%d" % (round(self._overlay_px(size[0])),
                                    round(self._overlay_px(size[1]))))

        def close():
            if before_close is not None and before_close() is False:
                return
            if takes_keys:
                self.search_dialog_open = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", close)
        # "break" so the key handler bound with bind_all does not also see the
        # Escape that shut this window - by then the dialog has already given
        # the main window its keys back
        win.bind("<Escape>", lambda e: (close(), "break")[1])

        frame = tk.Frame(win, padx=padding[0], pady=padding[1])
        frame.pack(fill=tk.BOTH, expand=True)
        return win, frame, close

    def _show_dialog(self, win, position=None, grab: bool = True):
        """
        Map a dialog once it is finished and placed.

        Every dialog here is built while withdrawn and shown through this
        method: a Toplevel is otherwise mapped where the window manager first
        puts it and moved to its own position only afterwards, which is seen as
        the window flashing in the corner of the screen before it settles. The
        grab has to wait for the same reason - a window that is not yet
        viewable cannot take one.

        Parameters
        ----------
        win : tk.Toplevel
            The dialog to show
        position : tuple, optional
            Screen position; centred on the main window when not given
        grab : bool
            Whether the dialog takes the input grab (modal dialogs)
        """
        win.update_idletasks()
        if position is None:
            root = self.rt._root
            position = (root.winfo_x() + (root.winfo_width() - win.winfo_width()) // 2,
                        root.winfo_y() + (root.winfo_height() - win.winfo_height()) // 2)
        # Never off the top or the left: a dialog taller or wider than the
        # window it is centred on would otherwise be given a negative corner,
        # putting its title bar - the only handle for moving it - off the screen
        win.geometry("+%d+%d" % (max(0, position[0]), max(0, position[1])))
        win.deiconify()
        if grab:
            win.wait_visibility()
            win.grab_set()

    def export_video_dialog(self):
        """
        Open the time-lapse video export dialog: renders N frames from the
        current observation time, advancing by a configurable number of
        simulated minutes per frame, into an MP4 (H.264, NVENC hardware
        encoding). See MoonRenderer.start_video_export for the mechanics.
        """
        if self.rt is None:
            return

        exporting = {"active": False}

        def still_exporting():
            """
            Closing during an export only asks for it to be cancelled; the
            dialog stays open to show how it ended, and closes then.
            """
            if not exporting["active"]:
                return True
            self.cancel_video_export()
            return False

        win, main_frame, on_close = self._dialog_window(
            "Export time-lapse video", padding=(15, 10), before_close=still_exporting)

        tk.Label(main_frame,
                 text=f"Starts at the current observation time: {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}",
                 anchor='w').pack(fill=tk.X, pady=(0, 6))

        grid = tk.Frame(main_frame)
        grid.pack(fill=tk.X)

        frames_var = tk.StringVar(value="120")
        step_var = tk.StringVar(value=str(self.time_step_minutes))
        # NVENC settings are fixed for the session by the first export
        cfg = self._video_encoder_cfg
        fps_var = tk.StringVar(value=str(cfg[0]) if cfg else "25")
        bitrate_var = tk.StringVar(value=f"{cfg[1]:g}" if cfg else "16")

        rows = [
            ("Frames:", frames_var, "(2 - 100000)", True),
            ("Minutes per frame:", step_var, "(negative goes back in time)", True),
            ("Playback FPS:", fps_var, "(1 - 60)", cfg is None),
            ("Bitrate (Mbit/s):", bitrate_var, "(1 - 60)", cfg is None),
        ]
        entries = []
        for i, (label, var, hint, enabled) in enumerate(rows):
            tk.Label(grid, text=label, anchor='e').grid(row=i, column=0, sticky='e', pady=2)
            e = tk.Entry(grid, textvariable=var, width=10,
                         state='normal' if enabled else 'disabled')
            e.grid(row=i, column=1, padx=5, pady=2, sticky='w')
            tk.Label(grid, text=hint, fg='gray').grid(row=i, column=2, sticky='w', pady=2)
            entries.append(e)

        if cfg is not None:
            tk.Label(main_frame, fg='gray', anchor='w',
                     text="FPS and bitrate were fixed by this session's first export\n"
                          "(PlotOptiX limitation); restart MoonRTX to change them."
                     ).pack(fill=tk.X, pady=(4, 0))

        # Live summary of video length and simulated time span
        summary_var = tk.StringVar()

        def update_summary(*args):
            try:
                n = int(frames_var.get())
                step = int(step_var.get())
                fps = int(fps_var.get())
                span_h = n * step / 60.0
                summary_var.set(f"Video length: {n / fps:.1f} s   "
                                f"simulated span: {span_h:+.1f} h ({span_h / 24:+.2f} days)")
            except (ValueError, ZeroDivisionError):
                summary_var.set("")

        for var in (frames_var, step_var, fps_var):
            var.trace_add('write', update_summary)
        update_summary()
        tk.Label(main_frame, textvariable=summary_var, anchor='w').pack(fill=tk.X, pady=(6, 0))

        # The status bar is not part of the ray-traced image, so the local time
        # has to be drawn into the frames themselves to appear in the video
        burn_time_var = tk.BooleanVar(value=True)
        # Themed, so the little box follows the display (see the launcher)
        burn_time_cb = ttk.Checkbutton(main_frame, variable=burn_time_var,
                                       text="Show local time")
        burn_time_cb.pack(fill=tk.X, pady=(4, 0))

        time_corner_var = tk.StringVar(value=self.VIDEO_TIME_CORNER)
        caption_corner_var = tk.StringVar(value=self.VIDEO_CAPTION_CORNER)

        def corner_row(parent, var, command):
            """Row of radio buttons selecting one of the four frame corners."""
            row = tk.Frame(parent)
            row.pack(fill=tk.X)
            buttons = []
            for corner in self.VIDEO_CORNERS:
                rb = ttk.Radiobutton(row, text=corner, value=corner, variable=var,
                                     command=command)
                rb.pack(side=tk.LEFT)
                buttons.append(rb)
            return buttons

        # The two labels must not share a corner; picking the one already taken
        # swaps them, which always leaves both settings valid
        previous = {"time": time_corner_var.get(), "caption": caption_corner_var.get()}

        def corner_chosen(which):
            def handler():
                var = time_corner_var if which == "time" else caption_corner_var
                other_var = caption_corner_var if which == "time" else time_corner_var
                other = "caption" if which == "time" else "time"
                if var.get() == other_var.get():
                    other_var.set(previous[which])
                previous[which] = var.get()
                previous[other] = other_var.get()
            return handler

        time_corner_buttons = corner_row(main_frame, time_corner_var, corner_chosen("time"))

        caption_frame = tk.Frame(main_frame)
        caption_frame.pack(fill=tk.X, pady=(6, 0))
        tk.Label(caption_frame, text="Caption:", anchor='w').pack(side=tk.LEFT)
        caption_var = tk.StringVar()
        caption_entry = tk.Entry(caption_frame, textvariable=caption_var)
        caption_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        caption_corner_buttons = corner_row(main_frame, caption_corner_var,
                                            corner_chosen("caption"))

        # The compass, the locator and the field-of-view frame are drawn on the
        # canvas over the render, not into it, so they reach the video the same
        # way the time does. Offered as they stand: whichever of them is on
        # screen when the export starts is what goes into the frames, each
        # redrawn for the time its own frame shows.
        showing = (self.compass_visible or self.locator_visible
                   or self.fov_overlay_visible)
        burn_overlays_var = tk.BooleanVar(value=showing)
        burn_overlays_cb = ttk.Checkbutton(
            main_frame, variable=burn_overlays_var,
            text="Show overlays (compass, locator, field of view)")
        burn_overlays_cb.pack(fill=tk.X, pady=(4, 0))
        if not showing:
            burn_overlays_cb.config(state='disabled')

        def limit_caption(*args):
            text = caption_var.get()
            if len(text) > self.VIDEO_CAPTION_MAX_CHARS:
                caption_var.set(text[:self.VIDEO_CAPTION_MAX_CHARS])

        caption_var.trace_add('write', limit_caption)

        status_var = tk.StringVar()
        status_label = tk.Label(main_frame, textvariable=status_var, anchor='w')
        status_label.pack(fill=tk.X, pady=(6, 0))

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        def set_exporting(active: bool):
            exporting["active"] = active
            state = 'disabled' if active else 'normal'
            for i, e in enumerate(entries):
                # FPS/bitrate stay disabled once the encoder is configured
                locked = self._video_encoder_cfg is not None and i >= 2
                e.config(state='disabled' if (active or locked) else 'normal')
            burn_time_cb.config(state=state)
            # Nothing to burn in leaves it disabled whatever the export is doing
            burn_overlays_cb.config(
                state='disabled' if (active or not showing) else 'normal')
            caption_entry.config(state=state)
            for rb in time_corner_buttons + caption_corner_buttons:
                rb.config(state=state)
            export_btn.config(state=state)
            cancel_btn.config(state='normal' if active else 'disabled')

        def on_progress(frame, total, dt_local):
            status_var.set(f"Rendering frame {frame} / {total}   "
                           f"{dt_local.strftime('%Y-%m-%d %H:%M')}")

        def on_done(error):
            set_exporting(False)
            if error:
                status_label.config(fg='red')
                status_var.set(f"Export stopped: {error}")
            else:
                status_label.config(fg='black')
                status_var.set("Export finished.")

        def on_export():
            try:
                n = int(frames_var.get())
                step = int(step_var.get())
                fps = int(fps_var.get())
                bitrate = float(bitrate_var.get())
                if not (2 <= n <= 100000): raise ValueError("frames out of range")
                if not (1 <= abs(step) <= 1440): raise ValueError("minutes per frame out of range")
                if not (1 <= fps <= 60): raise ValueError("FPS out of range")
                if not (1 <= bitrate <= 60): raise ValueError("bitrate out of range")
            except ValueError as e:
                status_label.config(fg='red')
                status_var.set(f"Invalid settings: {e}")
                return

            filename = filedialog.asksaveasfilename(
                parent=win,
                initialdir=".",
                title="Save time-lapse video as",
                initialfile=f"{self.get_default_filename()}_x{n}.mp4",
                defaultextension=".mp4",
                filetypes=(("MP4 video", "*.mp4"),)
            )
            if not filename:
                return

            status_label.config(fg='black')
            status_var.set("Starting export...")
            set_exporting(True)
            error = self.start_video_export(filename, n, step, fps, bitrate,
                                            on_progress, on_done,
                                            burn_time=burn_time_var.get(),
                                            caption=caption_var.get(),
                                            time_corner=time_corner_var.get(),
                                            caption_corner=caption_corner_var.get(),
                                            burn_overlays=burn_overlays_var.get())
            if error is not None:
                set_exporting(False)
                status_label.config(fg='red')
                status_var.set(error)

        export_btn = tk.Button(btn_frame, text="Export...", command=on_export, width=12)
        export_btn.pack(side=tk.LEFT)
        cancel_btn = tk.Button(btn_frame, text="Cancel export", command=self.cancel_video_export,
                               width=13, state='disabled')
        cancel_btn.pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Close", command=on_close, width=10).pack(side=tk.RIGHT)

        # Early warning when the FFmpeg DLLs the encoder depends on are not
        # findable. This mirrors the native loader's search (avcodec & co. in
        # PATH) but is only a heuristic - DLLs can also live e.g. in the
        # Python directory - so Export stays enabled and the engine's
        # encoder_is_open() check remains the definitive (and fail-safe) gate.
        if not _ffmpeg_dlls_findable():
            status_label.config(fg='red')
            status_var.set("No FFmpeg libraries (e.g. avcodec*.dll for Windows) found.\n"
                           "The export will most likely fail to start.\n"
                           "Install latest FFmpeg shared libraries for your OS.")

        self._show_dialog(win)

    def show_help_dialog(self):
        """Show a help window with keyboard and mouse shortcuts."""
        if self.rt is None:
            return

        # If already open, just bring it to front
        if hasattr(self, '_help_dialog') and self._help_dialog is not None:
            try:
                if self._help_dialog.winfo_exists():
                    self._help_dialog.lift()
                    self._help_dialog.focus_set()
                    return
            except Exception:
                pass

        def forget_it():
            self._help_dialog = None

        # Not kept above the main window and not taking its keys: the point of
        # it is to be read while the Moon is being driven
        help_win, main_frame, on_close = self._dialog_window(
            "Help - Keys and mouse", padding=(12, 6), takes_keys=False,
            over_main=False, before_close=forget_it)
        self._help_dialog = help_win

        # The list is put on a canvas so that it can be scrolled. Its lettering
        # is asked for in points and so follows the display's dots per inch,
        # while the number of lines does not change: at 300% scaling the window
        # it wanted was taller than the screen, and opened with its first and
        # last rows cut off above and below. Where there is room for the whole
        # list the canvas is simply made as tall as its contents and no
        # scrollbar appears, which is every ordinary screen - nothing about the
        # window changes there.
        viewport = tk.Canvas(main_frame, highlightthickness=0,
                             bg=help_win.cget('bg'))
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL,
                                 command=viewport.yview)
        viewport.configure(yscrollcommand=scrollbar.set)
        body = tk.Frame(viewport, bg=help_win.cget('bg'))
        viewport.create_window(0, 0, window=body, anchor='nw')
        viewport.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Entries from F1 to M/N use a fixed-width key column so hyphens align
        aligned_lines = [
            ("F1", "Help"),
            ("F2", "Toggle Moon ephemeris panel"),
            ("F3", "Set up the eyepiece / camera field of view frame"),
            ("F4", "Toggle parallactic mode (maintains Moon aligned to celestial north)"),
            ("F5", "NSWE view orientation"),
            ("F6", "NSEW view orientation"),
            ("F7", "SNEW view orientation"),
            ("F8", "SNWE view orientation"),
            ("F9", "Set time to now (in the session timezone)"),
            ("F10", "Set time to now + start auto-advance"),
            ("F11", "Toggle full screen"),
            ("F12", "Save image"),
            ("1-9", "Create/Remove pin (when pins are ON)"),
            ("0", "Toggle pins ON/OFF"),
            ("P", "Toggle labels for all Moon features"),
            ("L", "Toggle standard labels"),
            ("S", "Toggle spot labels"),
            ("G", "Toggle selenographic grid"),
            ("Y", "Toggle markers for sub-solar and sub-Earth points"),
            ("B", "Toggle the field of view frame (set it up with F3)"),
            ("C", "Toggle compass showing how far the view is turned (rotated) from default"),
            ("R", "Toggle locator showing where on the Moon the view is"),
            ("F", "Search for Moon features (craters, mounts etc.)"),
            ("X", "Find clair-obscur events (Lunar X, Jewelled Handle, Rupes Recta ...)"),
            ("U", "Chart when the Moon is up over the coming month"),
            ("K", "Open observation planner (terminator / libration) for Moon feature in status bar"),
            ("I", "Open USGS web page for Moon feature in status bar"),
            ("O", "Open user defined web page (Wikipedia by default) for Moon feature in status bar"),
            ("T", "Open date/time window"),
            ("V", "Export time-lapse video (MP4)"),
            ("A/Z", "Increase/Decrease brightness"),
            ("E/D", "Increase/Decrease gamma correction (0.5 - 5.0)"),
            ("H/J", "Roll view around current view direction"),
            ("Q/W", "Go back/forward in time by step minutes (hold the key to get an animation effect)"),
            ("M/N", "Increase/Decrease time step by 1 minute (max is 1440 - 1 day)"),
        ]

        # Remaining entries have longer keys, no fixed-width alignment
        other_lines = [
            ("Shift + M/N", "Increase/Decrease time step by 60 minutes (max is 1440 - 1 day)"),
            ("Escape", "Leave full screen"),
            ("Home", "Reset camera and time to initial state"),
            ("End", "Reset camera to default state (useful after starting with `--init-view` parameter)"),
            ("Space", "Center and fix view on point under cursor"),
            ("Arrows", "Move view"),
            ("Ctrl + Left/Right", "Rotate view around Moon's polar axis"),
            ("Ctrl + Up/Down", "Rotate view around Moon's equatorial axis"),
            ("Hold and drag left mouse button", "Rotate the eye around Moon"),
            ("Hold and drag right mouse button", "Rotate Moon around the eye (move view)"),
            ("Hold Shift + right mouse button and drag up/down", "Move eye backward/forward"),
            ("Hold Ctrl + drag left mouse button", "Measure distance and elevation difference on Moon surface"),
            ("Hold Shift + left mouse button and drag up/down", "Zoom out/in (more reliable)"),
            ("Mouse wheel up/down", "Zoom in/out (less reliable)"),
        ]

        # Find max key width for aligned section
        max_key_len = max(len(k) for k, _ in aligned_lines if k)

        for key, desc in aligned_lines:
            row = tk.Frame(body)
            row.pack(fill=tk.X)
            key_label = tk.Label(row, text=key, width=max_key_len, anchor='e', font=('Consolas', 9, 'bold'))
            key_label.pack(side=tk.LEFT)
            tk.Label(row, text=" - " + desc, anchor='w', font=('Consolas', 9)).pack(side=tk.LEFT)

        for key, desc in other_lines:
            row = tk.Frame(body)
            row.pack(fill=tk.X)
            key_label = tk.Label(row, text=key, anchor='e', font=('Consolas', 9, 'bold'))
            key_label.pack(side=tk.LEFT)
            tk.Label(row, text=" - " + desc, anchor='w', font=('Consolas', 9)).pack(side=tk.LEFT)

        # As tall as the list where the screen has room for it, and as tall as
        # the screen has room for where it does not.
        #
        # The room is not guessed at. The window is first made as tall as the
        # whole list and asked how tall that makes it, which counts the padding
        # round the list; what the window manager adds on top of that - the
        # title bar and the border - is read off the main window, whose frame
        # corner and client corner are both there to be asked. So the only thing
        # taken on trust is that the screen's usable area starts at its top edge.
        body.update_idletasks()
        wanted_w, wanted_h = body.winfo_reqwidth(), body.winfo_reqheight()
        viewport.config(width=wanted_w, height=wanted_h,
                        scrollregion=(0, 0, wanted_w, wanted_h))
        help_win.update_idletasks()

        root = self.rt._root
        caption = max(0, root.winfo_rooty() - root.winfo_y())
        top = round(self._overlay_px(self.HELP_TOP_PX))
        excess = help_win.winfo_reqheight() - (screen_size()[1] - caption - 2 * top)
        if excess > 0:
            viewport.config(height=wanted_h - excess)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=viewport)
            help_win.resizable(False, True)
            # A row at a time, so the wheel moves the list by whole lines
            rows = body.winfo_children()
            viewport.config(yscrollincrement=rows[0].winfo_reqheight() if rows else 1)

            def on_wheel(event):
                # Three lines to the notch, as Windows scrolls everything else
                viewport.yview_scroll(3 * (-event.delta // 120), "units")

            # Bound on the window, not on the canvas: the wheel arrives at
            # whichever label is under the pointer, and a label does nothing
            # with it, so it travels up to here. The main window's own wheel is
            # bound on its render canvas and is not disturbed.
            help_win.bind("<MouseWheel>", on_wheel)

        # Centred across the main window as any other dialog is, but at the top
        # of the screen rather than centred down it
        help_win.update_idletasks()
        left = root.winfo_x() + (root.winfo_width() - help_win.winfo_reqwidth()) // 2
        self._show_dialog(help_win, position=(left, top), grab=False)

    def save_image_dialog(self):
        """
        Open a save dialog with a custom default filename.
        """
        if self.rt is None:
            return
        
        default_name = self.get_default_filename()
        
        filename = filedialog.asksaveasfilename(
            initialdir=".",
            title="Save output as image",
            initialfile=f"{default_name}.jpg",
            defaultextension=".jpg",
            filetypes=(
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("TIFF 8-bit files", "*.tif"),
                ("TIFF 16-bit files", "*.tiff")
            )
        )
        if filename:
            fname, fext = os.path.splitext(filename)
            bps = "Bps16" if fext.lower() == ".tiff" else "Bps8"
            # The compass, the locator and the field-of-view frame are drawn on
            # the canvas over the render and are not in the buffer the ray
            # tracer saves, so whichever of them is on screen is composited into
            # the file instead. With none showing - or if that goes wrong - the
            # ray tracer writes the file itself, exactly as it always has.
            if not self.save_render_with_overlays(filename, bps):
                self.rt.save_image(filename, bps=bps)
            print(f"Saved: {filename}")

    def get_default_filename(self) -> str:
        """
        Generate a default filename for saving screenshots.
        
        Format: datetime_lat+XX.XXXXXX_lon+XX.XXXXXX_view<orientation>_cam<base64>
        
        The camera parameters (eye, target, up, fov) are encoded into a compact
        base64 string for a shorter filename while remaining fully reversible.
        
        Returns
        -------
        str
            Default filename (without extension)
        """
        parts = []
        
        # 1. Local time in ISO format (replace colons with dots for filename compatibility)
        # Format: YYYY-MM-DDTHH.MM.SS+HH.MM (colons replaced with dots)
        # Truncated to seconds: parse_init_view turns every dot back into a
        # colon, so a fractional part would come back as "SS:ffffff", which only
        # parses at all through a leniency of the older ISO reader
        iso_str = self.dt_local.isoformat(timespec='seconds')
        iso_str = iso_str.replace(':', '.')
        parts.append(iso_str)
        
        # 2. Latitude
        parts.append(f"lat{self.observer.lat:+.6f}")
        
        # 3. Longitude
        parts.append(f"lon{self.observer.lon:+.6f}")
        
        # 4. View orientation
        parts.append(f"view{self.view_orientation}")

        # 5. Parallactic mode flag (0 = OFF, 1 = ON)
        parts.append(f"par{1 if self.parallactic_mode else 0}")

        # 6. Current camera parameters (at the time of screenshot) - encoded as base64
        if self.rt is not None:
            try:
                cam = self.rt.get_camera(self.CAMERA_NAME)
                if cam is not None:
                    camera = Camera(eye=cam["Eye"], target=cam["Target"], up=cam["Up"], fov=self.rt._optix.get_camera_fov(0))
                    camera_encoded = encode_camera(camera)
                    parts.append(f"cam{camera_encoded}")
                else:
                    parts.append("nocam")
            except Exception as e:
                print(f"Error getting camera: {e}")
                parts.append("nocam")
        else:
            parts.append("nocam")
        
        return "_".join(parts)

    def search_feature_dialog(self):
        """
        Open a search dialog to find Moon features by name.
        """
        if self.rt is None:
            return
        
        # A list of names, so it opens at a readable size and can be made
        # bigger. Its parts carry their own padding, so the frame they go in
        # adds none of its own.
        search_win, main_frame, on_close = self._dialog_window(
            "Search Moon Feature", padding=(0, 0), size=(400, 340))

        # Search entry
        frame = tk.Frame(main_frame)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(frame, text="Search:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        entry = tk.Entry(frame, textvariable=search_var, width=40)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Results listbox with scrollbar
        list_frame = tk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Store matching features
        matching_features = []
        
        def update_results(*args):
            nonlocal matching_features
            query = search_var.get().lower().strip()
            listbox.delete(0, tk.END)
            matching_features.clear()
            
            if not query:
                return
            
            for feature in self.moon_features:
                if query in feature.name.lower():
                    matching_features.append(feature)
                    diameter_km = feature.diameter_km
                    listbox.insert(tk.END, f"{feature.name} ({diameter_km:.2f} km)")
        
        def selected_feature():
            selection = listbox.curselection()
            if not selection and listbox.size() > 0:
                listbox.selection_set(0)
                selection = (0,)
            if selection and matching_features:
                return matching_features[selection[0]]
            return None

        def on_select(event=None):
            selection = listbox.curselection()
            if selection and matching_features:
                feature = matching_features[selection[0]]
                self.center_on_feature(feature)
                on_close()

        def on_planner():
            feature = selected_feature()
            if feature is None:
                return
            on_close()
            self.observation_planner_dialog(feature)

        def on_key(event):
            if event.keysym == 'Return':
                # If listbox has selection, use it; otherwise select first
                if not listbox.curselection() and listbox.size() > 0:
                    listbox.selection_set(0)
                on_select()
            elif event.keysym == 'Escape':
                on_close()
            elif event.keysym == 'Down':
                if listbox.size() > 0:
                    listbox.focus_set()
                    if not listbox.curselection():
                        listbox.selection_set(0)
        
        search_var.trace_add('write', update_results)
        entry.bind('<Key>', on_key)
        listbox.bind('<Double-Button-1>', on_select)
        listbox.bind('<Return>', on_select)

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(btn_frame, text="Observation Planner", command=on_planner).pack(side=tk.RIGHT)

        self._show_dialog(search_win)

        # Focused only now the window is on screen: a widget in a window that
        # is still withdrawn cannot take focus, and the request is simply lost,
        # leaving the first thing typed to go nowhere
        entry.focus_set()

    def sync_datetime_dialog(self):
        """
        Put the moment now being shown into the date/time window, if it is open.

        Its fields would otherwise stand at whatever they were last set to while
        the Moon moved on without them - and pressing Set would then take the
        view back to a time the user had left behind.
        """
        if getattr(self, "_datetime_dialog_show", None) is not None:
            self._datetime_dialog_show(self.dt_local)

    def open_datetime_dialog(self):
        """
        Open a dialog to set date, time, and timezone.
        The dialog stays open and syncs with Q/W key time changes.
        """
        if self.rt is None:
            return
        
        # If already open, just bring it to front
        if self.datetime_dialog is not None and self.datetime_dialog.winfo_exists():
            self.datetime_dialog.lift()
            self.datetime_dialog.focus_set()
            return
        
        def forget_it():
            self.datetime_dialog = None
            self._datetime_dialog_show = None
            self.datetime_dialog_focused = False

        def on_focus_in(event):
            self.datetime_dialog_focused = True

        def on_focus_out(event):
            self.datetime_dialog_focused = False

        # Non-modal and stays open; it keeps the main window's keys, taking
        # only the ones its spinboxes are made of (see datetime_dialog_takes_key)
        dt_win, main_frame, on_close = self._dialog_window(
            "Date/Time", padding=(15, 5), takes_keys=False, before_close=forget_it)
        self.datetime_dialog = dt_win

        dt_win.bind("<FocusIn>", on_focus_in)
        dt_win.bind("<FocusOut>", on_focus_out)
        
        # Get current local time for later use
        current_dt_local = self.dt_local
        
        # Date and Time rows using grid for proper alignment
        grid_frame = tk.Frame(main_frame)
        grid_frame.pack(fill=tk.X, pady=3)
        
        def digits_only(proposed: str, digits: str) -> bool:
            """
            Let a spinbox hold nothing but digits, and no more than it has room
            for. Everything else typed at one is meant for the main window and
            is left alone here to reach it (see custom_key_handler).
            """
            return proposed == "" or (proposed.isdigit() and len(proposed) <= int(digits))

        accepts_digits = dt_win.register(digits_only)

        def spin_row(row: int, label: str, separator: str, parts: list) -> list:
            """
            Build one row of spinboxes separated by a repeated character.

            Parameters
            ----------
            row : int
                Row in grid_frame
            label : str
                Row caption
            separator : str
                Character drawn between the spinboxes
            parts : list
                One (value, low, high, digits, wrap) tuple per spinbox

            Returns
            -------
            list
                One (variable, spinbox) pair per part, in the order given
            """
            tk.Label(grid_frame, text=label, anchor='w').grid(row=row, column=0, sticky='e', pady=2)
            spins = []
            for i, (value, low, high, digits, wrap) in enumerate(parts):
                # Both rows share one grid, a column per part and a column per
                # separator, so the boxes of one line up with those of the other
                # without their widths having to be matched by hand
                column = 1 + 2 * i
                if i:
                    tk.Label(grid_frame, text=separator).grid(row=row, column=column - 1, pady=2)
                var = tk.StringVar(value=f"%0{digits}d" % value)
                spin = tk.Spinbox(grid_frame, textvariable=var, from_=low, to=high, width=digits + 1,
                                  format=f"%0{digits}.0f", wrap=wrap,
                                  validate='key', validatecommand=(accepts_digits, '%P', digits))
                # Stretched to the column, which the widest box in it has set
                spin.grid(row=row, column=column, pady=2, sticky='ew', padx=(5 if i == 0 else 0, 0))
                spins.append((var, spin))
            return spins

        # Date row. The year stops where the bundled Skyfield kernels do, and
        # the day is trimmed to the length of the month chosen (see below)
        (year_var, _), (month_var, _), (day_var, day_spin) = spin_row(
            0, "Date:", "-",
            [(current_dt_local.year, SKYFIELD_MOON_FRAME_START_UTC.year,
              SKYFIELD_MOON_FRAME_END_UTC.year - 1, 4, False),
             (current_dt_local.month, 1, 12, 2, True),
             (current_dt_local.day, 1, 31, 2, True)])

        # Spelled out because the order of the parts is the one thing the boxes
        # cannot show: a reader used to month first would otherwise have to guess
        tk.Label(grid_frame, text="(YYYY-MM-DD)", fg='gray').grid(row=0, column=6, sticky='w',
                                                                  padx=(5, 0), pady=2)

        # Time row. The offset is deliberately not shown: the date set above
        # may fall the other side of a daylight saving change from the one in
        # force now, and a label read at the wrong moment would name the wrong
        # offset. The time is on the session's clock whatever the date.
        (hour_var, hour_spin), (minute_var, _), (second_var, _) = spin_row(
            1, "Local Time:", ":",
            [(current_dt_local.hour, 0, 23, 2, True),
             (current_dt_local.minute, 0, 59, 2, True),
             (current_dt_local.second, 0, 59, 2, True)])

        def clamp_day_to_month(*_):
            """Keep the day within the chosen month, so 31 Jan then Feb gives 28 or 29."""
            try:
                last = calendar.monthrange(int(year_var.get()), int(month_var.get()))[1]
            except ValueError:
                return  # half-typed year or month; the next keystroke settles it
            day_spin.config(to=last)
            if int(day_var.get() or 0) > last:
                day_var.set(f"{last:02d}")

        year_var.trace_add('write', clamp_day_to_month)
        month_var.trace_add('write', clamp_day_to_month)
        clamp_day_to_month()


        # Error label
        error_var = tk.StringVar()
        error_label = tk.Label(main_frame, textvariable=error_var, fg='red')
        error_label.pack(fill=tk.X, pady=2)
        
        # Button frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        def show_datetime(dt):
            """Put a moment into the six spinboxes."""
            for var, value in ((year_var, dt.year), (month_var, dt.month), (day_var, dt.day),
                               (hour_var, dt.hour), (minute_var, dt.minute), (second_var, dt.second)):
                var.set(f"{value:04d}" if var is year_var else f"{value:02d}")

        # Reached by sync_datetime_dialog whenever the clock moves elsewhere
        self._datetime_dialog_show = show_datetime

        def go_to_time():
            """Apply the selected date/time in local timezone."""
            try:
                # A spinbox can still be typed into, so the parts are validated
                # rather than trusted; datetime rejects anything out of range
                new_dt_naive = datetime(int(year_var.get()), int(month_var.get()), int(day_var.get()),
                                        int(hour_var.get()), int(minute_var.get()), int(second_var.get()))

                # Read on the observer's clock, so the daylight saving rules
                # of the chosen date decide the offset (see from_observer_clock)
                new_dt_local = self.from_observer_clock(new_dt_naive)

                # Update the view
                self.update_view(new_dt_local)
                
                # Reset auto-advance counter when time is manually set
                if self._auto_advance_var and self._auto_advance_var.get():
                    self._auto_advance_elapsed = 0
                
                # Update status bar
                self._update_all_status_panels()
                
                error_var.set("")
                
            except Exception as e:
                error_var.set(f"Error: {str(e)}")
        
        def set_now():
            """Set to the current time on the observer's clock."""
            # In the session's timezone, not this machine's: the fields are read
            # back as wall clock in that zone (see from_observer_clock), so the
            # system reading would land on the wrong instant for a session
            # planned elsewhere
            show_datetime(datetime.now(self.dt_local.tzinfo))

        def sync_from_renderer():
            """Sync dialog fields with current renderer time."""
            show_datetime(self.dt_local)
        
        tk.Button(btn_frame, text="Now", command=set_now, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Sync with Moon", command=sync_from_renderer, width=16).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Set", command=go_to_time, width=10).pack(side=tk.RIGHT, padx=5)

        # Enter applies the reading wherever it is typed: bound on the window
        # rather than on each spinbox, so it answers from all six and from the
        # buttons too. Both Enters, the keypad having a key of its own.
        for enter in ('<Return>', '<KP_Enter>'):
            dt_win.bind(enter, lambda event: go_to_time())
        
        # Sized to what it holds rather than to a fixed guess, so the row of
        # buttons ends where the window does. The size is still pinned: without
        # it the window would grow the moment an error message appears.
        dt_win.update_idletasks()
        dt_win.geometry(f"{dt_win.winfo_reqwidth()}x{dt_win.winfo_reqheight()}")

        # Near the top-right of the main window rather than centred, so it
        # does not cover the Moon while the time is being set
        self._show_dialog(dt_win, (self.rt._root.winfo_x() + self.rt._root.winfo_width()
                                   - dt_win.winfo_reqwidth() - 50,
                                   self.rt._root.winfo_y() + 100), grab=False)
        
        # Focus on the hour for quick editing, but leave what is in it alone:
        # selected, the first digit typed would replace the whole reading
        hour_spin.focus_set()
