"""
DialogsMixin: dialog windows (help, search, save, datetime) for MoonRenderer.
"""

import os
import glob
import base64
import struct
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

from moonrtx import astro
from moonrtx.shared_types import Camera, MoonFeature


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
        win.geometry(f"+{position[0]}+{position[1]}")
        win.deiconify()
        if grab:
            win.wait_visibility()
            win.grab_set()

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
    CLAIR_OBSCUR_SCAN_DAYS = 120
    CLAIR_OBSCUR_STEP_MINUTES = 30
    CLAIR_OBSCUR_ALL_EVENTS = "All events"
    # Filters last chosen in the dialog. Declared on the class so the first
    # opening has defaults; changing one stores it on the instance, so both hold
    # for the rest of the session and start over on the next run.
    _clair_obscur_filter = CLAIR_OBSCUR_ALL_EVENTS
    _clair_obscur_visible_only = True

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

        # Reuse the search-dialog flag: it blocks main-window key handling
        # for this dialog in exactly the same way
        self.search_dialog_open = True

        win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        win.withdraw()
        win.title("Clair-obscur events")
        win.transient(self.rt._root)
        win.resizable(False, False)

        def on_close():
            self.search_dialog_open = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        win.bind('<Escape>', lambda e: on_close())

        main_frame = tk.Frame(win, padx=12, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        font = ('Consolas', 9)
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

        tk.Label(main_frame, anchor='w', font=font,
                 text=f"Shapes drawn by the terminator, over the next "
                      f"{self.CLAIR_OBSCUR_SCAN_DAYS} days"
                 ).pack(fill=tk.X)

        all_events = self.CLAIR_OBSCUR_ALL_EVENTS
        event_names = [all_events] + [e.name for e in astro.CLAIR_OBSCUR_EVENTS]
        # Resume the filter this session was last left on, unless it names an
        # event the catalogue no longer has
        filter_var = tk.StringVar(
            value=self._clair_obscur_filter if self._clair_obscur_filter in event_names
            else all_events)

        filter_row = tk.Frame(main_frame)
        filter_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(filter_row, text="Show:", anchor='w').pack(side=tk.LEFT)
        option = tk.OptionMenu(filter_row, filter_var, *event_names, command=lambda _: rescan())
        option.config(width=max(len(name) for name in event_names), anchor='w')
        option.pack(side=tk.LEFT, padx=(4, 12))

        visible_only_var = tk.BooleanVar(value=self._clair_obscur_visible_only)
        # Naming the column ties the filter to the figure it acts on
        tk.Checkbutton(filter_row, variable=visible_only_var, anchor='w',
                       text=f"Only when the Moon altitude (h☾) is at least "
                            f"{self.PLANNER_MOON_ALT_MIN:.0f}° in my sky",
                       command=lambda: rescan()).pack(side=tk.LEFT)

        desc_var = tk.StringVar()
        desc_label = tk.Label(main_frame, textvariable=desc_var, justify=tk.LEFT,
                              anchor='nw', font=font, height=3)
        desc_label.pack(fill=tk.X, pady=(4, 6))

        tk.Label(main_frame, text=header, font=('Consolas', 9, 'bold'),
                 anchor='w').pack(fill=tk.X)

        list_frame = tk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=font,
                             width=len(header) + 2, height=16)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        win.update_idletasks()
        desc_label.config(wraplength=listbox.winfo_reqwidth())

        def sky_of(o):
            if o["observer_sun_alt"] > 0.0:
                return "day"
            return "twilight" if o["observer_sun_alt"] > -12.0 else "night"

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
                               f"{visible:<16}{sun_alt:<8}{moon_alt:<8}  {sky_of(o)}")
            listbox.selection_set(0)
            show_description()

        def go_to(event=None):
            selection = listbox.curselection()
            if not events or not selection or selection[0] >= len(events):
                return
            o = events[selection[0]]
            on_close()
            self.update_view(self.in_observer_clock(o["peak"]))
            if self._auto_advance_var and self._auto_advance_var.get():
                self._auto_advance_elapsed = 0
            self._update_all_status_panels()
            self.center_on_lat_lon(o["lat"], o["lon"])

        listbox.bind('<<ListboxSelect>>', show_description)
        listbox.bind('<Double-Button-1>', go_to)
        listbox.bind('<Return>', go_to)

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        tk.Button(btn_frame, text="Go to selected", command=go_to, width=16).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Close", command=on_close, width=10).pack(side=tk.RIGHT)

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

        # Reuse the search-dialog flag: it blocks main-window key handling
        # for this dialog in exactly the same way
        self.search_dialog_open = True

        win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        win.withdraw()
        win.title(f"Observation Planner - {feature.name}")
        win.transient(self.rt._root)
        win.resizable(False, False)

        def on_close():
            self.search_dialog_open = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        win.bind('<Escape>', lambda e: on_close())

        main_frame = tk.Frame(win, padx=12, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        font = ('Consolas', 9)
        windows = []                     # results currently listed

        # The sky column is padded to its longest value ("twilight"), so the
        # header length already covers the widest row and the listbox sized
        # from it never clips one
        terminator_header = (f"{'Best time (local)':<22}{'Event':<9}{'Window (local)':<29}"
                             f"{'Sun@feat':>7}{'Moon alt':>10}  {'Sky':<8}")
        libration_header = (f"{'Best time (local)':<22}{'Window (local)':<29}{'Presented':>10}"
                            f"{'Libr L':>9}{'Libr B':>9}{'Sun@feat':>10}{'Moon alt':>10}  {'Sky':<8}")

        tk.Label(main_frame, anchor='w', font=font,
                 text=f"{feature.name}  (lat {feature.lat:.2f}°, lon {feature.lon:.2f}°)"
                      f"  -  next {self.PLANNER_SCAN_DAYS} days"
                 ).pack(fill=tk.X)

        mode_var = tk.StringVar(value="terminator")
        mode_row = tk.Frame(main_frame)
        mode_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(mode_row, text="Show:", anchor='w').pack(side=tk.LEFT)
        for value, label in (("terminator", "near the terminator"),
                             ("libration", "best presented (libration)")):
            tk.Radiobutton(mode_row, text=label, value=value, variable=mode_var,
                           command=lambda: rescan()).pack(side=tk.LEFT)

        desc_var = tk.StringVar()
        # Wrapped to the width of the list below (set once that knows its size)
        # rather than at hand-placed newlines, so the text fills the dialog.
        # The fixed height keeps it from resizing when the mode changes.
        desc_label = tk.Label(main_frame, textvariable=desc_var, justify=tk.LEFT,
                              anchor='nw', font=font, height=3)
        desc_label.pack(fill=tk.X, pady=(4, 6))

        header_var = tk.StringVar()
        tk.Label(main_frame, textvariable=header_var, font=('Consolas', 9, 'bold'),
                 anchor='w').pack(fill=tk.X)

        list_frame = tk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=font,
                             width=max(len(terminator_header), len(libration_header)) + 2,
                             height=16)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        win.update_idletasks()
        desc_label.config(wraplength=listbox.winfo_reqwidth())

        def sky_of(w):
            if w["observer_sun_alt"] > 0.0:
                return "day"
            return "twilight" if w["observer_sun_alt"] > -12.0 else "night"

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
                                   f"{w['moon_alt']:>9.0f}°  {sky_of(w)}")
                else:
                    listbox.insert(tk.END,
                                   f"{best:%Y-%m-%d %a %H:%M}  {w['event']:<9}{span}"
                                   f"{w['sun_alt']:>6.1f}°{w['moon_alt']:>9.0f}°  {sky_of(w)}")
            listbox.selection_set(0)

        def go_to(event=None):
            selection = listbox.curselection()
            if not windows or not selection or selection[0] >= len(windows):
                return
            target = self.in_observer_clock(windows[selection[0]]["best"])
            on_close()
            self.update_view(target)
            if self._auto_advance_var and self._auto_advance_var.get():
                self._auto_advance_elapsed = 0
            self._update_all_status_panels()
            self.center_on_feature(feature)

        listbox.bind('<Double-Button-1>', go_to)
        listbox.bind('<Return>', go_to)

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        tk.Button(btn_frame, text="Go to selected", command=go_to, width=16).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Close", command=on_close, width=10).pack(side=tk.RIGHT)

        rescan()

        self._show_dialog(win)

    def export_video_dialog(self):
        """
        Open the time-lapse video export dialog: renders N frames from the
        current observation time, advancing by a configurable number of
        simulated minutes per frame, into an MP4 (H.264, NVENC hardware
        encoding). See MoonRenderer.start_video_export for the mechanics.
        """
        if self.rt is None:
            return

        # Reuse the search-dialog flag: it blocks main-window key handling
        # for this dialog in exactly the same way
        self.search_dialog_open = True

        win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        win.withdraw()
        win.title("Export time-lapse video")
        win.transient(self.rt._root)
        win.resizable(False, False)

        exporting = {"active": False}

        def on_close():
            if exporting["active"]:
                # Closing during export only requests cancellation; the dialog
                # stays open to show the final status and can be closed then
                self.cancel_video_export()
                return
            self.search_dialog_open = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        win.bind('<Escape>', lambda e: on_close())

        main_frame = tk.Frame(win, padx=15, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

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
        burn_time_cb = tk.Checkbutton(main_frame, variable=burn_time_var, anchor='w', text="Show local time")
        burn_time_cb.pack(fill=tk.X, pady=(4, 0))

        time_corner_var = tk.StringVar(value=self.VIDEO_TIME_CORNER)
        caption_corner_var = tk.StringVar(value=self.VIDEO_CAPTION_CORNER)

        def corner_row(parent, var, command):
            """Row of radio buttons selecting one of the four frame corners."""
            row = tk.Frame(parent)
            row.pack(fill=tk.X)
            buttons = []
            for corner in self.VIDEO_CORNERS:
                rb = tk.Radiobutton(row, text=corner, value=corner, variable=var,
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
                                            caption_corner=caption_corner_var.get())
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

        help_win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        help_win.withdraw()
        help_win.title("Help - Keys and mouse")
        help_win.resizable(False, False)
        self._help_dialog = help_win

        def on_close():
            self._help_dialog = None
            help_win.destroy()

        help_win.protocol("WM_DELETE_WINDOW", on_close)

        main_frame = tk.Frame(help_win, padx=15, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

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
            ("F11", "Export time-lapse video (MP4)"),
            ("F12", "Save image"),
            ("1-9", "Create/Remove pin (when pins are ON)"),
            ("G", "Toggle selenographic grid"),
            ("L", "Toggle standard labels"),
            ("S", "Toggle spot labels"),
            ("P", "Toggle pins ON/OFF"),
            ("B", "Toggle the field of view frame (set it up with F3)"),
            ("R", "Reset view and time to initial state"),
            ("V", "Reset view to that based on current time (useful after starting with --init-view parameter)"),
            ("C", "Center and fix view on point under cursor"),
            ("F", "Search for Moon features (craters, mounts etc.)"),
            ("K", "Open observation planner (terminator / libration) for Moon feature in status bar"),
            ("X", "Find clair-obscur events (Lunar X, Jewelled Handle, Rupes Recta ...)"),
            ("I", "Open USGS web page for Moon feature shown in status bar"),
            ("O", "Open user defined web page (Wiki by default) for Moon feature shown in status bar"),
            ("T", "Open date/time window"),
            ("A/Z", "Increase/Decrease brightness"),
            ("E/D", "Increase/Decrease gamma correction (0.5 - 5.0)"),
            ("H/J", "Roll view around current view direction"),
            ("Q/W", "Go back/forward in time by step minutes (hold the key to get an animation effect)"),
            ("M/N", "Increase/Decrease time step by 1 minute (max is 1440 - 1 day)"),
        ]

        # Remaining entries have longer keys, no fixed-width alignment
        other_lines = [
            ("Shift + M/N", "Increase/Decrease time step by 60 minutes (max is 1440 - 1 day)"),
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
            row = tk.Frame(main_frame)
            row.pack(fill=tk.X, pady=1)
            key_label = tk.Label(row, text=key, width=max_key_len, anchor='e', font=('Consolas', 9, 'bold'))
            key_label.pack(side=tk.LEFT)
            tk.Label(row, text=" - " + desc, anchor='w', font=('Consolas', 9)).pack(side=tk.LEFT)

        for key, desc in other_lines:
            row = tk.Frame(main_frame)
            row.pack(fill=tk.X, pady=1)
            key_label = tk.Label(row, text=key, anchor='e', font=('Consolas', 9, 'bold'))
            key_label.pack(side=tk.LEFT)
            tk.Label(row, text=" - " + desc, anchor='w', font=('Consolas', 9)).pack(side=tk.LEFT)

        # Close button
        tk.Button(main_frame, text="Close", command=on_close, width=10).pack(pady=(10, 0))

        self._show_dialog(help_win, grab=False)

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
            if fext.lower() == ".tiff":
                self.rt.save_image(filename, bps="Bps16")
            else:
                self.rt.save_image(filename, bps="Bps8")
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
        
        # Set flag to prevent main window key handling
        self.search_dialog_open = True
        
        # Create search window
        search_win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        search_win.withdraw()
        search_win.title("Search Moon Feature")
        search_win.geometry("400x340")
        search_win.transient(self.rt._root)
        
        def on_close():
            self.search_dialog_open = False
            search_win.destroy()
        
        search_win.protocol("WM_DELETE_WINDOW", on_close)
        
        # Search entry
        frame = tk.Frame(search_win)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(frame, text="Search:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        entry = tk.Entry(frame, textvariable=search_var, width=40)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        entry.focus_set()
        
        # Results listbox with scrollbar
        list_frame = tk.Frame(search_win)
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

        btn_frame = tk.Frame(search_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(btn_frame, text="Observation Planner", command=on_planner).pack(side=tk.RIGHT)

        self._show_dialog(search_win)

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
        
        # Create datetime window (non-modal, stays open)
        dt_win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        dt_win.withdraw()
        dt_win.title("Date/Time")
        dt_win.geometry("360x130")
        dt_win.transient(self.rt._root)
        dt_win.resizable(False, False)
        
        self.datetime_dialog = dt_win
        
        def on_close():
            self.datetime_dialog = None
            self.datetime_dialog_focused = False
            dt_win.destroy()
        
        def on_focus_in(event):
            self.datetime_dialog_focused = True
        
        def on_focus_out(event):
            self.datetime_dialog_focused = False
        
        dt_win.protocol("WM_DELETE_WINDOW", on_close)
        dt_win.bind("<FocusIn>", on_focus_in)
        dt_win.bind("<FocusOut>", on_focus_out)
        
        # Main frame with padding
        main_frame = tk.Frame(dt_win, padx=15, pady=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Get current local time for later use
        current_dt_local = self.dt_local
        
        # Date and Time rows using grid for proper alignment
        grid_frame = tk.Frame(main_frame)
        grid_frame.pack(fill=tk.X, pady=3)
        
        # Format timezone offset as +HH:MM or -HH:MM
        offset = current_dt_local.strftime('%z')  # e.g., +0100
        offset_formatted = f"{offset[:3]}:{offset[3:]}" if offset else ""  # e.g., +01:00
        
        # Date row
        tk.Label(grid_frame, text="Date:", anchor='w').grid(row=0, column=0, sticky='e', pady=2)
        date_var = tk.StringVar(value=current_dt_local.strftime('%Y-%m-%d'))
        date_entry = tk.Entry(grid_frame, textvariable=date_var, width=15)
        date_entry.grid(row=0, column=1, padx=5, pady=2)
        tk.Label(grid_frame, text="(YYYY-MM-DD)", fg='gray').grid(row=0, column=2, sticky='w', pady=2)
        
        # Time row
        tz_label_var = tk.StringVar(value=f"Local Time (UTC{offset_formatted}):")
        tk.Label(grid_frame, textvariable=tz_label_var, anchor='e').grid(row=1, column=0, sticky='w', pady=2)
        time_var = tk.StringVar(value=current_dt_local.strftime('%H:%M:%S'))
        time_entry = tk.Entry(grid_frame, textvariable=time_var, width=15)
        time_entry.grid(row=1, column=1, padx=5, pady=2)
        tk.Label(grid_frame, text="(HH:MM:SS)", fg='gray').grid(row=1, column=2, sticky='w', pady=2)
        
        # Error label
        error_var = tk.StringVar()
        error_label = tk.Label(main_frame, textvariable=error_var, fg='red')
        error_label.pack(fill=tk.X, pady=2)
        
        # Button frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        def go_to_time():
            """Apply the selected date/time in local timezone."""
            try:
                date_str = date_var.get().strip()
                time_str = time_var.get().strip()
                
                # Parse date and time
                dt_str = f"{date_str} {time_str}"
                try:
                    new_dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Try without seconds
                    new_dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
                
                # Read on the observer's clock, so the daylight saving rules
                # of the typed date decide the offset (see from_observer_clock)
                new_dt_local = self.from_observer_clock(new_dt_naive)
                # The typed date may fall the other side of a daylight saving
                # change, so the label follows the offset actually applied
                offset = new_dt_local.strftime('%z')
                tz_label_var.set(f"Local Time (UTC{offset[:3]}:{offset[3:]}):")
                
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
            now_local = datetime.now(self.dt_local.tzinfo)
            offset = now_local.strftime('%z')
            offset_fmt = f"{offset[:3]}:{offset[3:]}" if offset else ""
            tz_label_var.set(f"Local Time (UTC{offset_fmt}):")
            date_var.set(now_local.strftime('%Y-%m-%d'))
            time_var.set(now_local.strftime('%H:%M:%S'))
        
        def sync_from_renderer():
            """Sync dialog fields with current renderer time."""
            current_dt_local = self.dt_local
            date_var.set(current_dt_local.strftime('%Y-%m-%d'))
            time_var.set(current_dt_local.strftime('%H:%M:%S'))
        
        tk.Button(btn_frame, text="Now", command=set_now, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Sync with Moon", command=sync_from_renderer, width=16).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Set", command=go_to_time, width=10).pack(side=tk.RIGHT, padx=5)
        
        # Near the top-right of the main window rather than centred, so it
        # does not cover the Moon while the time is being set
        dt_win.update_idletasks()
        self._show_dialog(dt_win, (self.rt._root.winfo_x() + self.rt._root.winfo_width()
                                   - dt_win.winfo_width() - 50,
                                   self.rt._root.winfo_y() + 100), grab=False)
        
        # Focus on time entry for quick editing
        time_entry.focus_set()
        time_entry.select_range(0, tk.END)
