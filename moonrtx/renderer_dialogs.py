"""
DialogsMixin: dialog windows (help, search, save, datetime) for MoonRenderer.
"""

import os
import base64
import struct
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

from moonrtx import astro
from moonrtx.shared_types import Camera, MoonFeature

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

    # Terminator planner filter settings (see astro.find_terminator_windows):
    # a feature is worth observing while the Sun stands 0-12 degrees above it
    # (terrain lit, shadows long) and the Moon is usefully above the horizon
    PLANNER_SCAN_DAYS = 60
    PLANNER_SUN_ALT_MAX = 12.0
    PLANNER_MOON_ALT_MIN = 5.0

    def terminator_planner_dialog(self, feature: MoonFeature):
        """
        Show upcoming windows when the given feature is observable near the
        terminator (see astro.find_terminator_windows) and let the user jump
        the app time to a selected window.

        Parameters
        ----------
        feature : MoonFeature
            The feature to plan for (None is ignored, so the method can be
            called directly with the status-bar feature)
        """
        if self.rt is None or feature is None:
            return

        try:
            windows = astro.find_terminator_windows(
                self.dt_local, self.PLANNER_SCAN_DAYS, feature.lat, feature.lon,
                sun_alt_max=self.PLANNER_SUN_ALT_MAX,
                moon_alt_min=self.PLANNER_MOON_ALT_MIN)
        except ValueError as e:
            # Scan start outside the bundled ephemeris kernel range
            print(f"Terminator planner: {e}")
            return

        # Reuse the search-dialog flag: it blocks main-window key handling
        # for this dialog in exactly the same way
        self.search_dialog_open = True

        win = tk.Toplevel(self.rt._root)
        win.title(f"Terminator planner - {feature.name}")
        win.transient(self.rt._root)
        win.grab_set()
        win.resizable(False, False)

        def on_close():
            self.search_dialog_open = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

        main_frame = tk.Frame(win, padx=12, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            main_frame,
            text=(f"{feature.name} (lat {feature.lat:.2f}°, lon {feature.lon:.2f}°) - times in the next "
                  f"{self.PLANNER_SCAN_DAYS} days when the Sun stands\n"
                  f"0-{self.PLANNER_SUN_ALT_MAX:.0f}° above the feature (lit, long shadows) and the Moon is at least "
                  f"{self.PLANNER_MOON_ALT_MIN:.0f}° up in your sky."),
            justify=tk.LEFT, anchor='w', font=('Consolas', 9)
        ).pack(fill=tk.X, pady=(0, 8))

        local_tz = self.dt_local.tzinfo
        font = ('Consolas', 9)

        if not windows:
            tk.Label(main_frame, text="No opportunities found in the scanned period.",
                     font=font, anchor='w').pack(fill=tk.X, pady=6)
        else:
            header = (f"{'Best time (local)':<22}{'Event':<9}{'Window (local)':<29}"
                      f"{'Sun@feat':>7}{'Moon alt':>10}  Sky")
            tk.Label(main_frame, text=header, font=('Consolas', 9, 'bold'),
                     anchor='w').pack(fill=tk.X)

            list_frame = tk.Frame(main_frame)
            list_frame.pack(fill=tk.BOTH, expand=True)
            scrollbar = tk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=font,
                                 width=len(header) + 2, height=min(len(windows), 16))
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)

            for w in windows:
                best = w["best"].astimezone(local_tz)
                start = w["start"].astimezone(local_tz)
                end = w["end"].astimezone(local_tz)
                if w["observer_sun_alt"] > 0.0:
                    sky = "day"
                elif w["observer_sun_alt"] > -12.0:
                    sky = "twilight"
                else:
                    sky = "night"
                listbox.insert(tk.END,
                               f"{best:%Y-%m-%d %a %H:%M}  {w['event']:<9}"
                               f"{start:%m-%d %H:%M} .. {end:%m-%d %H:%M}   "
                               f"{w['sun_alt']:>6.1f}°{w['moon_alt']:>9.0f}°  {sky}")
            listbox.selection_set(0)

            def go_to(event=None):
                selection = listbox.curselection()
                if not selection:
                    return
                target = windows[selection[0]]["best"].astimezone(local_tz)
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
        if windows:
            tk.Button(btn_frame, text="Go to best time", command=go_to, width=16).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Close", command=on_close, width=10).pack(side=tk.RIGHT)
        win.bind('<Escape>', lambda e: on_close())

        # Center on main window
        win.update_idletasks()
        x = self.rt._root.winfo_x() + (self.rt._root.winfo_width() - win.winfo_width()) // 2
        y = self.rt._root.winfo_y() + (self.rt._root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

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
            ("F4", "Toggle parallactic mode (maintains Moon aligned to celestial north)"),
            ("F5", "NSWE view orientation"),
            ("F6", "NSEW view orientation"),
            ("F7", "SNEW view orientation"),
            ("F8", "SNWE view orientation"),
            ("F9", "Set time to now using system timezone"),
            ("F10", "Set time to now + start auto-advance"),
            ("F12", "Save image"),
            ("1-9", "Create/Remove pin (when pins are ON)"),
            ("G", "Toggle selenographic grid"),
            ("L", "Toggle standard labels"),
            ("S", "Toggle spot labels"),
            ("P", "Toggle pins ON/OFF"),
            ("R", "Reset view and time to initial state"),
            ("V", "Reset view to that based on current time (useful after starting with --init-view parameter)"),
            ("C", "Center and fix view on point under cursor"),
            ("F", "Search for Moon features (craters, mounts etc.)"),
            ("K", "Open terminator planner for Moon feature shown in status bar"),
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

        # Center on main window
        help_win.update_idletasks()
        x = self.rt._root.winfo_x() + (self.rt._root.winfo_width() - help_win.winfo_width()) // 2
        y = self.rt._root.winfo_y() + (self.rt._root.winfo_height() - help_win.winfo_height()) // 2
        help_win.geometry(f"+{x}+{y}")

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
        iso_str = self.dt_local.isoformat()
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
        search_win.title("Search Moon Feature")
        search_win.geometry("400x340")
        search_win.transient(self.rt._root)
        search_win.grab_set()
        
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
            self.terminator_planner_dialog(feature)

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
        tk.Button(btn_frame, text="Terminator planner", command=on_planner).pack(side=tk.RIGHT)

        # Center the window
        search_win.update_idletasks()
        x = self.rt._root.winfo_x() + (self.rt._root.winfo_width() - search_win.winfo_width()) // 2
        y = self.rt._root.winfo_y() + (self.rt._root.winfo_height() - search_win.winfo_height()) // 2
        search_win.geometry(f"+{x}+{y}")

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
        
        # Get current local time and its timezone for later use
        current_dt_local = self.dt_local
        local_tz = current_dt_local.tzinfo
        
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
                
                # Apply the fixed local timezone
                new_dt_local = new_dt_naive.replace(tzinfo=local_tz)
                
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
            """Set to current system local time."""
            nonlocal local_tz
            now_local = datetime.now().astimezone()
            local_tz = now_local.tzinfo
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
        
        # Position near the top-right of the main window
        dt_win.update_idletasks()
        x = self.rt._root.winfo_x() + self.rt._root.winfo_width() - dt_win.winfo_width() - 50
        y = self.rt._root.winfo_y() + 100
        dt_win.geometry(f"+{x}+{y}")
        
        # Focus on time entry for quick editing
        time_entry.focus_set()
        time_entry.select_range(0, tk.END)
