from multiprocessing import Process
import calendar
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import os
import json

from moonrtx.moon_renderer import run_renderer
from moonrtx.shared_types import CameraParams

from moonrtx.main import (
    get_date_time_local,
    parse_init_view,
    check_elevation_file,
    check_color_file,
    check_starmap_file,
    check_gpu_architecture,
    DEFAULT_ELEVATION_FILE_LOCAL_PATH,
    APP_NAME,
    DEFAULT_COLOR_FILE_LOCAL_PATH,
    STARMAP_FILE_LOCAL_PATH,
    MOON_FEATURES_FILE_LOCAL_PATH,
    DATA_DIRECTORY_PATH,
    BASE_PATH,
    VALID_ORIENTATIONS,
    ORIENTATION_NSWE
)

# Generate UTC offset values for the timezone combobox (-12:00 to +14:00, 30-min steps)
_TZ_OFFSETS = []
for _total_min in range(-720, 841, 30):
    _h, _m = divmod(abs(_total_min), 60)
    _sign = '+' if _total_min >= 0 else '-'
    _TZ_OFFSETS.append(f"{_sign}{_h:02d}:{_m:02d}")


class CalendarPopup(tk.Toplevel):
    """Simple calendar popup for date selection."""

    def __init__(self, parent, year=None, month=None, day=None):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Select Date")
        self.resizable(False, False)

        self.result = None
        self.cal = calendar.Calendar(firstweekday=0)  # Monday first

        now = datetime.now()
        self.year = year or now.year
        self.month = month or now.month
        self.selected_day = day

        self._build_ui()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window()

    def _build_ui(self):
        nav = tk.Frame(self)
        nav.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(nav, text="\u25C0", command=self._prev_month, width=3).pack(side=tk.LEFT)
        self.month_label = tk.Label(nav, text="", font=("", 10, "bold"), width=18)
        self.month_label.pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="\u25B6", command=self._next_month, width=3).pack(side=tk.RIGHT)

        self.days_frame = tk.Frame(self)
        self.days_frame.pack(padx=5, pady=(0, 5))

        self._render_month()

    def _render_month(self):
        for w in self.days_frame.winfo_children():
            w.destroy()

        self.month_label.config(text=f"{calendar.month_name[self.month]} {self.year}")

        for i, day_name in enumerate(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]):
            tk.Label(self.days_frame, text=day_name, width=4, font=("", 9, "bold")).grid(row=0, column=i)

        today = datetime.now()
        for row_idx, week in enumerate(self.cal.monthdayscalendar(self.year, self.month)):
            for col_idx, day in enumerate(week):
                if day == 0:
                    tk.Label(self.days_frame, text="", width=4).grid(row=row_idx + 1, column=col_idx)
                else:
                    btn = tk.Button(self.days_frame, text=str(day), width=4,
                                    command=lambda d=day: self._select(d))
                    if day == self.selected_day:
                        btn.config(relief=tk.SUNKEN, bg="lightblue")
                    elif (day == today.day and self.month == today.month
                          and self.year == today.year):
                        btn.config(fg="blue")
                    btn.grid(row=row_idx + 1, column=col_idx)

    def _prev_month(self):
        if self.month == 1:
            self.month = 12
            self.year -= 1
        else:
            self.month -= 1
        self.selected_day = None
        self._render_month()

    def _next_month(self):
        if self.month == 12:
            self.month = 1
            self.year += 1
        else:
            self.month += 1
        self.selected_day = None
        self._render_month()

    def _select(self, day):
        self.result = f"{self.year:04d}-{self.month:02d}-{day:02d}"
        self.destroy()


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} - GUI Launcher")
        self.resizable(False, False)
        self._build_ui()

    def _build_ui(self):

        frm = tk.Frame(self)
        frm.pack(padx=10, pady=10)

        tk.Label(frm, text="Observer latitude:").grid(row=0, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Observer longitude:").grid(row=1, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Elevation (meters):").grid(row=2, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Time with timezone:").grid(row=3, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Elevation file:").grid(row=4, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Color file:").grid(row=5, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Downscale:").grid(row=6, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Brightness:").grid(row=7, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Gamma:").grid(row=8, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Time step (minutes):").grid(row=9, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="View orientation:").grid(row=10, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Init view parameter:").grid(row=11, column=0, sticky=tk.E, pady=2)

        self.lat_dir_var = tk.StringVar(value="N")
        self.lon_dir_var = tk.StringVar(value="E")

        self.lat_decimal_frame = tk.Frame(frm)
        self.lat_decimal_frame.grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Combobox(self.lat_decimal_frame, width=2, state="readonly",
                     values=["N", "S"], textvariable=self.lat_dir_var).pack(side=tk.LEFT, padx=(0, 4))
        self.lat_decimal = tk.Entry(self.lat_decimal_frame)
        self.lat_decimal.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lon_decimal_frame = tk.Frame(frm)
        self.lon_decimal_frame.grid(row=1, column=1, sticky=tk.EW, pady=2)
        ttk.Combobox(self.lon_decimal_frame, width=2, state="readonly",
                     values=["E", "W"], textvariable=self.lon_dir_var).pack(side=tk.LEFT, padx=(0, 4))
        self.lon_decimal = tk.Entry(self.lon_decimal_frame)
        self.lon_decimal.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.elevation_entry = tk.Entry(frm, width=5)
        self.elevation_entry.grid(row=2, column=1, sticky=tk.EW, pady=2)
        self.elevation_entry.insert(0, "0")

        # Date / Time / Timezone widgets
        self.time_frame = tk.Frame(frm)
        self.time_frame.grid(row=3, column=1, sticky=tk.W, pady=2)

        now = datetime.now().astimezone()

        self.date_entry = tk.Entry(self.time_frame, width=12)
        self.date_entry.pack(side=tk.LEFT)
        self.date_entry.insert(0, now.strftime("%Y-%m-%d"))

        tk.Button(self.time_frame, text="\u25BC", width=2, command=self._open_calendar).pack(side=tk.LEFT, padx=(1, 4))

        self.hour_var = tk.StringVar(value=f"{now.hour:02d}")
        self.minute_var = tk.StringVar(value=f"{now.minute:02d}")
        self.second_var = tk.StringVar(value=f"{now.second:02d}")

        self.hour_spin = tk.Spinbox(self.time_frame, from_=0, to=23, width=3,
                                    textvariable=self.hour_var, format="%02.0f", wrap=True)
        self.hour_spin.pack(side=tk.LEFT)
        tk.Label(self.time_frame, text=":").pack(side=tk.LEFT)
        self.minute_spin = tk.Spinbox(self.time_frame, from_=0, to=59, width=3,
                                      textvariable=self.minute_var, format="%02.0f", wrap=True)
        self.minute_spin.pack(side=tk.LEFT)
        tk.Label(self.time_frame, text=":").pack(side=tk.LEFT)
        self.second_spin = tk.Spinbox(self.time_frame, from_=0, to=59, width=3,
                                      textvariable=self.second_var, format="%02.0f", wrap=True)
        self.second_spin.pack(side=tk.LEFT)

        tk.Label(self.time_frame, text="  TZ:").pack(side=tk.LEFT)
        tz_offset = now.strftime("%z")  # e.g. '+0100'
        tz_str = f"{tz_offset[:3]}:{tz_offset[3:]}" if tz_offset else "+00:00"
        self.tz_combo = ttk.Combobox(self.time_frame, width=7, values=_TZ_OFFSETS)
        self.tz_combo.pack(side=tk.LEFT)
        self.tz_combo.set(tz_str)

        self.elevation_file = tk.Entry(frm, width=5)
        self.elevation_file.insert(0, DEFAULT_ELEVATION_FILE_LOCAL_PATH)
        self.elevation_file.grid(row=4, column=1, sticky=tk.EW, pady=2)

        self.color_file = tk.Entry(frm, width=5)
        self.color_file.insert(0, DEFAULT_COLOR_FILE_LOCAL_PATH)
        self.color_file.grid(row=5, column=1, sticky=tk.EW, pady=2)

        self.downscale = tk.Entry(frm, width=5)
        self.downscale.grid(row=6, column=1, sticky=tk.EW, pady=2)
        self.downscale.insert(0, 3)

        self.brightness = tk.Entry(frm, width=5)
        self.brightness.grid(row=7, column=1, sticky=tk.EW, pady=2)
        self.brightness.insert(0, 80)

        self.gamma_entry = tk.Entry(frm, width=5)
        self.gamma_entry.grid(row=8, column=1, sticky=tk.EW, pady=2)
        self.gamma_entry.insert(0, "3.2")

        self.time_step_minutes = tk.Entry(frm, width=5)
        self.time_step_minutes.grid(row=9, column=1, sticky=tk.EW, pady=2)
        self.time_step_minutes.insert(0, 15)
        
        self.init_orientation = ttk.Combobox(frm, width=5, state="readonly", values=VALID_ORIENTATIONS)
        self.init_orientation.grid(row=10, column=1, sticky=tk.EW, pady=2)
        self.init_orientation.set(ORIENTATION_NSWE)
        
        self.init_view = tk.Entry(frm, width=5)
        self.init_view.grid(row=11, column=1, sticky=tk.EW, pady=2)

        self.coord_mode = tk.StringVar(value='decimal')
        tk.Radiobutton(frm, text="Decimal", variable=self.coord_mode, value='decimal').grid(row=0, column=2, sticky=tk.W, padx=(4, 0))
        tk.Radiobutton(frm, text="Sexagesimal", variable=self.coord_mode, value='sexagesimal').grid(row=1, column=2, sticky=tk.W, padx=(4, 0))
        tk.Label(frm, text="(sea level = 0)", fg="gray").grid(row=2, column=2, sticky=tk.W, padx=(4, 0))
        tk.Label(frm, text="(0.5 - 5.0)", fg="gray").grid(row=8, column=2, sticky=tk.W, padx=(4, 0))

        def _set_time_now():
            n = datetime.now().astimezone()
            self.date_entry.delete(0, tk.END)
            self.date_entry.insert(0, n.strftime("%Y-%m-%d"))
            self.hour_var.set(f"{n.hour:02d}")
            self.minute_var.set(f"{n.minute:02d}")
            self.second_var.set(f"{n.second:02d}")
            tz_off = n.strftime("%z")
            self.tz_combo.set(f"{tz_off[:3]}:{tz_off[3:]}" if tz_off else "+00:00")

        tk.Button(frm, text="Now", width=12, command=_set_time_now).grid(row=3, column=2, sticky=tk.W, pady=2, padx=(4, 0))
        tk.Button(frm, text="Browse", width=12, command=self.browse_elevation).grid(row=4, column=2, sticky=tk.W, pady=2, padx=(4, 0))
        tk.Button(frm, text="Browse", width=12, command=self.browse_color).grid(row=5, column=2, sticky=tk.W, pady=2, padx=(4, 0))


        self.lat_sexa_frame = tk.Frame(frm)
        ttk.Combobox(self.lat_sexa_frame, width=2, state="readonly",
                     values=["N", "S"], textvariable=self.lat_dir_var).grid(row=0, column=0, padx=(0, 4))
        self.lat_deg = tk.Entry(self.lat_sexa_frame, width=6)
        self.lat_min = tk.Entry(self.lat_sexa_frame, width=4)
        self.lat_sec = tk.Entry(self.lat_sexa_frame, width=6)
        self.lat_deg.grid(row=0, column=1)
        tk.Label(self.lat_sexa_frame, text="°").grid(row=0, column=2)
        self.lat_min.grid(row=0, column=3)
        tk.Label(self.lat_sexa_frame, text="'").grid(row=0, column=4)
        self.lat_sec.grid(row=0, column=5)
        tk.Label(self.lat_sexa_frame, text='"').grid(row=0, column=6)

        self.lon_sexa_frame = tk.Frame(frm)
        ttk.Combobox(self.lon_sexa_frame, width=2, state="readonly",
                     values=["E", "W"], textvariable=self.lon_dir_var).grid(row=0, column=0, padx=(0, 4))
        self.lon_deg = tk.Entry(self.lon_sexa_frame, width=6)
        self.lon_min = tk.Entry(self.lon_sexa_frame, width=4)
        self.lon_sec = tk.Entry(self.lon_sexa_frame, width=6)
        self.lon_deg.grid(row=0, column=1)
        tk.Label(self.lon_sexa_frame, text="°").grid(row=0, column=2)
        self.lon_min.grid(row=0, column=3)
        tk.Label(self.lon_sexa_frame, text="'").grid(row=0, column=4)
        self.lon_sec.grid(row=0, column=5)
        tk.Label(self.lon_sexa_frame, text='"').grid(row=0, column=6)

        # Run button and status
        self.run_btn = tk.Button(self, text=f"Run {APP_NAME}", command=self.on_run)
        self.run_btn.pack(padx=10, fill=tk.X)

        # Preset controls
        preset_frame = tk.Frame(self)
        preset_frame.pack(padx=10, pady=(10, 10), anchor=tk.W)

        tk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_name_entry = tk.Entry(preset_frame, width=15)
        self.preset_name_entry.pack(side=tk.LEFT, padx=(2, 4))
        tk.Button(preset_frame, text="Save", width=6, command=self._save_preset).pack(side=tk.LEFT, padx=2)

        self.preset_combobox = ttk.Combobox(preset_frame, width=15, state="readonly")
        self.preset_combobox.pack(side=tk.LEFT, padx=(10, 4))
        tk.Button(preset_frame, text="Load", width=6, command=self._load_preset).pack(side=tk.LEFT, padx=2)

        # Status label for progress messages
        self.status_label = tk.Label(preset_frame, text="", fg="blue", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))

        self._refresh_preset_list()

        # start with decimal visible; sexagesimal frames are not gridded
        def update_coord_mode(*args):
            mode = self.coord_mode.get()
            if mode == 'sexagesimal':
                # hide decimal frames
                self.lat_decimal_frame.grid_remove()
                self.lon_decimal_frame.grid_remove()
                # show sexagesimal frames
                self.lat_sexa_frame.grid(row=0, column=1, sticky=tk.W, pady=2)
                self.lon_sexa_frame.grid(row=1, column=1, sticky=tk.W, pady=2)
            else:
                # hide sexagesimal frames
                self.lat_sexa_frame.grid_remove()
                self.lon_sexa_frame.grid_remove()
                # show decimal frames
                self.lat_decimal_frame.grid(row=0, column=1, sticky=tk.EW, pady=2)
                self.lon_decimal_frame.grid(row=1, column=1, sticky=tk.EW, pady=2)

        # attach trace to update UI when radio changes
        self.coord_mode.trace_add('write', update_coord_mode)

    def _get_presets_dir(self):
        """Get the presets directory path, creating it if it doesn't exist."""
        presets_dir = os.path.join(BASE_PATH, "presets")
        if not os.path.exists(presets_dir):
            os.makedirs(presets_dir)
        return presets_dir

    def _get_preset_list(self):
        """Get list of available preset names (without .json extension)."""
        presets_dir = self._get_presets_dir()
        presets = []
        for f in os.listdir(presets_dir):
            if f.endswith(".json"):
                presets.append(f[:-5])  # Remove .json extension
        return sorted(presets)

    def _refresh_preset_list(self):
        """Refresh the preset combobox with available presets."""
        presets = self._get_preset_list()
        self.preset_combobox["values"] = presets

    def _save_preset(self):
        """Save current settings to a preset file."""
        preset_name = self.preset_name_entry.get().strip()
        if not preset_name:
            messagebox.showerror("Error", "Please enter a preset name.")
            return

        # Collect current settings
        settings = {
            "coord_mode": self.coord_mode.get(),
            "lat_dir": self.lat_dir_var.get(),
            "lon_dir": self.lon_dir_var.get(),
            "lat_decimal": self.lat_decimal.get(),
            "lon_decimal": self.lon_decimal.get(),
            "elevation": self.elevation_entry.get(),
            "lat_deg": self.lat_deg.get(),
            "lat_min": self.lat_min.get(),
            "lat_sec": self.lat_sec.get(),
            "lon_deg": self.lon_deg.get(),
            "lon_min": self.lon_min.get(),
            "lon_sec": self.lon_sec.get(),
            "time": self._get_time_iso(),
            "elevation_file": self.elevation_file.get(),
            "color_file": self.color_file.get(),
            "downscale": self.downscale.get(),
            "brightness": self.brightness.get(),
            "gamma": self.gamma_entry.get(),
            "time_step_minutes": self.time_step_minutes.get(),
            "init_orientation": self.init_orientation.get(),
            "init_view": self.init_view.get(),
        }

        # Save to file
        presets_dir = self._get_presets_dir()
        filepath = os.path.join(presets_dir, f"{preset_name}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            self._refresh_preset_list()
            # Select the newly saved preset in the combobox
            presets = self._get_preset_list()
            if preset_name in presets:
                self.preset_combobox.current(presets.index(preset_name))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {e}")

    def _load_preset(self):
        """Load settings from the selected preset."""
        preset_name = self.preset_combobox.get()
        if not preset_name:
            messagebox.showerror("Error", "Please select a preset to load.")
            return

        presets_dir = self._get_presets_dir()
        filepath = os.path.join(presets_dir, f"{preset_name}.json")
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"Preset file not found: {preset_name}")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                settings = json.load(f)

            # Apply settings to UI
            self.coord_mode.set(settings.get("coord_mode", "decimal"))

            # Decimal latitude: detect sign, set direction, store absolute value
            lat_dec_str = settings.get("lat_decimal", "")
            try:
                lat_val = float(lat_dec_str) if lat_dec_str else None
            except ValueError:
                lat_val = None
            if lat_val is not None and lat_val < 0:
                self.lat_dir_var.set("S")
                lat_dec_str = str(abs(lat_val))
            else:
                self.lat_dir_var.set(settings.get("lat_dir", "N"))

            self.lat_decimal.delete(0, tk.END)
            self.lat_decimal.insert(0, lat_dec_str)

            # Decimal longitude: detect sign, set direction, store absolute value
            lon_dec_str = settings.get("lon_decimal", "")
            try:
                lon_val = float(lon_dec_str) if lon_dec_str else None
            except ValueError:
                lon_val = None
            if lon_val is not None and lon_val < 0:
                self.lon_dir_var.set("W")
                lon_dec_str = str(abs(lon_val))
            else:
                self.lon_dir_var.set(settings.get("lon_dir", "E"))

            self.lon_decimal.delete(0, tk.END)
            self.lon_decimal.insert(0, lon_dec_str)

            self.elevation_entry.delete(0, tk.END)
            self.elevation_entry.insert(0, settings.get("elevation", settings.get("altitude", "0")))

            # Sexagesimal latitude: detect negative degrees, set direction, store absolute
            lat_deg_str = settings.get("lat_deg", "")
            try:
                lat_deg_val = int(lat_deg_str) if lat_deg_str else None
            except ValueError:
                lat_deg_val = None
            if lat_deg_val is not None and lat_deg_val < 0:
                self.lat_dir_var.set("S")
                lat_deg_str = str(abs(lat_deg_val))

            self.lat_deg.delete(0, tk.END)
            self.lat_deg.insert(0, lat_deg_str)

            self.lat_min.delete(0, tk.END)
            self.lat_min.insert(0, settings.get("lat_min", ""))

            self.lat_sec.delete(0, tk.END)
            self.lat_sec.insert(0, settings.get("lat_sec", ""))

            # Sexagesimal longitude: detect negative degrees, set direction, store absolute
            lon_deg_str = settings.get("lon_deg", "")
            try:
                lon_deg_val = int(lon_deg_str) if lon_deg_str else None
            except ValueError:
                lon_deg_val = None
            if lon_deg_val is not None and lon_deg_val < 0:
                self.lon_dir_var.set("W")
                lon_deg_str = str(abs(lon_deg_val))

            self.lon_deg.delete(0, tk.END)
            self.lon_deg.insert(0, lon_deg_str)

            self.lon_min.delete(0, tk.END)
            self.lon_min.insert(0, settings.get("lon_min", ""))

            self.lon_sec.delete(0, tk.END)
            self.lon_sec.insert(0, settings.get("lon_sec", ""))

            self._set_time_from_iso(settings.get("time", ""))

            self.elevation_file.delete(0, tk.END)
            self.elevation_file.insert(0, settings.get("elevation_file", ""))

            self.color_file.delete(0, tk.END)
            self.color_file.insert(0, settings.get("color_file", DEFAULT_COLOR_FILE_LOCAL_PATH))

            self.downscale.delete(0, tk.END)
            self.downscale.insert(0, settings.get("downscale", "3"))

            self.brightness.delete(0, tk.END)
            self.brightness.insert(0, settings.get("brightness", "80"))

            self.gamma_entry.delete(0, tk.END)
            self.gamma_entry.insert(0, settings.get("gamma", "3.2"))

            self.time_step_minutes.delete(0, tk.END)
            self.time_step_minutes.insert(0, settings.get("time_step_minutes", "15"))

            self.init_orientation.set(settings.get("init_orientation", ORIENTATION_NSWE))

            self.init_view.delete(0, tk.END)
            self.init_view.insert(0, settings.get("init_view", ""))

            # Update preset name entry with loaded preset name
            self.preset_name_entry.delete(0, tk.END)
            self.preset_name_entry.insert(0, preset_name)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {e}")

    def _get_time_iso(self):
        """Construct an ISO 8601 datetime string from the date/time/tz widgets."""
        date_str = self.date_entry.get().strip()
        hour = int(self.hour_var.get().strip() or 0)
        minute = int(self.minute_var.get().strip() or 0)
        second = int(self.second_var.get().strip() or 0)
        tz = self.tz_combo.get().strip() or "+00:00"
        return f"{date_str}T{hour:02d}:{minute:02d}:{second:02d}{tz}"

    def _set_time_from_iso(self, iso_str):
        """Populate date/time/tz widgets from an ISO 8601 datetime string."""
        if not iso_str:
            return
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            self.date_entry.delete(0, tk.END)
            self.date_entry.insert(0, dt.strftime("%Y-%m-%d"))
            self.hour_var.set(f"{dt.hour:02d}")
            self.minute_var.set(f"{dt.minute:02d}")
            self.second_var.set(f"{dt.second:02d}")
            offset = dt.strftime("%z")
            if offset:
                self.tz_combo.set(f"{offset[:3]}:{offset[3:]}")
        except (ValueError, AttributeError):
            pass

    def _open_calendar(self):
        """Open a calendar popup and set the date entry to the selected date."""
        try:
            parts = self.date_entry.get().strip().split("-")
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            year, month, day = None, None, None
        popup = CalendarPopup(self, year, month, day)
        if popup.result:
            self.date_entry.delete(0, tk.END)
            self.date_entry.insert(0, popup.result)

    def browse_elevation(self):
        path = filedialog.askopenfilename(initialdir=DATA_DIRECTORY_PATH, title="Select elevation file")
        if path:
            self.elevation_file.delete(0, tk.END)
            self.elevation_file.insert(0, path)

    def browse_color(self):
        path = filedialog.askopenfilename(initialdir=DATA_DIRECTORY_PATH, title="Select color file")
        if path:
            self.color_file.delete(0, tk.END)
            self.color_file.insert(0, path)

    def on_run(self):

        init_view_str = self.init_view.get().strip()

        init_camera_params = None
        if init_view_str:
            init_view = parse_init_view(init_view_str)
            if init_view is None:
                messagebox.showerror("Error", "Could not parse init-view string.")
                return
            dt_local = init_view.dt_local
            lat = init_view.lat
            lon = init_view.lon
            init_orientation = init_view.orientation
            init_camera_params = CameraParams(
                eye=init_view.eye,
                target=init_view.target,
                up=init_view.up,
                fov=init_view.fov,
            )
        else:
            time_iso = self._get_time_iso()
            dt_local, error = get_date_time_local(time_iso)
            if error is not None:
                messagebox.showerror("Error", f"Incorrect time: {error}")
                return
            init_orientation = self.init_orientation.get()
            # Read latitude/longitude according to selected coordinate format
            def parse_sexa_fields(deg_w, min_w, sec_w, name, deg_min, deg_max):
                try:
                    deg_s = deg_w.get().strip()
                    if deg_s == "":
                        raise ValueError(f"{name} degrees missing")
                    deg = int(deg_s)
                    minutes = int(min_w.get().strip() or 0)
                    seconds = float(sec_w.get().strip() or 0)
                except ValueError as e:
                    messagebox.showerror("Error", f"Invalid {name} sexagesimal value: {e}")
                    return None
                
                # Validate degrees range
                if deg < deg_min or deg > deg_max:
                    messagebox.showerror("Error", f"Invalid {name} degrees. Must be between {deg_min} and {deg_max}.")
                    return None
                
                # Validate minutes range (0-59)
                if minutes < 0 or minutes > 59:
                    messagebox.showerror("Error", f"Invalid {name} minutes. Must be between 0 and 59.")
                    return None
                
                # Validate seconds range (0-60)
                if seconds < 0 or seconds > 60:
                    messagebox.showerror("Error", f"Invalid {name} seconds. Must be between 0 and 60.")
                    return None
                
                sign = -1 if deg < 0 else 1
                result = sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)
                return round(result, 4)

            if self.coord_mode.get() == 'decimal':
                lat_str = self.lat_decimal.get().strip()
                lon_str = self.lon_decimal.get().strip()
                if not lat_str or not lon_str:
                    messagebox.showerror("Error", "Latitude and longitude are mandatory unless init-view is provided.")
                    return
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                except ValueError:
                    messagebox.showerror("Error", "Latitude and longitude must be numbers.")
                    return
            else:
                lat = parse_sexa_fields(self.lat_deg, self.lat_min, self.lat_sec, "latitude", 0, 90)
                if lat is None:
                    return
                lon = parse_sexa_fields(self.lon_deg, self.lon_min, self.lon_sec, "longitude", 0, 180)
                if lon is None:
                    return

        if not (lon >= 0.0 and lon <= 180.0):
            messagebox.showerror("Error", "Invalid longitude. Must be between 0 and 180 degrees.")
            return
        if not (lat >= 0.0 and lat <= 90.0):
            messagebox.showerror("Error", "Invalid latitude. Must be between 0 and 90 degrees.")
            return
        
        if self.lat_dir_var.get() == "S":
            lat = -lat
        if self.lon_dir_var.get() == "W":
            lon = -lon

        try:
            elevation = int(self.elevation_entry.get().strip() or 0)
        except ValueError:
            messagebox.showerror("Error", "Elevation must be an integer (meters).")
            return
        if not (0 <= elevation <= 100000):
            messagebox.showerror("Error", "Invalid elevation. Must be between 0 and 100000 meters.")
            return

        try:
            downscale = int(self.downscale.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Downscale must be a positive integer.")
            return
        if downscale < 1:
            messagebox.showerror("Error", "Downscale must be a positive integer.")
            return

        try:
            brightness = int(self.brightness.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Brightness must be an integer.")
            return
        if not (brightness >= 0 and brightness <= 500):
            messagebox.showerror("Error", "Invalid brightness. Must be between 0 and 500.")
            return

        try:
            gamma = float(self.gamma_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Gamma must be a number.")
            return
        if not (0.5 <= gamma <= 5.0):
            messagebox.showerror("Error", "Invalid gamma. Must be between 0.5 and 5.0.")
            return

        try:
            time_step_minutes = int(self.time_step_minutes.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Time step must be a positive integer.")
            return
        if not (time_step_minutes >= 1 and time_step_minutes <= 1440):
            messagebox.showerror("Error", "Invalid time step. Must be between 1 and 1440 minutes.")
            return

        self._set_status("Checking GPU architecture...")
        self.update_idletasks()
        if not check_gpu_architecture():
            self._set_status("")
            messagebox.showerror("Error", "No compatible RTX GPU found.")
            return

        elevation_file = self.elevation_file.get().strip()
        self._set_status("Checking elevation file...")
        self.update_idletasks()
        if not check_elevation_file(elevation_file):
            self._set_status("")
            messagebox.showerror("Error", "Elevation file is not present or downloading default file failed.")
            return

        color_file = self.color_file.get().strip()
        self._set_status("Checking color file...")
        self.update_idletasks()
        if not check_color_file(color_file):
            self._set_status("")
            messagebox.showerror("Error", "Color file is not present or downloading default file failed.")
            return
        
        self._set_status("Checking starmap file...")
        self.update_idletasks()
        if not check_starmap_file():
            self._set_status("")
            messagebox.showerror("Error", "Starmap file is not present and download failed.")
            return
        
        self._set_status("Starting renderer...")
        self.update_idletasks()
        
        # Disable the Run button while renderer is running
        self.run_btn.config(state=tk.DISABLED)
        
        p = Process(
            target=run_renderer,
            args=(
                dt_local,
                lat,
                lon,
                elevation,
                elevation_file,
                color_file,
                STARMAP_FILE_LOCAL_PATH,
                MOON_FEATURES_FILE_LOCAL_PATH,
                downscale,
                brightness,
                APP_NAME,
                init_camera_params,
                time_step_minutes,
                init_orientation,
                gamma)
        )
        p.start()
        
        # Start a thread to monitor when the renderer process ends
        def monitor_process(process):
            process.join()  # Wait for process to finish
            print(f"Renderer process ended with exit code: {process.exitcode}")
            # Re-enable the Run button and clear status (must use after() for thread-safe UI update)
            def on_process_end():
                self.run_btn.config(state=tk.NORMAL)
                self._set_status("")
            self.after(0, on_process_end)
        
        monitor_thread = threading.Thread(target=monitor_process, args=(p,), daemon=True)
        monitor_thread.start()
        
        self._set_status("Renderer running...")

    def _set_status(self, text):
        """Update the status label with the given text."""
        self.status_label.config(text=text)

def main():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()