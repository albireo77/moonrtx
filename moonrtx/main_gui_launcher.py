from multiprocessing import Process
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
    COLOR_FILE_LOCAL_PATH,
    STARMAP_FILE_LOCAL_PATH,
    MOON_FEATURES_FILE_LOCAL_PATH,
    DATA_DIRECTORY_PATH
)

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} - GUI Launcher")
        self._build_ui()

    def _build_ui(self):

        frm = tk.Frame(self)
        frm.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        frm.columnconfigure(1, weight=1)

        tk.Label(frm, text="Observer latitude:").grid(row=0, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Observer longitude:").grid(row=1, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Time with timezone:").grid(row=2, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Elevation file:").grid(row=3, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Downscale:").grid(row=4, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Brightness:").grid(row=5, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Time step (minutes):").grid(row=6, column=0, sticky=tk.E, pady=2)
        tk.Label(frm, text="Init view:").grid(row=7, column=0, sticky=tk.E, pady=2)

        self.lat_decimal = tk.Entry(frm, width=60)
        self.lat_decimal.grid(row=0, column=1, sticky=tk.W, pady=2)

        self.lon_decimal = tk.Entry(frm, width=60)
        self.lon_decimal.grid(row=1, column=1, sticky=tk.W, pady=2)

        self.time = tk.Entry(frm, width=60)
        self.time.grid(row=2, column=1, sticky=tk.W, pady=2)
        self.time.insert(0, datetime.now().astimezone().isoformat(timespec="seconds"))

        self.elevation_file = tk.Entry(frm, width=60)
        self.elevation_file.insert(0, DEFAULT_ELEVATION_FILE_LOCAL_PATH)
        self.elevation_file.grid(row=3, column=1, sticky=tk.W, pady=2)

        self.downscale = tk.Entry(frm, width=60)
        self.downscale.grid(row=4, column=1, sticky=tk.W, pady=2)
        self.downscale.insert(0, 3)

        self.brightness = tk.Entry(frm, width=60)
        self.brightness.grid(row=5, column=1, sticky=tk.W, pady=2)
        self.brightness.insert(0, 120)

        self.time_step_minutes = tk.Entry(frm, width=60)
        self.time_step_minutes.grid(row=6, column=1, sticky=tk.W, pady=2)
        self.time_step_minutes.insert(0, 15)
        
        self.init_view = tk.Entry(frm, width=60)
        self.init_view.grid(row=7, column=1, sticky=tk.W, pady=2)

        self.coord_mode = tk.StringVar(value='decimal')
        tk.Radiobutton(frm, text="Decimal", variable=self.coord_mode, value='decimal').grid(row=0, column=2, sticky=tk.W)
        tk.Radiobutton(frm, text="Sexagesimal", variable=self.coord_mode, value='sexagesimal').grid(row=1, column=2, sticky=tk.W)

        def _set_time_now():
            self.time.delete(0, tk.END)
            self.time.insert(0, datetime.now().astimezone().isoformat(timespec="seconds"))

        tk.Button(frm, text="Now", width=12, command=_set_time_now).grid(row=2, column=2, sticky=tk.W, pady=2, padx=4)
        tk.Button(frm, text="Browse", width=12, command=self.browse_elevation).grid(row=3, column=2, sticky=tk.W, pady=2, padx=4)

        self.lat_sexa_frame = tk.Frame(frm)
        self.lat_deg = tk.Entry(self.lat_sexa_frame, width=6)
        self.lat_min = tk.Entry(self.lat_sexa_frame, width=4)
        self.lat_sec = tk.Entry(self.lat_sexa_frame, width=6)
        self.lat_deg.grid(row=0, column=0)
        tk.Label(self.lat_sexa_frame, text="°").grid(row=0, column=1)
        self.lat_min.grid(row=0, column=2)
        tk.Label(self.lat_sexa_frame, text="'").grid(row=0, column=3)
        self.lat_sec.grid(row=0, column=4)
        tk.Label(self.lat_sexa_frame, text='"').grid(row=0, column=5)

        self.lon_sexa_frame = tk.Frame(frm)
        self.lon_deg = tk.Entry(self.lon_sexa_frame, width=6)
        self.lon_min = tk.Entry(self.lon_sexa_frame, width=4)
        self.lon_sec = tk.Entry(self.lon_sexa_frame, width=6)
        self.lon_deg.grid(row=0, column=0)
        tk.Label(self.lon_sexa_frame, text="°").grid(row=0, column=1)
        self.lon_min.grid(row=0, column=2)
        tk.Label(self.lon_sexa_frame, text="'").grid(row=0, column=3)
        self.lon_sec.grid(row=0, column=4)
        tk.Label(self.lon_sexa_frame, text='"').grid(row=0, column=5)

        # Run button and status
        self.run_btn = tk.Button(frm, text=f"Run {APP_NAME}", command=self.on_run)
        self.run_btn.grid(row=8, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))

        # Preset controls (row 9)
        preset_frame = tk.Frame(frm)
        preset_frame.grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        tk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_name_entry = tk.Entry(preset_frame, width=15)
        self.preset_name_entry.pack(side=tk.LEFT, padx=(2, 4))
        tk.Button(preset_frame, text="Save", width=6, command=self._save_preset).pack(side=tk.LEFT, padx=2)

        self.preset_combobox = ttk.Combobox(preset_frame, width=15, state="readonly")
        self.preset_combobox.pack(side=tk.LEFT, padx=(10, 4))
        tk.Button(preset_frame, text="Load", width=6, command=self._load_preset).pack(side=tk.LEFT, padx=2)

        self._refresh_preset_list()

        # start with decimal visible; sexagesimal frames are not gridded
        def update_coord_mode(*args):
            mode = self.coord_mode.get()
            if mode == 'sexagesimal':
                # hide decimal entries
                self.lat_decimal.grid_remove()
                self.lon_decimal.grid_remove()
                # show sexagesimal frames
                self.lat_sexa_frame.grid(row=0, column=1, sticky=tk.W, pady=2)
                self.lon_sexa_frame.grid(row=1, column=1, sticky=tk.W, pady=2)
            else:
                # hide sexagesimal frames
                self.lat_sexa_frame.grid_remove()
                self.lon_sexa_frame.grid_remove()
                # show decimal entries
                self.lat_decimal.grid(row=0, column=1, sticky=tk.W, pady=2)
                self.lon_decimal.grid(row=1, column=1, sticky=tk.W, pady=2)

        # attach trace to update UI when radio changes
        self.coord_mode.trace_add('write', update_coord_mode)

    def _get_presets_dir(self):
        """Get the presets directory path, creating it if it doesn't exist."""
        presets_dir = os.path.join(os.path.dirname(__file__), "presets")
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
            "lat_decimal": self.lat_decimal.get(),
            "lon_decimal": self.lon_decimal.get(),
            "lat_deg": self.lat_deg.get(),
            "lat_min": self.lat_min.get(),
            "lat_sec": self.lat_sec.get(),
            "lon_deg": self.lon_deg.get(),
            "lon_min": self.lon_min.get(),
            "lon_sec": self.lon_sec.get(),
            "time": self.time.get(),
            "elevation_file": self.elevation_file.get(),
            "downscale": self.downscale.get(),
            "brightness": self.brightness.get(),
            "time_step_minutes": self.time_step_minutes.get(),
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

            self.lat_decimal.delete(0, tk.END)
            self.lat_decimal.insert(0, settings.get("lat_decimal", ""))

            self.lon_decimal.delete(0, tk.END)
            self.lon_decimal.insert(0, settings.get("lon_decimal", ""))

            self.lat_deg.delete(0, tk.END)
            self.lat_deg.insert(0, settings.get("lat_deg", ""))

            self.lat_min.delete(0, tk.END)
            self.lat_min.insert(0, settings.get("lat_min", ""))

            self.lat_sec.delete(0, tk.END)
            self.lat_sec.insert(0, settings.get("lat_sec", ""))

            self.lon_deg.delete(0, tk.END)
            self.lon_deg.insert(0, settings.get("lon_deg", ""))

            self.lon_min.delete(0, tk.END)
            self.lon_min.insert(0, settings.get("lon_min", ""))

            self.lon_sec.delete(0, tk.END)
            self.lon_sec.insert(0, settings.get("lon_sec", ""))

            self.time.delete(0, tk.END)
            self.time.insert(0, settings.get("time", ""))

            self.elevation_file.delete(0, tk.END)
            self.elevation_file.insert(0, settings.get("elevation_file", ""))

            self.downscale.delete(0, tk.END)
            self.downscale.insert(0, settings.get("downscale", "3"))

            self.brightness.delete(0, tk.END)
            self.brightness.insert(0, settings.get("brightness", "120"))

            self.time_step_minutes.delete(0, tk.END)
            self.time_step_minutes.insert(0, settings.get("time_step_minutes", "15"))

            self.init_view.delete(0, tk.END)
            self.init_view.insert(0, settings.get("init_view", ""))

            # Update preset name entry with loaded preset name
            self.preset_name_entry.delete(0, tk.END)
            self.preset_name_entry.insert(0, preset_name)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {e}")

    def browse_elevation(self):
        path = filedialog.askopenfilename(initialdir=DATA_DIRECTORY_PATH, title="Select elevation file")
        if path:
            self.elev_entry.delete(0, tk.END)
            self.elev_entry.insert(0, path)
            # ensure it's shown as normal text (not placeholder)
            self.elev_entry.config(fg="black")

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
            init_camera_params = CameraParams(
                eye=init_view.eye,
                target=init_view.target,
                up=init_view.up,
                fov=init_view.fov,
            )
        else:
            dt_local, error = get_date_time_local(self.time.get().strip())
            if error is not None:
                messagebox.showerror("Error", f"Incorrect time: {error}")
                return
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
                return round(result, 3)

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
                lat = parse_sexa_fields(self.lat_deg, self.lat_min, self.lat_sec, "latitude", -90, 90)
                if lat is None:
                    return
                lon = parse_sexa_fields(self.lon_deg, self.lon_min, self.lon_sec, "longitude", -180, 180)
                if lon is None:
                    return

        if not (lon >= -180.0 and lon <= 180.0):
            messagebox.showerror("Error", "Invalid longitude. Must be between -180 and 180 degrees.")
            return
        if not (lat >= -90.0 and lat <= 90.0):
            messagebox.showerror("Error", "Invalid latitude. Must be between -90 and 90 degrees.")
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
            time_step_minutes = int(self.time_step_minutes.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Time step must be a positive integer.")
            return
        if not (time_step_minutes >= 1 and time_step_minutes <= 1440):
            messagebox.showerror("Error", "Invalid time step. Must be between 1 and 1440 minutes.")
            return

        if not check_gpu_architecture():
            messagebox.showerror("Error", "No compatible RTX GPU found.")
            return

        elevation_file = self.elevation_file.get().strip()
        if not check_elevation_file(elevation_file):
            messagebox.showerror("Error", "Elevation file is not present or downloading default file failed.")
            return

        if not check_color_file():
            messagebox.showerror("Error", "Surface color file is not present and download failed.")
            return
        
        if not check_starmap_file():
            messagebox.showerror("Error", "Starmap file is not present and download failed.")
            return
        
        # Disable the Run button while renderer is running
        self.run_btn.config(state=tk.DISABLED)
        
        p = Process(
            target=run_renderer,
            args=(
                dt_local,
                lat,
                lon,
                elevation_file,
                COLOR_FILE_LOCAL_PATH,
                STARMAP_FILE_LOCAL_PATH,
                MOON_FEATURES_FILE_LOCAL_PATH,
                downscale,
                brightness,
                APP_NAME,
                init_camera_params,
                time_step_minutes)
        )
        p.start()
        
        # Start a thread to monitor when the renderer process ends
        def monitor_process(process):
            process.join()  # Wait for process to finish
            print(f"Renderer process ended with exit code: {process.exitcode}")
            # Re-enable the Run button (must use after() for thread-safe UI update)
            self.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
        
        monitor_thread = threading.Thread(target=monitor_process, args=(p,), daemon=True)
        monitor_thread.start()

def main():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
