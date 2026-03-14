"""
StatusMixin: status bar and info panel update methods for MoonRenderer.
"""

import tkinter as tk

from moonrtx.constants import ORIENTATION_NSWE


class _ToolTip:
    """Simple tooltip for tkinter widgets."""
    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._tw = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, event=None):
        x = self._widget.winfo_rootx() + self._widget.winfo_width() // 2
        y = self._widget.winfo_rooty() - 24
        self._tw = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        tk.Label(tw, text=self._text, background='#ffffe0', relief='solid',
                 borderwidth=1, font=('Segoe UI', 9)).pack()

    def _hide(self, event=None):
        if self._tw:
            self._tw.destroy()
            self._tw = None


class StatusMixin:
    """Mixin providing status bar and info panel methods for MoonRenderer."""

    # ---- Status panel update methods ----

    def _update_status_observer(self):
        if self._status_observer_var:
            if self.observer_lat is not None and self.observer_lon is not None:
                lat_dir = 'N' if self.observer_lat >= 0 else 'S'
                lon_dir = 'E' if self.observer_lon >= 0 else 'W'
                self._status_observer_var.set(
                    f"Observer: {abs(self.observer_lat):.4f}\u00b0{lat_dir} {abs(self.observer_lon):.4f}\u00b0{lon_dir}")
            else:
                self._status_observer_var.set("Observer:")

    def _update_status_view(self):
        if self._status_view_var:
            self._status_view_var.set(f"View: {self.orientation_mode}")

    def _update_status_time(self):
        if self._status_time_var and self.dt_local:
            offset = self.dt_local.strftime('%z')
            offset_fmt = f"{offset[:3]}:{offset[3:]}" if offset else ""
            self._status_time_var.set(
                f"Time: {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}{offset_fmt} (step {self.time_step_minutes} min)")

    def _update_info_moon(self):
        """Update the info panel with current Moon ephemeris data."""
        if self.moon_ephem is None:
            return
        e = self.moon_ephem
        if self._info_az_var:
            az_d = int(e.az)
            az_m = int((e.az - az_d) * 60)
            az_s = (e.az - az_d - az_m / 60) * 3600
            self._info_az_var.set(f"Az:  {az_d:3d}\u00b0{az_m:02d}'{az_s:04.1f}\"")
        if self._info_alt_var:
            alt_sign = '+' if e.alt >= 0 else '-'
            alt_abs = abs(e.alt)
            alt_d = int(alt_abs)
            alt_m = int((alt_abs - alt_d) * 60)
            alt_s = (alt_abs - alt_d - alt_m / 60) * 3600
            self._info_alt_var.set(f"Alt: {alt_sign}{alt_d:02d}\u00b0{alt_m:02d}'{alt_s:04.1f}\"")
        if self._info_ra_var:
            ra_total_h = e.ra / 15.0
            ra_h = int(ra_total_h)
            ra_m = int((ra_total_h - ra_h) * 60)
            ra_s = (ra_total_h - ra_h - ra_m / 60) * 3600
            self._info_ra_var.set(f"RA:  {ra_h:02d}h{ra_m:02d}m{ra_s:04.1f}s")
        if self._info_dec_var:
            dec_sign = '+' if e.dec >= 0 else '-'
            dec_abs = abs(e.dec)
            dec_d = int(dec_abs)
            dec_m = int((dec_abs - dec_d) * 60)
            dec_s = (dec_abs - dec_d - dec_m / 60) * 3600
            self._info_dec_var.set(f"DEC: {dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\"")
        if self._info_phase_var:
            self._info_phase_var.set(f"Phase ∠: {e.phase:6.2f}°")
        if self._info_sun_sep_var:
            self._info_sun_sep_var.set(f"Sun ∠:  {e.sun_separation:6.3f}°")
        if self._info_distance_var:
            self._info_distance_var.set(f"Dist: {e.distance:,.0f} km".replace(",", " "))
        if self._info_libr_l_var:
            self._info_libr_l_var.set(f"Libr L: {e.libr_long:+5.2f}°")
        if self._info_libr_b_var:
            self._info_libr_b_var.set(f"Libr B: {e.libr_lat:+5.2f}°")

    def _update_status_measured(self):
        if self._status_measured_var:
            if self.measured_distance is not None:
                self._status_measured_var.set(f"Measured: {self.measured_distance:7.2f} km")
            else:
                self._status_measured_var.set("")

    def _update_info_coords(self, lat=None, lon=None):
        """Update selenographic coordinates in the status bar coords panel."""
        if self._status_coords_var:
            if lat is not None and lon is not None:
                lat_dir = 'N' if lat >= 0 else 'S'
                lon_dir = 'E' if lon >= 0 else 'W'
                self._status_coords_var.set(
                    f"Lat: {abs(lat):5.2f}°{lat_dir} Lon: {abs(lon):6.2f}°{lon_dir}")
            else:
                self._status_coords_var.set("")

    def _update_status_feature(self, feature_text: str = ""):
        """Update feature name in the status bar."""
        if self._status_feature_var:
            self._status_feature_var.set(feature_text)

    def _update_status_brightness(self):
        if self._status_brightness_var:
            self._status_brightness_var.set(f"Brightness: {self.brightness}")

    def _update_status_gamma(self):
        if self._status_gamma_var:
            self._status_gamma_var.set(f"Gamma: {self.gamma:.1f}")

    def _update_status_pins(self):
        if self._status_pins_var:
            self._status_pins_var.set(f"Pins {'ON' if self.pins_visible else 'OFF'}")

    def _update_all_status_panels(self):
        self._update_status_observer()
        self._update_status_view()
        self._update_status_time()
        self._update_status_measured()
        self._update_status_feature()
        self._update_status_brightness()
        self._update_status_gamma()
        self._update_status_pins()
        self._update_info_moon()
        self._update_info_coords()

    def toggle_info_panel(self):
        """Toggle the Moon info panel visibility."""
        self.show_info_panel = not self.show_info_panel
        if self._info_frame is not None:
            if self.show_info_panel:
                self._info_frame.place(relx=0.0, rely=1.0, anchor='sw', x=6, y=-6)
            else:
                self._info_frame.place_forget()

    def _on_launch_finished(self, rt):
        """Callback to maximize window and set title on first launch."""
        if not self._window_maximized:
            self._window_maximized = True
            # Schedule maximize and title change on the main thread
            def init_window():
                rt._root.state('zoomed')
                rt._root.title(self.app_name)

                # Hide FPS panel from status bar
                if hasattr(rt, '_status_fps'):
                    rt._status_fps.grid_remove()

                # Build multi-panel status bar replacing the single label
                if hasattr(rt, '_status_action'):
                    grid_info = rt._status_action.grid_info()
                    parent = rt._status_action.master
                    rt._status_action.grid_remove()

                    status_frame = tk.Frame(parent)

                    self._status_observer_var = tk.StringVar()
                    self._status_view_var = tk.StringVar()
                    self._status_time_var = tk.StringVar()
                    self._status_measured_var = tk.StringVar()
                    self._status_feature_var = tk.StringVar()
                    self._status_brightness_var = tk.StringVar()
                    self._status_gamma_var = tk.StringVar()
                    self._status_pins_var = tk.StringVar()
                    self._status_coords_var = tk.StringVar()

                    self._auto_advance_var = tk.BooleanVar(value=False)

                    font = ("Consolas", 9)
                    panels = [
                        (self._status_pins_var,        8),
                        (self._status_brightness_var, 15),
                        (self._status_gamma_var,      10),
                        (self._status_feature_var,    41),
                        (self._status_coords_var,     26),
                        (self._status_measured_var,   20),
                        (None,                        47),  # placeholder for time panel
                        (self._status_view_var,       10),
                        (self._status_observer_var,   30)
                    ]
                    for var, w in panels:
                        if var is None:
                            # Build composite time panel: label + checkbox
                            time_panel = tk.Frame(status_frame, relief='sunken', borderwidth=1)
                            tk.Label(
                                time_panel,
                                textvariable=self._status_time_var,
                                font=font,
                                anchor='w',
                                width=w,
                            ).pack(side='left')
                            bg = time_panel.cget('bg')
                            cb = tk.Checkbutton(
                                time_panel,
                                text='\u25B6',
                                variable=self._auto_advance_var,
                                font=font,
                                indicatoron=False,
                                selectcolor=bg,
                                command=self._on_auto_advance_toggle,
                            )
                            cb.pack(side='right', padx=(2, 0))
                            _ToolTip(cb, 'Auto-advance time (every step minutes)')
                            time_panel.pack(side='right', padx=16)
                        else:
                            tk.Label(
                                status_frame,
                                textvariable=var,
                                font=font,
                                anchor='w',
                                width=w,
                                relief='sunken',
                                borderwidth=1,
                            ).pack(side='right', padx=16)

                # Build info panel (bottom-left overlay on canvas)
                if hasattr(rt, '_canvas'):
                    info_font = ("Consolas", 9)
                    info_fg = "#808080"
                    info_bg = "#010104"
                    info_width = 17  # Fixed width in chars (fits DEC: +89°59'59.9")

                    self._info_az_var = tk.StringVar(value="Az:")
                    self._info_alt_var = tk.StringVar(value="Alt:")
                    self._info_ra_var = tk.StringVar(value="RA:")
                    self._info_dec_var = tk.StringVar(value="DEC:")
                    self._info_phase_var = tk.StringVar(value="Ph:")
                    self._info_sun_sep_var = tk.StringVar(value="Sun:")
                    self._info_distance_var = tk.StringVar(value="Dist:")
                    self._info_libr_l_var = tk.StringVar(value="LbL:")
                    self._info_libr_b_var = tk.StringVar(value="LbB:")

                    info_frame = tk.Frame(rt._canvas, bg=info_bg, padx=6, pady=4)
                    self._info_frame = info_frame
                    info_vars = [
                        self._info_az_var,
                        self._info_alt_var,
                        self._info_ra_var,
                        self._info_dec_var,
                        self._info_phase_var,
                        self._info_sun_sep_var,
                        self._info_distance_var,
                        self._info_libr_l_var,
                        self._info_libr_b_var,
                    ]
                    for var in info_vars:
                        tk.Label(
                            info_frame,
                            textvariable=var,
                            font=info_font,
                            fg=info_fg,
                            bg=info_bg,
                            anchor='w',
                            width=info_width,
                        ).pack(anchor='w')
                    info_frame.place(relx=0.0, rely=1.0, anchor='sw', x=6, y=-6)

                # Add 4-char left padding to shift panels right
                status_frame.grid(
                    row=int(grid_info['row']),
                    column=int(grid_info['column']),
                    columnspan=int(grid_info.get('columnspan', 1)),
                    sticky='we',
                    padx=(4, 0), pady=0
                )

                # Bind mouse wheel for zoom
                if hasattr(rt, '_canvas'):
                    rt._canvas.bind('<MouseWheel>', self._mouse_wheel_handler)

                # F10 is intercepted by the window manager on Windows
                # (activates menu bar), so bind it explicitly on _root
                # and return 'break' to suppress the default behaviour.
                def _f10_handler(event):
                    self.set_time_to_now_and_auto_advance()
                    return 'break'
                rt._root.bind('<F10>', _f10_handler)
                
                # Apply initial view orientation to plotoptix
                if self.orientation_mode != ORIENTATION_NSWE:
                    rt._view_orientation = self.orientation_mode
                    # Update grid labels for initial orientation if grid exists
                    if self.moon_grid is not None and self.moon_grid_visible:
                        self.update_grid_labels_for_orientation()
                
                self._update_all_status_panels()
            rt._root.after_idle(init_window)
