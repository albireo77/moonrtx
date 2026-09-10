"""
StatusMixin: status bar and info panel update methods for MoonRenderer.
"""

import math
import tkinter as tk
import tkinter.font as tkfont
import webbrowser
from typing import Optional

from moonrtx import astro
from moonrtx.display import bring_to_front
from moonrtx.view_orientation import VIEW_ORIENTATIONS
from moonrtx.shared_types import MoonFeature


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


def timezone_name(dt_local) -> str:
    """
    The session's timezone as it should be shown: the IANA name when the clock
    carries rules, which is what MoonRTX runs on, and whatever else the tzinfo
    can be called when it does not.
    """
    tz = dt_local.tzinfo
    return getattr(tz, "key", None) or dt_local.strftime("%Z") or str(tz)


class StatusMixin:
    """Mixin providing status bar and info panel methods for MoonRenderer."""

    @staticmethod
    def _dms(value: float) -> tuple[int, int, float]:
        d = int(value)
        m = int((value - d) * 60)
        s = (value - d - m / 60) * 3600
        return d, m, s

    # The panels are given their widths in characters, and a character grows
    # with the display. At 300% scaling the nine of them together ask for 4504
    # pixels of a 3840-pixel screen, and nothing refuses: the canvas the Moon is
    # drawn on spans the same grid columns, so it is laid out that wide as well.
    # The right of it then hangs off the screen, taking the compass with it, and
    # the Moon - drawn at the middle of the canvas - sits a third of a thousand
    # pixels right of the middle of what can be seen.
    #
    # So the lettering is stepped down until the row fits across. It costs a
    # point or two and nothing else: at 300% a point is three pixels, so what is
    # left is still far larger than the whole bar is on an ordinary screen,
    # where it fits at its full size and nothing is changed at all.
    STATUS_MIN_FONT_SIZE = 6

    # Character width of the full-screen panel, taken from the one row that is
    # always the same length: the date and time. Everything else of a fixed
    # shape is shorter - the coordinates 13, the measurements 14 - and a feature
    # name is trimmed to fit rather than allowed to widen the panel, which costs
    # little: 11 of the bundled catalogue's 4488 names are longer than this.
    # Fixed on purpose. The panel is anchored by its right edge, so a width that
    # followed its contents would step sideways every time the cursor crossed a
    # differently named crater.
    STATUS_PANEL_WIDTH = 19

    # The window is two rows: the canvas across the top, and along the bottom
    # PlotOptiX's own selection readout beside the panels this mixin builds.
    # Only the canvas row carries any weight, so taking the bottom one out gives
    # the picture the whole window - which is what full screen does with it.
    STATUS_BAR_ROW = 1

    def _hide_status_row(self):
        """
        Take away the row along the bottom, remembering what was in it.

        For renderer_fullscreen, which needs the row gone but has no business
        knowing what is in it. Only what the row is actually managing is
        remembered, so the two readouts removed at start-up - PlotOptiX's
        frames-per-second panel and the action label these panels replaced - are
        not in the list and are never put back by _show_status_row.
        """
        root = self.rt._root
        self._windowed_status = root.grid_slaves(row=self.STATUS_BAR_ROW)
        for widget in self._windowed_status:
            widget.grid_remove()

    def _show_status_row(self):
        """
        Put back what _hide_status_row took, and nothing else.

        grid_remove keeps each widget's place - its row, column, span and
        padding - so each goes back exactly where it was.
        """
        for widget in self._windowed_status:
            widget.grid()
        self._windowed_status = []

    def _fit_status_bar(self, frame, available: int) -> Optional[int]:
        """
        Shrink the lettering of the status bar until it fits across the screen,
        and say what size it settled on - or None if it never had to.

        Every piece of lettering under the frame is scaled by the same ratio, so
        the smaller sign the Sun altitude is written with stays smaller than the
        figure beside it.
        """
        frame.update_idletasks()
        if frame.winfo_reqwidth() <= available:
            return None

        lettered = []
        stack = [frame]
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            try:
                spec = widget.cget("font")
            except tk.TclError:                 # a frame carries no lettering
                continue
            if not spec:
                continue
            actual = tkfont.Font(root=frame, font=spec).actual()
            lettered.append((widget, actual["family"], actual["size"],
                             actual["weight"] == "bold"))

        base = max((size for _w, _f, size, _b in lettered), default=0)
        if base <= self.STATUS_MIN_FONT_SIZE:
            return None

        for size in range(base - 1, self.STATUS_MIN_FONT_SIZE - 1, -1):
            for widget, family, was, bold in lettered:
                smaller = max(self.STATUS_MIN_FONT_SIZE,
                              int(round(was * size / base)))
                widget.config(font=(family, smaller, "bold") if bold
                              else (family, smaller))
            frame.update_idletasks()
            if frame.winfo_reqwidth() <= available:
                return size
        return self.STATUS_MIN_FONT_SIZE

    # ---- Status panel update methods ----

    def _update_status_parallactic(self):
        if self._status_parallactic_var:
            state = "ON" if self.parallactic_mode else "OFF"
            self._status_parallactic_var.set(f"Parallactic Mode: {state}")

    def _update_status_view(self):
        if self._status_view_var:
            self._status_view_var.set(f"View: {self.view_orientation}")

    def _update_status_time(self):
        if self._status_time_var and self.dt_local:
            offset = self.dt_local.strftime('%z')
            offset_fmt = f"{offset[:3]}:{offset[3:]}" if offset else ""
            self._status_time_var.set(
                f"{self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}{offset_fmt} (step {self.time_step_minutes} min)")
        if self._status_panel_datetime_var is not None and self.dt_local:
            # No zone and no step: the panel says when the picture is of, and
            # the observer's clock is the only one it ever shows
            self._status_panel_datetime_var.set(
                self.dt_local.strftime('%Y-%m-%d %H:%M:%S'))

    def _update_info_moon(self):
        """Update the info panel with current Moon ephemeris data."""
        if self.moon_ephem is None or self._info_az_var is None:
            return
        e = self.moon_ephem

        az_d, az_m, az_s = self._dms(e.az)
        self._info_az_var.set(f"Az:  {az_d:3d}°{az_m:02d}'{az_s:04.1f}\"")

        alt_sign = '+' if e.alt >= 0 else '-'
        alt_d, alt_m, alt_s = self._dms(abs(e.alt))
        self._info_alt_var.set(f"Alt: {alt_sign}{alt_d:02d}°{alt_m:02d}'{alt_s:04.1f}\"")
        if self._info_alt_label is not None:
            self._info_alt_label.configure(fg=self._info_alt_negative_fg if e.alt < 0 else self._info_fg)

        ra_h, ra_m, ra_s = self._dms(e.ra / 15.0 % 24)
        self._info_ra_var.set(f"RA:   {ra_h:02d}h{ra_m:02d}m{ra_s:04.1f}s")

        dec_sign = '+' if e.dec >= 0 else '-'
        dec_d, dec_m, dec_s = self._dms(abs(e.dec))
        self._info_dec_var.set(f"DEC: {dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\"")

        self._info_phase_name_var.set(f"{e.phase_name:>17}")
        self._info_phase_var.set(f"Phase ∠: {e.phase_angle:7.3f}°")
        self._info_age_var.set(f"Age:  {e.age_days:6.2f} days")
        self._info_elongation_var.set(f"Sun ∠:   {e.elongation:7.3f}°")
        self._info_distance_var.set(f"Dist:  {math.floor(e.distance + 0.5):,.0f} km".replace(",", " "))
        # Apparent diameter the Moon is actually rendered at (see moon_camera_distance),
        # from the same topocentric distance as the row above: 29.4' to 33.5' geocentric,
        # up to 34.1' for a perigee Moon in the zenith
        self._info_diameter_var.set(f"Diameter: {math.degrees(2 * self.moon_apparent_radius()) * 60:6.2f}'")
        self._info_illum_var.set(f"💡:       {(1 + math.cos(math.radians(e.phase_angle))) * 50.0:6.2f}%")
        self._info_geo_libr_l_var.set(f"⊕ Libr L: {e.libr_long_geo:+6.3f}°")
        self._info_geo_libr_b_var.set(f"⊕ Libr B: {e.libr_lat_geo:+6.3f}°")
        self._info_topo_libr_l_var.set(f"⌖ Libr L: {e.libr_long_topo:+6.3f}°")
        self._info_topo_libr_b_var.set(f"⌖ Libr B: {e.libr_lat_topo:+6.3f}°")
        self._info_colong_var.set(f"Colongit: {e.colongitude:6.2f}°")

    def _update_status_measured(self):
        if self._status_measured_var:
            measured_text = "             " if self.measured_distance is None else f"d: {self.measured_distance:7.2f} km"
            measured_text += "" if self.measured_height_diff is None else f"  Δh: {self.measured_height_diff:6.0f} m"
            self._status_measured_var.set(measured_text)
            if self._status_panel_distance_var is not None:
                # A row each. The label of the one and the number of the other
                # are given a column more than the status bar allows them, so
                # that the two numbers end in the same place.
                self._status_panel_distance_var.set(
                    "" if self.measured_distance is None
                    else f"d: {self.measured_distance:7.2f} km")
                self._status_panel_height_var.set(
                    "" if self.measured_height_diff is None
                    else f"Δh: {self.measured_height_diff:7.0f} m")

    def _update_info_coords(self, lat=None, lon=None):
        """
        Update the status bar coords panel: selenographic position of the
        point under the cursor, followed by the Sun's altitude above the local
        horizon there - the angle that sets how long shadows are (see
        astro.sun_altitude_at). Both are read from the cursor, so the panel is
        refreshed on mouse motion and cleared when the view changes under it.
        """
        if not self._status_coords_var:
            return

        def show_sun(visible: bool):
            if self._status_coords_sun_label is not None:
                self._status_coords_sun_label.config(
                    fg=self._status_coords_sun_fg if visible else self._status_coords_sun_bg)

        if lat is None or lon is None:
            self._status_coords_var.set("")
            self._status_coords_alt_var.set("")
            self._set_status_panel_coords("", "", "")
            show_sun(False)
            return

        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        coords = f"Lat: {abs(lat):5.2f}°{lat_dir} Lon: {abs(lon):6.2f}°{lon_dir}"
        # A row each in the full-screen panel. The latitude is given the
        # longitude's width, which it does not need, so that the two numbers
        # stand in one column instead of a digit apart.
        lat_row = f"Lat: {abs(lat):6.2f}°{lat_dir}"
        lon_row = f"Lon: {abs(lon):6.2f}°{lon_dir}"
        if self.moon_ephem is None:
            self._status_coords_var.set(coords)
            self._status_coords_alt_var.set("")
            self._set_status_panel_coords(lat_row, lon_row, "")
            show_sun(False)
            return

        # Split so the Sun sign can be drawn as a subscript of the "h"
        # (see the coords panel in _on_launch_finished)
        self._status_coords_var.set(f"{coords}  h")
        show_sun(True)
        sun_alt = astro.sun_altitude_at(self.moon_ephem.subsolar_lat,
                                        self.moon_ephem.subsolar_lon, lat, lon)
        altitude = f": {sun_alt:+5.1f}°"
        self._status_coords_alt_var.set(altitude)
        # The Sun sign sits beside the "h" here rather than under it as the
        # status bar has it: that is three labels in two font sizes, and a row
        # of this panel is one label and so one font. The two spaces after the
        # colon bring its degree sign into the column the other two rows keep.
        self._set_status_panel_coords(lat_row, lon_row,
                                    f"h☉:   {sun_alt:+6.1f}°")

    def _set_status_panel_coords(self, lat_row: str, lon_row: str, sun_row: str):
        """
        The three coordinate rows of the full-screen panel, when there is one.

        Always all three together: they are read from the cursor and are either
        all known or none of them, and the Sun's altitude needs an ephemeris
        besides, so it alone can be blank while the other two are not.
        """
        if self._status_panel_lat_var is None:
            return
        self._status_panel_lat_var.set(lat_row)
        self._status_panel_lon_var.set(lon_row)
        self._status_panel_sun_var.set(sun_row)

    def _update_status_feature(self, feature: Optional[MoonFeature] = None):
        """Update feature name in the status bar and remember the active feature."""
        self._status_feature = feature
        if self._status_feature_var:
            feature_text = "" if feature is None else f"{feature.name} (⌀ = {feature.diameter_km:.2f} km)"
            self._status_feature_var.set(feature_text)
            if self._status_panel_feature_var is not None:
                # The name alone: the diameter is in the status bar and in the
                # search results, and it is what made this row the long one.
                # Trimmed to the panel's width, so that nothing in the panel can
                # change the panel's width - see STATUS_PANEL_WIDTH.
                name = "" if feature is None else feature.name
                self._status_panel_feature_var.set(
                    name[:self.STATUS_PANEL_WIDTH])

    def _open_feature_url(self, url: str, feature_name: str) -> bool:
        try:
            return bool(webbrowser.open_new_tab(url))
        except Exception as exc:
            print(f"Failed to open page for {feature_name}: {exc}")
            return False

    def open_status_feature_usgs_page(self) -> bool:
        """Open the USGS page for the feature currently shown in the status bar."""
        feature = self._status_feature
        if feature is None or feature.feature_id is None:
            return False
        return self._open_feature_url(
            f"https://planetarynames.wr.usgs.gov/Feature/{feature.feature_id}",
            feature.name,
        )

    def open_status_feature_www_page(self) -> bool:
        """Open the user-defined web page for the feature shown in the status bar."""
        feature = self._status_feature
        if feature is None or not feature.www_address:
            return False
        url = feature.www_address
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        return self._open_feature_url(url, feature.name)

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
        self._update_status_parallactic()
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

    def toggle_status_panel(self):
        """
        Show or hide the status panel: the same four readings the status
        bar carries, in the opposite corner of the picture.

        Worth having twice because the status bar is not part of the picture.
        It is a row of widgets under the canvas, so it goes with the window
        frame in full screen and has never appeared in a saved image or a video
        frame. This panel is drawn on the canvas, so it is in all three.
        """
        self.show_status_panel = not self.show_status_panel
        if self._status_panel_frame is not None:
            if self.show_status_panel:
                self._status_panel_frame.place(relx=1.0, rely=1.0, anchor='se',
                                             x=-6, y=-6)
            else:
                self._status_panel_frame.place_forget()

    def window_title(self) -> str:
        lat = self.observer.lat
        lon = self.observer.lon
        elevation_m = self.observer.elevation_m
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        lat_str = f"{abs(lat):.4f}".rstrip('0').rstrip('.')
        lon_str = f"{abs(lon):.4f}".rstrip('0').rstrip('.')
        from moonrtx.main import APP_NAME
        return (f"{APP_NAME}        🕒 {timezone_name(self.dt_local)}"
                f"        👁️ {lat_str}°{lat_dir}   {lon_str}°{lon_dir}"
                f"   (elevation: {elevation_m} m)")

    def _on_launch_finished(self, rt):
        """Callback to maximize window and set title on first launch."""
        if not self._window_maximized:
            self._window_maximized = True
            # Schedule maximize and title change on the main thread
            def init_window():
                rt._root.state('zoomed')

                # Before the status bar below rather than after it: everything
                # that follows repaints the window at its windowed size, so
                # switched on at the end full screen arrived visibly late. The
                # row itself still has to wait - see _full_screen_window.
                if self.initial_fullscreen:
                    self._full_screen_window()
                rt._root.title(self.window_title())

                # Hide FPS panel from status bar
                if hasattr(rt, '_status_fps'):
                    rt._status_fps.grid_remove()

                # Build multi-panel status bar replacing the single label
                if hasattr(rt, '_status_action'):
                    grid_info = rt._status_action.grid_info()
                    parent = rt._status_action.master
                    rt._status_action.grid_remove()

                    status_frame = tk.Frame(parent)

                    self._status_parallactic_var = tk.StringVar()
                    self._status_view_var = tk.StringVar()
                    self._status_time_var = tk.StringVar()
                    self._status_measured_var = tk.StringVar()
                    self._status_feature_var = tk.StringVar()
                    self._status_brightness_var = tk.StringVar()
                    self._status_gamma_var = tk.StringVar()
                    self._status_pins_var = tk.StringVar()
                    self._status_coords_var = tk.StringVar()
                    self._status_coords_alt_var = tk.StringVar()

                    self._auto_advance_var = tk.BooleanVar(value=False)

                    font = ("Consolas", 9)
                    sun_font = ("Consolas", 7)      # subscript Sun sign
                    # Panels are packed to the right, so this list runs from
                    # the right edge leftwards, and the bar keeps the width it
                    # has always had: what one panel takes, another gives up.
                    # The coords panel took room for the Sun altitude (26 -> 39)
                    # and the feature panel gave it (46 -> 32). The time panel
                    # has since gone from 47 to 41: it was labelled "Time:" and
                    # is not any more, a date needing no announcement. The width
                    # is measured rather than guessed - the longest it ever holds
                    # is a date, an offset and a step of 1440 minutes, and that
                    # comes to exactly 41 cells, so 40 would clip it.
                    #
                    # Those six cells went to the feature panel (32 -> 38),
                    # which now holds the whole database: of the 4442 names the
                    # bar can show, with the size beside them, 17 used to be too
                    # long and none is. It is a close thing, mind - the longest
                    # of them, "Promontorium Heraclides", comes to 265 px in a
                    # panel of 266 - so a name longer than that one would want
                    # this widened again rather than quietly clipped.
                    panels = [
                        (self._status_pins_var,        8),
                        (self._status_brightness_var, 15),
                        (self._status_gamma_var,      10),
                        (self._status_feature_var,    38),
                        ("coords",                    39),  # composite, see below
                        (self._status_measured_var,   27),
                        (None,                        41),  # placeholder for time panel
                        (self._status_view_var,       10),
                        (self._status_parallactic_var, 21)
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
                                text='▶',
                                variable=self._auto_advance_var,
                                font=font,
                                indicatoron=False,
                                selectcolor=bg,
                                command=self._on_auto_advance_toggle,
                            )
                            cb.pack(side='right', padx=(2, 0))
                            _ToolTip(cb, 'Auto-advance time (every step minutes)')
                            time_panel.pack(side='right', padx=16)
                        elif var == "coords":
                            # Composite coords panel. A Label carries a single
                            # font, and Unicode has no subscript Sun sign, so
                            # the three parts are separate labels: the sign in
                            # a smaller font, bottom-aligned against the taller
                            # ones, which sets it below the baseline of the "h"
                            # it belongs to.
                            coords_panel = tk.Frame(status_frame, relief='sunken', borderwidth=1)
                            tk.Label(coords_panel, textvariable=self._status_coords_var,
                                     font=font, anchor='w', width=29, borderwidth=0
                                     ).pack(side='left')
                            # Sized to the glyph (a fixed width in character
                            # cells of the small font would leave a gap before
                            # the colon, the sign being wider than a cell) and
                            # blanked by colour rather than by clearing its
                            # text, so the panel never changes width.
                            sun = tk.Label(coords_panel, text="☉", font=sun_font,
                                           anchor='sw', borderwidth=0, padx=0)
                            sun.pack(side='left', anchor='s')
                            self._status_coords_sun_label = sun
                            self._status_coords_sun_fg = sun.cget('fg')
                            self._status_coords_sun_bg = sun.cget('bg')
                            tk.Label(coords_panel, textvariable=self._status_coords_alt_var,
                                     font=font, anchor='w', width=8, borderwidth=0
                                     ).pack(side='left')
                            coords_panel.pack(side='right', padx=16)
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

                    # The bar is built at its full size and then brought
                    # within the window, which on an ordinary screen it already
                    # is (see STATUS_MIN_FONT_SIZE). self.width is what the
                    # window was opened at, and what the canvas settles back to
                    # once nothing is asking for more.
                    settled = self._fit_status_bar(status_frame, self.width)
                    if settled is not None:
                        print(f"Status bar lettering reduced to {settled} pt "
                              f"to fit a {self.width} px window")

                # Build info panel (bottom-left overlay on canvas)
                if hasattr(rt, '_canvas'):
                    info_font = ("Consolas", 9)
                    info_fg = "#808080"
                    info_alt_negative_fg = "#404040"
                    # The colour of the Moon's own night side, so that a panel
                    # reads as a patch of unlit surface rather than as a hole
                    # cut in the sky. Measured off a render at gamma 2.2 and
                    # brightness 80: over 700,000 pixels of the unlit disk it
                    # runs from 0 to about 10, with a median of 2 and three
                    # quarters of it at 3 or below, while the sky beside it is a
                    # flat 0. Taken at that median, so the panel is the shade
                    # the unlit surface most often is, one step above the sky.
                    # Both panels use it.
                    info_bg = "#020202"
                    info_width = 17  # Fixed width in chars (fits DEC: +89°59'59.9")

                    self._info_fg = info_fg
                    self._info_alt_negative_fg = info_alt_negative_fg
                    self._info_alt_label = None

                    self._info_az_var = tk.StringVar(value="Az:")
                    self._info_alt_var = tk.StringVar(value="Alt:")
                    self._info_ra_var = tk.StringVar(value="RA:")
                    self._info_dec_var = tk.StringVar(value="DEC:")
                    self._info_distance_var = tk.StringVar(value="Dist:")
                    self._info_diameter_var = tk.StringVar(value="Diameter:")
                    self._info_geo_libr_l_var = tk.StringVar(value="Geo LbL:")
                    self._info_geo_libr_b_var = tk.StringVar(value="Geo LbB:")
                    self._info_topo_libr_l_var = tk.StringVar(value="Topo LbL:")
                    self._info_topo_libr_b_var = tk.StringVar(value="Topo LbB:")
                    self._info_colong_var = tk.StringVar(value="Colongit:")
                    self._info_illum_var = tk.StringVar(value="Illuminated:")
                    self._info_elongation_var = tk.StringVar(value="Elongation:")
                    self._info_phase_var = tk.StringVar(value="Ph:")
                    self._info_age_var = tk.StringVar(value="Age:")
                    self._info_phase_name_var = tk.StringVar(value="Phase:")

                    info_frame = tk.Frame(rt._canvas, bg=info_bg, padx=6, pady=4)
                    self._info_frame = info_frame
                    info_vars = [
                        self._info_az_var,
                        self._info_alt_var,
                        self._info_ra_var,
                        self._info_dec_var,
                        self._info_distance_var,
                        self._info_diameter_var,
                        self._info_geo_libr_l_var,
                        self._info_geo_libr_b_var,
                        self._info_topo_libr_l_var,
                        self._info_topo_libr_b_var,
                        self._info_colong_var,
                        self._info_illum_var,
                        self._info_elongation_var,
                        self._info_phase_var,
                        self._info_age_var,
                        self._info_phase_name_var,
                    ]
                    for var in info_vars:
                        label = tk.Label(
                            info_frame,
                            textvariable=var,
                            font=info_font,
                            fg=info_fg,
                            bg=info_bg,
                            anchor='w',
                            width=info_width,
                        )
                        label.pack(anchor='w')
                        if var is self._info_alt_var:
                            self._info_alt_label = label
                    info_frame.place(relx=0.0, rely=1.0, anchor='sw', x=6, y=-6)

                    # The status panel: the same lettering and ground as the
                    # ephemeris panel, in the opposite corner. Unlike that one it
                    # starts hidden, having nothing to say that the status bar is
                    # not already saying while there is a status bar to say it.
                    self._status_panel_datetime_var = tk.StringVar()
                    self._status_panel_distance_var = tk.StringVar()
                    self._status_panel_height_var = tk.StringVar()
                    self._status_panel_lat_var = tk.StringVar()
                    self._status_panel_lon_var = tk.StringVar()
                    self._status_panel_sun_var = tk.StringVar()
                    self._status_panel_feature_var = tk.StringVar()

                    status_panel_frame = tk.Frame(rt._canvas, bg=info_bg,
                                                padx=6, pady=4)
                    self._status_panel_frame = status_panel_frame
                    for var in (self._status_panel_feature_var,
                                self._status_panel_datetime_var,
                                self._status_panel_lat_var,
                                self._status_panel_lon_var,
                                self._status_panel_sun_var,
                                self._status_panel_distance_var,
                                self._status_panel_height_var):
                        tk.Label(
                            status_panel_frame,
                            textvariable=var,
                            font=info_font,
                            fg=info_fg,
                            bg=info_bg,
                            anchor='w',
                            width=self.STATUS_PANEL_WIDTH,
                        ).pack(anchor='w')
                    if self.show_status_panel:
                        status_panel_frame.place(relx=1.0, rely=1.0, anchor='se',
                                               x=-6, y=-6)

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
                if self.view_orientation != VIEW_ORIENTATIONS[0]:
                    rt._view_orientation = self.view_orientation
                    # Update grid labels for initial orientation if grid exists
                    if self.moon_grid is not None and self.moon_grid_visible:
                        self.update_grid_labels_for_orientation()

                self._update_all_status_panels()

                # From here the lettering follows the magnification: the poll
                # needs the window, and the window is only now up
                self._schedule_label_scale_poll()

                # The row can only go now: the panels above were built into
                # it, and were told where it sits by asking the label they
                # replace, which a removed widget can no longer answer. The
                # window does not move for this - it has been full screen since
                # the top of this callback - only the canvas grows.
                if self.initial_fullscreen:
                    self._hide_status_row()

                # And the keyboard. PlotOptiX builds this window on its own
                # thread while whatever started the renderer still holds the
                # focus, so it opens without it - and it binds its key handler
                # with bind_all, which reaches every widget of this window and
                # none of any other. So with the focus elsewhere F1 and the rest
                # never arrive at all, and the window has to be clicked before
                # it answers anything. Claimed last, when there is nothing left
                # to build that could take it back.
                bring_to_front(rt._root)
            rt._root.after_idle(init_window)