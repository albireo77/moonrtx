"""
FovMixin: eyepiece / camera field-of-view overlay for MoonRenderer.

Draws the field an optical setup covers on the sky as a frame over the rendered
Moon. This only says anything true because the Moon is rendered at the apparent
size it really has on the date (see MoonRenderer.moon_camera_distance), so the
frame answers the question imagers actually ask: does the Moon fit my camera at
this focal length, tonight - a perigee Moon overflows a frame that an apogee one
leaves room in.

The frame is drawn on the Tk canvas rather than into the scene: it belongs to
the observer's equipment, not to the Moon, so it must not rotate with the view
roll, take part in the lighting, or move with the surface. The side effect is
that it does not appear in images saved with F12, which capture the ray-traced
image alone.
"""

import math
import tkinter as tk
from typing import Optional

import numpy as np

from moonrtx.view_orientation import VIEW_ORIENTATION_NSEW, VIEW_ORIENTATION_SNWE


class FovMixin:
    """Mixin providing the field-of-view overlay and its setup dialog."""

    FOV_COLOR = "#00e5ff"
    FOV_TEXT_COLOR = "#00e5ff"
    FOV_LINE_WIDTH = 2
    FOV_TEXT_FONT = ("Consolas", 10)
    # The screen scale changes with the camera FOV (wheel and Shift+drag zoom),
    # with the camera distance (Shift+right drag, and the apparent size of the
    # date) and with the window size. Some of those are handled inside PlotOptiX,
    # where there is no hook to react to, so the frame is refreshed by a light
    # poll instead: it reads the camera and moves canvas items, nothing more.
    FOV_REFRESH_MS = 200

    def _init_fov_overlay(self):
        """Reset the overlay state; called from MoonRenderer.__init__."""
        self.fov_overlay_visible = False
        # Last values used, so reopening the dialog resumes where it left off.
        # Defaults describe a common planetary setup: a 2000 mm telescope with a
        # 10 mm/68 degree eyepiece, and an APS-C sensor on the same scope.
        self.fov_setup = {
            "mode": "eyepiece",         # "eyepiece" or "camera"
            "focal_mm": 2000.0,
            "sensor_w_mm": 23.5,
            "sensor_h_mm": 15.7,
            "rotation_deg": 0.0,
            "eyepiece_mm": 10.0,
            "afov_deg": 68.0,
        }
        self._fov_items = []
        self._fov_refresh_id = None

    # ---- geometry ----

    def screen_px_per_radian(self) -> Optional[float]:
        """
        Screen scale in pixels per radian of real sky angle.

        Derived from the Moon itself - its rendered radius in pixels against its
        true angular radius - so it follows the zoom, the camera distance and the
        apparent size of the date without assuming anything about any of them.
        """
        if self.rt is None or self.moon_ephem is None:
            return None

        fov_deg = self.rt._optix.get_camera_fov(0)
        if fov_deg <= 0 or self.rt._height <= 0:
            return None

        # Moon centre is the scene origin, so the eye vector is the distance
        eye_distance = float(np.linalg.norm(self.rt.get_camera(self.CAMERA_NAME)["Eye"]))
        if eye_distance <= self.MOON_RADIUS:
            return None

        moon_radius_px = (self.rt._height / 2) \
            * math.tan(math.asin(self.MOON_RADIUS / eye_distance)) \
            / math.tan(math.radians(fov_deg) / 2)
        return moon_radius_px / self.moon_apparent_radius()

    def fov_angles(self) -> tuple[float, float]:
        """
        Width and height of the configured field in radians (equal for an
        eyepiece, whose field is a circle).

        A sensor of size s at focal length f subtends 2 arctan(s / 2f). An
        eyepiece shows its apparent field divided by the magnification, the
        textbook approximation that ignores the eyepiece's own distortion.
        """
        setup = self.fov_setup
        focal = setup["focal_mm"]

        if setup["mode"] == "eyepiece":
            magnification = focal / setup["eyepiece_mm"]
            true_field = math.radians(setup["afov_deg"]) / magnification
            return true_field, true_field

        return (2 * math.atan(setup["sensor_w_mm"] / (2 * focal)),
                2 * math.atan(setup["sensor_h_mm"] / (2 * focal)))

    def _fov_screen_rotation(self) -> float:
        """
        Frame rotation as it appears on screen, in radians. The mirrored view
        orientations reverse the sense of a camera angle, exactly as a star
        diagonal does to the real field.
        """
        rotation = math.radians(self.fov_setup["rotation_deg"])
        if self.view_orientation in (VIEW_ORIENTATION_NSEW, VIEW_ORIENTATION_SNWE):
            return -rotation
        return rotation

    def _fov_summary(self, width_rad: float, height_rad: float) -> str:
        """One line describing the setup, its field and the Moon's size in it."""
        setup = self.fov_setup
        moon_arcmin = math.degrees(2 * self.moon_apparent_radius()) * 60

        if setup["mode"] == "eyepiece":
            magnification = setup["focal_mm"] / setup["eyepiece_mm"]
            head = (f"{setup['focal_mm']:g} mm + {setup['eyepiece_mm']:g} mm "
                    f"({magnification:.0f}x, {setup['afov_deg']:g} deg AFOV)")
            field = _format_angle(width_rad)
        else:
            head = (f"{setup['focal_mm']:g} mm + "
                    f"{setup['sensor_w_mm']:g} x {setup['sensor_h_mm']:g} mm")
            field = f"{_format_angle(width_rad)} x {_format_angle(height_rad)}"

        return f"{head}  ->  {field}   (Moon {moon_arcmin:.1f}')"

    # ---- drawing ----

    def _clear_fov_items(self):
        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is not None:
            for item in self._fov_items:
                canvas.delete(item)
        self._fov_items = []

    def _draw_fov_overlay(self):
        """Redraw the frame from the current camera and setup."""
        self._clear_fov_items()

        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is None or not self.fov_overlay_visible:
            return

        scale = self.screen_px_per_radian()
        if scale is None:
            return

        try:
            width_rad, height_rad = self.fov_angles()
        except (ZeroDivisionError, ValueError):
            return

        width_px = width_rad * scale
        height_px = height_rad * scale
        centre_x = self.rt._width / 2
        centre_y = self.rt._height / 2

        if self.fov_setup["mode"] == "eyepiece":
            radius = width_px / 2
            self._fov_items.append(canvas.create_oval(
                centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius,
                outline=self.FOV_COLOR, width=self.FOV_LINE_WIDTH))
        else:
            angle = self._fov_screen_rotation()
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            half_w, half_h = width_px / 2, height_px / 2
            corners = []
            for dx, dy in ((-half_w, -half_h), (half_w, -half_h),
                           (half_w, half_h), (-half_w, half_h)):
                corners += [centre_x + dx * cos_a - dy * sin_a,
                            centre_y + dx * sin_a + dy * cos_a]
            self._fov_items.append(canvas.create_polygon(
                corners, outline=self.FOV_COLOR, width=self.FOV_LINE_WIDTH, fill=""))

        text = self._fov_summary(width_rad, height_rad)
        if max(width_px, height_px) > max(self.rt._width, self.rt._height):
            text += "   (zoom out to see the whole frame)"
        self._fov_items.append(canvas.create_text(
            centre_x, 10, text=text, anchor='n',
            fill=self.FOV_TEXT_COLOR, font=self.FOV_TEXT_FONT))

    def _fov_refresh_tick(self):
        self._fov_refresh_id = None
        if not self.fov_overlay_visible:
            return
        self._draw_fov_overlay()
        self._schedule_fov_refresh()

    def _schedule_fov_refresh(self):
        if self.rt is None or self.rt._root is None:
            return
        self._fov_refresh_id = self.rt._root.after(self.FOV_REFRESH_MS, self._fov_refresh_tick)

    def show_fov_overlay(self, visible: bool = True):
        """Show or hide the field-of-view frame."""
        if self.rt is None:
            return

        self.fov_overlay_visible = visible

        if self._fov_refresh_id is not None and self.rt._root is not None:
            self.rt._root.after_cancel(self._fov_refresh_id)
            self._fov_refresh_id = None

        if visible:
            self._draw_fov_overlay()
            self._schedule_fov_refresh()
        else:
            self._clear_fov_items()

    def toggle_fov_overlay(self):
        """Toggle the field-of-view frame."""
        self.show_fov_overlay(not self.fov_overlay_visible)

    # ---- setup dialog ----

    def fov_overlay_dialog(self):
        """
        Open the field-of-view setup dialog. Opening it shows the frame, and
        every edit redraws it, so the numbers can be tried against the Moon as
        they are typed.
        """
        if self.rt is None:
            return

        # Reuse the search-dialog flag: it blocks main-window key handling for
        # this dialog in exactly the same way
        self.search_dialog_open = True

        win = tk.Toplevel(self.rt._root)
        # Built withdrawn and shown by _show_dialog once positioned
        win.withdraw()
        win.title("Field of view")
        win.transient(self.rt._root)
        win.resizable(False, False)

        def on_close():
            self.search_dialog_open = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        win.bind('<Escape>', lambda e: on_close())

        main_frame = tk.Frame(win, padx=15, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        setup = self.fov_setup
        mode_var = tk.StringVar(value=setup["mode"])
        focal_var = tk.StringVar(value=f"{setup['focal_mm']:g}")
        sensor_w_var = tk.StringVar(value=f"{setup['sensor_w_mm']:g}")
        sensor_h_var = tk.StringVar(value=f"{setup['sensor_h_mm']:g}")
        rotation_var = tk.StringVar(value=f"{setup['rotation_deg']:g}")
        eyepiece_var = tk.StringVar(value=f"{setup['eyepiece_mm']:g}")
        afov_var = tk.StringVar(value=f"{setup['afov_deg']:g}")

        mode_frame = tk.Frame(main_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        for label, value in (("Eyepiece", "eyepiece"), ("Camera sensor", "camera")):
            tk.Radiobutton(mode_frame, text=label, variable=mode_var, value=value,
                           command=lambda: apply(), anchor='w').pack(side=tk.LEFT, padx=(0, 12))

        grid = tk.Frame(main_frame)
        grid.pack(fill=tk.X)

        # Spin ranges only bound the arrows; any positive value can still be
        # typed, and the rotation wraps so the frame can be spun through 180
        # degrees in either direction without stopping at the ends
        rows = [
            ("Telescope focal length:", focal_var, "mm", None, (50, 20000, 10, False)),
            ("Sensor width:", sensor_w_var, "mm", "camera", None),
            ("Sensor height:", sensor_h_var, "mm", "camera", None),
            ("Camera rotation:", rotation_var, "deg", "camera", (-180, 180, 5, True)),
            ("Eyepiece focal length:", eyepiece_var, "mm", "eyepiece", (1, 60, 1, False)),
            ("Eyepiece apparent field:", afov_var, "deg", "eyepiece", (20, 120, 1, False)),
        ]
        field_widgets = []
        for i, (label, var, unit, only_mode, spin) in enumerate(rows):
            tk.Label(grid, text=label, anchor='e').grid(row=i, column=0, sticky='e', pady=2)
            if spin is None:
                widget = tk.Entry(grid, textvariable=var, width=10)
            else:
                low, high, step, wrap = spin
                widget = tk.Spinbox(grid, textvariable=var, width=8, from_=low, to=high,
                                    increment=step, wrap=wrap)
            widget.grid(row=i, column=1, padx=5, pady=2, sticky='w')
            tk.Label(grid, text=unit, fg='gray').grid(row=i, column=2, sticky='w', pady=2)
            field_widgets.append((widget, only_mode))

        status_var = tk.StringVar()
        status_label = tk.Label(main_frame, textvariable=status_var, anchor='w', justify='left')
        status_label.pack(fill=tk.X, pady=(8, 0))

        def apply(*args):
            """Validate the fields, store them and redraw the frame."""
            for widget, only_mode in field_widgets:
                widget.config(state='normal' if only_mode in (None, mode_var.get()) else 'disabled')
            mode = mode_var.get()
            values = dict(self.fov_setup, mode=mode)

            # Only the fields the chosen mode uses are read, so a half-typed
            # value left in the other mode's fields cannot block the redraw
            fields = [("focal_mm", focal_var, "telescope focal length")]
            if mode == "camera":
                fields += [("sensor_w_mm", sensor_w_var, "sensor width"),
                           ("sensor_h_mm", sensor_h_var, "sensor height")]
            else:
                fields += [("eyepiece_mm", eyepiece_var, "eyepiece focal length"),
                           ("afov_deg", afov_var, "apparent field")]
            try:
                for key, var, name in fields:
                    value = float(var.get())
                    if value <= 0:
                        raise ValueError(f"{name} must be above zero")
                    values[key] = value
                if mode == "camera":
                    # Any angle, including negative, is a valid rotation
                    values["rotation_deg"] = float(rotation_var.get())
            except ValueError as e:
                status_label.config(fg='red')
                status_var.set(f"Invalid settings: {e}")
                return

            self.fov_setup = values
            self.show_fov_overlay(True)
            width_rad, height_rad = self.fov_angles()
            status_label.config(fg='black')
            status_var.set(self._fov_summary(width_rad, height_rad))

        for var in (focal_var, sensor_w_var, sensor_h_var, rotation_var, eyepiece_var, afov_var):
            var.trace_add('write', apply)

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Button(btn_frame, text="Hide frame", width=12,
                  command=lambda: (self.show_fov_overlay(False), on_close())).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Close", width=10, command=on_close).pack(side=tk.RIGHT)

        apply()
        self._show_dialog(win)


def _format_angle(angle_rad: float) -> str:
    """Degrees for wide fields, arcminutes for the narrow ones."""
    degrees = math.degrees(angle_rad)
    return f"{degrees:.2f} deg" if degrees >= 2.0 else f"{degrees * 60:.1f}'"
