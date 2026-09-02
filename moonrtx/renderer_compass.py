"""
CompassMixin: a small globe in the corner showing how the view has been turned.

Three keys turn the Moon away from the position it is shown in by default: H/J
roll the view, and Ctrl with the arrow keys swings the camera around the Moon's
polar and equatorial axes. A few presses of each and the surface on screen is no
longer where the eye expects it, with nothing on the disk itself to say how far
it has moved - the terminator and the limb look much the same whichever way
round the globe is.

The compass answers that. In yellow, the equator and the prime meridian as the
default view of this moment shows them - the view the V key returns to, so the
libration and the parallactic roll of the date are already in the yellow. Over
the same disk, in orange, the same two circles where they lie now, with N at the
pole the meridian ends on. The difference between the two colours is exactly what
the three rotation keys have done: reset the view and the orange lands under the
yellow, which is then all that shows.

The part of each line on the near side of the globe is drawn solid and the part
behind it dashed, so the two halves of a rotation are told apart. Both colours
come from a camera - the live one and the default one - so the mirrored view
orientations apply to both. Like the field-of-view frame this is drawn on the Tk
canvas rather than into the scene, so it does not appear in images saved with
F12 or in exported video.
"""

import math
from typing import Optional

import numpy as np

from moonrtx.view_orientation import (FLIP_HORIZONTAL_VIEW_ORIENTATIONS,
                                      FLIP_VERTICAL_VIEW_ORIENTATIONS)


class CompassMixin:
    """Mixin providing the view-orientation globe."""

    COMPASS_REFERENCE_COLOR = "#ffd24a"     # the default position, in yellow
    COMPASS_CURRENT_COLOR = "#ff7a1a"       # orange, well clear of the reference gold
    COMPASS_LINE_WIDTH = 3                  # both sets of circles
    COMPASS_LIMB_WIDTH = 1                  # the outline around them
    COMPASS_FAR_DASH = (2, 3)               # behind the globe, so drawn broken
    COMPASS_FONT = ("Consolas", 10, "bold")
    COMPASS_MARGIN_PX = 16                  # from the corner of the window
    COMPASS_LABEL_GAP_PX = 3                # between the letter and the limb
    # Half of the upper-right quarter of the window, as a square
    COMPASS_SIZE_FRACTION = 0.25
    COMPASS_POINTS = 121                    # samples per drawn circle
    # The camera is read on a light poll, as the field-of-view frame is: the
    # view also moves under the mouse and under PlotOptiX's own handlers, where
    # there is nothing to hook. A redraw is skipped while nothing has moved.
    COMPASS_REFRESH_MS = 200

    def _init_compass_overlay(self):
        """Reset the overlay state; called from MoonRenderer.__init__."""
        self.compass_visible = False
        self._compass_items = []
        self._compass_refresh_id = None
        self._compass_last_view = None

    # ---- geometry ----

    @staticmethod
    def _compass_basis(eye, target, up) -> Optional[tuple]:
        """
        A camera's screen axes in scene coordinates: (right, up, forward).

        Screen right is forward x up, the convention the view navigation works
        in (see NavigationMixin.pan_view), and the up returned is the one truly
        perpendicular to the view direction rather than the camera's own, which
        need not be.
        """
        eye = np.array(eye, dtype=float)
        target = np.array(target, dtype=float)
        up = np.array(up, dtype=float)

        forward = target - eye
        norm = np.linalg.norm(forward)
        if norm == 0.0:
            return None
        forward = forward / norm

        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm == 0.0:                     # looking straight along the up axis
            return None
        right = right / norm

        return right, np.cross(right, forward), forward

    def _compass_live_basis(self) -> Optional[tuple]:
        """Screen axes of the camera the user is looking through."""
        if self.rt is None:
            return None
        cam = self.rt.get_camera(self.CAMERA_NAME)
        return self._compass_basis(cam["Eye"], cam["Target"], cam["Up"])

    def _compass_default_basis(self) -> Optional[tuple]:
        """
        Screen axes of the default view of this moment - what the V key gives
        back. The Moon's own orientation is not part of a camera, so the
        libration and the parallactic roll of the date reach the reference
        lines through moon_rotation, the same way they reach the live ones.
        """
        if self.rt is None or self.moon_ephem is None:
            return None
        cam = self.default_camera
        return self._compass_basis(cam.eye, cam.target, cam.up)

    def _compass_screen_points(self, body_points: np.ndarray, basis) -> Optional[np.ndarray]:
        """
        Where unit vectors of the Moon's own frame fall on the compass disk.

        Returns one (x, y, near) row per point: x to the right and y upwards in
        the picture, both in units of the globe radius, and near true for the
        half of the sphere turned towards the eye. The mirrored view
        orientations are applied here, PlotOptiX flipping the finished image
        rather than the camera (see tkoptix), so the compass turns with what is
        actually on screen.
        """
        if basis is None or self.moon_rotation is None:
            return None
        right, up, forward = basis

        scene = body_points @ self.moon_rotation.T          # body frame to scene
        x = scene @ right
        y = scene @ up
        near = scene @ forward < 0.0                        # forward runs away from the eye

        if self.view_orientation in FLIP_HORIZONTAL_VIEW_ORIENTATIONS:
            x = -x
        if self.view_orientation in FLIP_VERTICAL_VIEW_ORIENTATIONS:
            y = -y
        return np.column_stack((x, y, near))

    @staticmethod
    def _compass_equator(points: int) -> np.ndarray:
        """The equator as unit vectors of the Moon's frame, once round."""
        lon = np.linspace(0.0, 2 * math.pi, points)
        return np.column_stack((np.sin(lon), -np.cos(lon), np.zeros_like(lon)))

    @staticmethod
    def _compass_meridian(points: int) -> np.ndarray:
        """
        The prime meridian, once round: over the north pole to longitude 180
        and back under the south pole, so it circles the globe as the equator
        does rather than stopping at the poles.
        """
        angle = np.linspace(0.0, 2 * math.pi, points)
        return np.column_stack((np.zeros_like(angle), -np.cos(angle), np.sin(angle)))

    def _compass_placement(self) -> Optional[tuple]:
        """Centre and radius of the globe, in canvas pixels."""
        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is None:
            return None

        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:                       # not laid out yet
            width, height = self.rt._width, self.rt._height

        size = min(width, height) * self.COMPASS_SIZE_FRACTION
        margin = self.COMPASS_MARGIN_PX
        centre_x = width - margin - size / 2
        centre_y = margin + size / 2
        # Room for the N outside the limb, so the whole thing keeps to its square
        radius = size / 2 - self.COMPASS_FONT[1] - self.COMPASS_LABEL_GAP_PX
        if radius <= 0:
            return None
        return centre_x, centre_y, radius

    # ---- drawing ----

    def _clear_compass_items(self):
        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is not None:
            for item in self._compass_items:
                canvas.delete(item)
        self._compass_items = []

    def _draw_compass_circle(self, canvas, screen, centre_x, centre_y, radius, colour, width):
        """
        Draw one projected circle, its far half broken.

        The points are walked in order and cut into runs on the same side of the
        globe, so each run is one canvas line: a circle seen edge-on crosses
        from one side to the other twice, and drawing it as a single line would
        either lose the break or cost an item per segment.
        """
        run = []
        run_near = None
        for x, y, near in screen:
            near = bool(near)
            if run_near is not None and near != run_near:
                # The crossing point belongs to both runs, so the line does not
                # fall apart at the limb
                run.append((x, y))
                self._compass_draw_run(canvas, run, run_near, centre_x, centre_y, radius, colour, width)
                run = [(x, y)]
            else:
                run.append((x, y))
            run_near = near
        self._compass_draw_run(canvas, run, bool(run_near), centre_x, centre_y, radius, colour, width)

    def _compass_draw_run(self, canvas, run, near, centre_x, centre_y, radius, colour, width):
        """One canvas line through the points of a run, in globe-radius units."""
        if len(run) < 2:
            return
        coords = []
        for x, y in run:
            coords += [centre_x + x * radius, centre_y - y * radius]
        self._compass_items.append(canvas.create_line(
            *coords, fill=colour, width=width,
            dash=None if near else self.COMPASS_FAR_DASH))

    def _draw_compass(self):
        """Redraw the globe from the current camera."""
        self._clear_compass_items()

        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is None or not self.compass_visible:
            return

        placement = self._compass_placement()
        if placement is None:
            return
        centre_x, centre_y, radius = placement

        # The limb, which is a circle from wherever it is seen
        self._compass_items.append(canvas.create_oval(
            centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius,
            outline=self.COMPASS_REFERENCE_COLOR, width=self.COMPASS_LIMB_WIDTH))

        # The two circles twice over: as the default view of this moment shows
        # them, and as the view being looked through does
        equator = self._compass_equator(self.COMPASS_POINTS)
        meridian = self._compass_meridian(self.COMPASS_POINTS)
        # The current pair goes down first and the reference over it, both being
        # the same weight now: at the default view they fall on each other, and
        # yellow on top is the plainest way of saying so
        for basis, colour in ((self._compass_live_basis(), self.COMPASS_CURRENT_COLOR),
                              (self._compass_default_basis(), self.COMPASS_REFERENCE_COLOR)):
            for body_points in (equator, meridian):
                screen = self._compass_screen_points(body_points, basis)
                if screen is None:
                    continue
                self._draw_compass_circle(canvas, screen, centre_x, centre_y, radius,
                                          colour, self.COMPASS_LINE_WIDTH)

        self._draw_compass_pole(canvas, centre_x, centre_y, radius)

    def _draw_compass_pole(self, canvas, centre_x, centre_y, radius):
        """
        Write N at the north pole of the current globe, at the end of its
        meridian. Set outwards from the middle of the disk so it does not sit
        on the line it belongs to, and simply above the middle in the one view
        where outwards means nothing - straight down onto the pole.
        """
        pole = self._compass_screen_points(np.array([[0.0, 0.0, 1.0]]),
                                           self._compass_live_basis())
        if pole is None:
            return

        x, y = float(pole[0][0]), float(pole[0][1])
        length = math.hypot(x, y)
        out_x, out_y = (0.0, -1.0) if length < 1e-6 else (x / length, -y / length)
        offset = self.COMPASS_FONT[1] * 0.9 + self.COMPASS_LABEL_GAP_PX
        self._compass_items.append(canvas.create_text(
            centre_x + x * radius + out_x * offset, centre_y - y * radius + out_y * offset,
            text="N", fill=self.COMPASS_CURRENT_COLOR, font=self.COMPASS_FONT))

    def _compass_view_state(self):
        """
        A reading that changes whenever the compass would look different: the
        camera, the Moon's own orientation, the mirroring and the window size.
        Compared between ticks so a still view is not redrawn 5 times a second.
        """
        if self.rt is None:
            return None
        cam = self.rt.get_camera(self.CAMERA_NAME)
        canvas = getattr(self.rt, "_canvas", None)
        return (tuple(cam["Eye"]), tuple(cam["Target"]), tuple(cam["Up"]),
                None if self.moon_rotation is None else self.moon_rotation.tobytes(),
                self.view_orientation,
                (canvas.winfo_width(), canvas.winfo_height()) if canvas is not None else None)

    def _compass_refresh_tick(self):
        self._compass_refresh_id = None
        if not self.compass_visible:
            return
        state = self._compass_view_state()
        if state != self._compass_last_view:
            self._compass_last_view = state
            self._draw_compass()
        self._schedule_compass_refresh()

    def _schedule_compass_refresh(self):
        if self.rt is None or self.rt._root is None:
            return
        self._compass_refresh_id = self.rt._root.after(self.COMPASS_REFRESH_MS,
                                                       self._compass_refresh_tick)

    def show_compass(self, visible: bool = True):
        """Show or hide the orientation globe."""
        if self.rt is None:
            return

        self.compass_visible = visible

        if self._compass_refresh_id is not None and self.rt._root is not None:
            self.rt._root.after_cancel(self._compass_refresh_id)
            self._compass_refresh_id = None

        if visible:
            self._compass_last_view = self._compass_view_state()
            self._draw_compass()
            self._schedule_compass_refresh()
        else:
            self._clear_compass_items()

    def toggle_compass(self):
        """Toggle the orientation globe."""
        self.show_compass(not self.compass_visible)
