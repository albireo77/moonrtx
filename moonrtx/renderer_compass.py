"""
CompassMixin: a small globe in the corner showing how the view has been turned.

Three keys turn the Moon away from the position it is shown in by default: H/J
roll the view, and Ctrl with the arrow keys swings the camera around the Moon's
polar and equatorial axes. A few presses of each and the surface on screen is no
longer where the eye expects it, with nothing on the disk itself to say how far
it has moved - the terminator and the limb look much the same whichever way
round the globe is.

The compass answers that. Two globes are drawn over each other, each as its
equatorial plane and, out of the middle of it, two rays: one to the north pole and
one to longitude 0 on the equator, each bumped and named at its tip, with the
meridian arc between those tips closing them into a quarter of a globe. In grey,
the globe as the default view of this moment shows it - the view the V key returns
to, so the libration and the parallactic roll of the date are already in the
grey. In blue, the globe as it lies now. The difference between the two colours
is exactly what the rotation keys have done: reset the view and the blue lands
under the grey, which is then all that shows.

The planes are filled rather than outlined, in the stipple a Tk canvas has to use
for want of transparency. An equator projects to an ellipse however the globe is
turned, so the fill says which way the plane is tilted and how near edge-on it is
at a glance - the reading the rings of a planet give. It says nothing about a spin
about the poles, though, and neither does the polar axis: both are unchanged by
one. The ray across the plane is what shows that, swinging round with the globe,
its named end saying which way it has gone.

Under the globe the same difference is written out in three numbers: how far the
middle of the view has been carried in longitude and in latitude, and how far the
picture has been twisted. They describe where the view stands rather than which
keys were pressed - Ctrl with an arrow turns about an axis fixed to the Moon, so
away from the default longitude it twists the picture as well as moving it. All
three read zero when the view is at its default.

Both colours come from a camera - the live one and the default one - so the
mirrored view orientations apply to both. Like the field-of-view frame this is
drawn on the Tk canvas rather than into the scene, so it does not appear in
images saved with F12 or in exported video.
"""

import math
from typing import Optional

import numpy as np

from moonrtx.view_orientation import (FLIP_HORIZONTAL_VIEW_ORIENTATIONS,
                                      FLIP_VERTICAL_VIEW_ORIENTATIONS)


class CompassMixin:
    """Mixin providing the view-orientation globe."""

    # The locator's two colours, so that the pair of overlays reads as one set:
    # the grey it draws its rim and graticule in for the globe as it is meant to
    # stand, and the blue it draws the field in for the globe as it stands now
    COMPASS_REFERENCE_COLOR = "#9aa7bd"     # the default position, in grey
    COMPASS_CURRENT_COLOR = "#6772ab"       # blue, well clear of the reference grey
    COMPASS_LINE_WIDTH = 3                  # the axis of each globe
    COMPASS_DOT_RADIUS = 4                  # the bumps along it and on the rim
    # The equatorial plane is filled rather than merely outlined, and the fill
    # is stippled: a Tk canvas has no transparency, so a dither is the only way
    # for the disk underneath - and the lines crossing it - to show through.
    COMPASS_DISK_STIPPLE = "gray25"
    # Lettering over the render has no background it can count on: the ground
    # beneath it runs from the black of the sky to the white of a lit highland,
    # and no one colour is legible on both - the readings were measured at a
    # fifth over one on ordinary grey, where four and a half is what small text
    # wants. So each is written a second time, in black, at every one of these
    # offsets, and the coloured text laid on top. The letter then carries its
    # own dark rim wherever it goes.
    COMPASS_HALO_COLOR = "#000000"
    COMPASS_HALO_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0),
                            (1, 0), (-1, 1), (0, 1), (1, 1))
    COMPASS_FONT = ("Consolas", 10, "bold")
    COMPASS_VALUE_FONT = ("Consolas", 10)
    # Clear of the globe by a bump and a label, the plane reaching the rim when it
    # is turned full on to the eye
    COMPASS_VALUE_GAP_PX = 20
    # Nearer the middle of the disk than this, in units of the radius, the pole is
    # turned so nearly at the eye that the angle it is drawn at means nothing
    COMPASS_POLE_ON_LIMIT = 0.05
    COMPASS_MARGIN_PX = 16                  # from the corner of the window
    COMPASS_LABEL_GAP_PX = 3                # between the letter and the limb
    # Half of the upper-right quarter of the window, as a square
    COMPASS_SIZE_FRACTION = 0.25
    COMPASS_POINTS = 181                    # samples round each equator
    # Two points of the lunar frame, as unit vectors: the north pole, which the
    # axis is drawn through, and longitude 0 on the equator, where the prime
    # meridian crosses it. The second rides round the rim of the plane as the
    # globe is spun about its poles, which nothing else here would show.
    COMPASS_NORTH = (0.0, 0.0, 1.0)
    COMPASS_PRIME = (0.0, -1.0, 0.0)
    COMPASS_ARC_POINTS = 46                 # samples along the quarter between them
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

    def _compass_arc(self, points: int) -> np.ndarray:
        """
        The quarter of the prime meridian running from the north pole down to
        longitude 0 on the equator - the ground the two rays leave uncovered.

        The two ends are unit vectors at right angles to each other, so turning
        one into the other by cosine and sine sweeps the great circle between
        them at constant speed.
        """
        angle = np.linspace(0.0, math.pi / 2, points)
        return (np.outer(np.cos(angle), np.array(self.COMPASS_NORTH))
                + np.outer(np.sin(angle), np.array(self.COMPASS_PRIME)))

    def _compass_view_centre(self, basis) -> Optional[tuple]:
        """
        Selenographic latitude and longitude of the point at the middle of a view.

        The eye lies along the view direction reversed, and the Moon is at the
        origin of the scene, so that direction taken back into the lunar frame is
        the point turned towards it.
        """
        if basis is None or self.moon_rotation is None:
            return None
        _right, _up, forward = basis

        body = self.moon_rotation.T @ -forward
        lat = math.degrees(math.asin(max(-1.0, min(1.0, float(body[2])))))
        # The frame the labels and pins are placed in: longitude 0 towards -Y
        return lat, math.degrees(math.atan2(float(body[0]), float(-body[1])))

    def _compass_pole_angle(self, basis) -> Optional[float]:
        """
        The angle the north pole is drawn at, measured from straight up towards the
        right of the picture. None when the pole is turned so nearly at the eye
        that it projects onto the middle of the disk, where it has no angle.

        Read on the picture rather than in the scene, mirrored view orientations
        included, so it answers the twist actually seen.
        """
        screen = self._compass_screen_points(np.array([self.COMPASS_NORTH], dtype=float),
                                             basis)
        if screen is None:
            return None
        x, y = float(screen[0][0]), float(screen[0][1])
        if math.hypot(x, y) < self.COMPASS_POLE_ON_LIMIT:
            return None
        return math.degrees(math.atan2(x, y))

    @staticmethod
    def _compass_turn(degrees_apart: float) -> float:
        """An angle brought into -180 to 180, a turn being the shorter way round."""
        return (degrees_apart + 180.0) % 360.0 - 180.0

    def _compass_readings(self) -> Optional[list]:
        """
        How far the view has been turned from its default, in three numbers.

        lon and lat are where the middle of the view has been carried over the
        surface, against the point the default view has there - which is the
        sub-Earth point, so the reading is a departure from what libration alone
        would show, and stays put as the clock runs. rot is the twist of the
        picture: the angle the north pole is drawn at, against the angle the
        default view draws it at.

        They say where the view stands, not which keys were pressed to get there.
        H and J move rot alone, but Ctrl with an arrow turns about an axis fixed to
        the Moon rather than to the screen, and away from the default longitude
        that axis is oblique to the view: the same key then tips and twists at
        once. A quarter turn in longitude puts the equatorial axis straight at the
        eye, where Ctrl+Up and Ctrl+Down are a pure roll and move nothing but rot.
        Carrying the view round a closed loop over the surface leaves it turned as
        well, which is a property of the sphere and not of the reading.
        """
        live, default = self._compass_live_basis(), self._compass_default_basis()
        centre, centre_default = (self._compass_view_centre(live),
                                  self._compass_view_centre(default))
        if centre is None or centre_default is None:
            return None

        readings = [("lon", self._compass_turn(centre[1] - centre_default[1])),
                    ("lat", self._compass_turn(centre[0] - centre_default[0]))]
        angle, angle_default = (self._compass_pole_angle(live),
                                self._compass_pole_angle(default))
        readings.append(("rot", None if angle is None or angle_default is None
                         else self._compass_turn(angle - angle_default)))
        return readings

    def _rimmed_text(self, canvas, x, y, text, fill, font, anchor="n") -> list:
        """
        Write text with a dark rim round it, and hand back every piece of it.

        Tk has no outline for text, so the rim is the same string written once
        more at each offset around it and the wanted colour laid over the lot.
        Cheap enough at this size, and the only thing that makes small lettering
        hold up over ground that is black in one place and white in another.

        The pieces are returned rather than kept here, so that the locator can
        rim its own lettering with the same hand - both overlays draw on the one
        canvas and face the same ground, and each keeps its own list of what it
        has drawn so that it can take it away again.
        """
        items = []
        for dx, dy in self.COMPASS_HALO_OFFSETS:
            items.append(canvas.create_text(
                x + dx, y + dy, text=text, anchor=anchor,
                fill=self.COMPASS_HALO_COLOR, font=font))
        items.append(canvas.create_text(
            x, y, text=text, anchor=anchor, fill=fill, font=font))
        return items

    def _draw_compass_readings(self, canvas, centre_x, centre_y, radius):
        """
        Write the three readings under the globe, in the colour of the globe they
        describe, each rimmed in black so that it can be read over ground of any
        brightness. All three stand at zero when the view is at its default,
        which is the same thing the blue lying under the grey says.
        """
        readings = self._compass_readings()
        if readings is None:
            return

        line_height = self.COMPASS_VALUE_FONT[1] + 4
        top = centre_y + radius + self.COMPASS_VALUE_GAP_PX
        for row, (name, value) in enumerate(readings):
            # A fixed width for the number, so the three read as a column
            written = "   --" if value is None else f"{value:+6.1f}"
            self._compass_items.extend(self._rimmed_text(
                canvas, centre_x, top + row * line_height,
                f"\u0394{name} {written}\u00b0",
                self.COMPASS_CURRENT_COLOR, self.COMPASS_VALUE_FONT))

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

    def _compass_polygon(self, screen, centre_x, centre_y, radius) -> list:
        """Projected points as the flat list of canvas coordinates Tk takes."""
        coords = []
        for x, y, _near in screen:
            coords += [centre_x + x * radius, centre_y - y * radius]
        return coords

    def _draw_compass_disk(self, canvas, coords, colour):
        """
        Fill the plane a circle bounds.

        The equator projects to an ellipse however the globe is turned, and
        filling it says at a glance which way the plane is tilted and how far
        edge-on it is - the same reading the rings of a planet give.
        """
        self._compass_items.append(canvas.create_polygon(
            *coords, fill=colour, stipple=self.COMPASS_DISK_STIPPLE, outline=""))

    def _compass_bump(self, canvas, x, y, colour):
        """One dot on a line, drawn as a disk of its own colour."""
        bump = self.COMPASS_DOT_RADIUS
        self._compass_items.append(canvas.create_oval(
            x - bump, y - bump, x + bump, y + bump, fill=colour, outline=""))

    def _compass_at(self, body_point, basis, centre_x, centre_y, radius):
        """One point of the lunar frame, in canvas pixels."""
        screen = self._compass_screen_points(np.array([body_point], dtype=float), basis)
        if screen is None:
            return None
        x, y = float(screen[0][0]), float(screen[0][1])
        return centre_x + x * radius, centre_y - y * radius

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

        # Each globe is drawn as its equatorial plane, the axis through it, the
        # point where the prime meridian crosses the equator, and the north pole
        # the axis ends on. The current globe goes down first and the reference
        # over it: at the default view the two fall on each other, and grey on
        # top is the plainest way of saying so.
        equator = self._compass_equator(self.COMPASS_POINTS)
        globes = []
        for basis, colour in ((self._compass_live_basis(), self.COMPASS_CURRENT_COLOR),
                              (self._compass_default_basis(), self.COMPASS_REFERENCE_COLOR)):
            screen = self._compass_screen_points(equator, basis)
            if screen is None:
                continue
            globes.append((basis, colour,
                           self._compass_polygon(screen, centre_x, centre_y, radius)))

        # Both planes are laid down before either axis, so a fill never dims a
        # line drawn before it
        for _basis, colour, coords in globes:
            self._draw_compass_disk(canvas, coords, colour)
        for basis, colour, _coords in globes:
            self._draw_compass_axis(canvas, centre_x, centre_y, radius, basis, colour)
            self._draw_compass_prime(canvas, centre_x, centre_y, radius, basis, colour)
            self._draw_compass_arc(canvas, centre_x, centre_y, radius, basis, colour)
            self._draw_compass_pole(canvas, centre_x, centre_y, radius, basis, colour)

        self._draw_compass_readings(canvas, centre_x, centre_y, radius)

    def _draw_compass_axis(self, canvas, centre_x, centre_y, radius, basis, colour):
        """
        The axis the globe turns on, from the middle of the disk - where it meets
        the equatorial plane - out to the north pole, bumped at that end.

        The half it is drawn on says as much as the whole: which way the axis
        leans, and how far it is tipped towards the eye - foreshortened to nothing
        when the pole is turned straight at us. Drawn through, the two globes
        would cross in the middle of a small picture and be read as one shape.
        """
        north = self._compass_at(self.COMPASS_NORTH, basis, centre_x, centre_y, radius)
        if north is None:
            return

        self._compass_items.append(canvas.create_line(
            centre_x, centre_y, north[0], north[1],
            fill=colour, width=self.COMPASS_LINE_WIDTH))
        self._compass_bump(canvas, north[0], north[1], colour)

    def _draw_compass_prime(self, canvas, centre_x, centre_y, radius, basis, colour):
        """
        The prime meridian where it crosses the equatorial plane: a ray from the
        centre out to longitude 0, bumped and named at its end.

        Neither the plane nor the polar axis is changed by a spin about the poles,
        so without this line Ctrl+Left/Right would move nothing here. It swings
        round with the globe instead, and its label says which way it has gone.

        At the default view longitude 0 faces the eye and the line is foreshortened
        to a point on the centre of the disk, which is the reading not spun; the
        label then goes above it, there being no outwards to speak of.
        """
        end = self._compass_at(self.COMPASS_PRIME, basis, centre_x, centre_y, radius)
        if end is None:
            return

        self._compass_items.append(canvas.create_line(
            centre_x, centre_y, end[0], end[1],
            fill=colour, width=self.COMPASS_LINE_WIDTH))
        self._compass_bump(canvas, end[0], end[1], colour)
        self._compass_label(canvas, end[0], end[1], centre_x, centre_y,
                            "0", colour, (0.0, -1.0))

    def _draw_compass_arc(self, canvas, centre_x, centre_y, radius, basis, colour):
        """
        The meridian arc between the ends of the two rays, closing them into one
        shape: a quarter of the globe, drawn as it is really seen.

        Where the rays give two directions, the arc gives the surface they span,
        and it is the part of the drawing that bulges out of the flat as the globe
        turns. At the default view the meridian is edge-on and the arc lies along
        the polar ray, which is the reading everything is where it started.
        """
        screen = self._compass_screen_points(self._compass_arc(self.COMPASS_ARC_POINTS),
                                             basis)
        if screen is None:
            return
        self._compass_items.append(canvas.create_line(
            *self._compass_polygon(screen, centre_x, centre_y, radius),
            fill=colour, width=self.COMPASS_LINE_WIDTH))

    def _compass_label(self, canvas, x, y, centre_x, centre_y, text, colour, fallback):
        """
        Write a label just outside a point, set away from the middle of the disk
        so it clears the line it belongs to. A point on that middle has no
        outwards to speak of, and takes the direction given instead.
        """
        away_x, away_y = x - centre_x, y - centre_y
        length = math.hypot(away_x, away_y)
        out_x, out_y = fallback if length < 1e-6 else (away_x / length, away_y / length)
        offset = self.COMPASS_FONT[1] * 0.9 + self.COMPASS_LABEL_GAP_PX
        self._compass_items.append(canvas.create_text(
            x + out_x * offset, y + out_y * offset,
            text=text, fill=colour, font=self.COMPASS_FONT))

    def _draw_compass_pole(self, canvas, centre_x, centre_y, radius, basis, colour):
        """
        Write N at the north pole of the globe, at the end of its axis. Set
        outwards from the middle of the disk so it does not sit on the line it
        belongs to, and simply above the middle in the one view where outwards
        means nothing - straight down onto the pole.
        """
        north = self._compass_at(self.COMPASS_NORTH, basis, centre_x, centre_y, radius)
        if north is None:
            return

        self._compass_label(canvas, north[0], north[1], centre_x, centre_y,
                            "N", colour, (0.0, -1.0))

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
