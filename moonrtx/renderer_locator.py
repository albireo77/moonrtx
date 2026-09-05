"""
LocatorMixin: a small disk in the corner saying where on the Moon the view is.

Magnified far enough, one piece of the surface looks much like another. The
field-of-view frame says how much of the Moon is in the picture and the compass
says which way the globe has been turned, but neither says which part of the
face is being looked at, and at thirty times the eye loses the thread within a
couple of drags: every ridge is a ridge, and the limb that would place them is
long off the screen.

The locator answers that. The Moon is drawn small in the corner as it stands to
the eye - the whole disk of the half the camera can see, lit as that half really
is, the terminator falling where it really falls - and the edges of the picture
are traced onto it in the colour of the field-of-view frame. What is on screen is
inside that outline.

The half the camera can see, and not the half Earth sees. The arrow keys carry
the eye round the Moon, and a few presses take it far enough that it is looking
at another part of the globe altogether, lit quite differently: at fifty degrees
round, the face in the picture was measured a sixth lit while the face Earth sees
was near two thirds. A disk drawn for Earth's half then shows a phase and a
terminator with no likeness to the picture beside it, which is worse than no disk
at all. Drawn for the eye, the two agree - and while the eye has not been carried
anywhere, which is most of the time, the two halves are the same half and nothing
about the disk changes.

The outline is the true shape of what is being seen rather than a guess from
the middle of it: a field near the limb comes out foreshortened, as it should,
and one on the terminator is seen to be there. It is found by sweeping out from
the middle of the view - in each direction, how far the surface runs before it
leaves the picture or turns out of sight round the limb - so it is bounded by
the edge of the picture where the picture is the narrower and by the Moon's own
edge where the Moon is. At the default view everything is on screen and the
outline is the limb itself; magnifying shrinks it from there down to a patch,
with no step anywhere along the way. Aimed off the Moon altogether, nothing is
drawn but the disk.

Day and night are both laid on, pale and black, in the same stipple, so the
picture behind shows through each alike and the terminator between them is the
one the ground really has. The equator and the prime meridian are drawn across
the face with a 0 beside where they cross, which says how the globe is turned and
how far from the middle of the face the view has gone; N and S mark the poles,
outside the rim while a pole lies near it and on the pole itself once the eye has
climbed far enough to bring it inboard.

The projection is the compass's, which is what makes the two agree - the same
default camera of the same moment, and the same handling of the mirrored view
orientations. Like the compass this is drawn on the Tk canvas rather than into
the scene, so it does not appear in images saved with F12 or in exported video.
"""

import math
from typing import Optional

import numpy as np


class LocatorMixin:
    """Mixin providing the where-on-the-Moon inset."""

    LOCATOR_LIMB_COLOR = "#9aa7bd"          # the rim, and the terminator on it
    LOCATOR_LIT_COLOR = "#e2e7ef"           # daylight, pale
    LOCATOR_DARK_COLOR = "#000000"          # and night, laid on the same way
    # Stippled for the same reason the compass planes are: a Tk canvas has no
    # transparency, and magnified far enough the corner this sits in is surface
    # rather than sky, which the reader should still be able to see. Both sides
    # carry the same stipple, so neither hides what is behind it more than the
    # other and the two read as one disk lit from one side.
    LOCATOR_FACE_STIPPLE = "gray50"
    # A dark, quiet blue. The rim round the lettering (see
    # CompassMixin._rimmed_text) is what settled how dark it could be: a letter
    # is no longer read against the Moon but against its own black edge, and
    # small text wants four and a half to one against that, which puts a floor
    # under it - below about a 116 out of 255 in lightness the letter sinks into
    # its own outline. This sits just clear of that floor, at four and a half on
    # the rim, and keeps far enough from the reference grey not to be taken for
    # it. The lines carry no rim and are still read against the ground, where a
    # colour this dark shows well on the highlands and less well on the maria.
    LOCATOR_FIELD_COLOR = "#6772ab"
    LOCATOR_FIELD_WIDTH = 2
    LOCATOR_LINE_WIDTH = 1
    LOCATOR_FONT = ("Consolas", 10)
    LOCATOR_LABEL_FONT = ("Consolas", 10, "bold")
    LOCATOR_SIZE_FRACTION = 0.18            # of the shorter side of the window
    LOCATOR_MARGIN_PX = 16                  # from the corner of the window
    LOCATOR_LABEL_GAP_PX = 4                # between the rim and the N
    # Where the push outside the rim begins to fade in. Nearer the middle than
    # this the pole has room for its letter and keeps it; from here to the limb
    # the letter is carried out past the rim, a little further for every step
    # the pole takes towards the edge.
    LOCATOR_POLE_AT_RIM = 0.93
    # How far past the limb a pole may be and still be named. The libration
    # alone carries a pole up to seven degrees over the edge, so a rule that
    # named only what the eye can strictly see would drop N from an ordinary
    # view of the near side for half of every month; and a pole a little way
    # round the back still lies in the direction the letter is written, which
    # is what the letter is for. Past this the pole is well gone and so is it.
    LOCATOR_POLE_BEHIND = 12.0              # degrees
    # The equator stands at right angles to the Moon's axis, and the prime
    # meridian to the point a quarter turn east of where they cross - which is
    # the third of these, the origin of the graticule
    LOCATOR_AXIS = (0.0, 0.0, 1.0)
    LOCATOR_EAST = (1.0, 0.0, 0.0)
    LOCATOR_ORIGIN = (0.0, -1.0, 0.0)
    # The 0 is written clear of the crossing rather than on it - a little north
    # of the equator and a little east of the prime meridian - so that neither
    # line runs through the glyph. Given as an angle over the ground and not as
    # pixels on the canvas, so it keeps to that quarter however the disk is
    # turned, and mirrors along with the picture when the view orientation does.
    LOCATOR_ORIGIN_OFFSET = 5.0             # degrees, north and east
    # Clear of the N wherever the pole is lying, the reading being written
    # straight below the disk and the pole free to come round and point there
    LOCATOR_VALUE_GAP_PX = 26
    LOCATOR_POINTS = 121                    # samples along the limb, and the terminator
    # Directions swept out from the middle of the view to find the outline, and
    # how many times each is halved to place the point it ends at
    LOCATOR_SWEEP_POINTS = 96
    LOCATOR_SWEEP_STEPS = 20
    # A field smaller than this is drawn as a ring of this size instead of its
    # true outline. At fifty times the disk it is a pixel across, and a mark too
    # small to find is no better than none.
    LOCATOR_MIN_FIELD_PX = 8
    # A cross product shorter than this has no direction worth taking from it:
    # the Sun along the sight line, leaving a face all day or all night with no
    # terminator to hinge on, or a sweep starting where the camera happens to
    # point straight along one of its own axes
    LOCATOR_TOO_SHORT = 1e-6
    # The camera is read on a light poll, as the compass and the field-of-view
    # frame are: the view moves under the mouse and under PlotOptiX's own
    # handlers, where there is nothing to hook.
    LOCATOR_REFRESH_MS = 200

    def _init_locator(self):
        """Reset the overlay state; called from MoonRenderer.__init__."""
        self.locator_visible = False
        self._locator_items = []
        self._locator_refresh_id = None
        self._locator_last_view = None

    # ---- geometry ----

    def _locator_sunward(self) -> Optional[np.ndarray]:
        """
        The sub-solar point as a unit vector of the Moon's own frame - the place
        the Sun stands overhead, and so the pole of the terminator.
        """
        if self.moon_ephem is None:
            return None
        lat = math.radians(self.moon_ephem.subsolar_lat)
        lon = math.radians(self.moon_ephem.subsolar_lon)
        return np.array([math.cos(lat) * math.sin(lon),
                         -math.cos(lat) * math.cos(lon),
                         math.sin(lat)])

    def _locator_disk_basis(self) -> Optional[tuple]:
        """
        The screen axes of the disk: the half of the Moon the eye can see, lying
        the way the picture lies.

        The projection looks along the line from the middle of the Moon out to
        the eye, so the disk is the half that eye has in front of it. Where the
        camera is aimed does not come into it - a view swung off to one side
        still stands over the same half - and neither does how far it is zoomed.
        Only carrying the eye round the globe turns this disk.

        The camera's own up is kept, so the disk lies the same way up as the
        picture: the terminator on the one runs the way it runs on the other,
        and the pole leans the same way.
        """
        if self.rt is None:
            return None

        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"], dtype=float)
        length = float(np.linalg.norm(eye))
        if length < self.LOCATOR_TOO_SHORT:
            return None
        forward = -eye / length                 # from the eye towards the Moon

        right = np.cross(forward, np.array(cam["Up"], dtype=float))
        if np.linalg.norm(right) < self.LOCATOR_TOO_SHORT:
            return None                         # looking along its own up axis
        right = right / np.linalg.norm(right)
        return right, np.cross(right, forward), forward

    def _locator_sightline(self, basis) -> Optional[np.ndarray]:
        """
        The direction a view looks, in the Moon's own frame.

        Everything else here is worked out in that frame rather than the scene's,
        because the two circles being drawn are properties of the Moon and the
        Sun rather than of the camera, and because the compass projection - which
        this borrows, so that the two overlays cannot disagree - takes its points
        that way round.
        """
        if basis is None or self.moon_rotation is None:
            return None
        return self.moon_rotation.T @ basis[2]

    @staticmethod
    def _locator_half_circle(start: np.ndarray, through: np.ndarray,
                             points: int) -> np.ndarray:
        """
        Half a great circle: from start, through `through`, to the opposite of
        start.

        The two arguments are perpendicular unit vectors, and turning one into
        the other by cosine and sine sweeps the circle between them at a constant
        speed. Which half comes out is decided by which way `through` points,
        which is how the caller picks the lit half of the limb and the near half
        of the terminator.
        """
        angle = np.linspace(0.0, math.pi, points)
        return np.outer(np.cos(angle), start) + np.outer(np.sin(angle), through)

    def _locator_origin_label(self) -> tuple:
        """
        Where the 0 is written: off the crossing of the equator and the prime
        meridian, into the quarter north of the one and east of the other.
        """
        offset = math.radians(self.LOCATOR_ORIGIN_OFFSET)
        return (math.cos(offset) * math.sin(offset),
                -math.cos(offset) * math.cos(offset),
                math.sin(offset))

    def _locator_whole_limb(self, forward: np.ndarray) -> np.ndarray:
        """The limb, all the way round, for a face that is wholly day or night."""
        across = np.cross(forward, np.array(self.LOCATOR_AXIS, dtype=float))
        if np.linalg.norm(across) < self.LOCATOR_TOO_SHORT:
            across = np.cross(forward, np.array(self.LOCATOR_EAST, dtype=float))
        across = across / np.linalg.norm(across)
        along = np.cross(forward, across)
        return np.concatenate((
            self._locator_half_circle(across, along, self.LOCATOR_POINTS),
            self._locator_half_circle(-across, -along, self.LOCATOR_POINTS)))

    def _locator_face(self, basis, lit: bool) -> Optional[np.ndarray]:
        """
        One side of the disk - daylight or night - as a closed ring of unit
        vectors of the Moon's frame, or None when there is none of it to draw.

        Two half circles bound either. The limb is the circle at right angles to
        the sight line and the terminator the one at right angles to the
        sub-solar direction, and the two cross where a point is at right angles
        to both - the horns of the phase. Taking the wanted half of the limb and
        the near half of the terminator, from horn to horn and back, closes the
        shape, and it comes out as the phase really is: gibbous or crescent, and
        tilted the way the real one is tilted, which a half-ellipse fitted to the
        disk would only manage upright. The two sides share the terminator, so
        they meet along it exactly and neither overlaps the other.

        A full face has no horns to hinge on - the sub-solar direction lies along
        the sight line - and is answered with the whole limb for the side that
        has it all, and nothing for the other.
        """
        forward = self._locator_sightline(basis)
        sunward = self._locator_sunward()
        if forward is None or sunward is None:
            return None

        horn = np.cross(forward, sunward)
        length = float(np.linalg.norm(horn))
        if length < self.LOCATOR_TOO_SHORT:
            # The Sun is along the sight line: the face is all day or all night.
            # forward runs from the eye towards the Moon, so a sub-solar point
            # facing the eye - the full moon - has it pointing the other way.
            full_day = float(sunward @ forward) < 0.0
            return self._locator_whole_limb(forward) if full_day == lit else None
        horn = horn / length

        # Along the limb from one horn to the other, keeping to the side wanted
        along_limb = np.cross(forward, horn)
        if (float(along_limb @ sunward) < 0.0) == lit:
            along_limb = -along_limb
        # and back along the terminator, keeping to the side turned towards us
        along_terminator = np.cross(sunward, horn)
        if float(along_terminator @ forward) > 0.0:
            along_terminator = -along_terminator

        return np.concatenate((
            self._locator_half_circle(horn, along_limb, self.LOCATOR_POINTS),
            self._locator_half_circle(-horn, along_terminator, self.LOCATOR_POINTS)))

    def _locator_great_circle(self, pole, forward: np.ndarray) -> Optional[np.ndarray]:
        """
        The half of a great circle that is turned towards the eye, named by the
        pole it stands at right angles to.

        The equator has the Moon's own axis for a pole and the prime meridian
        the point a quarter turn east on the equator. Half of any such circle
        faces the eye and half is behind, and the two halves meet where it
        crosses the limb - at right angles to the pole and the sight line both.
        Seen exactly edge-on the circle is the limb itself, and there is nothing
        of it to draw that the rim does not already say.
        """
        if forward is None:
            return None
        pole = np.array(pole, dtype=float)
        start = np.cross(pole, forward)
        if np.linalg.norm(start) < self.LOCATOR_TOO_SHORT:
            return None
        start = start / np.linalg.norm(start)

        # The point of the circle nearest the eye, which the near half runs over
        through = -forward - pole * float(pole @ -forward)
        if np.linalg.norm(through) < self.LOCATOR_TOO_SHORT:
            return None
        return self._locator_half_circle(start, through / np.linalg.norm(through),
                                         self.LOCATOR_POINTS)

    # ---- what is on screen ----

    def _locator_camera(self) -> Optional[tuple]:
        """
        The live camera, as everything the tests below need of it.

        The field PlotOptiX reports is the vertical one, so the horizontal half
        angle is that one stretched by the shape of the window.
        """
        if self.rt is None or self.moon_rotation is None or self.rt._height <= 0:
            return None

        cam = self.rt.get_camera(self.CAMERA_NAME)
        basis = self._compass_basis(cam["Eye"], cam["Target"], cam["Up"])
        fov = self.rt._optix.get_camera_fov(0)
        if basis is None or fov <= 0.0:
            return None

        right, up, forward = basis
        tan_up = math.tan(math.radians(fov) / 2)
        return (np.array(cam["Eye"], dtype=float), right, up, forward,
                tan_up * (self.rt._width / self.rt._height), tan_up)

    def _locator_on_screen(self, scene: np.ndarray, camera) -> np.ndarray:
        """
        Which of these points of the surface are actually in the picture.

        Two things have to hold. The point must be on the side of the globe the
        camera can see, which for a point at the surface means the eye is
        further from it than the globe is wide across the sight line; and it
        must fall inside the frame, which for a pinhole camera is a matter of
        how far off the axis it lies for its depth.
        """
        eye, right, up, forward, tan_right, tan_up = camera

        from_eye = scene - eye
        depth = from_eye @ forward
        safe = np.where(depth > 0.0, depth, 1.0)
        return ((scene @ eye > self.MOON_RADIUS ** 2)
                & (depth > 0.0)
                & (np.abs(from_eye @ right) <= safe * tan_right)
                & (np.abs(from_eye @ up) <= safe * tan_up))

    def _locator_anchor(self, camera) -> Optional[np.ndarray]:
        """
        A point of the surface to sweep the outline out from: where the middle
        of the picture meets the globe.

        Aimed off the Moon there is no such point, and the nearest place on the
        surface to the sight line is taken instead - which is in the picture
        whenever any of the globe is, unless the globe has left the frame
        altogether, and that is answered with None.
        """
        eye, _right, _up, forward, _tan_right, _tan_up = camera

        towards = float(forward @ eye)
        gap = towards * towards - float(eye @ eye) + self.MOON_RADIUS ** 2
        if gap >= 0.0 and -towards - math.sqrt(gap) > 0.0:
            point = eye + (-towards - math.sqrt(gap)) * forward
        else:
            # The foot of the perpendicular from the middle of the Moon onto the
            # sight line, brought in to the surface
            point = eye - towards * forward
            length = float(np.linalg.norm(point))
            if length == 0.0:
                return None
            point = point / length * self.MOON_RADIUS

        return point if self._locator_on_screen(point[None, :], camera)[0] else None

    def _locator_field(self) -> Optional[np.ndarray]:
        """
        The outline of what is on screen, as unit vectors of the Moon's frame.

        Swept out from the middle of the view: in each direction around it, the
        surface is followed away until it leaves the picture or turns out of
        sight round the limb, and where it stops is a point of the outline.

        Bounded by the edge of the picture where the picture is the narrower and
        by the Moon's own edge where the Moon is, which is what makes it behave.
        Tracing the frame alone only works while the whole frame is on the globe
        - below about two and a half times it is not - and the honest answer
        there is not a mark in the middle but an outline larger than the disk
        can hold: at the default view everything is on screen, and the outline
        is the limb itself. Swept this way it shrinks from the whole disk down
        to a patch as the view is magnified, without a step anywhere.

        How far to follow is found by halving the interval. The region is
        bounded and holds the middle of the view, so each direction leaves it
        exactly once, and twenty halvings put the crossing to well inside a
        millionth of a turn.
        """
        camera = self._locator_camera()
        if camera is None:
            return None
        anchor = self._locator_anchor(camera)
        if anchor is None:
            return None

        # Two directions across the surface at the anchor, to sweep between
        along = np.cross(anchor, camera[1])         # the camera's right
        if np.linalg.norm(along) < self.LOCATOR_TOO_SHORT:
            along = np.cross(anchor, camera[2])     # its up, the other being useless
        along = along / np.linalg.norm(along)
        across = np.cross(anchor, along)
        across = across / np.linalg.norm(across)

        radial = anchor / self.MOON_RADIUS
        turn = np.linspace(0.0, 2 * math.pi, self.LOCATOR_SWEEP_POINTS, endpoint=False)
        heading = (np.outer(np.cos(turn), along) + np.outer(np.sin(turn), across))

        near = np.zeros(len(turn))                  # inside, to begin with
        far = np.full(len(turn), math.pi)           # and outside at the far side
        for _ in range(self.LOCATOR_SWEEP_STEPS):
            middle = (near + far) / 2
            walked = (np.cos(middle)[:, None] * radial
                      + np.sin(middle)[:, None] * heading) * self.MOON_RADIUS
            inside = self._locator_on_screen(walked, camera)
            near = np.where(inside, middle, near)
            far = np.where(inside, far, middle)

        edge = (np.cos(near)[:, None] * radial + np.sin(near)[:, None] * heading)
        return edge @ self.moon_rotation

    def _locator_centre(self) -> Optional[np.ndarray]:
        """
        The middle of the picture on the globe, as a unit vector of its frame,
        or None when the view is aimed past the limb.
        """
        camera = self._locator_camera()
        if camera is None:
            return None
        eye, _right, _up, forward, _tan_right, _tan_up = camera

        towards = float(forward @ eye)
        gap = towards * towards - float(eye @ eye) + self.MOON_RADIUS ** 2
        if gap < 0.0:
            return None
        step = -towards - math.sqrt(gap)
        if step <= 0.0:
            return None
        point = eye + step * forward
        return point / np.linalg.norm(point) @ self.moon_rotation

    # ---- drawing ----

    def _locator_placement(self) -> Optional[tuple]:
        """
        Centre and radius of the disk, in canvas pixels.

        The upper left, the other three corners being taken: the compass has the
        upper right, the ephemeris panel the lower left, and the field-of-view
        frame writes its summary across the top middle.
        """
        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is None:
            return None

        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:                       # not laid out yet
            width, height = self.rt._width, self.rt._height

        radius = min(width, height) * self.LOCATOR_SIZE_FRACTION / 2
        if radius <= 0:
            return None
        # Room for the N outside the rim, and on every side of it: in the
        # ordinary mount the disk turns through the night with the parallactic
        # angle, so the pole comes round to point anywhere at all
        margin = self.LOCATOR_MARGIN_PX + self._locator_label_room()
        return margin + radius, margin + radius, radius

    def _locator_label_room(self) -> float:
        """How far outside the rim the N reaches, letter and gap together."""
        return self.LOCATOR_LABEL_GAP_PX + 2 * self.LOCATOR_LABEL_FONT[1]

    def _draw_locator_letter(self, canvas, basis, point, text, colour,
                             centre_x, centre_y, radius, push_out=False):
        """
        A letter written against a place on the globe, or nothing when that
        place is round the back.

        A pole is usually within a few degrees of the limb - the eye stands over
        the equator or near it - so there is no room to write it on the face,
        and push_out sends it out past the rim in the direction the pole lies,
        which is where a map would put it anyway. Carried far enough north or
        south the pole comes inboard, and then it is written where it is.

        Between the two the letter slides rather than steps. Sending it to a
        fixed place outside the rim for every pole beyond a certain reach left
        it standing still through some twenty degrees of tilt - the pole moving
        under it all the while - and then dropping back onto the pole in one
        jump of twenty pixels. So the push is faded in over that last stretch
        instead: nought where the pole is still inboard, and full only when the
        pole has reached the limb.

        The graticule origin is always on the face and is never pushed out.
        """
        place = np.array([point], dtype=float)
        screen = self._compass_screen_points(place, basis)
        forward = self._locator_sightline(basis)
        # How far round the back the place is: nought at the limb, and rising to
        # one at the point dead opposite the eye. A pole exactly on the limb is
        # seen edge-on and belongs on the map - with the eye on the equator both
        # poles are that case, so a bare test on the sign would keep one and drop
        # the other on nothing but rounding - and a pole a little further over
        # still earns its letter, which is what push_out allows for.
        if screen is None or forward is None:
            return
        behind = float(place[0] @ forward)
        allowed = (math.sin(math.radians(self.LOCATOR_POLE_BEHIND)) if push_out
                   else self.LOCATOR_TOO_SHORT)
        if behind > allowed:
            return
        x, y = float(screen[0][0]), float(screen[0][1])
        reach = math.hypot(x, y)

        out = reach * radius                    # on the place itself
        if push_out:
            # How far the place has gone towards the limb, and past it: nought
            # where the push begins, one at the limb, and held at one beyond it.
            # Measured on how far round the back the place is rather than on how
            # far out it projects, because the projection turns back on itself at
            # the limb - a pole twelve degrees behind projects no further out
            # than one twelve degrees in front - and a letter steered by that
            # would come creeping back inboard as the pole slipped away.
            edge = math.sqrt(max(0.0, 1.0 - self.LOCATOR_POLE_AT_RIM ** 2))
            part = min(1.0, max(0.0, (behind + edge) / edge))
            # Squared, so the push comes in from nothing: a straight ramp would
            # begin it at full speed and the letter would change pace where the
            # push starts, which is the jump again in a smaller way. Eased at
            # the far end too it would instead dawdle as the pole reached the
            # limb, so only the near end is eased and the push goes on growing
            # to the last.
            beyond = radius + self.LOCATOR_LABEL_GAP_PX + self.LOCATOR_LABEL_FONT[1]
            out += part * part * (beyond - out)
        if reach < self.LOCATOR_TOO_SHORT:      # the eye straight over the pole
            at = (centre_x, centre_y)
        else:
            at = (centre_x + x / reach * out, centre_y - y / reach * out)
        # Rimmed, like the compass readings: N, S and the 0 fall wherever the
        # globe puts them, which may be black sky or lit highland
        self._locator_items.extend(self._rimmed_text(
            canvas, at[0], at[1], text, colour, self.LOCATOR_LABEL_FONT,
            anchor="center"))

    def _clear_locator_items(self):
        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is not None:
            for item in self._locator_items:
                canvas.delete(item)
        self._locator_items = []

    def _locator_screen(self, body_points: np.ndarray, basis,
                        centre_x: float, centre_y: float,
                        radius: float) -> Optional[tuple]:
        """
        Unit vectors of the Moon's frame as canvas coordinates on the disk.

        The places come from the compass's projection, which applies the
        mirrored view orientations while it is about it, so the disk turns with
        what is actually on screen.
        """
        screen = self._compass_screen_points(np.asarray(body_points, dtype=float),
                                             basis)
        if screen is None:
            return None
        coords = []
        for x, y, _near in screen:
            coords += [centre_x + x * radius, centre_y - y * radius]
        return coords

    def _draw_locator_field(self, canvas, coords):
        """
        The picture's own edge on the disk, or a ring where it is too small to
        see.

        The disk being drawn for the half the eye has in front of it, everything
        in the picture is on the near side of it, and the outline is one closed
        line in one style.
        """
        xs, ys = coords[0::2], coords[1::2]
        style = {"outline": self.LOCATOR_FIELD_COLOR, "fill": "",
                 "width": self.LOCATOR_FIELD_WIDTH}
        if max(max(xs) - min(xs), max(ys) - min(ys)) < self.LOCATOR_MIN_FIELD_PX:
            x, y = sum(xs) / len(xs), sum(ys) / len(ys)
            mark = self.LOCATOR_MIN_FIELD_PX / 2
            self._locator_items.append(canvas.create_oval(
                x - mark, y - mark, x + mark, y + mark, **style))
        else:
            self._locator_items.append(canvas.create_polygon(*coords, **style))

    def _locator_reading(self, centre: Optional[np.ndarray]) -> str:
        """
        Where the middle of the view stands, in selenographic coordinates.

        The status bar answers for the point under the cursor, which is a
        different question and needs a hand on the mouse to ask; this is where
        the picture is aimed, and it is the number to write down.
        """
        if centre is None:
            return "off the Moon"
        lat = math.degrees(math.asin(max(-1.0, min(1.0, float(centre[2])))))
        lon = math.degrees(math.atan2(float(centre[0]), float(-centre[1])))
        # A hair below zero reads as "-0.0", which looks like a fault rather
        # than like the middle of the disk
        lat, lon = round(lat, 1) or 0.0, round(lon, 1) or 0.0
        # Named, as the compass names its own three, and each number given the
        # width of the widest it can be - ninety of latitude, a hundred and
        # eighty of longitude - so the pair holds still instead of shuffling
        # sideways every time a digit comes or goes
        return f"lat {lat:+5.1f}°  lon {lon:+6.1f}°"

    def _draw_locator(self):
        """Redraw the disk and the field from the current camera."""
        self._clear_locator_items()

        canvas = getattr(self.rt, "_canvas", None) if self.rt is not None else None
        if canvas is None or not self.locator_visible:
            return

        placement = self._locator_placement()
        basis = self._locator_disk_basis()
        if placement is None or basis is None:
            return
        centre_x, centre_y, radius = placement

        # Day and night first, so everything else is drawn over them. Both go on
        # in the same stipple: the one is pale and the other black, and the
        # picture behind shows through each of them equally.
        for lit, colour in ((True, self.LOCATOR_LIT_COLOR),
                            (False, self.LOCATOR_DARK_COLOR)):
            face = self._locator_face(basis, lit)
            if face is None:
                continue
            coords = self._locator_screen(face, basis, centre_x, centre_y, radius)
            if coords is not None:
                self._locator_items.append(canvas.create_polygon(
                    *coords, fill=colour, stipple=self.LOCATOR_FACE_STIPPLE,
                    outline=self.LOCATOR_LIMB_COLOR, width=self.LOCATOR_LINE_WIDTH))

        # The rim. Every point of the limb is at right angles to the sight line,
        # so it projects to the circle the disk is drawn in and needs no working
        # out.
        self._locator_items.append(canvas.create_oval(
            centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius,
            outline=self.LOCATOR_LIMB_COLOR, width=self.LOCATOR_LINE_WIDTH))

        # The equator and the prime meridian, which say at a glance how the
        # globe is turned and how far the view is from the middle of the face
        forward = self._locator_sightline(basis)
        for pole in (self.LOCATOR_AXIS, self.LOCATOR_EAST):
            circle = self._locator_great_circle(pole, forward)
            if circle is None:
                continue
            coords = self._locator_screen(circle, basis, centre_x, centre_y, radius)
            if coords is not None:
                self._locator_items.append(canvas.create_line(
                    *coords, fill=self.LOCATOR_LIMB_COLOR,
                    width=self.LOCATOR_LINE_WIDTH))

        self._draw_locator_letter(canvas, basis, self._locator_origin_label(), "0",
                                  self.LOCATOR_LIMB_COLOR,
                                  centre_x, centre_y, radius)
        for point, letter in ((self.LOCATOR_AXIS, "N"),
                              (tuple(-v for v in self.LOCATOR_AXIS), "S")):
            self._draw_locator_letter(canvas, basis, point, letter,
                                      self.LOCATOR_LIT_COLOR,
                                      centre_x, centre_y, radius, push_out=True)

        field = self._locator_field()
        if field is not None:
            coords = self._locator_screen(field, basis, centre_x, centre_y, radius)
            if coords is not None:
                self._draw_locator_field(canvas, coords)

        centre = self._locator_centre()

        self._locator_items.extend(self._rimmed_text(
            canvas, centre_x, centre_y + radius + self.LOCATOR_VALUE_GAP_PX,
            self._locator_reading(centre), self.LOCATOR_FIELD_COLOR,
            self.LOCATOR_FONT))

    # ---- keeping up with the view ----

    def _locator_view_state(self):
        """
        A reading that changes whenever the locator would look different: the
        camera, the Moon's own orientation and lighting, the mirroring and the
        window size. Compared between ticks so a still view is not redrawn five
        times a second, every redraw being a dozen canvas items thrown away.
        """
        if self.rt is None:
            return None
        cam = self.rt.get_camera(self.CAMERA_NAME)
        canvas = getattr(self.rt, "_canvas", None)
        ephem = self.moon_ephem
        return (tuple(cam["Eye"]), tuple(cam["Target"]), tuple(cam["Up"]),
                None if self.moon_rotation is None else self.moon_rotation.tobytes(),
                None if ephem is None else (ephem.subsolar_lat, ephem.subsolar_lon),
                self.view_orientation,
                self.rt._optix.get_camera_fov(0),
                (canvas.winfo_width(), canvas.winfo_height()) if canvas is not None else None)

    def _locator_refresh_tick(self):
        self._locator_refresh_id = None
        if not self.locator_visible:
            return
        state = self._locator_view_state()
        if state != self._locator_last_view:
            self._locator_last_view = state
            self._draw_locator()
        self._schedule_locator_refresh()

    def _schedule_locator_refresh(self):
        if self.rt is None or self.rt._root is None:
            return
        self._locator_refresh_id = self.rt._root.after(self.LOCATOR_REFRESH_MS,
                                                       self._locator_refresh_tick)

    def show_locator(self, visible: bool = True):
        """Show or hide the where-on-the-Moon inset."""
        if self.rt is None:
            return

        self.locator_visible = visible

        if self._locator_refresh_id is not None and self.rt._root is not None:
            self.rt._root.after_cancel(self._locator_refresh_id)
            self._locator_refresh_id = None

        if visible:
            self._locator_last_view = self._locator_view_state()
            self._draw_locator()
            self._schedule_locator_refresh()
        else:
            self._clear_locator_items()

    def toggle_locator(self):
        """Toggle the where-on-the-Moon inset."""
        self.show_locator(not self.locator_visible)
