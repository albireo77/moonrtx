"""
ProfileMixin: the height of the ground along a measurement, drawn as a profile.

The Ctrl-drag measurement gives two numbers - how far, and how much higher the
end stands than the start - which says nothing of what lies between: whether a
scarp rises in one step or a slope climbs the whole way, whether the line
crosses a rim and drops into a crater on the way. The profile draws the ground
along the whole line, from the same elevation map the renderer displaces the
surface with, so it shows the terrain the picture is made of.

"""

import math
import tkinter as tk
from datetime import datetime, timedelta
import tkinter.font as tkfont
from tkinter import ttk
from typing import Optional

import numpy as np

from moonrtx import astro
from moonrtx.display import ToolTip
from moonrtx.view_orientation import (FLIP_HORIZONTAL_VIEW_ORIENTATIONS,
                                      FLIP_VERTICAL_VIEW_ORIENTATIONS)


class ProfileMixin:
    """Mixin drawing the elevation profile along a measurement."""

    PROFILE_COLOURS = {
        "ground": "#7a4e22",      # the profile line
        "fill": "#eadfcc",        # the ground under it
        "grid": "#d0d0d0",
        "readout": "#a06010",     # the place under the pointer, as in the graph
        "sunlit": "#fff2b0",      # the space above the ground in sunlight
        "shadow": "#d8d8d8",      # and in shadow: a shade under the window, over the grid
    }
    # How a sample's view towards the Sun is searched for ground standing in
    # the way: at this many places, closer together near the sample, where
    # the ground matters most, out to where the ray has climbed above the
    # highest ground on the Moon - though never further than this
    PROFILE_SUNLIT_STEPS = 400
    PROFILE_SUNLIT_REACH_KM = 500.0
    # While an end of the line is dragged, quicker: worked out at no more than
    # this many places along the line, and filled in between (see
    # _shadow_ceilings). Measured across Copernicus on the downscale-2 map,
    # 18 ms for 120 in full, and the top of the shadow within 2 px of it on
    # a plot 200 px tall; a shorter search from each place cost more in
    # accuracy than it saved in time
    PROFILE_SUNLIT_QUICK_SAMPLES = 300
    # The wheel zooms the profile in or out this much a notch, around the
    # pointer, no further in than this many samples across the view (see zoom)
    PROFILE_ZOOM_STEP = 1.05
    PROFILE_ZOOM_MIN_SAMPLES = 8
    # How long the wheel or a drag along the line has to pause before the
    # stretch in view is sampled afresh at the map's own detail (see refine)
    PROFILE_REFINE_MS = 150
    # Fewer still while an end of the line is dragged, every place moving with
    # it so none of the ground towards the Sun can be used again: 9.5 ms
    # rather than 19.5 for the same line, the top of the shadow within 2.7 px
    # of the full one for 99% of it, until the drag is let go
    PROFILE_SUNLIT_DRAG_SAMPLES = 150
    # The ground looked up towards the Sun is kept and used again while the
    # Sun's bearing from every place has turned by less than this - at 10 km
    # out, 44 m to the side, under a texel of the map - and the search still
    # reaches far enough, made this much longer than needed each time it is
    # looked up afresh (see _shadow_ceilings)
    PROFILE_LIGHT_REUSE_DEG = 0.25
    # When the ground under the pointer next goes into shadow or sunlight (see
    # light_change): looked for this far ahead - past the longest night, half
    # a lunation - with the Sun's place taken every so many minutes, a day of
    # them looked at together, and the moment pinned down between two of them
    # to well under a second. The Sun's course is worked out once and used
    # for as long as the clock stays within this many hours after its start
    PROFILE_LIGHT_DAYS = 16
    PROFILE_LIGHT_STEP_MINUTES = 30
    PROFILE_LIGHT_BATCH = 48
    PROFILE_LIGHT_HALVINGS = 14
    PROFILE_LIGHT_TRACK_REUSE_HOURS = 24
    # The ground towards the Sun searched at this many evenly spaced places
    # for it: some 11 m apart for a ridge 9 km off as the Sun clears it, where
    # the shadows' own spacing, close in and wide out, put its top 40 m out and
    # the moment 4 minutes early
    PROFILE_LIGHT_STEPS = 800
    PROFILE_LIGHT_REACH_SPARE = 1.25
    # The place under the profile's pointer, marked on the Moon in the main
    # window: the readout's colour made bright enough for the dark side of the
    # terminator, and rimmed in black for the bright highland, a size under the
    # measurement line's rings so it is not taken for an end
    PROFILE_MARK_COLOUR = "#ff9a1f"
    PROFILE_MARK_RADIUS_PX = 4
    PROFILE_MARK_TAG = "profile_mark"
    # Wide enough to follow a long line's detail, and as a share of the screen so
    # a 4K display gets the room it has rather than a FullHD-sized window
    PROFILE_WIDTH_FRACTION = 0.5
    PROFILE_PLOT_LINES = 14                 # plot height, in lines of lettering
    PROFILE_WINDOW = "profile"              # its name in DialogsMixin._window_memory
    # The map is sampled at twice its own spacing: enough that the bilinear
    # lookup's straight runs between texels draw as the map has them, more
    # adding nothing the map does not hold
    PROFILE_SAMPLES_PER_TEXEL = 2
    PROFILE_MIN_SAMPLES = 64
    PROFILE_MAX_SAMPLES = 2000
    # Whether the pointer's readout tells, on a line of its own, when the light
    # changes under it (see light_change) - worked out only when it is asked
    # for, being the one part of the readout that takes time. Off to begin
    # with; changing it stores it on the instance, so it holds for the rest of
    # the session and into the next, as the planner's own choices do (see
    # renderer_settings)
    _profile_show_light_change = False

    def _init_profile(self):
        """Reset the profile window state; called from MoonRenderer.__init__."""
        self._profile = None                # the open window and its redraw, or None
        # The ground looked up towards the Sun, by how many places it was
        # looked up for (see _shadow_ceilings)
        self._light_rays = {}
        # The Sun's course from a moment on (see _sun_track)
        self._profile_sun_track = None

    # ---- the numbers ----

    @staticmethod
    def _unit(lat: float, lon: float) -> np.ndarray:
        """
        The body-frame unit vector of a selenographic position - the app's own
        frame (see center_on_lat_lon): +Z north, longitude 0 towards -Y, +X east.
        """
        la, lo = math.radians(lat), math.radians(lon)
        return np.array([math.cos(la) * math.sin(lo),
                         -math.cos(la) * math.cos(lo),
                         math.sin(la)])

    def _along(self, start: tuple, end: tuple, fractions) -> tuple:
        """
        Where points that far along the great circle from start to end lie, as
        (lat, lon) arrays in degrees: 0 is start and 1 is end.

        By spherical interpolation, which spaces them evenly over the ground as
        a straight chord would not. The two ends must not be the same place.
        """
        a, b = self._unit(*start), self._unit(*end)
        angle = math.acos(max(-1.0, min(1.0, float(a @ b))))
        t = np.atleast_1d(np.asarray(fractions, dtype=float))
        points = (np.sin((1 - t) * angle)[:, None] * a
                  + np.sin(t * angle)[:, None] * b) / math.sin(angle)
        return (np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0))),
                np.degrees(np.arctan2(points[:, 0], -points[:, 1])))

    def elevation_profile(self, start: tuple, end: tuple, quick: int = 0,
                          part: tuple = (0.0, 1.0)) -> Optional[tuple]:
        """
        Heights along the great circle from start to end.

        Parameters
        ----------
        start, end : tuple
            (lat, lon) in selenographic degrees
        quick : int
            Work the light out the quicker way, at no more than this many
            places (see _shadow_ceilings); 0 for every place
        part : tuple
            The stretch of the line wanted, as shares of its length from the
            start: (0, 1) for the whole of it. Sampled as finely as a line of
            that stretch's length would be, so a stretch of a long line zoomed
            in on is given the map's own detail, which the whole line, held to
            PROFILE_MAX_SAMPLES, has to go without

        Returns
        -------
        tuple or None
            (distance_km, height_m, light): the distances measured from start
            along the surface, the heights, and how high the shadow reaches
            above each place (see _shadow_ceilings) - None before there is an
            ephemeris; None when the two points are the same place
        """
        a, b = self._unit(*start), self._unit(*end)
        if math.acos(max(-1.0, min(1.0, float(a @ b)))) < 1e-9:
            return None

        length_km = self.calculate_great_circle_distance(*start, *end)
        texel_km = 2 * math.pi * self.MOON_RADIUS_KM / self.elevation.shape[1]
        stretch_km = length_km * (part[1] - part[0])
        count = int(np.clip(math.ceil(stretch_km / texel_km * self.PROFILE_SAMPLES_PER_TEXEL),
                            self.PROFILE_MIN_SAMPLES, self.PROFILE_MAX_SAMPLES))

        t = np.linspace(part[0], part[1], count)
        lats, lons = self._along(start, end, t)
        # All at once rather than through get_elevation_m one by one: the same
        # float64 lookup to the last bit, but the profile wants up to a couple
        # of thousand heights every time it is drawn, and while an end of its
        # line is dragged it is drawn every MEASURE_PROFILE_LIVE_MS
        factor = self._ground_factor(lats, lons)
        heights = (factor * self.elevation_radius_scale - 1.0) * self.MOON_RADIUS_KM * 1000.0
        return t * length_km, heights, self._shadow_ceilings(lats, lons, heights, quick)

    def _shadow_ceilings(self, lats: np.ndarray, lons: np.ndarray,
                         heights_m: np.ndarray, quick: int = 0) -> Optional[tuple]:
        """
        How high above each of these places on the ground the shadow reaches:
        how far straight up from it one would have to go to see the Sun.

        Two heights each, in metres: above `full` all of the Sun's disk is in
        sight, below `some` none of it, and between the two lies the penumbra.
        Both are 0 where the ground itself is in full sunlight. Drawn above the
        profile they are the shadow's cross-section along the line, its upper
        edge the light ray grazing whatever casts it - at the Sun's altitude
        where the Sun shines along the line, gentler where it shines across.

        Worked out from the elevation map, not read off the rendered picture:
        from each place the ground is searched along the Sun's true direction -
        off the line as much as on it, so a shadow thrown across the line by a
        peak beside it is found as well - for how far each place ahead stands
        above this one's horizontal plane, the Moon curving away beneath it,
        and how far out along it. A ray at altitude a clears one standing r up
        and d out from a height of r - d tan a; the highest any of them asks
        for is the ceiling. For the Sun (the selenocentric subsolar point, the
        rays parallel) a is its altitude there, less its radius for the whole
        disk to be in sight and plus it for any of it.

        quick, a number of places, for the profile drawn again and again while
        an end of its line is dragged or the clock moves, which the full search,
        at about a tenth of a second for the longest lines, would hold back: it
        is made at no more than that many places, and filled in between.
        What is filled in is the top of the shadow, ground and ceiling
        together - made of straight rays, it runs straight between the places
        where it was worked out, where the ceiling alone follows every rise
        and dip of the ground under it. Between two places in sunlight it
        stays sunlit, so a dip between them is not taken for a shadow. Letting
        go of the drag draws it once more the full way.

        None before there is an ephemeris to say where the Sun is.
        """
        e = getattr(self, "moon_ephem", None)
        if e is None:
            return None
        if quick and len(lats) > quick:
            pick = np.unique(np.rint(np.linspace(0, len(lats) - 1, quick)).astype(int))
            light = self._shadow_ceilings(lats[pick], lons[pick], heights_m[pick])
            everywhere = np.arange(len(lats))
            filled = []
            for ceiling in light:
                top = np.interp(everywhere, pick, heights_m[pick] + ceiling)
                shaded = np.interp(everywhere, pick, (ceiling > 0).astype(float)) > 0
                filled.append(np.where(shaded, np.maximum(top - heights_m, 0.0), 0.0))
            return tuple(filled)
        radius_m = self.MOON_RADIUS_KM * 1000.0
        sun = self._unit(e.subsolar_lat, e.subsolar_lon)
        sun_radius = math.asin(self.SUN_RADIUS_KM / e.sun_distance)

        la, lo = np.radians(lats), np.radians(lons)
        up = np.column_stack((np.cos(la) * np.sin(lo), -np.cos(la) * np.cos(lo), np.sin(la)))
        sin_altitude = np.clip(up @ sun, -1.0, 1.0)
        altitude = np.arcsin(sin_altitude)
        # The way to the Sun along the ground; with the Sun straight overhead
        # there is none, and nothing can stand in its way either
        towards = sun - sin_altitude[:, None] * up
        length = np.linalg.norm(towards, axis=1)
        towards /= np.where(length > 1e-12, length, 1.0)[:, None]

        # The rays from the Sun's lower and upper edge, short of straight up
        # and down, where the tangent would turn over
        limit = math.pi / 2 - 1e-6
        tan_low = np.tan(np.clip(altitude - sun_radius, -limit, limit))
        tan_high = np.tan(np.clip(altitude + sun_radius, -limit, limit))

        # Past where the lower edge's ray has climbed above the highest ground
        # the map holds, nothing further on can be in the way of any of them
        peak_m = (self.elevation_radius_scale - 1.0) * radius_m
        rise = np.maximum(peak_m - heights_m, 0.0)
        reach = radius_m * (np.sqrt(tan_low ** 2 + 2.0 * rise / radius_m) - tan_low)
        reach = np.minimum(reach, self.PROFILE_SUNLIT_REACH_KM * 1000.0)

        # Each place ahead as seen from this one: how far above its horizontal
        # plane, and how far out along it. Looked up again only when it has to
        # be (see PROFILE_LIGHT_REUSE_DEG): as the clock moves, the Sun climbs
        # or sinks far faster than its bearing turns, and the ground towards it
        # is the same ground, seen at a new angle
        rays = self._light_rays.get(len(up))
        if (rays is None or not np.array_equal(rays["up"], up)
                or not np.array_equal(rays["heights"], heights_m)
                or np.any(reach > rays["reach"])
                or np.min(np.einsum("ij,ij->i", towards, rays["towards"]))
                < math.cos(math.radians(self.PROFILE_LIGHT_REUSE_DEG))):
            # With room to spare, so a Sun sinking a little lower still finds
            # the search long enough
            reach = np.minimum(reach * self.PROFILE_LIGHT_REACH_SPARE,
                               self.PROFILE_SUNLIT_REACH_KM * 1000.0)
            steps = (np.arange(1, self.PROFILE_SUNLIT_STEPS + 1) / self.PROFILE_SUNLIT_STEPS) ** 2
            angle = reach[:, None] * steps[None, :] / radius_m          # round the Moon's centre
            cos_a, sin_a = np.cos(angle)[..., None], np.sin(angle)[..., None]
            ahead = cos_a * up[:, None, :] + sin_a * towards[:, None, :]
            ahead_lat = np.degrees(np.arcsin(np.clip(ahead[..., 2], -1.0, 1.0)))
            ahead_lon = np.degrees(np.arctan2(ahead[..., 0], -ahead[..., 1]))
            ahead_r = (self._ground_factor(ahead_lat, ahead_lon)
                       * self.elevation_radius_scale * radius_m)
            here_r = radius_m + heights_m
            rays = {"up": up, "heights": np.array(heights_m), "towards": towards, "reach": reach,
                    "rising": ahead_r * np.cos(angle) - here_r[:, None],
                    "out": ahead_r * np.sin(angle)}
            self._light_rays[len(up)] = rays
        rising, out = rays["rising"], rays["out"]
        full = np.maximum((rising - out * tan_low[:, None]).max(axis=1), 0.0)
        some = np.maximum((rising - out * tan_high[:, None]).max(axis=1), 0.0)
        return full, some

    def light_change(self, lat: float, lon: float, height_m: float) -> Optional[tuple]:
        """
        When the ground at a place next goes into shadow, if it is in sunlight
        now, or into sunlight, if it is in shadow.

        Sunlight here is the Sun's centre standing above the ground's own
        horizon towards it - the middle of the penumbra, which starts a little
        before and ends a little after. The horizon is searched afresh for
        every place of the Sun, its bearing drifting as it climbs or sinks: a
        ridge beside the line can throw its shadow across the place hours after
        the Sun has risen over the ground straight ahead.

        The Sun's course is taken from the ephemeris every
        PROFILE_LIGHT_STEP_MINUTES (see _sun_track), a day of it looked at at
        once until the light changes, and the moment found by halving the step
        it falls in, the Sun's place in between drawn straight between its two
        ends - which over half an hour it is, to far better than the map can
        tell.

        Returns
        -------
        tuple or None
            (lit_now, hours): whether the place is in sunlight now, and how
            many hours until that changes - None for hours when it does not
            within PROFILE_LIGHT_DAYS. None altogether before there is an
            ephemeris.
        """
        track = self._sun_track()
        if track is None:
            return None
        hours, suns = track
        up = self._unit(lat, lon)

        def lit(sun_rows) -> np.ndarray:
            return self._sun_clearance(up, height_m, sun_rows) > 0.0

        lit_now = bool(lit(suns[:1])[0])
        first = 0
        while first < len(hours) - 1:
            # Each batch starts where the last one ended, so a change between
            # them is caught too
            last = min(first + self.PROFILE_LIGHT_BATCH, len(hours) - 1)
            changed = np.flatnonzero(lit(suns[first:last + 1]) != lit_now)
            if changed.size:
                after = first + int(changed[0])
                low, high = hours[after - 1], hours[after]
                for _ in range(self.PROFILE_LIGHT_HALVINGS):
                    middle = (low + high) / 2
                    share = (middle - hours[after - 1]) / (hours[after] - hours[after - 1])
                    sun = suns[after - 1] + (suns[after] - suns[after - 1]) * share
                    sun /= np.linalg.norm(sun)
                    if bool(lit(sun[None, :])[0]) == lit_now:
                        low = middle
                    else:
                        high = middle
                return lit_now, high
            first = last
        return lit_now, None

    def _sun_track(self) -> Optional[tuple]:
        """
        The Sun's course over the Moon from the moment shown on: (hours, suns),
        the hours since that moment and the body-frame unit vector to the Sun
        at each (see astro.subsolar_track).

        Worked out from a moment for PROFILE_LIGHT_DAYS and the next day, and
        used again while the clock stays within PROFILE_LIGHT_TRACK_REUSE_HOURS
        after that moment, the moment shown standing in for its first place -
        so a clock stepped while the pointer rests on the profile does not ask
        the ephemeris for the whole of it on every step.
        """
        when = getattr(self, "dt_local", None)
        if when is None or getattr(self, "moon_ephem", None) is None:
            return None
        track = self._profile_sun_track
        offset = None if track is None else (when - track[0]).total_seconds() / 3600.0
        if offset is None or not 0.0 <= offset <= self.PROFILE_LIGHT_TRACK_REUSE_HOURS:
            hours, lat, lon = astro.subsolar_track(
                when, self.PROFILE_LIGHT_DAYS + self.PROFILE_LIGHT_TRACK_REUSE_HOURS / 24.0,
                self.PROFILE_LIGHT_STEP_MINUTES)
            la, lo = np.radians(lat), np.radians(lon)
            suns = np.column_stack((np.cos(la) * np.sin(lo), -np.cos(la) * np.cos(lo), np.sin(la)))
            self._profile_sun_track = track = (when, hours, suns)
            offset = 0.0
        _start, hours, suns = track
        # From the moment shown: the place of the Sun then drawn straight
        # between the two places of the course either side of it
        after = int(np.searchsorted(hours, offset, side="right"))
        if after >= len(hours):
            return None
        share = (offset - hours[after - 1]) / (hours[after] - hours[after - 1])
        now = suns[after - 1] + (suns[after] - suns[after - 1]) * share
        now /= np.linalg.norm(now)
        end = min(len(hours), after + int(self.PROFILE_LIGHT_DAYS * 24 * 60
                                          / self.PROFILE_LIGHT_STEP_MINUTES))
        return (np.concatenate(([0.0], hours[after:end] - offset)),
                np.vstack((now[None, :], suns[after:end])))

    def _sun_clearance(self, up: np.ndarray, height_m: float, suns: np.ndarray) -> np.ndarray:
        """
        How far the Sun's centre stands above the ground's own horizon towards
        it, in radians, from one place on the ground, for each of these
        body-frame directions of the Sun: below 0, the place is in shadow.

        The ground towards the Sun is searched at PROFILE_LIGHT_STEPS evenly
        spaced places, out to where a ray at the Sun's altitude has climbed
        above the highest ground the map holds (as _shadow_ceilings searches,
        but evenly: the timing turns on the very top of whatever stands
        furthest out in the way).
        """
        radius_m = self.MOON_RADIUS_KM * 1000.0
        sin_altitude = np.clip(suns @ up, -1.0, 1.0)
        altitude = np.arcsin(sin_altitude)
        towards = suns - sin_altitude[:, None] * up[None, :]
        length = np.linalg.norm(towards, axis=1)
        towards /= np.where(length > 1e-12, length, 1.0)[:, None]
        limit = math.pi / 2 - 1e-6
        tan_altitude = np.tan(np.clip(altitude, -limit, limit))

        peak_m = (self.elevation_radius_scale - 1.0) * radius_m
        rise = max(peak_m - float(height_m), 0.0)
        reach = radius_m * (np.sqrt(tan_altitude ** 2 + 2.0 * rise / radius_m) - tan_altitude)
        reach = np.minimum(reach, self.PROFILE_SUNLIT_REACH_KM * 1000.0)

        steps = np.arange(1, self.PROFILE_LIGHT_STEPS + 1) / self.PROFILE_LIGHT_STEPS
        angle = reach[:, None] * steps[None, :] / radius_m          # round the Moon's centre
        ahead = (np.cos(angle)[..., None] * up[None, None, :]
                 + np.sin(angle)[..., None] * towards[:, None, :])
        ahead_lat = np.degrees(np.arcsin(np.clip(ahead[..., 2], -1.0, 1.0)))
        ahead_lon = np.degrees(np.arctan2(ahead[..., 0], -ahead[..., 1]))
        ahead_r = self._ground_factor(ahead_lat, ahead_lon) * self.elevation_radius_scale * radius_m
        rising = ahead_r * np.cos(angle) - (radius_m + float(height_m))
        out = ahead_r * np.sin(angle)
        return altitude - np.arctan2(rising, out).max(axis=1)

    def _canvas_xy_of(self, lat: float, lon: float) -> Optional[tuple]:
        """
        Where a place on the ground is in the main window, in canvas pixels; or
        None when the eye is not looking towards it (see _canvas_points).
        """
        points = self._canvas_points(np.array([lat]), np.array([lon]))
        return None if points is None else (float(points[0, 0]), float(points[0, 1]))

    def _canvas_points(self, lats: np.ndarray, lons: np.ndarray) -> Optional[np.ndarray]:
        """
        Where places on the ground are in the main window, as an (n, 2) array
        of canvas pixels; or None when the eye is not looking towards all of
        them.

        Each is projected at its own height through the live camera, so it
        lands where the rendered surface shows it: seen at a slant, towards
        the limb, high ground stands out towards the limb and low ground falls
        back from it, the more so the further from the disk's centre. The
        measurement line is drawn along its ground through these (see
        NavigationMixin._place_measure_line), and the profile's pointer is
        marked by one, so the mark sits on the line.

        Projected as PlotOptiX renders (see LocatorMixin._locator_camera and
        CompassMixin._compass_screen_points), the image mirrored afterwards for
        the flipped view orientations, and scaled into the window as
        TkOptiX._get_image_xy undoes it.
        """
        camera = self._locator_camera()
        if camera is None:
            return None
        eye, right, up, forward, tan_right, tan_up = camera

        # On their own ground, as the surface is displaced (see _aimed_at_ground)
        la, lo = np.radians(lats), np.radians(lons)
        units = np.column_stack((np.cos(la) * np.sin(lo), -np.cos(la) * np.cos(lo), np.sin(la)))
        radius = self.MOON_RADIUS * np.asarray(self._ground_factor(lats, lons), dtype=float)
        from_eye = (units * radius[:, None]) @ self.moon_rotation.T - eye
        depth = from_eye @ forward
        if np.any(depth <= 0.0):
            return None

        rt = self.rt
        x = (1.0 + (from_eye @ right) / (depth * tan_right)) / 2.0 * rt._width
        y = (1.0 - (from_eye @ up) / (depth * tan_up)) / 2.0 * rt._height
        if self.view_orientation in FLIP_HORIZONTAL_VIEW_ORIENTATIONS:
            x = rt._width - x
        if self.view_orientation in FLIP_VERTICAL_VIEW_ORIENTATIONS:
            y = rt._height - y
        if rt._fixed_size is not None:
            x = x * rt._image_scale + rt._image_at[0]
            y = y * rt._image_scale + rt._image_at[1]
        return np.column_stack((x, y))

    def _mark_profile_place(self, place: Optional[tuple]):
        """
        Mark a place on the profile's line in the main window, or take the mark
        away (None): where the pointer in the profile is, so a rise or a dip in
        it can be found on the Moon by looking rather than by its coordinates.

        Only while the line the profile is of is on the picture: once it has
        gone (see _keep_measure_line), a mark left without it would point at
        ground nobody measured.
        """
        if self.rt is None:
            return
        canvas = self.rt._canvas
        canvas.delete(self.PROFILE_MARK_TAG)
        if place is None or self._measure_line is None or not self._measure_line_graphed:
            return
        at = self._canvas_xy_of(*place)
        if at is None:
            return
        x, y = at
        r = self._overlay_px(self.PROFILE_MARK_RADIUS_PX)
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=self.PROFILE_MARK_COLOUR,
                           outline="black", width=max(1, round(self._overlay_px(1))),
                           tags=self.PROFILE_MARK_TAG)

    @staticmethod
    def _nice_step(span: float, lines: int) -> float:
        """A round step - 1, 2 or 5 times a power of ten - giving about that many lines."""
        if span <= 0:
            return 1.0
        raw = span / max(1, lines)
        power = 10 ** math.floor(math.log10(raw))
        for multiple in (1, 2, 5, 10):
            if raw <= multiple * power:
                return multiple * power
        return 10 * power

    # ---- the window ----

    def show_profile(self, start: tuple, end: tuple, live: bool = False,
                     light_only: bool = False) -> bool:
        """
        Draw the profile of the line from start to end, in the profile window -
        opening it if it is not already open - and say whether it was drawn.

        live while an end of the line is being dragged, or the clock is moving,
        and the profile drawn again and again: the light is worked out the
        quick way then (see _shadow_ceilings), and the end of it draws it once
        more the full way.

        light_only when only the moment has changed, the line being the same:
        its ground, grid and lettering are left as drawn and the light alone
        reshaped, so the profile keeps up with a held Q or W.
        """
        if self.rt is None or getattr(self, "elevation", None) is None:
            return False
        # A drag finds no ground to use again, every place moving with it, so
        # it is given fewer places than the clock, which finds most of them
        # (see PROFILE_LIGHT_REUSE_DEG)
        quick = 0
        if live:
            quick = self.PROFILE_SUNLIT_QUICK_SAMPLES if light_only else self.PROFILE_SUNLIT_DRAG_SAMPLES
        profile = self.elevation_profile(start, end, quick=quick)
        if profile is None:
            return False
        if self._profile is None:
            self._open_profile_window()
        self._profile["light" if light_only else "draw"](*profile, start, end)
        return True

    def close_profile(self):
        """
        Close the profile window, if it is open: the line it is the profile of
        has been taken off the picture, and a profile of nothing on it would
        read as the profile of whatever is there now.
        """
        if getattr(self, "_profile", None) is not None:
            self._profile["close"]()

    def _open_profile_window(self):
        colours = self.PROFILE_COLOURS
        # Where the window was left is kept by _dialog_window; how large its
        # plot was dragged, here (see DialogsMixin._window_memory)
        memory = self._window_memory(self.PROFILE_WINDOW)

        def before_close():
            memory["canvas"] = (width, height)
            self._profile = None
            self._mark_profile_place(None)
            # A finer sampling still waiting has no window left to draw into
            if state["refine_due"] is not None:
                canvas.after_cancel(state["refine_due"])

        # Not holding the renderer's keys and not modal: measuring again, and
        # stepping the clock, go on in the main window while the profile is up
        win, frame, close = self._dialog_window("Elevation and Illumination profile", takes_keys=False,
                                                 before_close=before_close,
                                                 remember=self.PROFILE_WINDOW)

        font = ('Consolas', 8)
        metrics = tkfont.Font(font=font)
        cell_w = metrics.measure('0')
        line_h = metrics.metrics('linespace')
        pad = max(2, round(cell_w * 2 / 3))
        line_w = max(1, cell_w // 5)
        end_r = max(3, round(cell_w * 0.6))     # the circles marking the line's ends

        # The summary on the left of the top row, the readout's choice on its right
        top_row = tk.Frame(frame)
        top_row.pack(fill=tk.X)
        summary_var = tk.StringVar()
        tk.Label(top_row, textvariable=summary_var, font=font, anchor='w').pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        light_change_var = tk.BooleanVar(value=self._profile_show_light_change)

        def light_change_toggled():
            self._profile_show_light_change = light_change_var.get()
            refresh_readout()

        ttk.Checkbutton(top_row, text="Light change", variable=light_change_var,
                        command=light_change_toggled).pack(side=tk.RIGHT)

        # What the window does not say on its face, as the graph keeps its own
        # (see feature_graph_dialog) - among it how finely the map holds the
        # ground: along a short line most of the curve lies between the map's
        # own heights, and looks more detailed than the map can know
        spacing_m = 2 * math.pi * self.MOON_RADIUS_KM * 1000.0 / self.elevation.shape[1]
        # Why the map is that coarse: the elevation downscale averages the full
        # map in square blocks, and a lower one keeps more of the ground
        downscale = self.downscale
        if downscale > 1:
            averaged = (f" - Elevation downscale {downscale} averages the\n"
                        f"   {spacing_m / downscale:.0f} m map in blocks of {downscale} × {downscale}")
        else:
            averaged = " - the full map, not downscaled"
        # A bullet to each entry, a line carried over indented by the three
        # spaces nearest the bullet's width in the tooltip's proportional font,
        # as the graph's Help is
        help_hint = (
            "• The ground along a line measured with the right Ctrl, from the\n"
            "   elevation map the surface is drawn with.\n"
            "• Heights are counted from the start of the line, where the drag began.\n"
            "• The place under the pointer is marked on the Moon with an orange dot.\n"
            f"• The map holds one height every {spacing_m:.0f} m{averaged}.\n"
            "• Between those heights the curve is blended from the nearest of\n"
            "   them, so a line only a few times that long shows less of the\n"
            "   ground than its shape suggests"
            + ("; a lower downscale shows more of it.\n" if downscale > 1 else ".\n") +
            "• Relief is the height from the lowest point of the line to the\n"
            "   highest: across a crater, its depth below the rim.\n"
            "• Slopes are true angles, however the profile stretches the heights,\n"
            "   taken along the line. The one under the pointer is signed: + rising\n"
            "   towards the end of the line, - falling. Steepest is the largest\n"
            "   slope anywhere on the line, up or down. A line crossing a slope at an\n"
            "   angle reads it gentler than it is, and the map's spacing smooths\n"
            "   a short steep step into a longer, gentler one.\n"
            "• Pale yellow is the space above the ground the Sun reaches, paler\n"
            "   in the penumbra, where only part of its disk is in sight. The\n"
            "   shadow is the grey wedge behind whatever casts it, its top\n"
            "   the grazing ray: at the Sun's true angle, stretched as the heights\n"
            "   are, and gentler where the Sun shines across the line. It is worked\n"
            "   out from the map along the Sun's true direction, so a shadow cast\n"
            "   across the line from beside it counts too; at the tips of long\n"
            "   shadows it can differ a little from the picture.\n"
            "• The filled circle is the start of the line, as on the Moon, and the\n"
            "   ring its end. ⇄ draws the profile the other way round - for a\n"
            "   line running right to left on the Moon - every number unchanged.\n"
            "• With Light change ticked, a line above the readout tells when the\n"
            "   ground under the pointer next goes into shadow, or into sunlight,\n"
            "   on the session's clock - as the Sun's centre crosses\n"
            "   the horizon its surroundings make, from the Sun's real course; the\n"
            "   penumbra starts a little before and ends a little after. Click to\n"
            "   take the clock there. Not told while the clock is moving.\n"
            "• The mouse wheel zooms in on the profile around the pointer, and back\n"
            "   out. Zoomed in, drag with the left button to move left and right\n"
            "   along it. The heights' scale fits what is in view, the numbers\n"
            "   above stay those of the line.\n"
            "• Drag the window's edges to make the profile larger.")
        help_label = tk.Label(top_row, text="Help", font=font, fg='#606060')
        help_label.pack(side=tk.RIGHT, padx=pad)
        ToolTip(help_label, help_hint)

        # Mirrors the drawing alone: distances and heights are still counted
        # from the start of the line, which the filled circle marks wherever
        # it ends up. Kept while the window is open, for every line drawn in it
        def flip():
            state.update(flipped=not state["flipped"])
            redraw_profile()

        # The arrows alone, the tooltip saying what they do; a themed button
        # being as wide as a word unless told otherwise
        flip_button = ttk.Button(top_row, text="⇄", width=3, command=flip)
        # Clear of the summary, close to Help
        flip_button.pack(side=tk.RIGHT, padx=(2 * cell_w, 0))
        ToolTip(flip_button, "Draw the profile the other way round, left to right")

        # The heights' labels kept clear of the circle on the line's end there
        label_w = metrics.measure('-10000 m') + 2 * pad + end_r
        plot_x0 = label_w
        # Two lines of room for the readout: the place under the pointer, and
        # above it when the light there changes
        plot_y0 = pad + 2 * line_h
        # Under the plot, a line for the distances
        below = pad + line_h + pad

        # The canvas's size and the plot's right and bottom edges, which follow
        # it - set by lay_out, first for the size the last window was left at
        # and again whenever it is dragged larger or smaller (see fit_size).
        # The first size is also the smallest it can be dragged to
        width = height = plot_x1 = plot_y1 = 0
        first_width = round(win.winfo_screenwidth() * self.PROFILE_WIDTH_FRACTION)
        first_height = plot_y0 + self.PROFILE_PLOT_LINES * line_h + below

        def lay_out(new_width: int, new_height: int):
            nonlocal width, height, plot_x1, plot_y1
            width = max(first_width, int(new_width))
            height = max(first_height, int(new_height))
            plot_x1 = width - pad - cell_w
            plot_y1 = height - below

        lay_out(*memory.get("canvas", (first_width, first_height)))

        canvas = tk.Canvas(frame, width=width, height=height,
                           highlightthickness=0, bg=win.cget('bg'))
        canvas.pack()

        # changes: when the light changes, by sample, as change_at found it at
        # the moment changes_at
        state = {"x": None, "h": None, "light": None, "light_items": None,
                 "slopes": None, "x_of": None, "y_of": None,
                 "base": 0.0, "pointer": None, "flipped": False,
                 "changes": {}, "changes_at": None, "view": None, "view_km": None,
                 "source": None, "fine": None, "refine_due": None}

        def signed(metres: float) -> str:
            """A height counted from the start of the line: signed, and plain 0 at it."""
            value = round(metres)
            return f"{value:+d} m" if value else "0 m"

        def signed_slope(degrees: float) -> str:
            """A slope along the line, as signed as the heights, and plain 0 when level."""
            value = round(degrees, 1)
            return f"{value:+.1f}°" if value else "0.0°"

        # The penumbra halfway between shadow and sunlight
        shadow = np.array(canvas.winfo_rgb(colours["shadow"])) / 65535.0
        sunlit = np.array(canvas.winfo_rgb(colours["sunlit"])) / 65535.0
        penumbra = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in (shadow + sunlit) / 2)

        def place_light():
            """Shape the light and the penumbra to the light worked out last."""
            _shadow_item, sunlit_item, penumbra_item = state["light_items"]
            in_view, source = state["in_view"], state["source"]
            full, some = (in_view(ceiling) for ceiling in source["light"])
            x_of, y_of = state["x_of"], state["y_of"]
            xs = x_of(in_view(source["x"])).tolist()
            heights = in_view(source["h"])

            def edge(ceiling):
                ys = np.clip(y_of(heights + ceiling), plot_y0, plot_y1)
                return np.column_stack((xs, ys)).ravel().tolist()

            full_edge, some_edge = edge(full), edge(some)
            # Closed along the top from whichever side the line ends at
            canvas.coords(sunlit_item, [xs[0], plot_y0] + full_edge + [xs[-1], plot_y0])
            backwards = [c for k in range(len(xs) - 1, -1, -1)
                         for c in some_edge[2 * k: 2 * k + 2]]
            canvas.coords(penumbra_item, full_edge + backwards)

        def update_light(distance_km: np.ndarray, height_m: np.ndarray, light: Optional[tuple],
                         start: tuple, end: tuple):
            """
            The light alone drawn again, for the same line at another moment -
            its ground, grid and lettering left as they are, which is most of
            the drawing; the whole of it where there is no light drawn yet to
            reshape, or the line is another.
            """
            if (light is None or state["light_items"] is None
                    or (start, end) != (state["start"], state["end"])):
                draw(distance_km, height_m, light, start, end)
                return
            state.update(light=light)
            # The finer stretch in view, if one is drawn, in the new light too:
            # the quick way while the clock is moving, as the line's own is
            fine = state["fine"]
            if fine is not None and state["source"] is fine:
                moving = getattr(self, "_measure_profile_settle", None) is not None
                stretch = self.elevation_profile(
                    start, end, quick=self.PROFILE_SUNLIT_QUICK_SAMPLES if moving else 0,
                    part=fine["part"])
                if stretch is not None and stretch[2] is not None:
                    fine["light"] = stretch[2]
            else:
                state["source"] = {"x": distance_km, "h": height_m, "light": light}
            place_light()
            # And the readout, which tells when the light changes from now
            refresh_readout()

        def draw(distance_km: np.ndarray, height_m: np.ndarray, light: Optional[tuple],
                 start: tuple, end: tuple):
            canvas.delete("all")
            # The stretch of the line in view: the whole of it, or the part the
            # wheel has zoomed in on (see zoom), for as long as the line is the
            # same - another line, or this one with an end moved, starts whole
            if (start, end) != (state.get("start"), state.get("end")):
                state.update(view=None)
            state.update(x=distance_km, h=height_m, light=light, start=start, end=end,
                         changes={}, changes_at=None)
            length = float(distance_km[-1])
            low, high = float(height_m.min()), float(height_m.max())
            km0, km1 = state["view"] if state["view"] is not None else (0.0, length)
            span = km1 - km0

            # What the view is drawn from: the stretch in view sampled as finely
            # as the map holds (see refine), once it has been for this view;
            # the line's own samples otherwise
            fine = state["fine"]
            if fine is not None and (fine["start"], fine["end"], fine["view"]) == (start, end, (km0, km1)):
                source = fine
            else:
                state.update(fine=None)
                source = {"x": distance_km, "h": height_m, "light": light}
            state.update(source=source)
            source_x, source_h = source["x"], source["h"]

            # The samples drawn: those in view and one past each end of it, at
            # about one to a pixel of the plot's width - a long line holds up
            # to PROFILE_MAX_SAMPLES of them, more than a plot of most widths
            # can show, and every point more in the ground's and the light's
            # outlines is work for Tk on every drawing, a drag's included. The
            # two outermost are brought in to the ends of the view (see
            # in_view). Read off, the profile is still read from every sample
            # (see refresh_readout)
            first = max(0, int(np.searchsorted(source_x, km0, side="right")) - 1)
            last = min(len(source_x) - 1, int(np.searchsorted(source_x, km1, side="left")))
            stride = max(1, int((last - first + 1) / max(1.0, plot_x1 - plot_x0)))
            drawn = np.unique(np.append(np.arange(first, last + 1, stride), last))

            def in_view(values: np.ndarray) -> np.ndarray:
                """Values at the samples drawn, the outermost two at the view's ends."""
                shown = np.array(values[drawn], dtype=float)
                shown[0] = np.interp(km0, source_x, values)
                shown[-1] = np.interp(km1, source_x, values)
                return shown

            # The heights in view set the height scale, so a small rise zoomed
            # in on is given height; a little room above and below the ground,
            # and a span of at least ten metres so a flat mare does not draw
            # its noise as mountains
            seen = np.concatenate((in_view(source_h), source_h[first + 1:last]))
            margin = max(5.0, 0.08 * (float(seen.max()) - float(seen.min())))
            y_min, y_max = float(seen.min()) - margin, float(seen.max()) + margin

            flipped = state["flipped"]

            def x_of(km):
                share = (km - km0) / span
                return plot_x0 + (plot_x1 - plot_x0) * (1.0 - share if flipped else share)

            def km_of(px):
                share = (px - plot_x0) / (plot_x1 - plot_x0)
                return km0 + (1.0 - share if flipped else share) * span

            def y_of(m):
                return plot_y1 - (plot_y1 - plot_y0) * (m - y_min) / (y_max - y_min)

            state.update(x_of=x_of, km_of=km_of, y_of=y_of, drawn=drawn, in_view=in_view,
                         view_km=(km0, km1))

            # Light and shadow first, under everything else: the plot in shadow,
            # then the space above the ground the Sun reaches, down to the top
            # of the shadow over each place, and the penumbra paler below that,
            # down to where none of the Sun is in sight. The ground is drawn
            # over them all. Made empty here and shaped by place_light, which
            # alone is called again as the clock moves (see update_light)
            state.update(light_items=None)
            if light is not None:
                state.update(light_items=(
                    canvas.create_rectangle(plot_x0, plot_y0, plot_x1, plot_y1,
                                            fill=colours["shadow"], outline=""),
                    canvas.create_polygon(0, 0, 0, 0, 0, 0, fill=colours["sunlit"], outline=""),
                    canvas.create_polygon(0, 0, 0, 0, 0, 0, fill=penumbra, outline="")))
                place_light()

            # Grid and axis labels. Heights are read from the start of the line,
            # which stands at 0, and the lines fall on round steps of that
            # reading - so one of them always passes through the start
            h_step = self._nice_step(y_max - y_min, 6)
            base = float(height_m[0])
            state.update(base=base)
            for k in range(math.ceil((y_min - base) / h_step),
                           math.floor((y_max - base) / h_step) + 1):
                h = k * h_step
                y = y_of(base + h)
                canvas.create_line(plot_x0, y, plot_x1, y, fill=colours["grid"], tags="grid")
                canvas.create_text(plot_x0 - pad - end_r, y, text=signed(h),
                                   anchor='e', font=font)
            # The unit stands at the far end of the axis from 0 - the right-hand
            # end, or the left once flipped - where the last number would run
            # into it whenever a step falls near the end of the line: such a
            # number is left out, its grid line kept. 0 itself is always shown
            unit_room = metrics.measure("km") + cell_w
            if flipped:
                canvas.create_text(plot_x0, plot_y1 + pad, text="km", anchor='nw', font=font)
            else:
                canvas.create_text(plot_x1, plot_y1 + pad, text="km", anchor='ne', font=font)
            d_step = self._nice_step(span, 10)
            decimals = max(0, -math.floor(math.log10(d_step)))
            d = math.ceil(km0 / d_step - 1e-9) * d_step
            while d <= km1 + 1e-9:
                x = x_of(d)
                canvas.create_line(x, plot_y0, x, plot_y1, fill=colours["grid"], tags="grid")
                text = f"{d:.{decimals}f}"
                half = metrics.measure(text) / 2
                clear = (x - half >= plot_x0 + unit_room) if flipped else (x + half <= plot_x1 - unit_room)
                if clear:
                    canvas.create_text(x, plot_y1 + pad, text=text, anchor='n', font=font)
                d += d_step

            # The ground: filled down to the bottom of the plot, then its edge
            xs = x_of(in_view(source_x)).tolist()
            ys = y_of(in_view(source_h)).tolist()
            outline = np.column_stack((xs, ys)).ravel().tolist()
            ground = canvas.create_polygon([xs[0], plot_y1] + outline + [xs[-1], plot_y1],
                                           fill=colours["fill"], outline="")
            # The grid over the ground rather than under it, so the heights of
            # the low places can be read across the fill as well
            canvas.tag_raise("grid", ground)
            canvas.create_line(outline, fill=colours["ground"], width=line_w)
            canvas.create_rectangle(plot_x0, plot_y0, plot_x1, plot_y1, outline="#808080")

            # The ends marked as on the Moon: the start filled, the end a ring.
            # In the line's colour rather than the measuring line's yellow,
            # which the sunlight behind would swallow
            # Only the ends in view, zoomed in
            r = end_r
            for (x, y), fill, shown in (((xs[0], ys[0]), colours["ground"], km0 <= 0.0),
                                        ((xs[-1], ys[-1]), "", km1 >= length)):
                if shown:
                    canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill,
                                       outline=colours["ground"], width=line_w)

            # What a profile drawn to fill its box hides: how steep the ground
            # really is, from one sample to the next, in degrees. The steepest
            # either way, as a scarp is as steep going down it as up
            slopes = np.degrees(np.arctan2(np.diff(height_m), np.diff(distance_km) * 1000.0))
            # And of what the view is drawn from, for the readout to read
            source_slopes = (slopes if source_x is distance_km else
                             np.degrees(np.arctan2(np.diff(source_h), np.diff(source_x) * 1000.0)))
            state.update(slopes=slopes, source_slopes=source_slopes)

            # Counted from the start of the line, as the axis is - all but the
            # relief, the height from the lowest point to the highest, which
            # is a crater's depth below its rim when the line crosses one
            rise = float(height_m[-1]) - base
            summary_var.set(
                f"{length:.1f} km, Δh {signed(rise)}, "
                f"lowest {signed(low - base)}, highest {signed(high - base)}, "
                f"relief {round(high - low)} m, "
                f"steepest {float(np.abs(slopes).max()):.1f}°")

            # Zoomed in on a line too long for its own samples to hold all the
            # map has, the stretch in view is sampled afresh once the wheel or
            # the drag has paused (see refine)
            if state["refine_due"] is not None:
                canvas.after_cancel(state["refine_due"])
                state.update(refine_due=None)
            texel_km = 2 * math.pi * self.MOON_RADIUS_KM / self.elevation.shape[1]
            coarser = length / (len(distance_km) - 1) > texel_km / self.PROFILE_SAMPLES_PER_TEXEL * 1.01
            if state["view"] is not None and source is not state["fine"] and coarser:
                state.update(refine_due=canvas.after(self.PROFILE_REFINE_MS, refine))

        def refine():
            """
            The stretch of the line in view sampled as finely as the map holds,
            heights and light, and the profile drawn again from it: once, after
            the wheel or a drag along the line has paused, so that zooming and
            moving stay quick while they go on.
            """
            state.update(refine_due=None)
            if state["view"] is None or state["x"] is None:
                return
            length = float(state["x"][-1])
            km0, km1 = state["view"]
            part = (km0 / length, km1 / length)
            stretch = self.elevation_profile(state["start"], state["end"], part=part)
            if stretch is None:
                return
            state.update(fine={"start": state["start"], "end": state["end"], "view": (km0, km1),
                               "part": part, "x": stretch[0], "h": stretch[1],
                               "light": stretch[2]})
            redraw_profile()

        def readout(event):
            state.update(pointer=event.x)
            refresh_readout()

        def pointer_left(_event):
            state.update(pointer=None)
            canvas.delete("readout")
            self._mark_profile_place(None)

        def moment(hours: float) -> datetime:
            """
            The moment that many hours after the one shown, on the observer's
            clock, rounded up to the whole second - so that going to it finds
            the change already made.
            """
            when = self.in_observer_clock(self.dt_local + timedelta(hours=hours))
            if when.microsecond:
                when = when.replace(microsecond=0) + timedelta(seconds=1)
            return when

        def moment_text(hours: float) -> str:
            return moment(hours).strftime("%Y-%m-%d %H:%M:%S")

        def change_at(km: float) -> Optional[tuple]:
            """
            When the ground at the profile's sample nearest km next goes into
            shadow or sunlight (see light_change): (lit_now, the moment, or None
            for none within PROFILE_LIGHT_DAYS) - or None when it is not asked
            for (the Light change box), while there is no light to tell, or
            while the clock is moving, so a held Q or W is not slowed by it; it
            is told once the clock stands still. Worked out the first time it is
            asked for at a moment, and kept for it: the pointer goes back and
            forth over the same samples.
            """
            if (not self._profile_show_light_change or state["light"] is None
                    or getattr(self, "_measure_profile_settle", None) is not None):
                return None
            if state["changes_at"] != self.dt_local:
                state.update(changes={}, changes_at=self.dt_local)
            sample = int(np.argmin(np.abs(state["x"] - km)))
            if sample not in state["changes"]:
                lat, lon = (float(v[0]) for v in
                            self._along(state["start"], state["end"],
                                        float(state["x"][sample]) / float(state["x"][-1])))
                found = self.light_change(lat, lon, float(state["h"][sample]))
                state["changes"][sample] = (None if found is None else
                                            (found[0], None if found[1] is None
                                             else moment(found[1])))
            return state["changes"][sample]

        def go_to_change(event):
            """A click on the profile: the clock taken to when the light changes there."""
            if state["x"] is None or not (plot_x0 <= event.x <= plot_x1):
                return
            if getattr(self, "_video_export", None) is not None:    # the export owns the clock
                return
            change = change_at(state["km_of"](event.x))
            if change is None or change[1] is None:
                return
            self._go_to_moment(change[1])
            # The light has been reshaped for the new moment (see
            # NavigationMixin._follow_measure_line); the readout follows it
            refresh_readout()

        def refresh_readout():
            """The readout for where the pointer is, drawn afresh, and its mark on the Moon."""
            canvas.delete("readout")
            px = state["pointer"]
            if state["x"] is None or px is None or not (plot_x0 <= px <= plot_x1):
                self._mark_profile_place(None)
                return
            km = state["km_of"](px)
            lat, lon = (float(v[0]) for v in
                        self._along(state["start"], state["end"], km / float(state["x"][-1])))
            self._mark_profile_place((lat, lon))
            # Read from what the view is drawn from: zoomed in on a long line,
            # the stretch sampled to the map's own detail (see refine)
            source = state["source"]
            m = float(np.interp(km, source["x"], source["h"]))
            x, y = state["x_of"](km), state["y_of"](m)
            canvas.create_line(x, plot_y0, x, plot_y1, fill=colours["readout"], tags="readout")
            canvas.create_oval(x - 2 * line_w, y - 2 * line_w, x + 2 * line_w, y + 2 * line_w,
                               outline=colours["readout"], width=line_w, tags="readout")
            anchor = 'sw' if x < (plot_x0 + plot_x1) / 2 else 'se'
            # The height read as the axis reads, from the start of the line, the
            # slope of the stretch the pointer is over, and where on the Moon
            # the point is when that is asked for, written as the status bar
            # writes it
            slopes = state["source_slopes"]
            stretch = int(np.clip(np.searchsorted(source["x"], km) - 1, 0, len(slopes) - 1))
            text = (f" {km:.1f} km  {signed(m - state['base'])}  "
                    f"{signed_slope(float(slopes[stretch]))} ")
            text += (f" {abs(lat):.2f}°{'N' if lat >= 0 else 'S'} "
                     f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'} ")
            canvas.create_text(x, plot_y0 - 1, text=text,
                               anchor=anchor, fill=colours["readout"], font=font,
                               tags="readout")
            # On the line above, when asked for: when the light there changes
            change = change_at(km)
            if change is not None:
                lit_now, when = change
                if when is None:
                    text = (f" {'lit' if lit_now else 'in shadow'} past "
                            f"{moment_text(self.PROFILE_LIGHT_DAYS * 24.0)} ")
                else:
                    text = (f" {'in shadow' if lit_now else 'lit'} at "
                            f"{when.strftime('%Y-%m-%d %H:%M:%S')} (click to go) ")
                canvas.create_text(x, plot_y0 - 1 - line_h, text=text,
                                   anchor=anchor, fill=colours["readout"], font=font,
                                   tags="readout")

        canvas.bind("<Motion>", readout)
        canvas.bind("<Leave>", pointer_left)

        def zoom(event):
            """
            The wheel: in on the stretch under the pointer, or back out - the
            pointer's place held where it is, and never out past the whole
            line or in past PROFILE_ZOOM_MIN_SAMPLES of its samples, finer
            than which the map holds nothing more to show.
            """
            if state["x"] is None or not (plot_x0 <= event.x <= plot_x1):
                return
            length = float(state["x"][-1])
            km0, km1 = state["view_km"]
            at = state["km_of"](event.x)
            smallest = min(length, self.PROFILE_ZOOM_MIN_SAMPLES * length / (len(state["x"]) - 1))
            span = km1 - km0
            new_span = min(length, max(smallest, span * self.PROFILE_ZOOM_STEP ** (-event.delta / 120)))
            new0 = min(max(at - (at - km0) * new_span / span, 0.0), length - new_span)
            state.update(view=None if new_span >= length else (new0, new0 + new_span))
            redraw_profile()

        # A drag sideways moves a zoomed-in profile along the line; a click
        # that does not move takes the clock to when the light changes there
        # (see go_to_change)
        held = {"x": None, "view": None, "moved": False}

        def press(event):
            held.update(x=event.x, view=state["view_km"], moved=False)

        def drag(event):
            state.update(pointer=event.x)
            if held["x"] is None or state["view"] is None:
                return
            dx = event.x - held["x"]
            if not held["moved"] and abs(dx) <= self._overlay_px(self.CLICK_SLOP_PX):
                return
            held.update(moved=True)
            km0, km1 = held["view"]
            span, length = km1 - km0, float(state["x"][-1])
            shift = dx / (plot_x1 - plot_x0) * span * (1.0 if state["flipped"] else -1.0)
            new0 = min(max(km0 + shift, 0.0), length - span)
            state.update(view=(new0, new0 + span))
            redraw_profile()

        def release(event):
            moved = held["moved"]
            held.update(x=None, moved=False)
            if not moved:
                go_to_change(event)

        canvas.bind("<MouseWheel>", zoom)
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", release)

        # Dragging the window's edges makes the plot wider, narrower, taller or
        # shorter, the profile drawn again to fill it - a long line's detail
        # spread out, a small rise given height (see _follow_window_size). A
        # profile draws quickly, so it follows the drag as it goes
        def fit(spare_w: int, spare_h: int):
            lay_out(width + spare_w, height + spare_h)
            return width, height

        def redraw_profile():
            if state["x"] is not None:
                draw(state["x"], state["h"], state["light"], state["start"], state["end"])
                refresh_readout()

        self._follow_window_size(frame, canvas, fit, redraw_profile)

        self._profile = {"win": win, "draw": draw, "light": update_light, "close": close}
        # Larger or smaller by either edge, but never smaller than it first
        # opens at, however large the last one was left. Not forced to the
        # front, as it never was: it holds none of the renderer's keys
        self._show_tool_window(win, self.PROFILE_WINDOW, resizable=(True, True),
                               shrink=(width - first_width, height - first_height),
                               focus=False)
