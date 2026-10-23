"""
LabelsMixin: grid, standard labels, spot labels, and illumination logic for MoonRenderer.

Each overlay group (grid lines, grid labels, standard labels, spot labels) is
merged into a single PlotOptiX graph geometry, so updating an overlay after a
time change is one update_graph call instead of one update_data call per line
segment (hundreds to thousands of calls before the merge).
"""

import numpy as np

from plotoptix.materials import m_flat

from moonrtx.shared_types import MoonFeature, MoonLabel
from moonrtx.view_orientation import FLIP_HORIZONTAL_VIEW_ORIENTATIONS, FLIP_VERTICAL_VIEW_ORIENTATIONS
from moonrtx.moon_grid import (
    LABEL_LIFT, create_moon_grid, create_standard_labels, create_spot_labels,
    create_grid_labels_for_orientation, merge_segments_to_graph
)

class LabelsMixin:
    """Mixin providing grid, label, and illumination methods for MoonRenderer."""

    GRID_LINE_RADIUS = 0.006    # Thin lines for grid
    GRID_LABEL_RADIUS = 0.012   # Slightly thicker lines for grid labels
    STANDARD_LABEL_RADIUS = 0.008  # Standard feature label thickness
    SPOT_LABEL_RADIUS = 0.008   # Spot feature label thickness
    # How far the Sun may be under a feature's horizon and its name still shown.
    # The terminator is not where light stops: past it the high ground still
    # catches the Sun, a peak of height h staying lit until the Sun is about
    # sqrt(2h/R) under the horizon - some 4 degrees for the Moon's 5 km peaks,
    # a few tens of kilometres per degree beyond the line. A name is worth
    # having as long as its feature may still show.
    LABEL_NIGHT_MARGIN_DEG = 2.0

    # Lettering on the surface is written in scene units, so magnifying the
    # surface magnifies it with everything else: at the far end of the zoom a
    # name written across a crater is written across the screen. Three sizes are
    # kept instead, the smaller ones cutting in as the magnification passes these
    # multiples of the default view. A long name covers some 220 km of ground at
    # the full size, and the view is 3474 km across at the default one, so each
    # size holds until the name reaches about a quarter of the width - which is
    # where it starts to be more in the way than of use. The N over the pole is
    # not in this: it stands for the globe, not for the ground, and is wanted
    # whole at every magnification (see moon_grid.create_moon_grid).
    LABEL_SCALES = (1.0, 0.5, 0.25)
    LABEL_SCALE_ABOVE = (4.0, 8.0)      # magnifications the smaller sizes start at
    # The zoom moves under the mouse wheel and under Shift with the right button,
    # the second of them inside PlotOptiX where there is nothing to hook, so the
    # magnification is read on a light poll - as the field-of-view frame is.
    LABEL_SCALE_POLL_MS = 250
    GRID_COLOR = [0.50, 0.50, 0.50]
    SPOT_LABEL_COLOR = [1.0, 0.9, 0.3]
    # The spot labels' yellow rather than a light grey of their own: grey is lost
    # against the brightly lit middle of the disk, where the ground is as pale as
    # the lettering, and the hue still reads there
    STANDARD_LABEL_COLOR = SPOT_LABEL_COLOR

    GRID_LINES_GEOM = "grid_lines"
    GRID_LABELS_GEOM = "grid_labels"
    STANDARD_LABELS_GEOM = "standard_labels_graph"
    SPOT_LABELS_GEOM = "spot_labels_graph"

    def _init_label_scale(self):
        """Reset the lettering-size state; called from MoonRenderer.__init__."""
        self._label_scale = self.LABEL_SCALES[0]
        self._label_scale_poll_id = None

    def label_scale(self) -> float:
        """
        The size the lettering is written at now, against its usual size.

        Held rather than computed on the spot: every overlay has to be built at
        one size, and the size may only change where they can all be rebuilt
        together (see _label_scale_tick).
        """
        return getattr(self, "_label_scale", self.LABEL_SCALES[0])

    def _label_scale_for_view(self) -> float:
        """The size the magnification of the moment asks for."""
        magnification = self.surface_magnification()
        size = self.LABEL_SCALES[0]
        for threshold, smaller in zip(self.LABEL_SCALE_ABOVE, self.LABEL_SCALES[1:]):
            if magnification >= threshold:
                size = smaller
        return size

    def _rebuild_lettering(self):
        """
        Write every overlay that carries lettering again at the current size.

        The same work a change of view orientation asks for, and done by the same
        methods: only what is switched on is rebuilt, and the grid keeps its lines
        and its N, only the numbers being redrawn.
        """
        if self.moon_grid is not None and self.moon_grid_visible:
            self.update_grid_labels_for_orientation()
        if self.standard_labels is not None and self.standard_labels_visible:
            self.update_standard_labels_for_view_orientation()
        if self.spot_labels is not None and self.spot_labels_visible:
            self.update_spot_labels_for_view_orientation()
        self.update_pins_for_view_orientation()
        self._draw_place_mark()

    def _label_scale_tick(self):
        """Follow the magnification, and rewrite the lettering when it moves a step."""
        self._label_scale_poll_id = None
        if self.rt is None:
            return

        wanted = self._label_scale_for_view()
        if wanted != self._label_scale:
            self._label_scale = wanted
            self._rebuild_lettering()

        # The names of everything else in view answer to the view itself, which
        # moves under the wheel and the mouse where nothing else would see it
        if self._catalogue_active():
            self.update_catalogue()
        # And the eye, for any way of moving it that is not followed more
        # closely (see _reaim_if_eye_moved)
        self._reaim_if_eye_moved()
        self._schedule_label_scale_poll()

    def _schedule_label_scale_poll(self):
        """Start the poll, or keep it going. Harmless before there is a window."""
        if self.rt is None or getattr(self.rt, "_root", None) is None:
            return
        if self._label_scale_poll_id is None:
            self._label_scale_poll_id = self.rt._root.after(
                self.LABEL_SCALE_POLL_MS, self._label_scale_tick)

    # ---- merged-graph helpers ----

    def _rotate_to_scene(self, pos: np.ndarray) -> np.ndarray:
        """Rotate body-frame vertices to scene coordinates with the current Moon rotation."""
        R = self.moon_rotation
        return pos if R is None else pos @ R.T

    # ---- aiming lettering at the ground it names ----
    #
    # Lettering is written on a sphere just above the highest peak (LABEL_LIFT),
    # so that no mountain hides it. The ground it names lies below that, by some
    # 12 km where the surface is at its mean height: looked at square on that
    # changes nothing, but towards the limb the line of sight crosses the
    # lettering's sphere well before it reaches the ground, and a name stood
    # radially above its crater is seen beside it - towards the limb, by some
    # 7 pixels on a 4K screen at Archytas, a crater 17 pixels across there.
    #
    # So each label is turned about the Moon's centre, as one piece, until the
    # point it names on its sphere lies on the line from the eye to the ground
    # itself: the name covers its crater from wherever it is seen, and is still
    # above every peak. The turn is small - at most some 7 degrees, where the
    # line of sight grazes the limb - so the lettering keeps its shape and its
    # slant. The labels keep their vertices unturned and are turned each time
    # they are put in the scene (_aimed_to_scene): with every step of the time,
    # which turns the Moon, and whenever the eye moves (_reaim_if_eye_moved) -
    # a frame after each step of a drag, straight after a key, and on the poll
    # for anything else - so the turn follows the view as it changes. A label
    # whose ground faces away from the eye is left as it is, behind the limb
    # with its ground: turned, it would be brought round into sight.
    #
    # Lying on the sphere, a label is seen as slanted as the ground under it:
    # at Archytas, 58 degrees north, its letters are squashed to about half
    # their height, and further out they are squashed flat. So each label is
    # also stood up, turned about the point it names until it stands square to
    # the line of sight: read as if it faced the eye from straight ahead. It
    # is laid so that it runs the way east runs on screen where it stands, so
    # it still follows the graticule as before. Stood up, the half of it nearer
    # the middle of the disk leans down towards the ground - by some 15 km at
    # Archytas, enough to sink into it - so the label is then slid towards the
    # eye along the same line of sight, just far enough that all of it is back
    # on or above the lettering's sphere. Moved along the line of sight it
    # stays where it is seen, still on its crater.

    def _unit_vectors(self, lat, lon) -> np.ndarray:
        """Body-frame unit position vectors of places, shape (n, 3)."""
        lat = np.radians(np.atleast_1d(np.asarray(lat, dtype=float)))
        lon = np.radians(np.atleast_1d(np.asarray(lon, dtype=float)))
        cos_lat = np.cos(lat)
        return np.column_stack((cos_lat * np.sin(lon), -cos_lat * np.cos(lon), np.sin(lat)))

    def _aimed_at_ground(self, pos: np.ndarray, lat, lon, counts) -> np.ndarray:
        """
        Body-frame label vertices, each label turned so the place it names is
        seen on its own ground from where the eye is now, and stood up to face
        the eye (see the note above). Notes where the eye was, for
        _reaim_if_eye_moved.

        Parameters
        ----------
        pos : np.ndarray
            Body-frame vertices of every label in the set, label after label
        lat, lon : array_like
            The place each label names, one per label
        counts : array_like
            How many of the vertices belong to each label
        """
        R = self.moon_rotation
        if R is None or self.rt is None or len(pos) == 0:
            return pos
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        eye = np.asarray(self.rt.get_camera(self.CAMERA_NAME)["Eye"], dtype=float)
        self._labels_aimed_eye = eye

        # Worked in the scene, where the eye is, and taken back to the body
        # frame at the end, R being a rotation
        named = self._unit_vectors(lat, lon) @ R.T
        on_ground = named * (self.MOON_RADIUS * np.asarray(self._ground_factor(lat, lon),
                                                           dtype=float))[:, None]
        sight = eye - on_ground
        sight /= np.linalg.norm(sight, axis=1, keepdims=True)

        # Where the line from the ground to the eye leaves the lettering's
        # sphere: the far root of |g + t s| = lifted, the ground being inside
        lifted = self.MOON_RADIUS * (1 + LABEL_LIFT)
        b = np.einsum('ij,ij->i', on_ground, sight)
        c = np.einsum('ij,ij->i', on_ground, on_ground) - lifted * lifted
        t = -b + np.sqrt(np.maximum(b * b - c, 0.0))
        aimed = on_ground + t[:, None] * sight
        aimed /= np.linalg.norm(aimed, axis=1, keepdims=True)

        away = np.einsum('ij,ij->i', named, sight) <= 0.0
        aimed[away] = named[away]

        # Stood up at the aimed point, square to the line of sight: each label
        # is written in its own frame - east along the line, north up, out of
        # the sphere - and that frame is laid on the one it is seen in, with
        # the line running the way east runs on screen there, up square to it
        # and out towards the eye. The lettering then reads face on, slanted
        # just as the graticule is where it stands
        la, lo = np.radians(lat), np.radians(lon)
        east = np.column_stack((np.cos(lo), np.sin(lo), np.zeros_like(lo))) @ R.T
        north = np.column_stack((-np.sin(la) * np.sin(lo), np.sin(la) * np.cos(lo),
                                 np.cos(la))) @ R.T
        along_line = east - np.einsum('ij,ij->i', east, sight)[:, None] * sight
        along_line /= np.linalg.norm(along_line, axis=1, keepdims=True)
        up = np.cross(sight, along_line)
        written = np.stack((east, north, named), axis=1)       # rows: the label's frame
        seen = np.stack((along_line, up, sight), axis=1)       # rows: where it goes
        turn = np.einsum('nki,nkj->nij', written, seen)
        # A label behind the limb is left lying where it is, with its ground
        turn[away] = np.eye(3)

        scene = pos @ R.T
        at = np.repeat(aimed * lifted, counts, axis=0)
        rel = scene - np.repeat(named * lifted, counts, axis=0)
        stood = at + np.einsum('vi,vij->vj', rel, np.repeat(turn, counts, axis=0))

        # Stood up, the half nearer the middle of the disk leans in towards the
        # ground, so the label is slid towards the eye along the line of sight
        # until no vertex is under the lettering's sphere: for each one inside
        # it, how far along the line it is to the sphere (the far root of
        # |p + d s| = lifted), and for each label the most any vertex needs
        towards = np.repeat(sight, counts, axis=0)
        ps = np.einsum('ij,ij->i', stood, towards)
        pp = np.einsum('ij,ij->i', stood, stood)
        need = np.where(pp < lifted * lifted,
                        -ps + np.sqrt(np.maximum(ps * ps + lifted * lifted - pp, 0.0)), 0.0)
        slide = np.zeros(len(named))
        np.maximum.at(slide, np.repeat(np.arange(len(named)), counts), need)
        slide[away] = 0.0
        stood += np.repeat(slide, counts)[:, None] * towards
        return stood @ R

    @staticmethod
    def _features_places(features: list[MoonFeature]) -> tuple[np.ndarray, np.ndarray]:
        """The places labels naming these features name: their centres."""
        return (np.array([f.lat for f in features], dtype=float),
                np.array([f.lon for f in features], dtype=float))

    def _aimed_to_scene(self, pos: np.ndarray, places, counts) -> np.ndarray:
        """
        A set of labels in the scene, each aimed at its ground from where the
        eye is now: what _rotate_to_scene is for every other overlay.

        Parameters
        ----------
        pos : np.ndarray
            Body-frame vertices of every label in the set, as written
        places : tuple
            (lat, lon) of the place each label names, one per label
        counts : array_like
            How many of the vertices belong to each label
        """
        return self._rotate_to_scene(self._aimed_at_ground(pos, *places, counts))

    def _eye_moved_since_aim(self) -> bool:
        """Whether the eye has moved since the lettering was last aimed."""
        if self.rt is None:
            return False
        eye = np.asarray(self.rt.get_camera(self.CAMERA_NAME)["Eye"], dtype=float)
        aimed = getattr(self, "_labels_aimed_eye", None)
        return aimed is None or not np.allclose(eye, aimed, rtol=0.0, atol=1e-9)

    def _reaim_if_eye_moved(self):
        """
        Aim every overlay naming a place again if the eye has moved: the
        standard, spot and catalogue names, the pins and the place mark. Called
        after whatever may have moved it, and does nothing if it did not, so
        the picture is not started again for nothing.
        """
        if self.rt is None or self.moon_rotation is None or not self._eye_moved_since_aim():
            return
        self._labels_aimed_eye = np.asarray(
            self.rt.get_camera(self.CAMERA_NAME)["Eye"], dtype=float)
        if not (self.standard_labels_visible or self.spot_labels_visible
                or (self.pins and self.pins_visible) or self._place_mark is not None
                or self._catalogue_active()):
            return
        # Under the padlock, so the accumulation restarts once for them all
        with self.rt._padlock:
            if self.standard_labels_visible:
                self.update_standard_labels_orientation()
            if self.spot_labels_visible:
                self.update_spot_labels_orientation()
            if self.pins_visible:
                self.update_pins_orientation()
            if self._place_mark is not None:
                self.update_place_mark_orientation()
            if self._catalogue_active():
                self.update_catalogue(moved=True)
        if self.rt._is_started:
            self.rt.refresh_scene()

    def _to_screen(self, body: np.ndarray) -> np.ndarray:
        """
        Map a body-frame direction to the screen axes (x right, y up).

        The scene puts the eye at -Y looking towards +Y, with +X to the right and
        +Z up, so the screen axes are the scene's X and Z - mirrored where the
        view orientation asks for it (E left instead of W, or S up instead of N).
        """
        scene = body if self.moon_rotation is None else self.moon_rotation @ body
        return np.array([
            -scene[0] if self.view_orientation in FLIP_HORIZONTAL_VIEW_ORIENTATIONS else scene[0],
            -scene[2] if self.view_orientation in FLIP_VERTICAL_VIEW_ORIENTATIONS else scene[2],
        ])

    def _glyph_flips(self, lat: float, lon: float) -> tuple[bool, bool]:
        """
        How the glyphs of a label at (lat, lon) have to be mirrored to be read.

        Glyphs are laid out in the tangent plane there, running east and standing
        north, so the lettering follows the graticule. Whether that comes out
        readable depends on where north ends up pointing on screen, which the
        observer's latitude, the hour and the view orientation all have a say in:
        from the southern hemisphere the Moon is seen the other way up, and the
        graticule turns through the night, so one answer cannot serve every label.

        East, north and the outward normal make a right-handed frame wherever they
        are taken, and so do right, up and the direction back to the eye for anyone
        looking at the label. One turns into the other; neither mirrors into it. So
        a glyph wants turning and never mirroring, and since two flips make a turn
        and one makes a mirror, the two go together. A label beyond the limb is no
        different: it is read by turning that side of the globe into view with Ctrl
        and the arrow keys, and turning about the poles leaves north standing where
        it stood.

        The one mirror that is wanted is the one that cancels another. A view
        orientation showing the picture mirrored - one flip of the two, not both -
        reverses every glyph with it, so the glyph goes in reversed to come out
        right.

        The answer is baked into the geometry when an overlay is built, and the
        overlay is built when it is switched on or when the view orientation
        changes - never on a time step - so the lettering holds still while time
        runs and turns only when the view does.
        """
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

        # The tangent the glyph stands along, in the body frame
        north = np.array([-sin_lat * sin_lon, sin_lat * cos_lon, cos_lat])
        turn = bool(self._to_screen(north)[1] < 0.0)

        mirrored_view = ((self.view_orientation in FLIP_HORIZONTAL_VIEW_ORIENTATIONS)
                         != (self.view_orientation in FLIP_VERTICAL_VIEW_ORIENTATIONS))
        return turn != mirrored_view, turn

    @staticmethod
    def _label_graph_arrays(labels: list[MoonLabel]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge all label segments into one graph.

        Returns
        -------
        tuple
            (pos, edges, counts): graph vertices and edges, plus the number of
            vertices belonging to each label (for per-label radii).
        """
        segments = [seg for label in labels for seg in label.segments]
        pos, edges = merge_segments_to_graph(segments)
        counts = np.array([sum(seg.shape[0] for seg in label.segments) for label in labels],
                          dtype=np.int64)
        return pos, edges, counts

    @staticmethod
    def _features_unit_vectors(features: list[MoonFeature]) -> np.ndarray:
        """Body-frame unit position vectors of features, shape (n, 3)."""
        lat = np.radians([f.lat for f in features])
        lon = np.radians([f.lon for f in features])
        cos_lat = np.cos(lat)
        return np.column_stack((cos_lat * np.sin(lon), -cos_lat * np.cos(lon), np.sin(lat)))

    def _lit_mask(self, units: np.ndarray) -> np.ndarray:
        """
        Boolean mask of features on the illuminated hemisphere, or no further
        past the terminator than LABEL_NIGHT_MARGIN_DEG, given the cached light
        position and current Moon rotation (vectorized over all features).
        """
        if self.light_pos is None or self.moon_rotation is None:
            # If we don't have light or rotation info, assume visible to avoid hiding labels
            return np.ones(units.shape[0], dtype=bool)
        light = np.asarray(self.light_pos, dtype=float)
        light_norm = np.linalg.norm(light)
        if light_norm == 0:
            return np.ones(units.shape[0], dtype=bool)
        # The dot product is the sine of the Sun's altitude over the feature's
        # centre: above 0 it is on the lit hemisphere, and down to the sine of
        # minus the margin it is close enough past the terminator to keep its name
        return (units @ (self.moon_rotation.T @ (light / light_norm))
                > -np.sin(np.radians(self.LABEL_NIGHT_MARGIN_DEG)))

    def _label_radii(self, units: np.ndarray, counts: np.ndarray, radius: float) -> np.ndarray:
        """Per-vertex radii hiding labels of features on the night side."""
        return np.repeat(np.where(self._lit_mask(units), radius, 0.0), counts).astype(np.float32)

    @staticmethod
    def _no_shadow_flat_material() -> dict:
        # Flat material with shadow rays passing through (base_color alpha 0
        # + transparent occlusion program), so overlays cast no shadow on the surface
        m = m_flat.copy()
        m["OcclusionProgram"] = "chit7_occlusion_transp.ptx::__closesthit__occlusion_transparency"
        m["VarFloat4"] = {"base_color": [1.0, 1.0, 1.0, 0.0]}
        return m

    # ---- orientation helpers ----

    def set_view_orientation(self, view_orientation: str):
        """
        Set the view orientation mode and update the status bar.

        Called when F5-F8 keys are pressed to match plotoptix internal orientation change.

        Parameters
        ----------
        view_orientation : str
        """
        self.view_orientation = view_orientation

        # Update grid labels if grid is visible
        if self.moon_grid is not None and self.moon_grid_visible:
            self.update_grid_labels_for_orientation()

        # Update standard labels if visible
        if self.standard_labels is not None and self.standard_labels_visible:
            self.update_standard_labels_for_view_orientation()

        # Update spot labels if visible
        if self.spot_labels is not None and self.spot_labels_visible:
            self.update_spot_labels_for_view_orientation()

        # Pin digits are mirrored like any other lettering
        self.update_pins_for_view_orientation()

        # As are the names beside the Sun and Earth markers
        if self.sub_points_visible:
            self.update_sub_points()

        self._update_status_view()

    def _rebuild_grid_labels_arrays(self):
        """Rebuild the merged vertex/edge arrays for all grid number labels and the N label."""
        segments = [seg for segs in self.moon_grid.lat_labels for seg in segs]
        segments += [seg for segs in self.moon_grid.lon_labels for seg in segs]
        segments += list(self.moon_grid.N)
        self._grid_labels_pos, self._grid_labels_edges = merge_segments_to_graph(segments)

    def update_grid_labels_for_orientation(self):
        """
        Update grid number labels to match current view orientation.

        Regenerates latitude and longitude number labels so they are
        always readable (not upside down) in the current view orientation.
        """
        if self.rt is None or self.moon_grid is None:
            return

        # Generate new labels with proper orientation
        lat_labels, lat_label_values, lon_labels, lon_label_values = create_grid_labels_for_orientation(
            moon_radius=self.MOON_RADIUS,
            lat_step=15.0,
            lon_step=15.0,
            offset=0.0,
            flips_at=self._glyph_flips,
            scale=self.label_scale()
        )

        # Update the moon_grid with new labels
        self.moon_grid = self.moon_grid._replace(
            lat_labels=lat_labels,
            lat_label_values=lat_label_values,
            lon_labels=lon_labels,
            lon_label_values=lon_label_values
        )

        # Flipping mirrors coordinates but keeps the segment structure, so the
        # edge indices stay valid and only vertex positions need an update
        self._rebuild_grid_labels_arrays()
        # The stroke goes with the size: a quarter-size number drawn at the full
        # thickness is a smudge
        self.rt.update_graph(self.GRID_LABELS_GEOM,
                             pos=self._rotate_to_scene(self._grid_labels_pos),
                             r=self.GRID_LABEL_RADIUS * self.label_scale())

    def update_standard_labels_for_view_orientation(self):
        """
        Update standard labels to match current view orientation.

        Regenerates standard labels so they are always readable
        (not upside down) in the current view orientation.
        """
        if self.rt is None or self.standard_labels is None or self.standard_label_features is None:
            return

        # Regenerate labels with proper orientation
        self.standard_labels = create_standard_labels(
            self.standard_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flips_at=self._glyph_flips,
            scale=self.label_scale()
        )

        self._standard_labels_pos, self._standard_labels_edges, self._standard_labels_counts = \
            self._label_graph_arrays(self.standard_labels)
        self.rt.update_graph(
            self.STANDARD_LABELS_GEOM,
            pos=self._aimed_to_scene(self._standard_labels_pos, self._standard_places,
                                     self._standard_labels_counts),
            r=self._label_radii(self._standard_units, self._standard_labels_counts, self.STANDARD_LABEL_RADIUS * self.label_scale()))

    def update_spot_labels_for_view_orientation(self):
        """
        Update spot labels to match current view orientation.

        Regenerates spot labels so they are always readable
        (not upside down) in the current view orientation.
        """
        if self.rt is None or self.spot_labels is None or self.spot_label_features is None:
            return

        # Regenerate labels with proper orientation
        self.spot_labels = create_spot_labels(
            self.spot_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flips_at=self._glyph_flips,
            scale=self.label_scale()
        )

        self._spot_labels_pos, self._spot_labels_edges, self._spot_labels_counts = \
            self._label_graph_arrays(self.spot_labels)
        self.rt.update_graph(
            self.SPOT_LABELS_GEOM,
            pos=self._aimed_to_scene(self._spot_labels_pos, self._spot_places,
                                     self._spot_labels_counts),
            r=self._label_radii(self._spot_units, self._spot_labels_counts, self.SPOT_LABEL_RADIUS * self.label_scale()))

    # ---- grid setup / show / hide ----

    def setup_moon_grid(self, lat_step: float = 15.0, lon_step: float = 15.0):
        """
        Create selenographic coordinate grid.

        Parameters
        ----------
        lat_step : float
            Spacing between latitude lines in degrees
        lon_step : float
            Spacing between longitude lines in degrees
        """
        if self.rt is None:
            print("Renderer not initialized")
            return

        # Generate grid data
        self.moon_grid = create_moon_grid(
            moon_radius=self.MOON_RADIUS,
            lat_step=lat_step,
            lon_step=lon_step,
            points_per_line=100,
            offset=0.0,
            flips_at=self._glyph_flips,
            scale=self.label_scale()
        )

        self.rt.update_material("grid_material", self._no_shadow_flat_material())

        # All grid lines in one graph geometry, all number labels in another
        self._grid_lines_pos, self._grid_lines_edges = merge_segments_to_graph(
            self.moon_grid.lat_lines + self.moon_grid.lon_lines)
        self._rebuild_grid_labels_arrays()

        self.rt.set_graph(self.GRID_LINES_GEOM,
                          pos=self._grid_lines_pos, edges=self._grid_lines_edges,
                          r=self.GRID_LINE_RADIUS, c=self.GRID_COLOR, mat="grid_material")
        self.rt.set_graph(self.GRID_LABELS_GEOM,
                          pos=self._grid_labels_pos, edges=self._grid_labels_edges,
                          r=self.GRID_LABEL_RADIUS * self.label_scale(), c=self.GRID_COLOR, mat="grid_material")

        self.moon_grid_visible = True

        self.update_moon_grid_orientation()

    def show_moon_grid(self, visible: bool = True):
        """
        Show or hide the selenographic grid.

        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return

        if self.moon_grid is None:
            if visible:
                self.setup_moon_grid()
            return

        # Toggle visibility by setting zero radius (hide) or restoring (show)
        self.rt.update_graph(self.GRID_LINES_GEOM, r=self.GRID_LINE_RADIUS if visible else 0.0)
        self.rt.update_graph(self.GRID_LABELS_GEOM,
                             r=self.GRID_LABEL_RADIUS * self.label_scale()
                             if visible else 0.0)

        self.moon_grid_visible = visible

        # When showing the grid, update its orientation to match current view and Moon position
        # This is needed in case view orientation or time changed while the grid was hidden
        if visible:
            self.update_grid_labels_for_orientation()  # View orientation for labels
            self.update_moon_grid_orientation()  # Moon rotation for grid lines

    def toggle_grid(self):
        """Toggle the selenographic grid visibility."""
        self.show_moon_grid(not self.moon_grid_visible)

    # ---- standard labels ----

    def setup_standard_labels(self):
        """
        Create standard feature labels for Moon features with standard_label=true.
        """
        if self.rt is None:
            print("Renderer not initialized")
            return

        # Get ALL features with standard_label=True (illumination checked during rendering)
        self.standard_label_features = [f for f in self.moon_features if f.standard_label]
        if not self.standard_label_features:
            return
        self.standard_labels = create_standard_labels(
            self.standard_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flips_at=self._glyph_flips,
            scale=self.label_scale()
        )

        self.rt.update_material("standard_label_material", self._no_shadow_flat_material())

        self._standard_units = self._features_unit_vectors(self.standard_label_features)
        self._standard_labels_pos, self._standard_labels_edges, self._standard_labels_counts = \
            self._label_graph_arrays(self.standard_labels)
        self._standard_places = self._features_places(self.standard_label_features)

        # All labels in one graph geometry; night-side labels hidden via zero vertex radii
        self.rt.set_graph(
            self.STANDARD_LABELS_GEOM,
            pos=self._aimed_to_scene(self._standard_labels_pos, self._standard_places,
                                     self._standard_labels_counts),
            edges=self._standard_labels_edges,
            r=self._label_radii(self._standard_units, self._standard_labels_counts, self.STANDARD_LABEL_RADIUS * self.label_scale()),
            c=self.STANDARD_LABEL_COLOR,
            mat="standard_label_material")

        self.standard_labels_visible = True

    def show_standard_labels(self, visible: bool = True):
        """
        Show or hide the standard feature labels.

        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return

        if self.standard_labels is None:
            if visible:
                self.setup_standard_labels()
            return

        self.standard_labels_visible = visible

        if visible:
            # Restores orientation and per-label illumination radii, in case
            # time or view orientation changed while labels were hidden
            self.update_standard_labels_for_view_orientation()
        else:
            self.rt.update_graph(self.STANDARD_LABELS_GEOM, r=0.0)

    def toggle_standard_labels(self):
        """Toggle the feature standard labels visibility."""
        self.show_standard_labels(not self.standard_labels_visible)

    # ---- spot labels ----

    def setup_spot_labels(self):
        """
        Create spot labels for Moon features with spot_label=true.
        """
        if self.rt is None:
            print("Renderer not initialized")
            return

        # Get ALL features with spot_label=True (illumination checked during rendering)
        self.spot_label_features = [f for f in self.moon_features if f.spot_label]
        if not self.spot_label_features:
            return
        self.spot_labels = create_spot_labels(
            self.spot_label_features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flips_at=self._glyph_flips,
            scale=self.label_scale()
        )

        self.rt.update_material("spot_label_material", self._no_shadow_flat_material())

        self._spot_units = self._features_unit_vectors(self.spot_label_features)
        self._spot_labels_pos, self._spot_labels_edges, self._spot_labels_counts = \
            self._label_graph_arrays(self.spot_labels)
        self._spot_places = self._features_places(self.spot_label_features)

        # All labels in one graph geometry; night-side labels hidden via zero vertex radii
        self.rt.set_graph(
            self.SPOT_LABELS_GEOM,
            pos=self._aimed_to_scene(self._spot_labels_pos, self._spot_places,
                                     self._spot_labels_counts),
            edges=self._spot_labels_edges,
            r=self._label_radii(self._spot_units, self._spot_labels_counts, self.SPOT_LABEL_RADIUS * self.label_scale()),
            c=self.SPOT_LABEL_COLOR,
            mat="spot_label_material")

        self.spot_labels_visible = True

    def show_spot_labels(self, visible: bool = True):
        """
        Show or hide the spot feature labels.

        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if self.rt is None:
            return

        if self.spot_labels is None:
            if visible:
                self.setup_spot_labels()
            return

        self.spot_labels_visible = visible

        if visible:
            # Restores orientation and per-label illumination radii, in case
            # time or view orientation changed while labels were hidden
            self.update_spot_labels_for_view_orientation()
        else:
            self.rt.update_graph(self.SPOT_LABELS_GEOM, r=0.0)

    def toggle_spot_labels(self):
        """Toggle the spot labels visibility."""
        self.show_spot_labels(not self.spot_labels_visible)

    def hide_pinned_labels(self):
        """
        Hide pinned labels.
        """
        if self.rt is None:
            return
        self._catalogue_pinned.clear()
        # Redrawn rather than switched off: show_catalogue(False) would take the
        # catalogue's own names (P) off with the pinned ones
        self._refresh_catalogue()
        # And the mark of a place the Find window went to, which stands in for
        # a name there was none of
        self.clear_place_mark()

    # ---- orientation updates (after time change) ----

    def update_spot_labels_orientation(self):
        """
        Update spot labels to match current Moon orientation.

        This should be called after update_view() to rotate the labels
        along with the Moon surface.
        """
        if self.rt is None or self.spot_labels is None:
            return

        if self.moon_rotation is None:
            return

        self.rt.update_graph(
            self.SPOT_LABELS_GEOM,
            pos=self._aimed_to_scene(self._spot_labels_pos, self._spot_places,
                                     self._spot_labels_counts),
            r=self._label_radii(self._spot_units, self._spot_labels_counts, self.SPOT_LABEL_RADIUS * self.label_scale()))

    def update_standard_labels_orientation(self):
        """
        Update standard labels to match current Moon orientation.

        This should be called after update_view() to rotate the labels
        along with the Moon surface.
        """
        if self.rt is None or self.standard_labels is None:
            return

        if self.moon_rotation is None:
            return

        self.rt.update_graph(
            self.STANDARD_LABELS_GEOM,
            pos=self._aimed_to_scene(self._standard_labels_pos, self._standard_places,
                                     self._standard_labels_counts),
            r=self._label_radii(self._standard_units, self._standard_labels_counts, self.STANDARD_LABEL_RADIUS * self.label_scale()))

    def update_moon_grid_orientation(self):
        """
        Update grid lines to match current Moon orientation.

        This should be called after update_view() to rotate the grid
        along with the Moon surface.
        """
        if self.rt is None or self.moon_grid is None or not self.moon_grid_visible:
            return

        if self.moon_rotation is None:
            return

        self.rt.update_graph(self.GRID_LINES_GEOM, pos=self._rotate_to_scene(self._grid_lines_pos))
        self.rt.update_graph(self.GRID_LABELS_GEOM, pos=self._rotate_to_scene(self._grid_labels_pos))
