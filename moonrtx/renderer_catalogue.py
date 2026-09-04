"""
CatalogueMixin: naming what is in view, out of the whole feature table.

The table holds some four and a half thousand features, and the disk is about a
thousand pixels across at the default view: written all at once their names would
cover the Moon twice over in ink. What is drawn instead is the largest of them
that are in view and in daylight, a capped number of names at a time, so the
naming holds a readable density wherever the view is and whatever it is magnified
to. Zooming in does not add names on top of the ones already there; it drops the
craters that have left the picture and takes up smaller ones in their place.

The names are the ones the standard labels use, written across the feature they
belong to. A feature already named by the standard or the spot labels is left to
them while those are on, so no name is drawn twice.

The features the table marks for a spot label - Plato and the landing sites, the
last of them a tenth of a kilometre across - are named whenever they are in view,
over and above the count: ranking by size would never reach them. There are only
a couple of dozen in the table, so they cost little.

Only the chosen names are built, and only when the choice changes - forty of them
cost about seven milliseconds, against the eight hundred the whole table would.
The choice is looked at on the same poll that follows the lettering size, so it
keeps up with the wheel and with a dragged view, neither of which the renderer
sees any other way.
"""

import math

import numpy as np

from moonrtx.moon_grid import create_standard_labels


class CatalogueMixin:
    """Mixin naming the features of the table that are in view."""

    CATALOGUE_GEOM = "catalogue_labels"
    # Dimmer than the standard labels, which stay the brighter of the two so the
    # names worth knowing still read first
    CATALOGUE_COLOR = [0.70, 0.70, 0.70]
    CATALOGUE_LABEL_RADIUS = 0.008
    # How many names at once. Enough to be worth having, few enough to read: the
    # picture holds about this many at the density the standard labels are drawn.
    CATALOGUE_LIMIT = 40
    # A name whose feature is just off the edge is worth drawing, the name itself
    # reaching back into the picture, so the field is taken slightly wider than it is
    CATALOGUE_VIEW_MARGIN = 1.1

    def _init_catalogue(self):
        """Reset the catalogue state; called from MoonRenderer.__init__."""
        self.catalogue_visible = False
        self._catalogue_units = None
        self._catalogue_diameters = None
        self._catalogue_drawn = None        # what is on screen, to spot a change
        self._catalogue_pos = None

    # ---- choosing ----

    def _catalogue_arrays(self):
        """
        Unit vectors and diameters of every feature in the table, built once.

        The table does not change while the app runs, so neither do these.
        """
        if self._catalogue_units is None:
            self._catalogue_units = self._features_unit_vectors(self.moon_features)
            self._catalogue_diameters = np.array([f.diameter_km for f in self.moon_features])
        return self._catalogue_units, self._catalogue_diameters

    def _catalogue_in_view(self, units: np.ndarray) -> np.ndarray:
        """
        Which features are turned towards the eye and fall inside the picture.

        The camera is a pinhole, so a point is in the picture when its offset from
        the axis is within the half-field its own depth gives. Being on the near
        side is asked separately: the far side projects into the disk just as the
        near side does, and a name written there would show through the Moon.
        """
        if self.rt is None or self.moon_rotation is None:
            return np.zeros(len(units), dtype=bool)

        cam = self.rt.get_camera(self.CAMERA_NAME)
        eye = np.array(cam["Eye"], dtype=float)
        target = np.array(cam["Target"], dtype=float)
        up = np.array(cam["Up"], dtype=float)

        forward = target - eye
        norm = np.linalg.norm(forward)
        eye_norm = np.linalg.norm(eye)
        fov = self.rt._optix.get_camera_fov(0)
        if norm == 0.0 or eye_norm == 0.0 or fov <= 0.0 or self.rt._height <= 0:
            return np.zeros(len(units), dtype=bool)
        forward = forward / norm

        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm == 0.0:
            return np.zeros(len(units), dtype=bool)
        right = right / norm
        up = np.cross(right, forward)

        scene = (units @ self.moon_rotation.T) * self.MOON_RADIUS
        near = scene @ (eye / eye_norm) > 0.0

        from_eye = scene - eye
        depth = from_eye @ forward
        half_height = np.maximum(depth, 0.0) * math.tan(math.radians(fov) / 2)
        half_width = half_height * (self.rt._width / self.rt._height)
        margin = self.CATALOGUE_VIEW_MARGIN
        inside = ((np.abs(from_eye @ right) <= half_width * margin)
                  & (np.abs(from_eye @ up) <= half_height * margin)
                  & (depth > 0.0))
        return near & inside

    def _catalogue_selection(self) -> np.ndarray:
        """
        The features to name: those in view, lit, and not already named by another
        overlay - the largest CATALOGUE_LIMIT of them, and every spot-marked one
        besides.

        Taking the largest is what keeps the density even. The small craters
        outnumber the large ones, but not as fast as the view shrinks when it is
        magnified, so a rule on size alone would name fewer and fewer of them the
        further in the view went; a count names the same many, and they are finer
        ones each time.

        The spot-marked features are added rather than ranked with the rest: they
        are the ones size can never reach, and letting them compete would have
        them crowd out the craters at the wide view instead.
        """
        if not self.moon_features:
            return np.empty(0, dtype=int)

        units, diameters = self._catalogue_arrays()
        taken = np.array([
            (f.standard_label and self.standard_labels_visible)
            or (f.spot_label and self.spot_labels_visible)
            for f in self.moon_features])

        shown = ~taken & self._lit_mask(units) & self._catalogue_in_view(units)
        marked = np.array([f.spot_label for f in self.moon_features])

        rest = np.flatnonzero(shown & ~marked)
        if rest.size > self.CATALOGUE_LIMIT:
            biggest = np.argsort(diameters[rest])[::-1][:self.CATALOGUE_LIMIT]
            rest = rest[biggest]
        return np.sort(np.concatenate((np.flatnonzero(shown & marked), rest)))

    # ---- drawing ----

    def update_catalogue(self, moved: bool = False):
        """
        Draw the names the view asks for, if they are not the ones already drawn.

        Touching the scene restarts the accumulation, and the picture is built up
        over many frames: an update on every poll, four times a second, was seen
        as a flicker even with the view standing still. So when the choice has not
        changed this does nothing at all - unless the Moon itself has moved, when
        the names that are drawn have to turn with it.

        Parameters
        ----------
        moved : bool
            The Moon has turned since the last call - a step in time - so the
            names need placing again even where the choice of them stands.
            update_overlays passes this; the poll that watches the view does not.
        """
        if self.rt is None or not self.catalogue_visible:
            return

        chosen = self._catalogue_selection()
        drawn = (tuple(chosen), self.label_scale(), self.view_orientation)
        if drawn == self._catalogue_drawn:
            if moved and self._catalogue_pos is not None:
                self.rt.update_graph(self.CATALOGUE_GEOM,
                                     pos=self._rotate_to_scene(self._catalogue_pos))
            return

        self._catalogue_drawn = drawn
        if chosen.size == 0:
            # Nothing in view worth naming - a view over the night side, most
            # likely, where the names go out as every other overlay's do
            self._catalogue_pos = None
            self._hide_catalogue()
            return

        features = [self.moon_features[i] for i in chosen]
        labels = create_standard_labels(
            features,
            moon_radius=self.MOON_RADIUS,
            offset=0.0,
            flips_at=self._glyph_flips,
            scale=self.label_scale()
        )
        pos, edges, counts = self._label_graph_arrays(labels)
        self._catalogue_pos = pos

        self.rt.update_material("catalogue_material", self._no_shadow_flat_material())
        self.rt.set_graph(
            self.CATALOGUE_GEOM,
            pos=self._rotate_to_scene(pos),
            edges=edges,
            r=np.repeat(self.CATALOGUE_LABEL_RADIUS * self.label_scale(),
                        counts.sum()).astype(np.float32),
            c=self.CATALOGUE_COLOR,
            mat="catalogue_material")

    def show_catalogue(self, visible: bool = True):
        """Show or hide the names of everything else in view."""
        if self.rt is None:
            return

        self.catalogue_visible = visible
        if visible:
            self._catalogue_drawn = None    # nothing on screen to compare against
            self.update_catalogue()
        else:
            self._hide_catalogue()

    def _hide_catalogue(self):
        """Take the names off, whether or not any were ever drawn."""
        try:
            self.rt.update_graph(self.CATALOGUE_GEOM, r=0.0)
        except Exception:                   # never drawn yet, so nothing to hide
            pass

    def toggle_catalogue(self):
        """Toggle the names of everything else in view."""
        self.show_catalogue(not self.catalogue_visible)
