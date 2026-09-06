"""
CanvasOverlayMixin: what the three corner overlays all need, in one place.

The compass, the locator and the field-of-view frame are drawn on the Tk canvas
over the render rather than into the scene, and each of them therefore needs the
same four things. A canvas to draw on, which does not exist until the window
does. A list of what it has drawn, so that it can take it away again. Lettering
that can be read over ground running from black sky to lit highland, which no
colour manages on its own and which is done here by giving every letter a dark
rim. And a way of noticing that the view has moved: the camera answers to the
mouse and to PlotOptiX's own handlers, where there is nothing to hook, so each
overlay looks at it on a light poll and redraws only when what it would draw has
changed.

Each of the three had all four written out for itself, the four differing between
them only in the names. They are here instead, and each overlay is left with the
part that is truly its own: what it draws, when it is switched on, and what else
besides the camera its picture depends on.

Nothing here draws anything. The rim is the one exception and it draws only what
it is told to, twice over.
"""

import os
import copy
from typing import Optional

import tkinter as tk

import cv2
import numpy as np

from .overlay_raster import FontBook, OverlaySurface, composite


class CanvasOverlayMixin:
    """Mixin carrying what the canvas overlays have in common."""

    # How often an overlay looks to see whether the view has moved. Five times a
    # second is quick enough that a drag does not visibly lag the picture, and
    # slow enough that a still view costs nothing worth measuring - the looking
    # itself is a camera read and a tuple comparison.
    OVERLAY_REFRESH_MS = 200

    # Lettering over the render has no background it can count on: the ground
    # beneath it runs from the black of the sky to the white of a lit highland,
    # and no one colour is legible on both - the compass readings were measured
    # at a fifth over one to one on ordinary grey, where four and a half is what
    # small text wants. So each is written a second time, in black, at every one
    # of these offsets, and the coloured text laid on top. The letter then
    # carries its own dark rim wherever it goes, and stands at nearly five to one
    # against that rim whatever is behind it.
    OVERLAY_HALO_COLOR = "#000000"
    OVERLAY_HALO_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0),
                            (1, 0), (-1, 1), (0, 1), (1, 1))

    # What each overlay calls the list of canvas items it is holding. Set aside
    # while one is being drawn into a picture instead of onto the window, so
    # that the window's own items are neither redrawn nor lost.
    OVERLAY_ITEM_LISTS = ("_compass_items", "_locator_items", "_fov_items")

    # ---- the canvas, and what has been put on it ----

    def _overlay_canvas(self):
        """
        What the overlays draw on: the window's canvas, or None while there is
        no window - or, while one is being drawn into a picture, that picture.

        Asked for rather than remembered: an overlay may be switched on before
        the window is up, and every drawing method has to cope with there being
        nowhere to draw yet. Being asked for is also what lets `overlay_image`
        put something else in the canvas's place for the length of one drawing,
        so that the overlays can be drawn into a saved image or a video frame
        without knowing they are (see overlay_raster).
        """
        offscreen = getattr(self, "_overlay_surface", None)
        if offscreen is not None:
            return offscreen
        return getattr(self.rt, "_canvas", None) if self.rt is not None else None

    def _clear_overlay(self, items) -> list:
        """
        Take these canvas items off the canvas, and hand back the empty list to
        keep in their place.

        Returned rather than emptied in place so that the caller's own list is
        plainly reassigned at the call site, which is where a reader looks to
        see what an overlay is holding.
        """
        canvas = self._overlay_canvas()
        if canvas is not None:
            for item in items:
                canvas.delete(item)
        return []

    # ---- lettering that can be read over anything ----

    def _rimmed_text(self, canvas, x, y, text, fill, font, anchor="n") -> list:
        """
        Write text with a dark rim round it, and hand back every piece of it.

        Tk has no outline for text, so the rim is the same string written once
        more at each offset around it and the wanted colour laid over the lot.
        Cheap enough at this size, and the only thing that makes small lettering
        hold up over ground that is black in one place and white in another.

        The pieces are returned rather than kept here: each overlay keeps its
        own list of what it has drawn, so that it can take its own away again
        without disturbing the others sharing the canvas.
        """
        items = []
        for dx, dy in self.OVERLAY_HALO_OFFSETS:
            items.append(canvas.create_text(
                x + dx, y + dy, text=text, anchor=anchor,
                fill=self.OVERLAY_HALO_COLOR, font=font))
        items.append(canvas.create_text(
            x, y, text=text, anchor=anchor, fill=fill, font=font))
        return items

    # ---- noticing that the view has moved ----

    def _overlay_view_state(self, extra=()) -> Optional[tuple]:
        """
        A reading that changes whenever an overlay drawn from the camera would
        look different: where the camera stands and which way it faces, the
        Moon's own orientation, the mirrored view orientations, and the size of
        the window.

        Compared between one beat of the poll and the next, so that a view
        standing still is not redrawn five times a second for nothing.

        What is here is what every one of them depends on. An overlay that
        depends on more says so in `extra` - the lighting, the field of view,
        the setup of an eyepiece - rather than having it added here, because a
        reading that carried everything would have each of them redrawing at the
        others' news: the compass does not change with the zoom, and should not
        be redrawn when only the zoom has.
        """
        if self.rt is None:
            return None
        cam = self.rt.get_camera(self.CAMERA_NAME)
        canvas = self._overlay_canvas()
        return (tuple(cam["Eye"]), tuple(cam["Target"]), tuple(cam["Up"]),
                None if self.moon_rotation is None else self.moon_rotation.tobytes(),
                self.view_orientation,
                (canvas.winfo_width(), canvas.winfo_height()) if canvas is not None
                else None) + tuple(extra)

    def _schedule_overlay(self, tick, every_ms: Optional[int] = None):
        """
        Ask for the next beat of an overlay's poll, and say which timer it is so
        that it can be stopped again.

        Nothing is scheduled when there is no window: the overlay will be
        started properly when one appears.
        """
        if self.rt is None or getattr(self.rt, "_root", None) is None:
            return None
        return self.rt._root.after(
            self.OVERLAY_REFRESH_MS if every_ms is None else every_ms, tick)

    def _cancel_overlay(self, pending):
        """Stop a poll that is waiting to fire, if one is."""
        if pending is not None and self.rt is not None:
            root = getattr(self.rt, "_root", None)
            if root is not None:
                try:
                    root.after_cancel(pending)
                except tk.TclError:     # the window went before the timer did
                    pass
        return None

    # ---- the same overlays, drawn into a picture ----

    def _init_overlay_raster(self):
        """Reset the offscreen drawing state; called from MoonRenderer.__init__."""
        self._overlay_surface = None
        self._overlay_fonts = None

    def _overlay_offscreen(self, width: int, height: int):
        """
        A stand-in for this renderer that draws into a picture of that size
        instead of onto the window.

        It is a shallow copy, so it answers every question about the view the
        same way - the same camera, the same Moon, the same moment - while
        holding the picture and the list of what has been drawn on it as its
        own. That is what keeps the two drawings out of each other's way:
        during an export a frame is drawn here, on the raytracing thread, while
        the window's poll goes on redrawing the same three overlays on the main
        thread, and neither can now take the other's canvas or lose track of
        the other's items.

        The fonts are the exception, kept on the renderer and shared: they are
        measured once and never change, and the first measuring has to happen
        where there is a Tk to ask (see overlay_raster.FontBook).
        """
        if self._overlay_fonts is None:
            root = getattr(self.rt, "_root", None) if self.rt is not None else None
            self._overlay_fonts = FontBook(root)

        stand_in = copy.copy(self)
        stand_in._overlay_surface = OverlaySurface(
            width, height, self._overlay_fonts)
        for name in self.OVERLAY_ITEM_LISTS:
            setattr(stand_in, name, [])
        return stand_in

    def overlay_image(self, width: int, height: int):
        """
        Everything the canvas overlays currently show, drawn at that size as an
        RGBA array - or None when none of them is switched on.

        This is what puts them into an image saved with F12 and into the frames
        of an exported video, neither of which carries the canvas. They are
        drawn afresh rather than copied off the window, so that the picture may
        be a different size from it and so that this can be done from the
        raytracing thread, where Tk may not be touched at all.
        """
        showing = [(getattr(self, "compass_visible", False), "_draw_compass"),
                   (getattr(self, "locator_visible", False), "_draw_locator"),
                   (getattr(self, "fov_overlay_visible", False),
                    "_draw_fov_overlay")]
        if not any(visible for visible, _draw in showing):
            return None

        stand_in = self._overlay_offscreen(width, height)
        for visible, draw in showing:
            if visible:
                getattr(stand_in, draw)()
        return stand_in._overlay_surface.rgba()

    # ---- and into a saved image ----

    # What the ray tracer can be asked for, by the extension the file is given.
    # A depth it cannot write is not offered by the save dialog, so an unknown
    # extension only means the plain save should handle it.
    OVERLAY_SAVE_DEPTHS = {".jpg": "Bps8", ".jpeg": "Bps8", ".png": "Bps8",
                           ".bmp": "Bps8", ".tif": "Bps8", ".tiff": "Bps16"}
    # JPEG at its usual settings averages the colour of every second pixel
    # away, which is most of what a one-pixel line is; the overlays are drawn
    # in thin coloured lines, so the colour is kept at full resolution here
    OVERLAY_SAVE_JPEG_QUALITY = 95

    def save_render_with_overlays(self, filename: str, bps: str) -> bool:
        """
        Write the render to a file with the canvas overlays laid over it, and
        say whether it was written.

        The ray tracer's own save writes the buffer it rendered, which is the
        picture without them - they are canvas items over the top of it, and
        nothing of the canvas reaches the file. So the image is taken out
        instead, the overlays are drawn into one of their own at the same size,
        and the two are composited here.

        False means nothing was written and the plain save should do it: no
        overlay is showing, or the file is of a kind not handled here, or
        something went wrong - in which case the picture still gets saved,
        without the overlays, which is what it would have been anyway.
        """
        if self.rt is None:
            return False
        try:
            overlay = self.overlay_image(self.rt._width, self.rt._height)
            if overlay is None:                     # nothing switched on
                return False

            extension = os.path.splitext(filename)[1].lower()
            if self.OVERLAY_SAVE_DEPTHS.get(extension) != bps:
                return False

            base = self.rt.get_rt_output(bps=bps, channels="RGB")
            if base is None:
                return False

            merged = composite(base, overlay)
            # OpenCV writes blue first, and encoding to memory rather than
            # letting it open the file keeps paths it cannot spell working
            params = ([int(cv2.IMWRITE_JPEG_QUALITY), self.OVERLAY_SAVE_JPEG_QUALITY,
                       int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                       int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)]
                      if extension in (".jpg", ".jpeg") else [])
            ok, encoded = cv2.imencode(
                extension, np.ascontiguousarray(merged[:, :, ::-1]), params)
            if not ok:
                return False
            with open(filename, "wb") as out:
                out.write(encoded.tobytes())
        except Exception as e:
            # The picture matters more than the overlays on it
            print(f"Overlays not saved into the image ({e}); saving without them")
            return False
        return True
