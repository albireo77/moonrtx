"""
The canvas overlays, drawn into an image instead of onto the window.

The compass, the locator and the field-of-view frame are Tk canvas items laid
over the render rather than geometry inside it, which is why they have never
appeared in an image saved with F12 or in an exported video: those carry the
ray-traced buffer, and the canvas is not part of it.

What is here draws them a second time, into a picture. `OverlaySurface` answers
to the handful of canvas calls the overlays make - create_line, create_oval,
create_polygon, create_text, and the two that ask how big the canvas is - so the
overlays can be run against it exactly as they are run against the window,
without knowing the difference and without a second copy of what they draw.
Which also means it holds no Tk of its own: it can be used from the raytracing
thread, which is where video frames are composed.

Two departures from what Tk puts on the screen, both deliberate:

  - Everything is drawn at SUPERSAMPLE times the asked-for size and scaled back
    down, so curves that step visibly on the screen come out smooth in a saved
    picture. Lettering is exempted from the growth that would otherwise cause:
    the size given to FreeType is chosen so that the text measures what Tk
    measured it at, times the supersampling, and it is placed from Tk's own
    ascent and line height. It therefore lands where it does on the screen,
    letter for letter, only cleaner.

  - Tk fills a stippled shape by painting every other pixel of it. At a
    supersampled size that pattern would be finer than the picture it ends up
    in, and scaling down would turn it into a flat wash anyway - which is what
    the eye makes of it on the screen. So a stipple is drawn as the wash
    directly: gray50 as half-opaque, gray25 as a quarter, and so on.
"""

import threading

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

# How much larger than the finished picture everything is drawn before being
# scaled back down. Three is enough that a circle reads as a circle; the cost is
# an RGBA buffer nine times the frame, which for a screenshot is momentary and
# for a video frame is nothing beside the accumulation cycle it accompanies.
SUPERSAMPLE = 3

# What Tk's stipple bitmaps come to when the eye stops resolving the pattern
STIPPLE_ALPHA = {"gray12": 0.125, "gray25": 0.25, "gray50": 0.5, "gray75": 0.75}

# Tk names a font by family, size in points and style; FreeType wants a file
FONT_FILES = {
    ("consolas", False): "consola.ttf", ("consolas", True): "consolab.ttf",
    ("courier new", False): "cour.ttf", ("courier new", True): "courbd.ttf",
    ("segoe ui", False): "segoeui.ttf", ("segoe ui", True): "segoeuib.ttf",
    ("arial", False): "arial.ttf", ("arial", True): "arialbd.ttf",
    ("tahoma", False): "tahoma.ttf", ("tahoma", True): "tahomabd.ttf",
}
FALLBACK_FONT_FILES = ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf")

# Points to pixels where there is no Tk to ask. Only reached when the very first
# lettering is drawn off the main thread, which in this program does not happen:
# the first frame of an export is composed before the raytracing thread has it.
DEFAULT_PIXELS_PER_POINT = 96.0 / 72.0

# The string the FreeType size is calibrated against. Monospaced text needs only
# one character; a mixed sample keeps a proportional family honest as well.
CALIBRATION_SAMPLE = "0123456789 NSEW lat lon +-."


def _colour(value, alpha: float = 1.0):
    """
    A Tk colour as RGBA, or None where Tk would draw nothing.

    An empty string is how a canvas item says it has no fill or no outline, and
    it must not become black.
    """
    if not value:
        return None
    try:
        rgb = ImageColor.getrgb(value)[:3]
    except ValueError:
        return None
    return rgb + (max(0, min(255, int(round(255 * alpha)))),)


def _points(args, scale: float):
    """
    The coordinates of a canvas item as a list of scaled (x, y) pairs.

    Tk takes them either loose - create_line(x0, y0, x1, y1) - or as one
    sequence, and the overlays use both.
    """
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        flat = list(args[0])
    else:
        flat = list(args)
    return [(float(flat[i]) * scale, float(flat[i + 1]) * scale)
            for i in range(0, len(flat) - 1, 2)]


class FontBook:
    """
    Fonts for the surface, measured as Tk measures them.

    Text is placed from Tk's ascent and line height rather than FreeType's, so
    that a label sits where the window puts it, and the FreeType size is chosen
    so the text is as wide as Tk drew it. Both are worked out once per font and
    kept: the first is a Tk call, which may only be made from the main thread,
    and by the time any other thread wants one the answer is already here.
    """

    def __init__(self, root=None):
        self._root = root
        self._metrics = {}          # spec -> (px, ascent, descent, linespace)
        self._faces = {}            # (file, size) -> ImageFont

    # ---- what Tk makes of a font ----

    @staticmethod
    def _parse(spec):
        """A Tk font spec as (family, points, bold)."""
        if isinstance(spec, str):
            parts = spec.split()
            family = parts[0] if parts else "Consolas"
            points = (int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit()
                      else 10)
            bold = any(p.lower() == "bold" for p in parts[2:])
            return family, points, bold
        spec = tuple(spec)
        family = spec[0] if spec else "Consolas"
        points = int(spec[1]) if len(spec) > 1 else 10
        bold = any(str(s).lower() == "bold" for s in spec[2:])
        return family, points, bold

    def _tk_metrics(self, spec):
        """
        Pixel size, ascent, descent and line height of a font, from Tk itself
        where it can be asked and from the outline where it cannot.

        A negative size is Tk's way of giving one in pixels already; a positive
        one is in points and follows the display.
        """
        key = tuple(spec) if not isinstance(spec, str) else spec
        if key in self._metrics:
            return self._metrics[key]

        family, points, bold = self._parse(spec)
        px = (int(round(points * DEFAULT_PIXELS_PER_POINT)) if points > 0
              else abs(points))
        got = None
        if (self._root is not None
                and threading.current_thread() is threading.main_thread()):
            try:
                import tkinter.font as tkfont
                metrics = tkfont.Font(root=self._root, font=spec).metrics()
                if points > 0:
                    px = int(round(points * float(self._root.winfo_fpixels("1p"))))
                got = (px, metrics["ascent"], metrics["descent"],
                       metrics["linespace"])
            except Exception:
                got = None
        if got is None:
            ascent, descent = self._face(family, bold, px).getmetrics()
            got = (px, ascent, descent, ascent + descent)

        self._metrics[key] = got
        return got

    # ---- and what FreeType is asked for in its place ----

    def _face(self, family: str, bold: bool, size: int):
        """The nearest outline file to a family and weight, at that pixel size."""
        named = FONT_FILES.get((family.lower(), bold))
        candidates = [named] if named else []
        candidates.append(family.lower().replace(" ", "")
                          + ("bd.ttf" if bold else ".ttf"))
        candidates.extend(FALLBACK_FONT_FILES)
        for name in candidates:
            cached = self._faces.get((name, size))
            if cached is not None:
                return cached
            try:
                face = ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
            self._faces[(name, size)] = face
            return face
        return ImageFont.load_default()

    def face(self, spec, scale: float):
        """
        The face to draw a font with at this magnification, and where the
        baseline of a line of it sits below the top of its box.

        The size is not simply the pixel size times the magnification: FreeType
        rounds an outline to whole pixels of advance, and the rounding at the
        larger size need not agree with Tk's at the smaller one. So of the sizes
        near it, the one measuring closest to what Tk measured times the
        magnification is taken - the text then grows with the picture and with
        nothing else.

        Returns (face, ascent, linespace), the last two already magnified.
        """
        px, ascent, _descent, linespace = self._tk_metrics(spec)
        family, _points, bold = self._parse(spec)

        wanted = self._face(family, bold, px).getlength(CALIBRATION_SAMPLE) * scale
        best, best_off = None, None
        for size in range(max(1, int(px * scale) - 3), int(px * scale) + 4):
            face = self._face(family, bold, size)
            off = abs(face.getlength(CALIBRATION_SAMPLE) - wanted)
            if best_off is None or off < best_off:
                best, best_off = face, off
        return best, ascent * scale, linespace * scale


class OverlaySurface:
    """
    Something the canvas overlays can draw on that is a picture, not a window.

    Only the calls the three of them actually make are answered. Anything else
    would be a silent omission from a saved image, so it is left to fail rather
    than quietly do nothing - except `delete`, which the overlays use to take
    down what they drew last time and which here has nothing to take down.
    """

    def __init__(self, width: int, height: int, fonts: FontBook,
                 scale: int = SUPERSAMPLE):
        self._width, self._height = int(width), int(height)
        self._scale = scale
        self._image = Image.new(
            "RGBA", (self._width * scale, self._height * scale), (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._image)
        self._fonts = fonts
        self._next_id = 0

    # ---- the canvas questions the overlays ask ----

    def winfo_width(self) -> int:
        return self._width

    def winfo_height(self) -> int:
        return self._height

    def delete(self, item):
        """Nothing here outlives one drawing, so there is nothing to remove."""

    # ---- and the marks they make ----

    def _item(self) -> int:
        """A number standing in for a canvas item id, so the caller has one."""
        self._next_id += 1
        return self._next_id

    def _line_width(self, width) -> int:
        return max(1, int(round(float(width) * self._scale)))

    def create_line(self, *args, fill=None, width=1, **_ignored):
        colour = _colour(fill)
        points = _points(args, self._scale)
        if colour is not None and len(points) > 1:
            self._draw.line(points, fill=colour, width=self._line_width(width),
                            joint="curve")
        return self._item()

    def create_oval(self, x0, y0, x1, y1, fill="", outline="", width=1,
                    stipple=None, **_ignored):
        scale = self._scale
        box = [min(x0, x1) * scale, min(y0, y1) * scale,
               max(x0, x1) * scale, max(y0, y1) * scale]
        inside = _colour(fill, STIPPLE_ALPHA.get(stipple, 1.0))
        edge = _colour(outline)
        if inside is not None or edge is not None:
            self._draw.ellipse(box, fill=inside, outline=edge,
                               width=self._line_width(width))
        return self._item()

    def create_polygon(self, *args, fill="", outline="", width=1, stipple=None,
                       **_ignored):
        points = _points(args, self._scale)
        inside = _colour(fill, STIPPLE_ALPHA.get(stipple, 1.0))
        edge = _colour(outline)
        if inside is not None and len(points) > 2:
            self._draw.polygon(points, fill=inside)
        # Drawn as a closed line of its own: PIL's polygon outline has no width
        if edge is not None and len(points) > 1:
            self._draw.line(points + [points[0]], fill=edge,
                            width=self._line_width(width), joint="curve")
        return self._item()

    def create_text(self, x, y, text="", anchor="center", fill=None, font=None,
                    **_ignored):
        colour = _colour(fill)
        if colour is None or not text:
            return self._item()

        face, ascent, linespace = self._fonts.face(font, self._scale)
        x, y = float(x) * self._scale, float(y) * self._scale
        wide = face.getlength(text)

        # Tk anchors a text item by its box; the letters are laid out from the
        # top of that box down to the baseline, which is what PIL draws from
        anchor = (anchor or "center").lower()
        if "w" in anchor:
            left = x
        elif "e" in anchor:
            left = x - wide
        else:
            left = x - wide / 2
        if "n" in anchor:
            top = y
        elif "s" in anchor:
            top = y - linespace
        else:
            top = y - linespace / 2

        self._draw.text((left, top + ascent), text, font=face, fill=colour,
                        anchor="ls")
        return self._item()

    # ---- what came of it ----

    def image(self):
        """
        The finished overlay, scaled back down to the size asked for.

        By averaging each block of pixels down to the one it stands for, which
        is what supersampling means and is also the quickest way of asking:
        a general resize of a frame-sized picture costs four times as much and
        rings around every edge into the bargain.
        """
        if self._scale == 1:
            return self._image
        return self._image.reduce(self._scale)

    def rgba(self) -> np.ndarray:
        """The finished overlay as a height x width x 4 array of bytes."""
        return np.asarray(self.image(), dtype=np.uint8)


def composite(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Lay an 8-bit RGBA overlay over an image, which may be of any depth.

    The image keeps its own type and range: the overlay is 8-bit because that is
    what a screen is, and a 16-bit save is not made shallower by carrying one.
    """
    if overlay.shape[:2] != base.shape[:2]:
        raise ValueError("overlay %r does not match the image %r"
                         % (overlay.shape[:2], base.shape[:2]))

    top = float(np.iinfo(base.dtype).max) if base.dtype.kind in "ui" else 1.0
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    colour = overlay[:, :, :3].astype(np.float32) / 255.0 * top
    out = base.astype(np.float32) * (1.0 - alpha) + colour * alpha
    if base.dtype.kind in "ui":
        return np.clip(np.rint(out), 0, top).astype(base.dtype)
    return out.astype(base.dtype)
