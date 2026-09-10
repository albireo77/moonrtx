"""
The screen the app draws on, and the declaration that it means real pixels.

Both live here rather than in the renderer because they are wanted before one
exists: main asks how wide the star map has to be while it is still deciding
what to download, and the DPI declaration has to be made before any window is
created at all - including the hidden one screen_size itself opens.
"""

import ctypes
import sys
import tkinter as tk
from typing import Optional

# The star map is loaded several times wider than the window: it wraps the whole
# sky, so only a fraction of it is ever on screen at once.
STARMAP_WIDTH_FACTOR = 6

# Screen widths the star map is prepared at. The processed map is cached under
# the width it was made for, so a laptop whose panel is 1728 pixels wide would
# keep a cache to itself and build another the first time it is plugged into an
# ordinary 1920 monitor - some minutes of work on a 16k source, and a second
# copy of the result on disk, for the sake of a couple of hundred pixels.
# Rounding up to a standard width has both share one. Anything wider than the
# last step is taken as it comes.
SCREEN_WIDTH_STEPS = (1280, 1366, 1440, 1600, 1920, 2256, 2560, 2880,
                      3440, 3840, 5120, 7680)

# What to leave for a taskbar or panel where the system will not say how much.
# Only reached away from Windows, there being nothing portable to ask: a window
# a little short of the screen is better than one with its status bar hidden
# underneath a panel.
ASSUMED_PANEL_HEIGHT = 40

_screen_size: Optional[tuple] = None


class _Rect(ctypes.Structure):
    """A Windows RECT, for the work area query below."""
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def make_dpi_aware():
    """
    Tell Windows this process works in real pixels. Call it before any window.

    A process that says nothing is taken for one written before display scaling
    existed, and Windows protects it by lying: on a 4K screen set to 150% it
    reports a 2560 x 1440 desktop, lets the app draw at that size, and stretches
    the result up to the real 3840 x 2160 panel. For most programs that is a
    kindness. For this one it is the worst of both - every pixel the ray tracer
    spent time on is thrown away and replaced by an interpolated one, and the
    detail the whole app exists to show is smeared before it is ever seen.

    Said plainly, the window is given the real panel to draw on. Tk reads the
    true DPI at the same time and scales its own lettering to match, the fonts
    here all being asked for in points, so the dialogs and the status bar keep
    the size they had and gain the sharpness. The picture costs more to make -
    on that 4K screen, two and a quarter times the pixels of the stretched one -
    which is the price of it being real.

    The system-wide declaration is the one made, not the per-monitor one. This
    app opens a single window maximised on the primary screen, and Tk does not
    resize its lettering when a window is dragged to a monitor of another DPI.
    Per-monitor awareness would therefore trade a picture Windows rescales for
    us on that rare drag - blurred, but the right size - for lettering that
    comes out the wrong size there and stays wrong.

    Failure is not worth reporting: it means this is not Windows, or that the
    answer has already been given - by an earlier call, or by the compatibility
    settings of a shortcut - and it cannot be given twice.
    """
    global _screen_size
    _screen_size = None         # anything measured before was measured in the lie

    if sys.platform != "win32":
        return

    try:
        # PROCESS_SYSTEM_DPI_AWARE, Windows 8.1 and later
        if ctypes.windll.shcore.SetProcessDpiAwareness(1) == 0:
            return
    except (AttributeError, OSError):
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()        # the original, Vista
    except (AttributeError, OSError):
        pass


def _work_area(screen_width: int, screen_height: int) -> tuple:
    """
    The screen less whatever the desktop keeps for itself - a taskbar, a dock, a
    panel - so that the window and its status bar are not left underneath it.

    Windows will say exactly, and says it in real pixels once the declaration
    above has been made, so a taskbar that is 40 pixels tall unscaled and 60 at
    150% is reported as the height it really is rather than as a guess that
    happens to be right on one machine. It also covers a taskbar moved to the
    side of the screen, and one set to hide itself, neither of which a guess at
    a height can. Elsewhere there is nothing portable to ask and a strip is
    assumed. An answer bigger than the screen, or an empty one, is not believed.
    """
    if sys.platform == "win32":
        try:
            rect = _Rect()
            SPI_GETWORKAREA = 0x0030
            if ctypes.windll.user32.SystemParametersInfoW(
                    SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                if 0 < width <= screen_width and 0 < height <= screen_height:
                    return width, height
        except (AttributeError, OSError):
            pass

    return screen_width, max(screen_height - ASSUMED_PANEL_HEIGHT, 1)


def bring_to_front(window) -> None:
    """
    Make a window the one the keyboard is talking to.

    Tk's lift and focus_force are not enough on Windows, and the reason is worth
    writing down. SetForegroundWindow is refused, silently, to a process that is
    not already the foreground one - and the renderer is never that: it is
    started from the launcher or from a console, and does not open its window
    until the maps have loaded, by which time any right to come forward that it
    inherited from its parent is long spent. focus_force only moves Tk's own
    idea of the focus, which is a different thing from the desktop's and not
    what decides where a key press is delivered.

    Measured with another process holding the foreground: lift and focus_force
    left the window in the background, flipping the topmost attribute left it in
    the background, and only the call below brought it forward. The way through
    is to borrow the foreground window's input queue for the length of the call.
    Two threads sharing a queue share a notion of which window is active, so the
    refusal does not apply. Nothing outside this process is changed by it, and
    nothing is left attached afterwards.

    Anywhere but Windows, and if any part of it fails, Tk's own way is all that
    happens - and the worst that leaves is a window in the background, which is
    where it was already.

    Parameters
    ----------
    window : tkinter.Misc
        The window to bring forward. It must be on the screen already: the
        handle Windows knows it by does not exist before that.
    """
    window.lift()
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            handle = int(window.wm_frame(), 16)
            theirs = user32.GetWindowThreadProcessId(
                user32.GetForegroundWindow(), None)
            ours = ctypes.windll.kernel32.GetCurrentThreadId()
            attached = user32.AttachThreadInput(theirs, ours, True)
            try:
                user32.BringWindowToTop(handle)
                user32.SetForegroundWindow(handle)
            finally:
                if attached:
                    user32.AttachThreadInput(theirs, ours, False)
        except (AttributeError, OSError, ValueError, tk.TclError):
            pass
    # Which widget inside the window the keys then reach is Tk's business, and
    # this is what settles it. PlotOptiX binds its key handler with bind_all, so
    # anything in the window will do, the window itself included.
    window.focus_force()


def screen_size() -> tuple:
    """
    The usable width and height of the primary screen, in real pixels.

    Read through a hidden Tk window - the same interpreter that will size the
    renderer window - and then kept: it is asked for once before the renderer
    exists and again while it is being built, and standing up a whole Tk
    interpreter twice to be told the same number is waste. make_dpi_aware
    forgets it, so a size measured before the declaration is never handed back
    after it.
    """
    global _screen_size
    if _screen_size is None:
        root = tk.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        _screen_size = _work_area(*size)
    return _screen_size


def starmap_target_width() -> int:
    """
    The width the star map is prepared at, and so the key its cache is stored
    under (see data_loader.load_starmap).

    A module-level function because main needs the answer before a renderer, and
    its window, exists: it decides from this whether the source has to be
    downloaded at all. The renderer asks the same question later, and has to be
    given the same answer, or it would rebuild what main found already there.
    """
    width = screen_size()[0]
    for step in SCREEN_WIDTH_STEPS:
        if width <= step:
            width = step
            break
    return width * STARMAP_WIDTH_FACTOR
