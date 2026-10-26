"""
TimeHistoryMixin: undo and redo of the moves of the clock - Ctrl+Z and Ctrl+Y,
and their buttons on the time toolbar (T).

A planner's Go, a finder's result, a click on the profile's light change, F9,
Home: each takes the clock somewhere else in one go, and there was no way back
to where it was but setting it again by hand. Every move of the clock now
remembers the moment it left, the last TIME_HISTORY_SIZE of them, and Ctrl+Z
goes back through them as a browser's Back does, Ctrl+Y forward again.

A run of moves close together - a held Q or W, a held arrow of the time
toolbar, a step after a step - is one move: what is remembered is where the
clock stood before the run began, not every moment it passed through. The
clock moving on its own is not remembered at all: auto-advance following the
time of day (see time_history_paused), and a video export, whose clock is
driven from the raytracing thread.
"""

from contextlib import contextmanager


class TimeHistoryMixin:
    """Mixin keeping the moments the clock was moved from, to go back to them."""

    # How many moments are kept to go back to
    TIME_HISTORY_SIZE = 10
    # Moves of the clock closer together than this make one run (see above):
    # past the gap between the steps of a held key, short of two presses
    TIME_HISTORY_RUN_MS = 700

    def _init_time_history(self):
        """Reset the history; called from MoonRenderer.__init__."""
        self._time_undo = []        # moments to go back to, the latest last
        self._time_redo = []        # and forward to, after going back
        self._time_run = None       # the timer ending a run of moves, while one goes on
        self._time_history_quiet = 0

    @contextmanager
    def time_history_paused(self):
        """Moves of the clock made inside this are not remembered."""
        self._time_history_quiet += 1
        try:
            yield
        finally:
            self._time_history_quiet -= 1

    def note_time_change(self, previous):
        """
        The clock has moved from previous to dt_local: remember previous to
        go back to, unless this move continues a run (see above), and forget
        what there was to go forward to - a new move starts a new future, as a
        new page does in a browser.
        """
        # Not by asking which thread this is: PlotOptiX runs the window's
        # event loop on a thread of its own rather than Python's main one, so
        # every move made from the window comes from there; the export, the
        # one mover from elsewhere, says so itself
        if (previous is None or previous == self.dt_local or self._time_history_quiet
                or getattr(self, "_video_export", None) is not None
                or self.rt is None):
            return
        if self._time_run is None:
            self._time_undo.append(previous)
            del self._time_undo[:-self.TIME_HISTORY_SIZE]
            self._time_redo.clear()
            self.sync_time_history_buttons()
        else:
            self.rt._root.after_cancel(self._time_run)
        self._time_run = self.rt._root.after(self.TIME_HISTORY_RUN_MS, self._end_time_run)

    def _end_time_run(self):
        self._time_run = None

    def record_time_jump(self, previous):
        """
        Remember previous to go back to, the clock having been moved from it
        by one deliberate act - Enter in a date or time field, or an arrow of
        one let go - as a move of its own, whatever came just before: unlike
        note_time_change, it never joins a run.
        """
        if (previous is None or previous == self.dt_local
                or getattr(self, "_video_export", None) is not None or self.rt is None):
            return
        if self._time_run is not None:
            self.rt._root.after_cancel(self._time_run)
            self._time_run = None
        self._time_undo.append(previous)
        del self._time_undo[:-self.TIME_HISTORY_SIZE]
        self._time_redo.clear()
        self.sync_time_history_buttons()

    @property
    def can_undo_time(self) -> bool:
        return bool(self._time_undo)

    @property
    def can_redo_time(self) -> bool:
        return bool(self._time_redo)

    def undo_time(self):
        """Ctrl+Z: the clock back to where it was before its last move."""
        self._step_time_history(self._time_undo, self._time_redo)

    def redo_time(self):
        """Ctrl+Y: the clock forward again to where Ctrl+Z took it from."""
        self._step_time_history(self._time_redo, self._time_undo)

    def _step_time_history(self, take_from: list, keep_in: list):
        """
        Go to the latest moment of take_from, keeping the moment left in
        keep_in to come back to; this move itself is not a new one.
        """
        if not take_from or getattr(self, "_video_export", None) is not None:
            return
        if self._time_run is not None:
            self.rt._root.after_cancel(self._time_run)
            self._time_run = None
        when = take_from.pop()
        keep_in.append(self.dt_local)
        del keep_in[:-self.TIME_HISTORY_SIZE]
        with self.time_history_paused():
            self._go_to_moment(when)
        self.sync_time_history_buttons()
