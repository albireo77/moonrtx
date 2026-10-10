"""
SettingsMixin: the choices made in the windows, kept from one run to the next.

The field-of-view setup is the one that costs most to lose - a telescope, an
eyepiece or a camera typed in again every run - and the ticks and spans left in
the planner, the graph and the profile go with it, and whether the scale bar
(Insert) is on. They are read when the
renderer starts and written when its window is closed, into a file of their own
beside the caches in the data folder.

Where a window was left on the screen is not kept: the same program is run on
screens of different sizes and scaling, and a place remembered on one can be off
the edge of the other.

What is read back is checked before it is used, each value against what its
window can show. A file from an older version, edited by hand, or cut short is
no reason not to start, so anything that does not pass is left at its default
and the rest is taken.
"""

import json
import math
import os


class SettingsMixin:
    """Mixin keeping the windows' choices between runs."""

    SETTINGS_FILE_NAME = "renderer_settings.json"

    def _settings_path(self) -> str:
        # Imported here, as skyfield_utils does, since main imports the renderer
        from moonrtx.main import DATA_DIRECTORY_PATH
        return os.path.join(DATA_DIRECTORY_PATH, self.SETTINGS_FILE_NAME)

    def _settings_checks(self) -> dict:
        """
        Every setting kept, by the attribute that holds it, with what a value
        read back has to be to be taken.
        """
        def flag(value):
            return isinstance(value, bool)

        return {
            "fov_setup": None,                      # checked field by field
            "_clair_obscur_filter": lambda v: isinstance(v, str),
            "_clair_obscur_visible_only": flag,
            "_eclipse_visible_only": flag,
            "_eclipse_penumbral": flag,
            "_planner_dark_only": flag,
            "_graph_view": lambda v: v in ("keep", "standard", "centre"),
            "_label_on_moon": flag,
            "_graph_show_moon_alt": flag,
            "_graph_span": lambda v: (isinstance(v, int) and not isinstance(v, bool)
                                      and v in self.GRAPH_ZOOM_SPANS),
            "_graph_step_minutes": lambda v: v is None or (
                isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 1440),
            "_profile_show_place": flag,
            "scale_bar_visible": flag,
        }

    def _fov_setup_from(self, stored) -> dict:
        """
        The field-of-view setup read back over the default one, field by field:
        each that is what the dialog itself would accept replaces the default.
        """
        setup = dict(self.fov_setup)
        if not isinstance(stored, dict):
            return setup
        for key, value in stored.items():
            if key not in setup:
                continue
            if key == "mode":
                if value in ("eyepiece", "camera"):
                    setup[key] = value
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                continue
            # Any angle is a rotation; every length and field must be above zero
            if key == "rotation_deg" or value > 0:
                setup[key] = float(value)
        return setup

    def _load_settings(self):
        """Take up the choices the last run left; called from MoonRenderer.__init__."""
        path = self._settings_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                stored = json.load(f)
        except (OSError, ValueError) as e:
            print(f"Settings not read from {path} ({e}); using the defaults")
            return
        if not isinstance(stored, dict):
            return

        for name, check in self._settings_checks().items():
            if name not in stored:
                continue
            value = stored[name]
            if name == "fov_setup":
                self.fov_setup = self._fov_setup_from(value)
            elif check(value):
                setattr(self, name, value)

    def save_settings(self):
        """
        Write the choices down for the next run. Written to a file aside and
        then put in place, so a run ended halfway through writing leaves the
        last good file rather than half of one.
        """
        path = self._settings_path()
        settings = {name: getattr(self, name) for name in self._settings_checks()}
        temp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            os.replace(temp_path, path)
        except (OSError, TypeError, ValueError) as e:
            print(f"Settings not saved to {path} ({e})")
            try:
                os.remove(temp_path)
            except OSError:
                pass
