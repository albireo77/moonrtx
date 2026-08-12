import math
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield import almanac
from skyfield.api import wgs84
from skyfield.positionlib import ICRF
from skyfield.framelib import ecliptic_frame, true_equator_and_equinox_of_date
from skyfield.trigonometry import position_angle_of

from moonrtx.skyfield_utils import (
    SKYFIELD_MOON_FRAME_END_UTC,
    SKYFIELD_MOON_FRAME_START_UTC,
    skyfield_ephemeris,
    skyfield_moon_frame,
    skyfield_timescale,
)
from moonrtx.shared_types import ClairObscurEvent, MoonEphemeris, Observer

RENDERER_TO_SKYFIELD_BODY_MATRIX = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=float,
)


def init(observer: Observer):
    global _observer, _observer_lat, _earth, _moon, _sun, _moon_frame, _timescale
    global _moon_phases_fn, _lunation_bounds
    ephemeris = skyfield_ephemeris()
    _moon_frame = skyfield_moon_frame()
    _timescale = skyfield_timescale()
    _earth = ephemeris["earth"]
    _moon = ephemeris["moon"]
    _sun = ephemeris["sun"]
    _observer = _earth + wgs84.latlon(
        latitude_degrees=observer.lat,
        longitude_degrees=observer.lon,
        elevation_m=observer.elevation_m
    )
    _observer_lat = observer.lat
    _moon_phases_fn = almanac.moon_phases(ephemeris)
    _lunation_bounds = None


def _moon_age_days(time) -> float:
    """
    Time since the previous new moon, in days.

    The true age, not the mean-rate estimate from elongation, which can differ
    by more than half a day. The new moons bracketing the current lunation
    are found with an almanac search and cached, so repeated calls during
    time animation cost only a comparison until the date leaves the cached lunation.
    """
    global _lunation_bounds
    tt = time.tt
    if _lunation_bounds is None or not (_lunation_bounds[0] <= tt < _lunation_bounds[1]):
        t0 = _timescale.tt_jd(tt - 31.0)
        t1 = _timescale.tt_jd(tt + 31.0)
        times, phases = almanac.find_discrete(t0, t1, _moon_phases_fn)
        new_moons = times.tt[phases == 0]   # phase index 0 = new moon
        prev = new_moons[new_moons <= tt].max()
        upcoming = new_moons[new_moons > tt]
        nxt = upcoming.min() if upcoming.size else prev + 29.53
        _lunation_bounds = (float(prev), float(nxt))
    return tt - _lunation_bounds[0]


def _validate_supported_datetime(dt_local: datetime) -> datetime:
    dt_utc = dt_local.astimezone(timezone.utc)
    if dt_utc < SKYFIELD_MOON_FRAME_START_UTC or dt_utc > SKYFIELD_MOON_FRAME_END_UTC:
        raise ValueError(
            "Moon ephemeris supports dates from "
            f"{SKYFIELD_MOON_FRAME_START_UTC.isoformat()} through "
            f"{SKYFIELD_MOON_FRAME_END_UTC.isoformat()} with the bundled Skyfield kernels; "
            f"received {dt_utc.isoformat()}."
        )
    return dt_utc


def _wrap_signed_degrees(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _colongitude_from_subsolar_longitude(subsolar_lon_deg: float) -> float:
    return (90.0 - _wrap_signed_degrees(subsolar_lon_deg)) % 360.0


def _normalize_np(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _parallactic_angle_deg(hour_angle_deg: float, dec_deg: float, lat_deg: float) -> float:
    hour_angle_rad = math.radians(hour_angle_deg)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)
    return math.degrees(math.atan2(
        math.sin(hour_angle_rad),
        math.tan(lat_rad) * math.cos(dec_rad) - math.sin(dec_rad) * math.cos(hour_angle_rad),
    ))


def _latlon_from_icrf(pos_au: np.ndarray, R_icrf_to_body: np.ndarray) -> tuple[float, float]:
    """Convert an ICRF position vector (AU) to body-frame (lat_deg, lon_deg)."""
    body_vec = R_icrf_to_body @ pos_au
    r = np.linalg.norm(body_vec)
    return (
        math.degrees(math.asin(body_vec[2] / r)),
        math.degrees(math.atan2(body_vec[1], body_vec[0])),
    )


def _rotation_matrix(
    R_moon: np.ndarray,
    R_equator: np.ndarray,
    moon_ra_deg: float,
    moon_dec_deg: float,
    q_deg: float,
) -> np.ndarray:
    ra_rad = math.radians(moon_ra_deg)
    dec_rad = math.radians(moon_dec_deg)
    sin_ra, cos_ra = math.sin(ra_rad), math.cos(ra_rad)
    sin_dec, cos_dec = math.sin(dec_rad), math.cos(dec_rad)

    moon_sight_date = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec], dtype=float)
    east_cel = np.array([-sin_ra, cos_ra, 0.0], dtype=float)
    north_cel = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec], dtype=float)

    q_rad = math.radians(q_deg)
    up_view = _normalize_np(math.sin(q_rad) * east_cel + math.cos(q_rad) * north_cel)
    right_view = _normalize_np(np.cross(moon_sight_date, up_view))
    view_basis = np.vstack([right_view, moon_sight_date, up_view])

    body_to_date = R_equator @ R_moon.T
    rotation_matrix = view_basis @ body_to_date @ RENDERER_TO_SKYFIELD_BODY_MATRIX
    return rotation_matrix


def _phase_name(moon: ICRF, sun: ICRF) -> str:

    _, moon_ecl_lon, _ = moon.frame_latlon(ecliptic_frame)
    _, sun_ecl_lon, _ = sun.frame_latlon(ecliptic_frame)
    delta = (moon_ecl_lon.degrees - sun_ecl_lon.degrees) % 360.0
    
    if (delta < 0.5) or (delta > 359.5):
        return "New Moon"
    elif delta < 89.5:
        return "Waxing Crescent"
    elif delta < 90.5:
        return "First Quarter"
    elif delta < 179.5:
        return "Waxing Gibbous"
    elif delta < 180.5:
        return "Full Moon"
    elif delta < 269.5:
        return "Waning Gibbous"
    elif delta < 270.5:
        return "Last Quarter"
    else:
        return "Waning Crescent"


def _body_altitude_at_feature(sub_lat_deg: np.ndarray, sub_lon_deg: np.ndarray,
                              lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Altitude of a body above the local lunar horizon at a selenographic
    location: 90 degrees minus the angular distance to the body's sub-point
    (spherical law of cosines). Vectorized over the sub-point arrays.

    With the subsolar point this is the Sun's altitude, which sets shadow
    length. With the sub-Earth point (that is, the libration) it measures how
    far inside the limb the feature lies, and equally how much it is
    foreshortened - the feature is compressed by cos of the angular distance,
    i.e. by sin of this altitude, across the line of sight.
    """
    b0 = np.radians(sub_lat_deg)
    l0 = np.radians(sub_lon_deg)
    b = math.radians(lat_deg)
    l = math.radians(lon_deg)
    sin_alt = np.sin(b0) * math.sin(b) + np.cos(b0) * math.cos(b) * np.cos(l - l0)
    return np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


def _scan_times(start_local: datetime, days: int, step_minutes: int) -> tuple:
    """
    Sample times for a planner scan, as a list of UTC datetimes and the
    matching Skyfield time array (clamped to the bundled kernel range).
    """
    start_utc = _validate_supported_datetime(start_local)
    end_utc = min(start_utc + timedelta(days=days), SKYFIELD_MOON_FRAME_END_UTC)
    n = max(int((end_utc - start_utc).total_seconds() // (step_minutes * 60)), 0)
    dts = [start_utc + timedelta(minutes=step_minutes * i) for i in range(n + 1)]
    return dts, _timescale.from_datetimes(dts)


def _sub_point(target, t, moon_at) -> tuple:
    """
    Selenographic latitude/longitude of the point where `target` stands
    overhead, vectorized over the time array. Geometric positions are enough:
    light-time shifts the sub-point by arcseconds, far below the hour-level
    resolution of a planner scan.
    """
    vec = (target.at(t) - moon_at).position.au
    body_vec = np.einsum('ijn,jn->in', _moon_frame.rotation_at(t), vec)
    r = np.linalg.norm(body_vec, axis=0)
    return (np.degrees(np.arcsin(body_vec[2] / r)),
            np.degrees(np.arctan2(body_vec[1], body_vec[0])))


def _split_windows(idx: np.ndarray) -> list:
    """Split indices of qualifying samples into runs of consecutive ones."""
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return [idx[s:e + 1] for s, e in zip(starts, ends)]


def find_terminator_windows(start_local: datetime, days: int,
                            feature_lat: float, feature_lon: float,
                            step_minutes: int = 60,
                            sun_alt_max: float = 12.0,
                            moon_alt_min: float = 5.0) -> list[dict]:
    """
    Find upcoming windows when a Moon feature can be observed near the
    terminator: the Sun is low over the feature (0..sun_alt_max degrees, so
    the terrain is lit with long shadows) while the Moon stands at least
    moon_alt_min degrees above the observer's horizon.

    The scan is a single vectorized Skyfield evaluation over the whole range
    (a per-sample calculate_moon_ephemeris loop would take tens of seconds
    for a 60-day scan; this takes a fraction of one). Consecutive qualifying
    samples are merged into windows.

    Parameters
    ----------
    start_local : datetime
        Timezone-aware start of the scan
    days : int
        Scan length in days (clamped to the bundled kernel range)
    feature_lat, feature_lon : float
        Selenographic position of the feature in degrees
    step_minutes : int
        Sample spacing; window edges are accurate to this resolution
    sun_alt_max : float
        Highest Sun altitude over the feature still considered "near the
        terminator" (12 degrees is roughly a day past sunrise/before sunset)
    moon_alt_min : float
        Minimum Moon altitude at the observer site

    Returns
    -------
    list[dict]
        One dict per window, sorted by time, with keys: "start", "end",
        "best" (UTC datetimes; "best" is the sample with the highest Moon
        altitude), "event" ("sunrise" or "sunset"), "sun_alt" and "moon_alt"
        (degrees at "best"), "observer_sun_alt" (degrees at "best", for
        judging sky darkness).
    """
    dts, t = _scan_times(start_local, days, step_minutes)

    observer_at = _observer.at(t)
    moon_alt, _, _ = observer_at.observe(_moon).apparent().altaz(temperature_C="standard")
    sun_alt_obs, _, _ = observer_at.observe(_sun).apparent().altaz(temperature_C="standard")
    moon_alt = moon_alt.degrees
    sun_alt_obs = sun_alt_obs.degrees

    subsolar_lat, subsolar_lon = _sub_point(_sun, t, _moon.at(t))
    sun_alt_f = _body_altitude_at_feature(subsolar_lat, subsolar_lon, feature_lat, feature_lon)

    ok = (sun_alt_f >= 0.0) & (sun_alt_f <= sun_alt_max) & (moon_alt >= moon_alt_min)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []

    windows = []
    for seg in _split_windows(idx):
        best = seg[np.argmax(moon_alt[seg])]
        windows.append({
            "start": dts[seg[0]],
            "end": dts[seg[-1]],
            "best": dts[best],
            # The Sun climbs over the feature after sunrise and sinks toward
            # sunset; within a window (at most a day) this is monotonic
            "event": "sunrise" if sun_alt_f[seg[-1]] >= sun_alt_f[seg[0]] else "sunset",
            "sun_alt": float(sun_alt_f[best]),
            "moon_alt": float(moon_alt[best]),
            "observer_sun_alt": float(sun_alt_obs[best]),
        })
    return windows


def find_libration_windows(start_local: datetime, days: int,
                           feature_lat: float, feature_lon: float,
                           step_minutes: int = 60,
                           sun_alt_min: float = 3.0,
                           moon_alt_min: float = 5.0,
                           max_results: int = 20) -> list[dict]:
    """
    Find upcoming windows when a Moon feature is best presented, that is when
    libration tilts it toward Earth. This is what decides whether a limb
    formation such as Mare Orientale shows any detail at all, or is not turned
    into view in the first place - for features near the middle of the disk
    libration hardly matters and the results are simply all similar.

    The figure of merit is the altitude of the Earth above the feature's own
    horizon: 90 degrees at the centre of the disk, 0 exactly on the limb (see
    _body_altitude_at_feature). It doubles as the foreshortening angle, the
    feature being squashed by its sine across the line of sight.

    A window additionally requires the feature to be sunlit and the Moon to be
    up at the observer's site, so the listed times are actually observable.

    Parameters
    ----------
    start_local : datetime
        Timezone-aware start of the scan
    days : int
        Scan length in days (clamped to the bundled kernel range)
    feature_lat, feature_lon : float
        Selenographic position of the feature in degrees
    step_minutes : int
        Sample spacing; window edges are accurate to this resolution
    sun_alt_min : float
        Minimum Sun altitude over the feature, so it is lit rather than in
        night or in the deepest grazing shadow
    moon_alt_min : float
        Minimum Moon altitude at the observer site
    max_results : int
        Cap on the number of windows returned

    Returns
    -------
    list[dict]
        One dict per window, best presented first, with keys: "start", "end",
        "best" (UTC datetimes; "best" is the sample where the feature is
        presented most favourably), "earth_alt" (degrees above the feature's
        horizon at "best" - the figure of merit), "libr_long" and "libr_lat"
        (topocentric libration there), "sun_alt" (Sun altitude over the
        feature), "moon_alt" and "observer_sun_alt" (degrees at "best").
    """
    dts, t = _scan_times(start_local, days, step_minutes)

    observer_at = _observer.at(t)
    moon_at = _moon.at(t)
    moon_alt, _, _ = observer_at.observe(_moon).apparent().altaz(temperature_C="standard")
    sun_alt_obs, _, _ = observer_at.observe(_sun).apparent().altaz(temperature_C="standard")
    moon_alt = moon_alt.degrees
    sun_alt_obs = sun_alt_obs.degrees

    # Sub-Earth point = the libration of the moment, seen from the observer
    # (topocentric, so the daily rocking of up to ~1 degree counts too)
    libr_lat, libr_lon = _sub_point(_observer, t, moon_at)
    earth_alt = _body_altitude_at_feature(libr_lat, libr_lon, feature_lat, feature_lon)

    subsolar_lat, subsolar_lon = _sub_point(_sun, t, moon_at)
    sun_alt_f = _body_altitude_at_feature(subsolar_lat, subsolar_lon, feature_lat, feature_lon)

    ok = (earth_alt > 0.0) & (sun_alt_f >= sun_alt_min) & (moon_alt >= moon_alt_min)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []

    windows = []
    for seg in _split_windows(idx):
        best = seg[np.argmax(earth_alt[seg])]
        windows.append({
            "start": dts[seg[0]],
            "end": dts[seg[-1]],
            "best": dts[best],
            "earth_alt": float(earth_alt[best]),
            "libr_long": _wrap_signed_degrees(float(libr_lon[best])),
            "libr_lat": float(libr_lat[best]),
            "sun_alt": float(sun_alt_f[best]),
            "moon_alt": float(moon_alt[best]),
            "observer_sun_alt": float(sun_alt_obs[best]),
        })

    windows.sort(key=lambda w: w["earth_alt"], reverse=True)
    return windows[:max_results]


# Clair-obscur ("light-dark") events: the shapes that appear for a few hours
# when the terminator lights only the high ground of a formation. Each window
# below is a Sun altitude range over the event's own point, which the Sun
# crosses at about 0.5 degrees per hour - so a window n degrees wide lasts
# roughly 2n hours. See ClairObscurEvent for why altitude and not colongitude.
#
# The Lunar X window is the one with a firm published trigger (colongitude
# 358.0-358.7), and this range reproduces it. The others are set from the
# geometry of the formation: a summit of height h catches the Sun while it is
# still arccos(R / (R + h)) below the local horizon, which is 4.8 degrees for
# the 6 km peaks of Montes Jura, hence the negative window of the Jewelled
# Handle. Tune them here if your own observations disagree.
CLAIR_OBSCUR_EVENTS = (
    ClairObscurEvent(
        "Lunar X", -25.9, 1.85, -0.3, 1.8, True,
        "The rims of La Caille, Blanchinus and Purbach lit into a bright X "
        "standing in the dark, a few hours before First Quarter."),
    ClairObscurEvent(
        "Lunar V", 14.5, 0.7, -0.3, 1.8, True,
        "A V of lit ridges near Ukert, formed within a couple of hours of the "
        "Lunar X and usually seen in the same session."),
    ClairObscurEvent(
        "Jewelled Handle", 44.1, -31.5, -4.8, -1.0, True,
        "The 6 km peaks of Montes Jura catching the Sun while the floor of "
        "Sinus Iridum is still in night, drawing a bright handle on the "
        "terminator."),
    ClairObscurEvent(
        "Eyes of Clavius", -58.4, -14.4, -0.5, 2.5, True,
        "Sunrise over Clavius, when the rims of the crater chain on its floor "
        "light up as a curving row of bright rings."),
    ClairObscurEvent(
        "Rupes Recta dark", -21.8, -7.8, 1.0, 8.0, True,
        "The Straight Wall casting its shadow after local sunrise, drawing a "
        "110 km dark line across Mare Nubium."),
    ClairObscurEvent(
        "Rupes Recta bright", -21.8, -7.8, 1.0, 8.0, False,
        "The same scarp around local sunset, its face now turned to the Sun "
        "and shining as a bright line instead."),
)


def find_clair_obscur_events(start_local: datetime, days: int,
                             step_minutes: int = 30,
                             moon_alt_min: float = 5.0,
                             events: tuple = CLAIR_OBSCUR_EVENTS) -> list[dict]:
    """
    Find upcoming clair-obscur events: the hours in which each pattern of
    CLAIR_OBSCUR_EVENTS stands, filtered to the ones the observer can actually
    see. These are timed by illumination alone, so a scan over Sun altitude at
    the event's point finds them exactly, and they are short - typically four
    to eight hours out of a lunation.

    One vectorized Skyfield evaluation serves every event: the subsolar point
    and the observer's Moon and Sun altitudes are computed once for the whole
    range, and each event only costs a spherical-triangle evaluation on top of
    that (see _body_altitude_at_feature).

    Parameters
    ----------
    start_local : datetime
        Timezone-aware start of the scan
    days : int
        Scan length in days (clamped to the bundled kernel range)
    step_minutes : int
        Sample spacing; window edges are accurate to this resolution
    moon_alt_min : float
        Minimum Moon altitude at the observer site for an occurrence to count
        as visible. Occurrences with no visible part are dropped; pass 0 to
        keep every occurrence regardless of the observer's sky.
    events : tuple of ClairObscurEvent
        Events to scan for, by default the whole catalogue

    Returns
    -------
    list[dict]
        One dict per occurrence, in time order, with keys: "event" and
        "description" (from the catalogue), "lat" and "lon" (where to look),
        "start", "end" (UTC datetimes bounding the pattern), "peak" (UTC, the
        sample where the illumination is closest to the middle of the event's
        window), "visible_start" and "visible_end" (the part of it with the
        Moon up, None when there is none), "peak_visible" (whether the Moon is
        up at "peak" itself), "sun_alt" (Sun altitude over the event at
        "peak", the illumination that forms the pattern), "highest" (UTC,
        where the Moon stands highest during the pattern) and "moon_alt" and
        "observer_sun_alt" (degrees there, not at "peak", so they describe the
        best moment to go out and never contradict moon_alt_min).
    """
    dts, t = _scan_times(start_local, days, step_minutes)

    observer_at = _observer.at(t)
    moon_alt, _, _ = observer_at.observe(_moon).apparent().altaz(temperature_C="standard")
    sun_alt_obs, _, _ = observer_at.observe(_sun).apparent().altaz(temperature_C="standard")
    moon_alt = moon_alt.degrees
    sun_alt_obs = sun_alt_obs.degrees

    subsolar_lat, subsolar_lon = _sub_point(_sun, t, _moon.at(t))

    occurrences = []
    for event in events:
        sun_alt = _body_altitude_at_feature(subsolar_lat, subsolar_lon, event.lat, event.lon)
        # The Sun climbs over a given point for half a lunation and sinks for
        # the other half, so the same altitude window is met twice; only the
        # half this event belongs to counts
        climbing = np.gradient(sun_alt) > 0
        ok = (sun_alt >= event.sun_alt_min) & (sun_alt <= event.sun_alt_max) \
            & (climbing == event.rising)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue

        middle = 0.5 * (event.sun_alt_min + event.sun_alt_max)
        for seg in _split_windows(idx):
            peak = seg[np.argmin(np.abs(sun_alt[seg] - middle))]
            # The illumination is reported at the peak, but the observer's sky
            # is reported where the Moon stands highest during the pattern:
            # that is the moment worth going out for, and it is the same test
            # the moon_alt_min filter applies - so a listed occurrence can
            # never show a Moon altitude below it, which the peak could
            visible = seg[moon_alt[seg] >= moon_alt_min]
            if moon_alt_min > 0.0 and visible.size == 0:
                continue
            highest = seg[np.argmax(moon_alt[seg])]
            occurrences.append({
                "event": event.name,
                "description": event.description,
                "lat": event.lat,
                "lon": event.lon,
                "start": dts[seg[0]],
                "end": dts[seg[-1]],
                "peak": dts[peak],
                "highest": dts[highest],
                "visible_start": dts[visible[0]] if visible.size else None,
                "visible_end": dts[visible[-1]] if visible.size else None,
                "peak_visible": bool(moon_alt[peak] >= moon_alt_min),
                "sun_alt": float(sun_alt[peak]),
                "moon_alt": float(moon_alt[highest]),
                "observer_sun_alt": float(sun_alt_obs[highest]),
            })

    occurrences.sort(key=lambda o: o["peak"])
    return occurrences


def sun_altitude_at(subsolar_lat: float, subsolar_lon: float, lat_deg: float, lon_deg: float) -> float:
    """
    Sun altitude above the local horizon at a selenographic point, in degrees:
    negative in lunar night, 0 at sunrise or sunset, 90 with the Sun overhead.
    It is what sets shadow length there - a shadow is its caster's height
    divided by the tangent of this angle.
    """
    return float(_body_altitude_at_feature(
        np.asarray(subsolar_lat), np.asarray(subsolar_lon),
        lat_deg, lon_deg))


def calculate_moon_ephemeris(dt_local: datetime, parallactic_mode: bool) -> MoonEphemeris:

    dt_utc = _validate_supported_datetime(dt_local)
    time = _timescale.from_datetime(dt_utc)

    earth_at = _earth.at(time)
    moon_at = _moon.at(time)
    sun_at = _sun.at(time)
    observer_at = _observer.at(time)

    moon_topo = observer_at.observe(_moon).apparent()
    sun_topo = observer_at.observe(_sun).apparent()

    moon_radec = moon_topo.radec(epoch="date")
    sun_radec = sun_topo.radec(epoch="date")
    moon_ra, moon_dec, _ = moon_radec
    moon_ra_deg = moon_ra.hours * 15.0
    moon_dec_deg = moon_dec.degrees

    # In non-parallactic-mount mode we rotate the view basis to follow the zenith, so the
    # parallactic angle q is applied as a rotation of the Moon-relative view basis. In
    # parallactic-mount mode we keep celestial north "up" in the view frame (no field
    # rotation to follow the zenith), so the view-basis rotation is computed with q = 0.
    if parallactic_mode:
        q_deg = 0.0
    else:
        moon_hour_angle, _, _ = moon_topo.hadec()
        moon_hour_angle_deg = moon_hour_angle.hours * 15.0
        q_deg = _parallactic_angle_deg(moon_hour_angle_deg, moon_dec_deg, _observer_lat)

    moon_alt, moon_az, _ = moon_topo.altaz(temperature_C="standard")

    elongation = moon_topo.separation_from(sun_topo).degrees
    bright_limb_angle_deg = position_angle_of(moon_radec, sun_radec).degrees - q_deg
    # Geometric geocentric positions differ from apparent ones by arcseconds,
    # far below the 0.5-degree phase-name bins, and skip two observe() chains.
    phase_name = _phase_name(moon_at - earth_at, sun_at - earth_at)

    # Pre-compute rotation matrices once; reused for libration, colongitude, and view matrix.
    R_moon = _moon_frame.rotation_at(time)
    R_equator = true_equator_and_equinox_of_date.rotation_at(time)

    earth_from_moon = earth_at - moon_at
    observer_from_moon = observer_at - moon_at
    libr_lat_geo, libr_lon_geo = _latlon_from_icrf(earth_from_moon.position.au, R_moon)
    libr_lat_topo, libr_lon_topo = _latlon_from_icrf(observer_from_moon.position.au, R_moon)

    sun_from_moon = sun_at - moon_at
    sun_lat_moon, sun_lon_moon = _latlon_from_icrf(sun_from_moon.position.au, R_moon)
    colongitude = _colongitude_from_subsolar_longitude(sun_lon_moon)

    # Phase angle is the Sun-Moon-observer angle; both direction vectors are
    # already available, so avoid phase_angle()'s internal Sun re-evaluation.
    sun_dir_au = sun_from_moon.position.au
    observer_dir_au = observer_from_moon.position.au
    phase_angle_deg = math.degrees(math.atan2(
        np.linalg.norm(np.cross(sun_dir_au, observer_dir_au)),
        np.dot(sun_dir_au, observer_dir_au),
    ))
    moon_distance_km = observer_from_moon.distance().km
    sun_distance_km = sun_topo.distance().km
    rotation_matrix = _rotation_matrix(R_moon, R_equator, moon_ra_deg, moon_dec_deg, q_deg)

    return MoonEphemeris(
        az=moon_az.degrees,
        alt=moon_alt.degrees,
        ra=moon_ra_deg,
        dec=moon_dec_deg,
        distance=moon_distance_km,
        sun_distance=sun_distance_km,
        phase_angle=phase_angle_deg,
        age_days=_moon_age_days(time),
        bright_limb_angle=_wrap_signed_degrees(bright_limb_angle_deg),
        libr_long_geo=_wrap_signed_degrees(libr_lon_geo),
        libr_lat_geo=libr_lat_geo,
        libr_long_topo=_wrap_signed_degrees(libr_lon_topo),
        libr_lat_topo=libr_lat_topo,
        elongation=elongation,
        phase_name=phase_name,
        colongitude=colongitude,
        subsolar_lat=sun_lat_moon,
        subsolar_lon=_wrap_signed_degrees(sun_lon_moon),
        rotation_matrix=rotation_matrix,
    )
