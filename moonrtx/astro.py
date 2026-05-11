import math
from datetime import datetime, timezone

import numpy as np
from skyfield.api import wgs84
from skyfield.framelib import ecliptic_frame, true_equator_and_equinox_of_date
from skyfield.trigonometry import position_angle_of

from moonrtx.skyfield_utils import (
    SKYFIELD_MOON_FRAME_END_UTC,
    SKYFIELD_MOON_FRAME_START_UTC,
    skyfield_ephemeris,
    skyfield_moon_frame,
    skyfield_timescale,
)
from moonrtx.shared_types import MoonEphemeris, Observer

RENDERER_TO_SKYFIELD_BODY_MATRIX = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=float,
)


def init(observer: Observer):
    global _observer, _observer_lat, _earth, _moon, _sun, _moon_frame, _timescale
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

def _phase_name(moon, sun) -> str:

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
    elif delta < 359.5:
        return "Waning Crescent"
    else:
        return "New Moon"


def calculate_moon_ephemeris(dt_local: datetime, parallactic_mode: bool) -> MoonEphemeris:

    dt_utc = _validate_supported_datetime(dt_local)
    time = _timescale.from_datetime(dt_utc)

    earth_at = _earth.at(time)
    moon_at = _moon.at(time)
    sun_at = _sun.at(time)
    observer_at = _observer.at(time)

    moon_geo = earth_at.observe(_moon).apparent()
    moon_topo = observer_at.observe(_moon).apparent()
    sun_geo = earth_at.observe(_sun).apparent()
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
    phase_name = _phase_name(moon_geo, sun_geo)

    # Pre-compute rotation matrices once; reused for libration, colongitude, and view matrix.
    R_moon = _moon_frame.rotation_at(time)
    R_equator = true_equator_and_equinox_of_date.rotation_at(time)

    earth_from_moon = earth_at - moon_at
    observer_from_moon = observer_at - moon_at
    libr_lat_geo, libr_lon_geo = _latlon_from_icrf(earth_from_moon.position.au, R_moon)
    libr_lat_topo, libr_lon_topo = _latlon_from_icrf(observer_from_moon.position.au, R_moon)

    sun_from_moon = sun_at - moon_at
    _, sun_lon_moon = _latlon_from_icrf(sun_from_moon.position.au, R_moon)
    colongitude = _colongitude_from_subsolar_longitude(sun_lon_moon)

    phase_angle_deg = moon_topo.phase_angle(_sun).degrees
    moon_distance_km = (moon_at - observer_at).distance().km
    rotation_matrix = _rotation_matrix(R_moon, R_equator, moon_ra_deg, moon_dec_deg, q_deg)

    return MoonEphemeris(
        az=moon_az.degrees,
        alt=moon_alt.degrees,
        ra=moon_ra_deg,
        dec=moon_dec_deg,
        distance=math.floor(moon_distance_km + 0.5),
        phase_angle=phase_angle_deg,
        bright_limb_angle=_wrap_signed_degrees(bright_limb_angle_deg),
        libr_long_geo=_wrap_signed_degrees(libr_lon_geo),
        libr_lat_geo=libr_lat_geo,
        libr_long_topo=_wrap_signed_degrees(libr_lon_topo),
        libr_lat_topo=libr_lat_topo,
        elongation=elongation,
        phase_name=phase_name,
        colongitude=colongitude,
        rotation_matrix=rotation_matrix,
    )
