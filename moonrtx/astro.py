import math
from datetime import datetime, timezone

import numpy as np
from skyfield.api import wgs84
from skyfield.framelib import ecliptic_frame, true_equator_and_equinox_of_date

from moonrtx.skyfield_utils import (
    SKYFIELD_MOON_FRAME_END_UTC,
    SKYFIELD_MOON_FRAME_START_UTC,
    skyfield_ephemeris as _skyfield_ephemeris,
    skyfield_moon_frame as _skyfield_moon_frame,
    skyfield_timescale as _skyfield_timescale,
)
from moonrtx.shared_types import MoonEphemeris
RENDERER_TO_SKYFIELD_BODY_MATRIX = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=float,
)


def _validate_supported_datetime(dt_utc: datetime) -> datetime:
    dt_utc = dt_utc.astimezone(timezone.utc)
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


def _unit_from_ra_dec(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    cos_dec = math.cos(dec_rad)
    return (
        cos_dec * math.cos(ra_rad),
        cos_dec * math.sin(ra_rad),
        math.sin(dec_rad),
    )


def _sky_basis(ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    east = np.array((-math.sin(ra_rad), math.cos(ra_rad), 0.0), dtype=float)
    north = np.array(
        (
            -math.sin(dec_rad) * math.cos(ra_rad),
            -math.sin(dec_rad) * math.sin(ra_rad),
            math.cos(dec_rad),
        ),
        dtype=float,
    )
    return east, north


def _normalize_np(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _matrix_to_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _parallactic_angle_deg(hour_angle_deg: float, dec_deg: float, lat_deg: float) -> float:
    hour_angle_rad = math.radians(hour_angle_deg)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)
    return math.degrees(math.atan2(
        math.sin(hour_angle_rad),
        math.tan(lat_rad) * math.cos(dec_rad) - math.sin(dec_rad) * math.cos(hour_angle_rad),
    ))


def topocentric_bright_limb_pa(
    sun_ra_deg: float,
    sun_dec_deg: float,
    moon_ra_deg: float,
    moon_dec_deg: float,
) -> float:
    sun_ra_rad = math.radians(sun_ra_deg)
    sun_dec_rad = math.radians(sun_dec_deg)
    moon_ra_rad = math.radians(moon_ra_deg)
    moon_dec_rad = math.radians(moon_dec_deg)

    return math.degrees(math.atan2(
        math.cos(sun_dec_rad) * math.sin(sun_ra_rad - moon_ra_rad),
        math.sin(sun_dec_rad) * math.cos(moon_dec_rad)
        - math.cos(sun_dec_rad) * math.sin(moon_dec_rad) * math.cos(sun_ra_rad - moon_ra_rad),
    )) % 360.0


def _rotation_matrix_and_axis_angle(
    time,
    moon_frame,
    moon_ra_deg: float,
    moon_dec_deg: float,
    q_deg: float,
) -> tuple[np.ndarray, float]:
    moon_sight_date = np.array(_unit_from_ra_dec(moon_ra_deg, moon_dec_deg), dtype=float)
    east_cel, north_cel = _sky_basis(moon_ra_deg, moon_dec_deg)

    q_rad = math.radians(q_deg)
    up_view = _normalize_np(math.sin(q_rad) * east_cel + math.cos(q_rad) * north_cel)
    right_view = _normalize_np(np.cross(moon_sight_date, up_view))
    view_basis = np.vstack([right_view, moon_sight_date, up_view])

    body_to_date = true_equator_and_equinox_of_date.rotation_at(time) @ moon_frame.rotation_at(time).T
    rotation_matrix = view_basis @ body_to_date @ RENDERER_TO_SKYFIELD_BODY_MATRIX

    pole_date = body_to_date @ np.array([0.0, 0.0, 1.0], dtype=float)
    pa_axis_topo = _wrap_signed_degrees(math.degrees(math.atan2(
        float(np.dot(pole_date, east_cel)),
        float(np.dot(pole_date, north_cel)),
    )))
    return rotation_matrix, pa_axis_topo


def calculate_moon_ephemeris(dt_utc: datetime, lat: float, lon: float, observer_elevation: int = 0) -> MoonEphemeris:
    
    dt_utc = _validate_supported_datetime(dt_utc)

    time = _skyfield_timescale().from_datetime(dt_utc)
    ephemeris = _skyfield_ephemeris()
    moon_frame = _skyfield_moon_frame()

    earth = ephemeris["earth"]
    moon = ephemeris["moon"]
    sun = ephemeris["sun"]
    observer = earth + wgs84.latlon(
        latitude_degrees=lat,
        longitude_degrees=lon,
        elevation_m=float(observer_elevation),
    )

    moon_geo = earth.at(time).observe(moon).apparent()
    moon_topo = observer.at(time).observe(moon).apparent()
    sun_geo = earth.at(time).observe(sun).apparent()
    sun_topo = observer.at(time).observe(sun).apparent()

    moon_ra, moon_dec, _ = moon_topo.radec(epoch="date")
    sun_ra, sun_dec, _ = sun_topo.radec(epoch="date")
    moon_ra_deg = moon_ra.hours * 15.0
    moon_dec_deg = moon_dec.degrees
    sun_ra_deg = sun_ra.hours * 15.0
    sun_dec_deg = sun_dec.degrees

    moon_hour_angle, _, _ = moon_topo.hadec()
    moon_hour_angle_deg = moon_hour_angle.hours * 15.0
    q_deg = _parallactic_angle_deg(moon_hour_angle_deg, moon_dec_deg, lat)

    moon_alt, moon_az, _ = moon_topo.altaz(temperature_C="standard")
    moon_alt_deg = moon_alt.degrees

    sun_moon_separation = moon_topo.separation_from(sun_topo).degrees
    phase_angle = moon_topo.phase_angle(sun).degrees
    bright_limb_pa = topocentric_bright_limb_pa(
        sun_ra_deg,
        sun_dec_deg,
        moon_ra_deg,
        moon_dec_deg,
    )

    _, moon_geo_lon, _ = moon_geo.frame_latlon(ecliptic_frame)
    _, sun_geo_lon, _ = sun_geo.frame_latlon(ecliptic_frame)
    delta_long = (moon_geo_lon.degrees - sun_geo_lon.degrees) % 360.0

    earth_from_moon = earth.at(time) - moon.at(time)
    observer_from_moon = observer.at(time) - moon.at(time)
    libr_lat_geo, libr_lon_geo, _ = earth_from_moon.frame_latlon(moon_frame)
    libr_lat_topo, libr_lon_topo, _ = observer_from_moon.frame_latlon(moon_frame)
    topocentric_distance_km = float((moon.at(time) - observer.at(time)).distance().km)

    rotation_matrix, pa_axis_topo = _rotation_matrix_and_axis_angle(
        time,
        moon_frame,
        moon_ra_deg,
        moon_dec_deg,
        q_deg,
    )
    pa_axis_view = (q_deg - pa_axis_topo) % 360.0

    sun_from_moon = sun.at(time) - moon.at(time)
    _, sun_lon_moon, _ = sun_from_moon.frame_latlon(moon_frame)
    colongitude = _colongitude_from_subsolar_longitude(float(sun_lon_moon.degrees))

    return MoonEphemeris(
        az=float(moon_az.degrees),
        alt=float(moon_alt_deg),
        ra=float(moon_ra_deg),
        dec=float(moon_dec_deg),
        distance=math.floor(topocentric_distance_km + 0.5),
        phase_angle=float(phase_angle),
        pa=float(bright_limb_pa),
        pa_axis_view=float(pa_axis_view),
        q=float(q_deg),
        libr_long_geo=float(_wrap_signed_degrees(libr_lon_geo.degrees)),
        libr_lat_geo=float(libr_lat_geo.degrees),
        libr_long_topo=float(_wrap_signed_degrees(libr_lon_topo.degrees)),
        libr_lat_topo=float(libr_lat_topo.degrees),
        sun_separation=float(sun_moon_separation),
        delta_long=float(delta_long),
        colongitude=float(colongitude),
        rotation_matrix=_matrix_to_tuple(rotation_matrix),
    )
