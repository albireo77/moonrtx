import math
from datetime import datetime

from pymeeus.Epoch import Epoch
from pymeeus.Moon import Moon
from pymeeus.Sun import Sun
from pymeeus.Earth import Earth
from pymeeus.Angle import Angle
from pymeeus import Coordinates

from moonrtx.shared_types import MoonEphemeris

EARTH = Earth()
EARTH_RADIUS_KM = 6378.14
AU_KM = 149597870.7  # 1 Astronomical Unit in km
JDE2000 = Epoch(2000, 1, 1.5)
I_RAD = math.radians(1.54242)  # Inclination of Moon's equator to ecliptic
SIN_I = math.sin(I_RAD)
COS_I = math.cos(I_RAD)
LIBRATION_LONGITUDE_BIAS_DEG = -0.00393272327219034
LIBRATION_LATITUDE_BIAS_DEG = -0.02312743538917065


def _apply_libration_biases(libr_long_deg: float, libr_lat_deg: float) -> tuple[float, float]:
    return (
        (libr_long_deg + LIBRATION_LONGITUDE_BIAS_DEG + 180.0) % 360.0 - 180.0,
        libr_lat_deg + LIBRATION_LATITUDE_BIAS_DEG,
    )


def _lunar_orientation_terms(epoch: Epoch):
    """Return the lunar arguments used by Meeus chapter 53."""
    t = (epoch - JDE2000) / 36525.0

    d_rad = math.radians((297.8501921 + (445267.1114034
                          + (-0.0018819
                             + (1.0 / 545868.0 - t / 113065000.0) * t) * t) * t) % 360.0)
    m_rad = math.radians((357.5291092 + (35999.0502909
                          + (-0.0001536 + t / 24490000.0) * t) * t) % 360.0)
    mprime_rad = math.radians((134.9633964 + (477198.8675055
                               + (0.0087414
                                  + (1.0 / 69699.9 + t / 14712000.0) * t) * t) * t) % 360.0)
    f_rad = math.radians((93.2720950 + (483202.0175233
                          + (-0.0036539
                             + (-1.0 / 3526000.0 + t / 863310000.0) * t) * t) * t) % 360.0)
    omega = Moon.longitude_mean_ascending_node(epoch)
    k1_rad = math.radians((119.75 + 131.849 * t) % 360.0)
    k2_rad = math.radians((72.56 + 20.186 * t) % 360.0)
    eccentricity = 1.0 + (-0.002516 - 0.0000074 * t) * t

    return d_rad, m_rad, mprime_rad, f_rad, omega, k1_rad, k2_rad, eccentricity


def _physical_libration_in_longitude(
    a_rad: float,
    b_rad: float,
    d_rad: float,
    m_rad: float,
    mprime_rad: float,
    f_rad: float,
    omega_rad: float,
    k1_rad: float,
    k2_rad: float,
    eccentricity: float,
) -> float:
    """Return the physical libration correction in longitude (degrees)."""
    rho = (-0.02752 * math.cos(mprime_rad)
           - 0.02245 * math.sin(f_rad)
           + 0.00684 * math.cos(mprime_rad - 2.0 * f_rad)
           - 0.00293 * math.cos(2.0 * f_rad)
           - 0.00085 * math.cos(2.0 * (f_rad - d_rad))
           - 0.00054 * math.cos(mprime_rad - 2.0 * d_rad)
           - 0.00020 * math.sin(mprime_rad + f_rad)
           - 0.00020 * math.cos(mprime_rad + 2.0 * f_rad)
           - 0.00020 * math.cos(mprime_rad - f_rad)
           + 0.00014 * math.cos(mprime_rad + 2.0 * (f_rad - d_rad)))

    sigma = (-0.02816 * math.sin(mprime_rad)
             + 0.02244 * math.cos(f_rad)
             - 0.00682 * math.sin(mprime_rad - 2.0 * f_rad)
             - 0.00279 * math.sin(2.0 * f_rad)
             - 0.00083 * math.sin(2.0 * (f_rad - d_rad))
             + 0.00069 * math.sin(mprime_rad - 2.0 * d_rad)
             + 0.00040 * math.cos(mprime_rad + f_rad)
             - 0.00025 * math.sin(2.0 * mprime_rad)
             - 0.00023 * math.sin(mprime_rad + 2.0 * f_rad)
             + 0.00020 * math.cos(mprime_rad - f_rad)
             + 0.00019 * math.sin(mprime_rad - f_rad)
             + 0.00013 * math.sin(mprime_rad + 2.0 * (f_rad - d_rad))
             - 0.00010 * math.cos(mprime_rad - 3.0 * f_rad))

    tau = (0.02520 * eccentricity * math.sin(m_rad)
           + 0.00473 * math.sin(2.0 * (mprime_rad - f_rad))
           - 0.00467 * math.sin(mprime_rad)
           + 0.00396 * math.sin(k1_rad)
           + 0.00276 * math.sin(2.0 * (mprime_rad - d_rad))
           + 0.00196 * math.sin(omega_rad)
           - 0.00183 * math.cos(mprime_rad - f_rad)
           + 0.00115 * math.sin(mprime_rad - 2.0 * d_rad)
           - 0.00096 * math.sin(mprime_rad - d_rad)
           + 0.00046 * math.sin(2.0 * (f_rad - d_rad))
           - 0.00039 * math.sin(mprime_rad - f_rad)
           - 0.00032 * math.sin(mprime_rad - m_rad - d_rad)
           + 0.00027 * math.sin(2.0 * (mprime_rad - d_rad) - m_rad)
           + 0.00023 * math.sin(k2_rad)
           - 0.00014 * math.sin(2.0 * d_rad)
           + 0.00014 * math.cos(2.0 * (mprime_rad - f_rad))
           - 0.00012 * math.sin(mprime_rad - 2.0 * f_rad)
           - 0.00012 * math.sin(2.0 * mprime_rad)
           + 0.00011 * math.sin(2.0 * (mprime_rad - m_rad - d_rad)))

    return -tau + (rho * math.cos(a_rad) + sigma * math.sin(a_rad)) * math.tan(b_rad)


def _ecliptic_to_cartesian(lon: Angle, lat: Angle, radius: float):
    lon_rad = lon.rad()
    lat_rad = lat.rad()
    cos_lat = math.cos(lat_rad)
    return (
        radius * cos_lat * math.cos(lon_rad),
        radius * cos_lat * math.sin(lon_rad),
        radius * math.sin(lat_rad),
    )


def _cartesian_to_ecliptic(x: float, y: float, z: float) -> tuple[float, float]:
    return (
        math.degrees(math.atan2(y, x)) % 360.0,
        math.degrees(math.atan2(z, math.hypot(x, y))),
    )


def _exact_topocentric_distance(
    distance_geo_km: float,
    ra_geo: Angle,
    dec_geo: Angle,
    local_sidereal_deg: float,
    lat: Angle,
    elevation_m: int = 0,
) -> float:
    """Return exact topocentric distance from geocentric equatorial coordinates."""
    theta = math.radians(local_sidereal_deg)
    rho_cosphi = EARTH.rho_cosphi(lat, float(elevation_m))
    rho_sinphi = EARTH.rho_sinphi(lat, float(elevation_m))

    obs_x = EARTH_RADIUS_KM * rho_cosphi * math.cos(theta)
    obs_y = EARTH_RADIUS_KM * rho_cosphi * math.sin(theta)
    obs_z = EARTH_RADIUS_KM * rho_sinphi

    ra_rad = ra_geo.rad()
    dec_rad = dec_geo.rad()
    cos_dec = math.cos(dec_rad)
    obj_x = distance_geo_km * cos_dec * math.cos(ra_rad)
    obj_y = distance_geo_km * cos_dec * math.sin(ra_rad)
    obj_z = distance_geo_km * math.sin(dec_rad)

    dx = obj_x - obs_x
    dy = obj_y - obs_y
    dz = obj_z - obs_z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(_dot(v, v))
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _unit_from_ra_dec(ra: Angle, dec: Angle) -> tuple[float, float, float]:
    ra_rad = ra.rad()
    dec_rad = dec.rad()
    cos_dec = math.cos(dec_rad)
    return (
        cos_dec * math.cos(ra_rad),
        cos_dec * math.sin(ra_rad),
        math.sin(dec_rad),
    )


def _sky_basis(
    ra: Angle,
    dec: Angle,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ra_rad = ra.rad()
    dec_rad = dec.rad()
    return (
        (-math.sin(ra_rad), math.cos(ra_rad), 0.0),
        (
            -math.sin(dec_rad) * math.cos(ra_rad),
            -math.sin(dec_rad) * math.sin(ra_rad),
            math.cos(dec_rad),
        ),
    )


def _subobserver_vector_body(libr_long_deg: float, libr_lat_deg: float) -> tuple[float, float, float]:
    libr_long_rad = math.radians(libr_long_deg)
    libr_lat_rad = math.radians(libr_lat_deg)
    cos_lat = math.cos(libr_lat_rad)
    return (
        math.sin(libr_long_rad) * cos_lat,
        -math.cos(libr_long_rad) * cos_lat,
        math.sin(libr_lat_rad),
    )


def _vector_to_librations(v: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = _normalize(v)
    return (
        (math.degrees(math.atan2(x, -y)) + 180.0) % 360.0 - 180.0,
        math.degrees(math.asin(max(-1.0, min(1.0, z)))),
    )


def _pole_position_angle(
    ra: Angle,
    dec: Angle,
    pole: tuple[float, float, float],
) -> float:
    east, north = _sky_basis(ra, dec)
    return (math.degrees(math.atan2(_dot(pole, east), _dot(pole, north))) + 180.0) % 360.0 - 180.0


def _topocentric_lunar_orientation(
    epoch: Epoch,
    moon_ra_geo: Angle,
    moon_dec_geo: Angle,
    moon_ra_topo: Angle,
    moon_dec_topo: Angle,
) -> tuple[float, float, float, float, float]:
    """Return geocentric librations, topocentric librations, and pole angle."""
    _, _, _, _, libr_long_geo_raw, libr_lat_geo_raw = Moon.moon_librations(epoch)
    pa_axis_geo = Moon.moon_position_angle_axis(epoch)

    sight_geo = _unit_from_ra_dec(moon_ra_geo, moon_dec_geo)
    sight_topo = _unit_from_ra_dec(moon_ra_topo, moon_dec_topo)
    earth_cel = (-sight_geo[0], -sight_geo[1], -sight_geo[2])
    observer_cel = (-sight_topo[0], -sight_topo[1], -sight_topo[2])

    east_geo, north_geo = _sky_basis(moon_ra_geo, moon_dec_geo)
    libr_lat_geo_rad = math.radians(float(libr_lat_geo_raw))
    pa_axis_geo_rad = math.radians(float(pa_axis_geo))
    cos_libr_lat_geo = math.cos(libr_lat_geo_rad)
    pole = _normalize(tuple(
        cos_libr_lat_geo * math.cos(pa_axis_geo_rad) * north_geo[i]
        + cos_libr_lat_geo * math.sin(pa_axis_geo_rad) * east_geo[i]
        - math.sin(libr_lat_geo_rad) * sight_geo[i]
        for i in range(3)
    ))

    subearth_body = _subobserver_vector_body(float(libr_long_geo_raw), float(libr_lat_geo_raw))
    east_body = _cross((0.0, 0.0, 1.0), subearth_body)
    if _dot(east_body, east_body) < 1e-15:
        east_body = _cross((0.0, 1.0, 0.0), subearth_body)
    east_body = _normalize(east_body)
    north_body = _normalize(_cross(subearth_body, east_body))

    east_cel = _cross(pole, earth_cel)
    if _dot(east_cel, east_cel) < 1e-15:
        east_cel = east_geo
    else:
        east_cel = _normalize(east_cel)
    north_cel = _normalize(_cross(earth_cel, east_cel))

    comp_east = _dot(east_cel, observer_cel)
    comp_north = _dot(north_cel, observer_cel)
    comp_radial = _dot(earth_cel, observer_cel)
    observer_body = (
        east_body[0] * comp_east + north_body[0] * comp_north + subearth_body[0] * comp_radial,
        east_body[1] * comp_east + north_body[1] * comp_north + subearth_body[1] * comp_radial,
        east_body[2] * comp_east + north_body[2] * comp_north + subearth_body[2] * comp_radial,
    )

    libr_long_geo, libr_lat_geo = _apply_libration_biases(
        float(libr_long_geo_raw),
        float(libr_lat_geo_raw),
    )
    libr_long_topo, libr_lat_topo = _apply_libration_biases(*_vector_to_librations(observer_body))
    return (
        libr_long_geo,
        libr_lat_geo,
        libr_long_topo,
        libr_lat_topo,
        _pole_position_angle(moon_ra_topo, moon_dec_topo, pole),
    )

def calculate_moon_ephemeris(dt_utc: datetime, lat: float, lon: float, observer_elevation: int = 0) -> MoonEphemeris:
    """
    Calculate Moon ephemeris for a given time and observer location (topocentric system)
    
    Parameters
    ----------
    dt_utc : datetime
        UTC time
    lat : float
        Observer latitude in degrees
    lon : float
        Observer longitude in degrees
    observer_elevation : int
        Observer elevation in meters above sea level
        
    Returns
    -------
    MoonEphemeris class
        Containing:
        - az, alt: Azimuth and altitude (topocentric)
        - ra, dec: Right ascension and declination (topocentric)
        - distance: Distance to in km (topocentric, rounded to nearest integer)
        - illum: Illumination fraction
        - phase: Topocentric phase angle (Sun-Moon-Observer angle, 0 = full, 180 = new)
        - pa: Topocentric position angle of the bright limb (from celestial north)
        - pa_axis_view: Apparent tilt of Moon's rotation axis in observer's view
        - q: Parallactic angle (tilt of celestial N from zenith)
        - libr_long_geo, libr_lat_geo: Geocentric librations
        - libr_long_topo, libr_lat_topo: Topocentric librations
        - sun_separation: Topocentric angular separation between Sun and Moon (degrees)
    """

    epoch = Epoch(dt_utc, utc=True)

    # UT1-based epoch for sidereal time computation
    # Epoch(utc=True) stores TT internally (adds ΔT ≈ 69s), but sidereal time
    # must be computed from UT1. Since UT1 ≈ UTC (within 0.9s), we use UTC directly.
    epoch_ut = Epoch(dt_utc)
    
    moon_ra_geo, moon_dec_geo, moon_distance, moon_parallax = Moon.apparent_equatorial_pos(epoch)
    
    # Nutation and obliquity (needed for apparent sidereal time)
    nut_lon = Coordinates.nutation_longitude(epoch)
    obl = Coordinates.true_obliquity(epoch)

    # Calculate local sidereal time for hour angle computation
    # LST = Greenwich Apparent Sidereal Time + observer longitude
    # Must use apparent (not mean) sidereal time to match apparent coordinates (which include nutation)
    # Must use UT-based epoch (not TT) since sidereal time is a function of Earth's rotation (UT1)
    lst_hours = (epoch_ut.apparent_sidereal_time(obl, nut_lon) * 24.0 + lon / 15.0) % 24.0
    lst_deg = lst_hours * 15.0  # convert to degrees

    observer_lat = Angle(lat)

    # Geocentric hour angle (needed for parallax correction)
    moon_ha_deg = (lst_deg - float(moon_ra_geo)) % 360.0
    if moon_ha_deg > 180:
        moon_ha_deg -= 360
    moon_ha = Angle(moon_ha_deg)

    # Apply topocentric parallax correction (Meeus ch. 40, oblate Earth)
    moon_distance_au = float(moon_distance) / AU_KM
    moon_ra, moon_dec = Earth.parallax_correction(
        moon_ra_geo, moon_dec_geo, observer_lat, moon_distance_au, moon_ha, float(observer_elevation)
    )

    # Recompute hour angle with topocentric RA
    moon_ha_deg = (lst_deg - float(moon_ra)) % 360.0
    if moon_ha_deg > 180:
        moon_ha_deg -= 360
    moon_ha = Angle(moon_ha_deg)

    moon_distance_topo = _exact_topocentric_distance(
        moon_distance,
        moon_ra_geo,
        moon_dec_geo,
        lst_deg,
        observer_lat,
        observer_elevation,
    )
    
    # Convert equatorial to horizontal coordinates
    moon_az, moon_alt_geo = Coordinates.equatorial2horizontal(moon_ha, moon_dec, observer_lat)
    moon_alt_app = Coordinates.refraction_true2apparent(moon_alt_geo)

    # Match Stellarium's displayed altitude behavior: once the refracted altitude
    # is far enough below the horizon, show the geometric altitude instead.
    moon_alt = moon_alt_app if float(moon_alt_app) > -2.0 else moon_alt_geo

    lambda_moon, beta_moon, _, _ = Moon.apparent_ecliptical_pos(epoch)
    sun_lon, sun_lat, sun_distance_au = Sun.apparent_geocentric_position(epoch)
    delta_long = lambda_moon - sun_lon

    sun_ra_geo, sun_dec_geo = Coordinates.ecliptical2equatorial(sun_lon, sun_lat, obl)
    sun_ha_deg = (lst_deg - float(sun_ra_geo)) % 360.0
    if sun_ha_deg > 180:
        sun_ha_deg -= 360
    sun_ha = Angle(sun_ha_deg)
    sun_ra, sun_dec = Earth.parallax_correction(
        sun_ra_geo, sun_dec_geo, observer_lat, float(sun_distance_au), sun_ha, float(observer_elevation)
    )

    sun_ha_deg = (lst_deg - float(sun_ra)) % 360.0
    if sun_ha_deg > 180:
        sun_ha_deg -= 360
    sun_ha = Angle(sun_ha_deg)
    sun_distance_topo = topocentric_distance(float(sun_distance_au) * AU_KM, observer_lat, sun_dec, sun_ha, observer_elevation)

    sun_moon_separation = topocentric_sun_moon_separation(sun_ra, sun_dec, moon_ra, moon_dec)
    phase_angle = topocentric_phase_angle(sun_moon_separation, sun_distance_topo, moon_distance_topo)
    pa = topocentric_bright_limb_pa(sun_ra, sun_dec, moon_ra, moon_dec)

    # Parallactic angle tells us how much celestial north is tilted from zenith
    q = Coordinates.parallactic_angle(moon_ha, moon_dec, observer_lat)

    libr_long_geo, libr_lat_geo, libr_long_topo, libr_lat_topo, pa_axis_topo = _topocentric_lunar_orientation(
        epoch,
        moon_ra_geo,
        moon_dec_geo,
        moon_ra,
        moon_dec,
    )
    pa_axis_view = float(q) - pa_axis_topo

    colongitude = calculate_colongitude(epoch, lambda_moon - nut_lon, beta_moon, moon_distance)
    moon_distance_topo_rounded = math.floor(moon_distance_topo + 0.5)

    return MoonEphemeris(
        az=(float(moon_az) + 180.0) % 360.0,      # Convert from Meeus convention (azimuth from South) to standard (from North)
        alt=float(moon_alt),
        ra=float(moon_ra),
        dec=float(moon_dec),
        distance=moon_distance_topo_rounded,
        phase_angle=phase_angle,
        pa=float(pa),
        pa_axis_view=float(pa_axis_view) % 360.0,
        q=float(q),
        libr_long_geo=libr_long_geo,
        libr_lat_geo=libr_lat_geo,
        libr_long_topo=libr_long_topo,
        libr_lat_topo=libr_lat_topo,
        sun_separation=sun_moon_separation,
        delta_long=float(delta_long) % 360.0,
        colongitude=colongitude
    )

def calculate_colongitude(
    epoch: Epoch,
    moon_lon_geo: Angle,
    moon_lat_geo: Angle,
    moon_distance_km: float,
) -> float:
    """
    Calculate the selenographic colongitude of the Sun.

    The colongitude is the selenographic longitude of the morning terminator.
    Co ≈ 0° at First Quarter, 90° at Full Moon, 180° at Last Quarter, 270° at New Moon.

    Uses the same selenographic projection as Moon.moon_librations() (Meeus ch. 53),
    but with the Sun direction evaluated at the Moon instead of at the Earth.
    This removes the Earth-Moon parallax error, which can shift colongitude by
    about 0.15°.
    """
    d_rad, m_rad, mprime_rad, f_rad, omega, k1_rad, k2_rad, eccentricity = _lunar_orientation_terms(epoch)

    sun_lon_geo, sun_lat_geo, sun_distance_au = Sun.geometric_geocentric_position(epoch)

    sun_x, sun_y, sun_z = _ecliptic_to_cartesian(sun_lon_geo, sun_lat_geo, sun_distance_au * AU_KM)
    moon_x, moon_y, moon_z = _ecliptic_to_cartesian(moon_lon_geo, moon_lat_geo, moon_distance_km)
    sun_lon_seleno, sun_lat_seleno = _cartesian_to_ecliptic(
        sun_x - moon_x,
        sun_y - moon_y,
        sun_z - moon_z,
    )

    w_sun_rad = math.radians(sun_lon_seleno + 180.0 - float(omega))
    sun_lat_seleno_rad = math.radians(sun_lat_seleno)
    sin_w = math.sin(w_sun_rad)
    cos_w = math.cos(w_sun_rad)
    sin_beta = math.sin(sun_lat_seleno_rad)
    cos_beta = math.cos(sun_lat_seleno_rad)

    a_sun = math.atan2(
        sin_w * cos_beta * COS_I - sin_beta * SIN_I,
        cos_w * cos_beta,
    )
    b_sun = math.asin(-sin_w * cos_beta * SIN_I - sin_beta * COS_I)
    l_sun = math.degrees(a_sun) % 360.0 - math.degrees(f_rad)
    l_sun += _physical_libration_in_longitude(
        a_sun,
        b_sun,
        d_rad,
        m_rad,
        mprime_rad,
        f_rad,
        omega.rad(),
        k1_rad,
        k2_rad,
        eccentricity,
    )

    return (450.0 - l_sun) % 360.0


def topocentric_sun_moon_separation(
    sun_ra: Angle, sun_dec: Angle,
    moon_ra: Angle, moon_dec: Angle
) -> float:
    """
    Topocentric angular separation between Sun and Moon (degrees).

    Uses the standard spherical angular separation formula.
    """
    sun_ra_rad = sun_ra.rad()
    sun_dec_rad = sun_dec.rad()
    moon_ra_rad = moon_ra.rad()
    moon_dec_rad = moon_dec.rad()

    cos_sep = (math.sin(sun_dec_rad) * math.sin(moon_dec_rad)
               + math.cos(sun_dec_rad) * math.cos(moon_dec_rad)
               * math.cos(sun_ra_rad - moon_ra_rad))
    cos_sep = max(-1.0, min(1.0, cos_sep))  # clamp for numerical safety
    return math.degrees(math.acos(cos_sep))


def topocentric_phase_angle(
    sun_moon_separation: float,
    sun_distance_km: float,
    moon_distance_km: float
) -> float:
    """
    Topocentric phase angle (Sun-Moon-Observer angle at the Moon vertex).

    In the Sun-Moon-Observer triangle:
      elongation ψ = Sun-Observer-Moon angle ≈ sun_moon_separation
      phase angle i = Sun-Moon-Observer angle
        Uses a direct atan2 form for speed and stable behavior near full/new Moon.

    Returns
    -------
    float
        Phase angle in degrees (0 = full, 180 = new).
    """
    d_sun = sun_distance_km
    d_moon = moon_distance_km
    psi_rad = math.radians(sun_moon_separation)

    return math.degrees(math.atan2(
        d_sun * math.sin(psi_rad),
        d_moon - d_sun * math.cos(psi_rad),
    ))


def topocentric_bright_limb_pa(
    sun_ra: Angle, sun_dec: Angle,
    moon_ra: Angle, moon_dec: Angle
) -> Angle:
    """
    Position angle of the bright limb (degrees, 0-360).

    PA of the Sun direction from the Moon, measured from celestial North
    toward East (standard position-angle convention).
    """
    sun_ra_rad = sun_ra.rad()
    sun_dec_rad = sun_dec.rad()
    moon_ra_rad = moon_ra.rad()
    moon_dec_rad = moon_dec.rad()

    return Angle(math.atan2(
        math.cos(sun_dec_rad) * math.sin(sun_ra_rad - moon_ra_rad),
        math.sin(sun_dec_rad) * math.cos(moon_dec_rad)
        - math.cos(sun_dec_rad) * math.sin(moon_dec_rad) * math.cos(sun_ra_rad - moon_ra_rad)
    ), radians=True)


def topocentric_distance(
    distance_geo_km: float,
    lat: Angle,
    dec_topo: Angle,
    ha_topo: Angle,
    elevation_m: int = 0
) -> float:
    """
    Compute topocentric distance Δ′ of the Moon (Meeus, ch. 40)

    Parameters
    ----------
    distance_geo_km : float
        Geocentric distance Δ (km)
    lat : Angle
        Observer latitude φ
    dec_topo : Angle
        Topocentric declination δ′
    ha_topo : Angle
        Topocentric hour angle H′
    elevation_m : int
        Observer elevation in meters above sea level

    Returns
    -------
    float
        Topocentric distance Δ′ in km
    """

    # Observer distance to Earth's center (oblate Earth + elevation), in km
    rho = EARTH.rho(lat)
    observer_radius_km = EARTH_RADIUS_KM * rho + elevation_m / 1000.0

    # Convert to radians
    phi = lat.rad()
    dec = dec_topo.rad()
    H = ha_topo.rad()

    # cos(z) where z is the zenith distance
    cos_z = (
        math.sin(phi) * math.sin(dec) +
        math.cos(phi) * math.cos(dec) * math.cos(H)
    )

    # Meeus formula (using oblate Earth radius)
    delta_prime = math.sqrt(
        distance_geo_km**2 +
        observer_radius_km**2 -
        2.0 * distance_geo_km * observer_radius_km * cos_z
    )

    return delta_prime