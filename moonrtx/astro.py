import math
from datetime import datetime

from pymeeus.Epoch import Epoch
from pymeeus.Moon import Moon
from pymeeus.Sun import Sun
from pymeeus.Earth import Earth
from pymeeus.Angle import Angle
from pymeeus import Coordinates

from moonrtx.shared_types import MoonEphemeris

EARTH_RADIUS_KM = 6378.14
AU_KM = 149597870.7  # 1 Astronomical Unit in km

def calculate_moon_ephemeris(dt_utc: datetime, lat: float, lon: float) -> MoonEphemeris:
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
        
    Returns
    -------
    MoonEphemeris class
        Containing:
        - az, alt: Azimuth and altitude (topocentric)
        - ra, dec: Right ascension and declination (topocentric)
        - distance: Distance to in km (topocentric)
        - illum: Illumination fraction
        - phase: Topocentric phase angle (Sun-Moon-Observer angle, 0 = full, 180 = new)
        - pa: Topocentric position angle of the bright limb (from celestial north)
        - pa_axis_view: Apparent tilt of Moon's rotation axis in observer's view
        - q: Parallactic angle (tilt of celestial N from zenith)
        - libr_long, libr_lat: Librations (in longitude and in latitude)
        - sun_separation: Topocentric angular separation between Sun and Moon (degrees)
    """

    epoch = Epoch(dt_utc, utc=True)
    
    moon_ra, moon_dec, moon_distance, moon_parallax = Moon.apparent_equatorial_pos(epoch)
    
    # Calculate local sidereal time for hour angle computation
    # LST = Greenwich Sidereal Time + observer longitude
    lst_hours = (epoch.mean_sidereal_time() * 24.0 + lon / 15.0) % 24.0
    lst_deg = lst_hours * 15.0  # convert to degrees

    observer_lat = Angle(lat)

    # Geocentric hour angle (needed for parallax correction)
    moon_ha_deg = (lst_deg - float(moon_ra)) % 360.0
    if moon_ha_deg > 180:
        moon_ha_deg -= 360
    moon_ha = Angle(moon_ha_deg)

    # Apply topocentric parallax correction (Meeus ch. 40, oblate Earth)
    moon_distance_au = float(moon_distance) / AU_KM
    moon_ra, moon_dec = Earth.parallax_correction(
        moon_ra, moon_dec, observer_lat, moon_distance_au, moon_ha
    )

    # Recompute hour angle with topocentric RA
    moon_ha_deg = (lst_deg - float(moon_ra)) % 360.0
    if moon_ha_deg > 180:
        moon_ha_deg -= 360
    moon_ha = Angle(moon_ha_deg)

    moon_distance_topo = topocentric_distance(moon_distance, observer_lat, moon_dec, moon_ha)
    
    # Convert equatorial to horizontal coordinates
    moon_az, moon_alt = Coordinates.equatorial2horizontal(moon_ha, moon_dec, observer_lat)
    
    illum_frac = Moon.illuminated_fraction_disk(epoch)

    # ---- Sun position (geocentric; Sun parallax < 9" so geocentric ≈ topocentric) ----
    sun_ra, sun_dec, sun_r_au = Sun.apparent_rightascension_declination_coarse(epoch)

    sun_moon_separation = topocentric_sun_moon_separation(sun_ra, sun_dec, moon_ra, moon_dec)
    phase_angle = topocentric_phase_angle(sun_moon_separation, sun_r_au, moon_distance_topo)
    pa = topocentric_bright_limb_pa(sun_ra, sun_dec, moon_ra, moon_dec)

    # Parallactic angle tells us how much celestial north is tilted from zenith
    q = Coordinates.parallactic_angle(moon_ha, moon_dec, observer_lat)

    pa_axis = Moon.moon_position_angle_axis(epoch)
    pa_axis_view = q - pa_axis

    _, _, _, _, libr_long_tot, libr_lat_tot = Moon.moon_librations(epoch)

    return MoonEphemeris(
        az=(float(moon_az) + 180.0) % 360.0,      # Convert from Meeus convention (azimuth from South) to standard (from North)
        alt=float(moon_alt),
        ra=float(moon_ra),
        dec=float(moon_dec),
        distance=moon_distance_topo,
        illum=illum_frac * 100,
        phase=phase_angle,
        pa=float(pa),
        pa_axis_view=float(pa_axis_view) % 360.0,
        q=float(q),
        libr_long=float(libr_long_tot),
        libr_lat=float(libr_lat_tot),
        sun_separation=sun_moon_separation
    )

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
    sun_r_au: float,
    moon_distance_km: float
) -> float:
    """
    Topocentric phase angle (Sun-Moon-Observer angle at the Moon vertex).

    In the Sun-Moon-Observer triangle:
      elongation ψ = Sun-Observer-Moon angle ≈ sun_moon_separation
      phase angle i = Sun-Moon-Observer angle
    Uses the sine rule after computing the Sun-Moon distance.

    Returns
    -------
    float
        Phase angle in degrees (0 = full, 180 = new).
    """
    d_sun = sun_r_au * AU_KM
    d_moon = moon_distance_km
    psi_rad = math.radians(sun_moon_separation)

    d_sun_moon = math.sqrt(d_sun**2 + d_moon**2 - 2 * d_sun * d_moon * math.cos(psi_rad))

    sin_i = d_sun * math.sin(psi_rad) / d_sun_moon if d_sun_moon > 0 else 0.0
    sin_i = max(-1.0, min(1.0, sin_i))
    phase_angle = math.degrees(math.asin(sin_i))

    # asin gives 0..90; the actual phase angle spans 0..180.
    # When elongation > 90° the Moon is between Earth and Sun → phase > 90°.
    if sun_moon_separation < 90.0:
        phase_angle = 180.0 - phase_angle

    return phase_angle


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
    ha_topo: Angle
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

    Returns
    -------
    float
        Topocentric distance Δ′ in km
    """

    # Observer distance to Earth's center (oblate Earth), in km
    rho = Earth().rho(lat)
    observer_radius_km = EARTH_RADIUS_KM * rho

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