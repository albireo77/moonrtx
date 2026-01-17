import math
from datetime import datetime

from pymeeus.Epoch import Epoch
from pymeeus.Moon import Moon
from pymeeus.Angle import Angle
from pymeeus import Coordinates

from moonrtx.shared_types import MoonEphemeris

EARTH_RADIUS_KM = 6378.14

def calculate_moon_ephemeris(dt_utc: datetime, lat: float, lon: float) -> MoonEphemeris:
    """
    Calculate Moon ephemeris for a given time and observer location (topocentric system)
    
    Parameters
    ----------
    dt_utc : date_time
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
        - phase: Phase angle (0 = new, 180 = full)
        - pa: Position angle of the bright limb (from celestial north)
        - pa_axis_view
        - q: Parallactic angle (tilt of celestial N from zenith)
        - libr_long, libr_lat: Librations (in longitude and in latitude)
    """

    epoch = Epoch(dt_utc, utc=True)
    
    moon_ra, moon_dec, moon_distance, moon_parallax = Moon.apparent_equatorial_pos(epoch)
    
    # Calculate local sidereal time for hour angle computation
    # LST = Greenwich Sidereal Time + observer longitude
    lst_hours = (epoch.mean_sidereal_time() * 24.0 + lon / 15.0) % 24.0
    lst_deg = lst_hours * 15.0  # convert to degrees

    observer_lat = Angle(lat)
    moon_ra, moon_dec = moon_topocentric_ra_dec(moon_ra, moon_dec, observer_lat, moon_parallax, lst_deg)
    
    # Moon hour angle
    moon_ha_deg = (lst_deg - float(moon_ra)) % 360.0
    if moon_ha_deg > 180:
        moon_ha_deg -= 360
    moon_ha = Angle(moon_ha_deg)

    moon_distance_topo = moon_topocentric_distance(moon_distance, observer_lat, moon_dec, moon_ha)
    
    # Convert equatorial to horizontal coordinates
    moon_az, moon_alt = Coordinates.equatorial2horizontal(moon_ha, moon_dec, observer_lat)
    
    illum_frac = Moon.illuminated_fraction_disk(epoch)
    # Calculate Moon phase angle
    # k = (1 + cos(i)) / 2, so i = arccos(2k - 1)
    phase_angle = math.degrees(math.acos(2 * illum_frac - 1))
    
    # Get position angle of the bright limb using pymeeus
    pa = Moon.position_bright_limb(epoch)
    
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
        pa_axis_view=float(pa_axis_view),
        q=float(q),
        libr_long=float(libr_long_tot),
        libr_lat=float(libr_lat_tot)
    )

def moon_topocentric_ra_dec(
    ra_deg: Angle,
    dec_deg: Angle,
    lat_deg: Angle,
    parallax: Angle,
    lst_deg: float):

    ra = ra_deg.rad()
    dec = dec_deg.rad()
    lat = lat_deg.rad()
    lst = math.radians(lst_deg)

    pi = parallax.rad()

    H = lst - ra

    sin_phi = math.sin(lat)
    cos_phi = math.cos(lat)
    sin_dec = math.sin(dec)
    cos_dec = math.cos(dec)

    sin_H = math.sin(H)
    cos_H = math.cos(H)

    delta_ra = math.atan2(
        -cos_phi * sin_H * math.sin(pi),
        cos_dec - cos_phi * cos_H * math.sin(pi)
    )

    delta_dec = math.atan2(
        -(sin_phi * cos_dec - cos_phi * sin_dec * cos_H) * math.sin(pi),
        1 - cos_phi * cos_dec * cos_H * math.sin(pi)
    )

    ra_topo = ra + delta_ra
    dec_topo = dec + delta_dec

    return Angle(ra_topo, radians=True), Angle(dec_topo, radians=True)

def moon_topocentric_distance(
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

    # Convert to radians
    phi = lat.rad()
    dec = dec_topo.rad()
    H = ha_topo.rad()

    # cos(z) where z is the zenith distance
    cos_z = (
        math.sin(phi) * math.sin(dec) +
        math.cos(phi) * math.cos(dec) * math.cos(H)
    )

    # Meeus formula
    delta_prime = math.sqrt(
        distance_geo_km**2 +
        EARTH_RADIUS_KM**2 -
        2.0 * distance_geo_km * EARTH_RADIUS_KM * cos_z
    )

    return delta_prime