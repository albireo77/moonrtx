import math
from datetime import datetime

from pymeeus.Epoch import Epoch
from pymeeus.Moon import Moon
from pymeeus.Angle import Angle
from pymeeus import Coordinates

from moonrtx.types import MoonEphemeris

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
    
    moon_ra_deg, moon_dec_deg, moon_distance, moon_parallax = Moon.apparent_equatorial_pos(epoch)
    
    # Calculate local sidereal time for hour angle computation
    # LST = Greenwich Sidereal Time + observer longitude
    lst_hours = (epoch.mean_sidereal_time() * 24.0 + lon / 15.0) % 24.0
    lst_deg = lst_hours * 15.0  # convert to degrees

    moon_ra_deg, moon_dec_deg = moon_topocentric_ra_dec_deg(
        moon_ra_deg, moon_dec_deg,
        moon_distance,
        lat, lon,
        lst_deg
    )

    moon_ra = Angle(moon_ra_deg)
    moon_dec = Angle(moon_dec_deg)
    
    # Moon hour angle
    moon_ha_deg = (lst_deg - float(moon_ra)) % 360.0
    if moon_ha_deg > 180:
        moon_ha_deg -= 360
    moon_ha = Angle(moon_ha_deg)
    
    # Convert equatorial to horizontal coordinates
    # Note: pymeeus equatorial2horizontal returns (azimuth, elevation)
    # where azimuth is measured westward from SOUTH (Meeus convention)
    # We need to add 180° to get azimuth from North
    observer_lat = Angle(lat)
    
    moon_az_meeus, moon_alt = Coordinates.equatorial2horizontal(moon_ha, moon_dec, observer_lat)
    
    # Convert from Meeus convention (azimuth from South) to standard (from North)
    moon_az_deg = (float(moon_az_meeus) + 180.0) % 360.0
    moon_alt_deg = float(moon_alt)
    
    illum_frac = Moon.illuminated_fraction_disk(epoch)
    # Calculate Moon phase angle
    # k = (1 + cos(i)) / 2, so i = arccos(2k - 1)
    phase_angle = math.degrees(math.acos(2 * illum_frac - 1))
    
    # Get position angle of the bright limb using pymeeus
    position_angle = float(Moon.position_bright_limb(epoch))
    
    # Parallactic angle tells us how much celestial north is tilted from zenith
    q = float(Coordinates.parallactic_angle(moon_ha, moon_dec, Angle(lat)))

    pa_axis = float(Moon.moon_position_angle_axis(epoch))

    lopt, bopt, lphys, bphys, ltot, btot = Moon.moon_librations(epoch)

    return MoonEphemeris(
        az=moon_az_deg,
        alt=moon_alt_deg,
        ra=moon_ra_deg,
        dec=moon_dec_deg,
        distance=moon_distance - EARTH_RADIUS_KM,
        illum=illum_frac * 100,
        phase=phase_angle,
        pa=position_angle,
        pa_axis_view=-pa_axis + q,
        q=q,
        libr_long=float(ltot),
        libr_lat=float(btot)
    )

def moon_topocentric_ra_dec_deg(
    ra_deg, dec_deg,
    distance_km,
    lat_deg, lon_deg,
    lst_deg):

    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lst = math.radians(lst_deg)

    pi = math.asin(EARTH_RADIUS_KM / distance_km)

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

    ra_topo_deg = math.degrees(ra_topo) % 360.0
    dec_topo_deg = math.degrees(dec_topo)

    return ra_topo_deg, dec_topo_deg