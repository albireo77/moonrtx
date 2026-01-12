import numpy as np
import math
from datetime import datetime

from pymeeus.Epoch import Epoch
from pymeeus.Moon import Moon
from pymeeus.Angle import Angle
from pymeeus import Coordinates

EARTH_RADIUS_KM = 6378.14

def calculate_moon_librations(date_time_utc: datetime) -> dict:
    lopt, bopt, lphys, bphys, ltot, btot = Moon.moon_librations(Epoch(date_time_utc, utc=True))
    return {
        'libration_longitude': float(ltot),
        'libration_latitude': float(btot)
    }

def calculate_moon_positions_topo(date_time_utc: datetime, lat: float, lon: float) -> dict:
    """
    Calculate topocentric Moon positions for a given time and observer location.
    
    Uses pymeeus library for astronomical calculations to be consistent with
    the libration calculations.
    
    Parameters
    ----------
    date_time_utc : date_time
        Time as datetime object
    lat : float
        Observer latitude in degrees
    lon : float
        Observer longitude in degrees  
        
    Returns
    -------
    dict
        Dictionary containing:
        - moon_alt, moon_az: Moon altitude and azimuth (topocentric)
        - moon_ra, moon_dec: Moon right ascension and declination (topocentric)
        - moon_distance: Distance to Moon in km (topocentric)
        - moon_phase: Moon phase angle (0 = new, 180 = full)
        - illumination: Moon illumination fraction
        - position_angle: Position angle of the bright limb (from celestial north)
        - position_angle_axis_view
        - parallactic_angle: Tilt of celestial N from zenith
    """

    epoch_utc = Epoch(date_time_utc)
    epoch_tt = Epoch(date_time_utc, utc=True)
    
    moon_ra_deg, moon_dec_deg, moon_distance, moon_parallax = Moon.apparent_equatorial_pos(epoch_tt)
    
    # Calculate local sidereal time for hour angle computation
    # LST = Greenwich Sidereal Time + observer longitude
    lst_hours = (epoch_utc.mean_sidereal_time() * 24.0 + lon / 15.0) % 24.0
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
    
    illum_frac = Moon.illuminated_fraction_disk(epoch_tt)
    # Calculate Moon phase angle
    # k = (1 + cos(i)) / 2, so i = arccos(2k - 1)
    phase_angle = np.degrees(np.arccos(2 * illum_frac - 1))
    
    # Get position angle of the bright limb using pymeeus
    position_angle = float(Moon.position_bright_limb(epoch_tt))
    
    # Parallactic angle tells us how much celestial north is tilted from zenith
    parallactic_angle = float(Coordinates.parallactic_angle(moon_ha, moon_dec, Angle(lat)))

    pa_axis = float(Moon.moon_position_angle_axis(epoch_tt))
    
    return {
        'moon_alt': moon_alt_deg,
        'moon_az': moon_az_deg,
        'moon_distance': moon_distance - EARTH_RADIUS_KM,
        'illumination': illum_frac * 100,
        'moon_phase': phase_angle,
        'moon_ra': moon_ra_deg,
        'moon_dec': moon_dec_deg,
        'position_angle': position_angle,  # PA of bright limb from celestial N
        'position_angle_axis_view': -pa_axis + parallactic_angle,
        'parallactic_angle': parallactic_angle  # tilt of celestial N from zenith
    }

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