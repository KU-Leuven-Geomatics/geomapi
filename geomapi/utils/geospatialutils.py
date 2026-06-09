"""
Geospatialutils - a Python library for processing coordinate system data.
"""

from xmlrpc.client import Boolean
# from osgeo import gdal #conda install gdal
# from osgeo import osr
# from osgeo import ogr

import PIL
from PIL.ExifTags import TAGS, GPSTAGS
import xml.etree.ElementTree as ET 
import math
from typing import List,Tuple
import numpy as np
from pyproj import Transformer
#IMPORT MODULES
import geomapi.utils as ut


def belgian_datum_to_wgs84(x, y):
    """
    Converts Belgian Datum 72 (EPSG:31370) coordinates to WGS84 (EPSG:4326).

    Parameters:
    x (float): The x-coordinate (Easting in meters).
    y (float): The y-coordinate (Northing in meters).

    Returns:
    tuple: (latitude, longitude) in WGS84 format.
    """
    # Define transformation from EPSG:31370 (Belgian Datum) to EPSG:4326 (WGS84)
    transformer = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    
    # Perform transformation
    lon, lat = transformer.transform(x, y)
    
    return lat, lon  # Return as (latitude, longitude)

def wgs84_to_belgian_datum(lat, lon):
    """
    Converts WGS84 (EPSG:4326) coordinates to Belgian Datum 72 (EPSG:31370).

    Parameters:
    lat (float): Latitude in WGS84.
    lon (float): Longitude in WGS84.

    Returns:
    tuple: (x, y) in Belgian Datum 72 (Lambert 1972) coordinate system.
    """
    # Define transformation from EPSG:4326 (WGS84) to EPSG:31370 (Belgian Datum)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
    
    # Perform transformation
    x, y = transformer.transform(lon, lat)  # always_xy=True ensures (lon, lat) input
    
    return x, y  # Return as (Easting, Northing)


def parse_exif_gps_data(data, reference:str) -> float:
    """Returns decimal degrees from world angles in exif data.\n

    Args:
        data (float or tuple): decimal degrees or (degrees, minutes, seconds) notation of world angles\n
        reference (str): direction character i.e. 'S','W','s' or 'w'.\n

    Returns:
        decimal degrees (float)
    """
    if type(data) is float:
        if reference in ['S','W','s','w']:
            return data*-1
        else:
            return data
    elif type(data) is tuple and len(data) == 3:
        value=dms_to_dd(data[0],data[1],data[2],reference)
        return value    
     
# NOTE use consistent naming sceme
def dms_to_dd(degrees:float, minutes:float, seconds:float, direction:str) -> float:
    """Convert world angles (degrees, minutes, seconds, direction) to decimal degrees.\n

    Args:
        1. degrees (float) \n
        2. minutes (float) \n
        3. seconds (float) \n
        4. direction (str): 'N' and 'E' are positive, 'W' and 'S' are negative  \n

    Returns:
        decimal degrees (float)
    """
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd

def dd_to_dms(deg) -> List[float]:
    """Convert decimal degrees to List[degrees, minutes, seconds] notation. \n

    **NOTE**: Direction is not determined (should be easy based on sign).

    Args:
        decimal degrees (float)

    Returns:
        List[degrees(float),minutes(float),seconds(float)]
    """
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]

def parse_dms(dms) :
    """Returns decimal degrees (float) notation of geospatial coordinates from various (degrees, minutes, seconds, direction) notations.\n

    Args (degrees, minutes, seconds,direction):
        1. dms (np.array): \n
        2. dms (tuple[float,float,float,float])\n
        3. dms (str): stringed list with values\n

    Raises:
        ValueError: 'dms.size!=4'

    Returns:
        decimal degrees (float)
    """
    try:
        if type(dms) is np.ndarray and dms.size==4:
            return dms_to_dd(dms[0],dms[1],dms[2], dms[3] )
        elif type(dms) is tuple and len(dms)==4:
            return dms_to_dd(dms[0],dms[1],dms[2], dms[3] )
        elif type(dms) is str and 'None' not in dms:
            temp=ut.validate_string(dms, ' ')
            temp=temp.replace("\n","")
            temp=temp.replace("\r","")
            temp=temp.split(' ')
            temp=[x for x in temp if x]
            if temp:
                res=np.asarray(temp) 
                return dms_to_dd(res[0],res[1],res[2], res[3] )
        return None  
    except:
        raise ValueError ('dms.size!=4')

def get_exif_data(img:PIL.Image)-> dict:
    """Returns a dictionary from the exif data of an Image (PIL) item. Also
    converts the GPS Tags.\n

    https://pillow.readthedocs.io/en/stable/reference/Image.html
    
    Args:
        Img (PIL.image)
    
    Returns:
        exifData (dict): dictionary containing all meta data of the PIL Image.
    """
    exifData = {}
    info = img._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exifData[decoded] = gps_data
            else:
                exifData[decoded] = value        
        return exifData      
    else:
        return None    