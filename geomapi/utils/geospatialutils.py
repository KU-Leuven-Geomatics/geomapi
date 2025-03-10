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
#IMPORT MODULES
import geomapi.utils as ut


# from osgeo import osr
# from ezdxf.math import Vec3

# # GPS track in WGS84, load_gpx_track() code see above
# gpx_points = list(load_gpx_track('track1.gpx'))

# # Create source coordinate system:
# src_datum = osr.SpatialReference()
# src_datum.SetWellKnownGeoCS('WGS84')

# # Create target coordinate system:
# target_datum = osr.SpatialReference()
# target_datum.SetWellKnownGeoCS('EPSG:3395')

# # Create transformation object:
# ct = osr.CoordinateTransform(src_datum, target_datum)

# # Create GeoProxy() object:
# geo_proxy = GeoProxy.parse({
#     'type': 'LineString',
#     'coordinates': gpx_points
# })

# # Apply a custom transformation function to all coordinates:
# geo_proxy.apply(lambda v: Vec3(ct.TransformPoint(v.x, v.y)))
# mport ezdxf
# from ezdxf.addons import geo
# from shapely.geometry import shape

# # Load DXF document including HATCH entities.
# doc = ezdxf.readfile('hatch.dxf')
# msp = doc.modelspace()

# # Test a single entity
# # Get the first DXF hatch entity:
# hatch_entity = msp.query('HATCH').first

# # Create GeoProxy() object:
# hatch_proxy = geo.proxy(hatch_entity)

# # Shapely supports the __geo_interface__
# shapely_polygon = shape(hatch_proxy)

# if shapely_polygon.is_valid:
#     ...
# else:
#     print(f'Invalid Polygon from {str(hatch_entity)}.')

# # Remove invalid entities by a filter function
# def validate(geo_proxy: geo.GeoProxy) -> bool:
#     # Multi-entities are divided into single entities:
#     # e.g. MultiPolygon is verified as multiple single Polygon entities.
#     if geo_proxy.geotype == 'Polygon':
#         return shape(geo_proxy).is_valid
#     return True

# # The gfilter() function let only pass compatible DXF entities
# msp_proxy = geo.GeoProxy.from_dxf_entities(geo.gfilter(msp))

# # remove all mappings for which validate() returns False

def belgian_datum_to_wgs84(latBel:float,lngBel:float) -> tuple:
    """
    Input parameters : Lat, Lng : latitude / longitude in decimal degrees and in Belgian 1972 datum
    Output parameters : LatWGS84, LngWGS84 : latitude / longitude in decimal degrees and in WGS84 datum
    source: http://zoologie.umons.ac.be/tc/algorithms.aspx
    """
    Haut = 0.0   
        
    #conversion to radians
    Lat = (math.pi / 180.0) * latBel
    Lng = (math.pi / 180.0) * lngBel
    
    SinLat = math.sin(Lat)
    SinLng = math.sin(Lng)
    CoSinLat = math.cos(Lat)
    CoSinLng = math.cos(Lng)
    
    dx = -125.8
    dy = 79.9
    dz = -100.5
    da = -251.0
    df = -0.000014192702
    
    LWf = 1.0 / 297.0
    LWa = 6378388
    LWb = (1.0 - LWf) * LWa
    LWe2 = (2.0 * LWf) - (LWf * LWf)
    Adb = 1.0 / (1.0 - LWf)
    
    Rn = LWa / math.sqrt(1.0 - LWe2 * SinLat * SinLat)
    Rm = LWa * (1.0 - LWe2) / (1 - LWe2 * Lat * Lat) ** 1.5
    
    DLat = -dx * SinLat * CoSinLng - dy * SinLat * SinLng + dz * CoSinLat
    DLat = DLat + da * (Rn * LWe2 * SinLat * CoSinLat) / LWa
    DLat = DLat + df * (Rm * Adb + Rn / Adb) * SinLat * CoSinLat
    DLat = DLat / (Rm + Haut)
    
    DLng = (-dx * SinLng + dy * CoSinLng) / ((Rn + Haut) * CoSinLat)
    Dh = dx * CoSinLat * CoSinLng + dy * CoSinLat * SinLng + dz * SinLat
    Dh = Dh - da * LWa / Rn + df * Rn * Lat * Lat / Adb
    
    LatWGS84 = ((Lat + DLat) * 180.0) / math.pi
    LngWGS84 = ((Lng + DLng) * 180.0) / math.pi
    return LatWGS84,LngWGS84

def wgs84_to_belgian_datum(LatWGS84: float,LngWGS84:float) -> tuple:
    """
    Input parameters : Lat, Lng : latitude / longitude in decimal degrees and in WGS84 datum
    Output parameters : LatBel, LngBel : latitude / longitude in decimal degrees and in Belgian datum
    source: http://zoologie.umons.ac.be/tc/algorithms.aspx
    """
    Haut = 0.0 
    
    #conversion to radians
    Lat = (math.pi / 180.0) * LatWGS84
    Lng = (math.pi / 180.0) * LngWGS84
    
    SinLat = math.sin(Lat)
    SinLng = math.sin(Lng)
    CoSinLat = math.cos(Lat)
    CoSinLng = math.cos(Lng)
    
    dx = 125.8
    dy = -79.9
    dz = 100.5
    da = 251.0
    df = 0.000014192702
    
    LWf = 1.0 / 297.0
    LWa = 6378388.0
    LWb = (1.0 - LWf) * LWa
    LWe2 = (2.0 * LWf) - (LWf * LWf)
    Adb = 1.0 / (1.0 - LWf)
    
    Rn = LWa / math.sqrt(1 - LWe2 * SinLat * SinLat)
    Rm = LWa * (1.0 - LWe2) / (1 - LWe2 * Lat * Lat) ** 1.5
    
    DLat = -dx * SinLat * CoSinLng - dy * SinLat * SinLng + dz * CoSinLat
    DLat = DLat + da * (Rn * LWe2 * SinLat * CoSinLat) / LWa
    DLat = DLat + df * (Rm * Adb + Rn / Adb) * SinLat * CoSinLat
    DLat = DLat / (Rm + Haut)
    
    DLng = (-dx * SinLng + dy * CoSinLng) / ((Rn + Haut) * CoSinLat)
    Dh = dx * CoSinLat * CoSinLng + dy * CoSinLat * SinLng + dz * SinLat
    Dh = Dh - da * LWa / Rn + df * Rn * Lat * Lat / Adb
    
    latBel = ((Lat + DLat) * 180.0) / math.pi
    lngBel = ((Lng + DLng) * 180.0) / math.pi
    return  latBel,lngBel

def spherical_coordinates_to_lambert72(latBel :float,lngBel:float) -> tuple:
    """
    Conversion from spherical coordinates to Lambert 72
    Input parameters : lat, lng (spherical coordinates)
    Spherical coordinates are in decimal degrees converted to Belgium datum!
    source: http://zoologie.umons.ac.be/tc/algorithms.aspx
    """
     
    LongRef  = 0.076042943        #'=4°21'24"983
    bLamb  = 6378388 * (1 - (1 / 297))
    aCarre  = 6378388 ** 2
    eCarre  = (aCarre - bLamb ** 2) / aCarre
    KLamb  = 11565915.812935
    nLamb  = 0.7716421928
 
    eLamb = math.sqrt(eCarre)
    eSur2 = eLamb / 2.0
    
    #conversion to radians
    lat = (math.pi / 180.0) * latBel
    lng = (math.pi / 180.0) * lngBel
 
    eSinLatitude  = eLamb * math.sin(lat)
    TanZDemi  = (math.tan((math.pi / 4) - (lat / 2))) * (((1 + (eSinLatitude)) / (1 - (eSinLatitude))) ** (eSur2))
       
 
    RLamb = KLamb * ((TanZDemi) ** nLamb)
 
    Teta = nLamb * (lng - LongRef)
 
    x = 150000 + 0.01256 + RLamb * math.sin(Teta - 0.000142043)
    y = 5400000 + 88.4378 - RLamb * math.cos(Teta - 0.000142043)
    return x,y

def lambert72_to_spherical_coordinates(x :float ,y:float) -> tuple: 
    """"
    Belgian Lambert 1972---> Spherical coordinates
    Input parameters : X, Y = Belgian coordinates in meters
    Output : latitude and longitude in Belgium Datum!
    source: http://zoologie.umons.ac.be/tc/algorithms.aspx
    """
    LongRef = 0.076042943  #      '=4°21'24"983
    nLamb  = 0.7716421928 #
    aCarre = 6378388 ** 2 #
    bLamb = 6378388 * (1 - (1 / 297)) #
    eCarre = (aCarre - bLamb ** 2) / aCarre #
    KLamb = 11565915.812935 #
    
    eLamb = math.sqrt(eCarre)
    eSur2 = eLamb / 2
    
    Tan1  = (x - 150000.01256) / (5400088.4378 - y)
    Lambda = LongRef + (1 / nLamb) * (0.000142043 + math.atan(Tan1))
    RLamb = math.sqrt((x - 150000.01256) ** 2 + (5400088.4378 - y) ** 2)
    
    TanZDemi = (RLamb / KLamb) ** (1 / nLamb)
    Lati1 = 2 * math.atan(TanZDemi)
    
    eSin=0.0
    Mult1=0.0
    Mult2=0.0
    Mult=0.0
    LatiN=0.0
    Diff=1     

    while abs(Diff) > 0.0000000277777:
        eSin = eLamb * math.sin(Lati1)
        Mult1 = 1 - eSin
        Mult2 = 1 + eSin
        Mult = (Mult1 / Mult2) ** (eLamb / 2)
        LatiN = (math.pi / 2) - (2 * (math.atan(TanZDemi * Mult)))
        Diff = LatiN - Lati1
        Lati1 = LatiN
    
    latBel = (LatiN * 180) / math.pi
    lngBel = (Lambda * 180) / math.pi
    return latBel,lngBel

# def wgs84_to_l72(lat:float,long:float) -> tuple:
#     SourceEPSG = 4326  
#     TargetEPSG = 31370

#     source = osr.SpatialReference()
#     source.ImportFromEPSG(SourceEPSG)

#     target = osr.SpatialReference()
#     target.ImportFromEPSG(TargetEPSG)


#     transform = osr.CoordinateTransformation(source, target)
#     point = ogr.Geometry(ogr.wkbPoint)
#     point.SetPoint_2D(0, float(Long), float(Lat))
#     point.Transform(transform)
    
#     x=point.GetX()
#     y=point.GetY()
    
#     return x,y 


# NOTE use different word than filter, because you are just inverting the negative coordinates
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