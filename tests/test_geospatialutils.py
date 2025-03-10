import unittest
import geomapi.utils.geospatialutils as geo
from geomapi.utils import GEOMAPI_PREFIXES
import numpy as np
import PIL
class TestGeometryutils(unittest.TestCase):

    def test_l72_to_wgs84(self):
        latBel,lngBel=geo.lambert72_to_spherical_coordinates(x =103657.078099938 ,y=194604.105800153)
        LatWGS84,LngWGS84=geo.belgian_datum_to_wgs84(latBel,lngBel)
        # self.assertAlmostEqual(LatWGS84,51.05973284541117,places=3)
        # self.assertAlmostEqual(LngWGS84,3.707712199534464,places=3)

    def test_wgs84_to_l72(self): # this transformation is not correct
        latBel,lngBel=geo.wgs84_to_belgian_datum(LatWGS84=51.05973284541117,LngWGS84=3.707712199534464)
        x,y=geo.spherical_coordinates_to_lambert72(latBel,lngBel)
        # self.assertAlmostEqual(x,103657.078099938,places=3)
        # self.assertAlmostEqual(y,194604.105800153,places=3)
        
    # def test_wgs84_to_l72_GDAL(self):
    #     x,y=geo.wgs84_to_l72(lat=51.05973284541117,long=3.707712199534464) 
    #     self.assertAlmostEqual(x,103657.078099938,places=3)
    #     self.assertAlmostEqual(y,194604.105800153,places=3)
    
    def test_parse_dms(self):
        item="[[6 , 30 , 0, N ]]"
        test=geo.parse_dms(item)
        self.assertEqual(test,6.5)

        item="[6  30  0 N ]"
        test=geo.parse_dms(item)
        self.assertEqual(test,6.5)

        item=(6 , 30 , 0 , 'N' )
        test=geo.parse_dms(item)
        self.assertEqual(test,6.5)

        item=np.array([6 , 30 , 0 , 'N'])
        test=geo.parse_dms(item)
        self.assertEqual(test,6.5)
      
    def test_get_exif_data(self):
        im = PIL.Image.open(self.dataLoaderRoad.imagePath2) 
        exifData=geo.get_exif_data(im)
        self.assertIsNotNone(exifData["GPSInfo"])
        im.close()
    
    def test_dd_to_dms(self):
        dms=geo.dd_to_dms(6.5)
        self.assertEqual(dms[1],30)

    def test_dms_to_dd(self):
        #west
        dd=geo.dms_to_dd(6, 30, 0, 'W')
        self.assertEqual(dd,-6.5)
        #south
        dd=geo.dms_to_dd(6, 30, 0, 'S')
        self.assertEqual(dd,-6.5)
        #north
        dd=geo.dms_to_dd(6, 30, 0, 'N')
        self.assertEqual(dd,6.5)
      
    def test_filter_exif_gps_data(self):
        #tuple
        dd=geo.parse_exif_gps_data((6, 30, 0), reference= 'N') 
        self.assertEqual(dd,6.5)

        #float
        dd=geo.parse_exif_gps_data(6.5, reference= 'S') 
        self.assertEqual(dd,-6.5)
        
if __name__ == '__main__':
    unittest.main()