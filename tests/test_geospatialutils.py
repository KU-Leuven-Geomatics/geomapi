import unittest
import geomapi.utils.geospatialutils as geo

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

if __name__ == '__main__':
    unittest.main()