import unittest
import numpy as np
import PIL
import time
import os
import sys
#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.geospatialutils as gsu

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from geomapi.utils import GEOMAPI_PREFIXES

class TestGeoSpacialutils(unittest.TestCase):

    ################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        
        cls.dataLoaderParking = DATALOADERPARKINGINSTANCE
        cls.dataLoaderRoad = DATALOADERROADINSTANCE

        #TIME TRACKING 
        et = time.time()
        print("startup time: "+str(et - st))
        print('{:50s} {:5s} '.format('tests','time'))
        print('------------------------------------------------------')

    @classmethod
    def tearDownClass(cls):
        #execute once after all tests
        # if os.path.exists(cls.dataLoaderParking.resourcePath):
        #     shutil.rmtree(cls.dataLoaderParking.resourcePath)  
        print('-----------------TearDown Class----------------------')   
 


        
################################## SETUP/TEARDOWN ######################
    def setUp(self):
        #execute before every test
        self.startTime = time.time()   

    def tearDown(self):
        #execute after every test
        t = time.time() - self.startTime
        print('{:50s} {:5s} '.format(self._testMethodName,str(t)))

################################## TEST FUNCTIONS ######################

    def test_belgian_datum_to_from_wgs84(self):
        x,y = 103824.557675, 194661.815941
        lat, lon = 51.060265, 3.710094
        xC,yC = gsu.wgs84_to_belgian_datum(lat, lon)
        latC, lonC = gsu.belgian_datum_to_wgs84(x,y)
        self.assertAlmostEqual(lon,lonC,places=3)
        self.assertAlmostEqual(lat,latC,places=3)
        self.assertAlmostEqual(x,xC,places=1) # The reverse transformation is les precise
        self.assertAlmostEqual(y,yC,places=1) # The reverse transformation is les precise
    
    def test_parse_dms(self):
        item="[[6 , 30 , 0, N ]]"
        test=gsu.parse_dms(item)
        self.assertEqual(test,6.5)

        item="[6  30  0 N ]"
        test=gsu.parse_dms(item)
        self.assertEqual(test,6.5)

        item=(6 , 30 , 0 , 'N' )
        test=gsu.parse_dms(item)
        self.assertEqual(test,6.5)

        item=np.array([6 , 30 , 0 , 'N'])
        test=gsu.parse_dms(item)
        self.assertEqual(test,6.5)
      
    def test_get_exif_data(self):
        im = PIL.Image.open(self.dataLoaderRoad.imagePath2) 
        exifData=gsu.get_exif_data(im)
        self.assertIsNotNone(exifData["GPSInfo"])
        im.close()
    
    def test_dd_to_dms(self):
        dms=gsu.dd_to_dms(6.5)
        self.assertEqual(dms[1],30)

    def test_dms_to_dd(self):
        #west
        dd=gsu.dms_to_dd(6, 30, 0, 'W')
        self.assertEqual(dd,-6.5)
        #south
        dd=gsu.dms_to_dd(6, 30, 0, 'S')
        self.assertEqual(dd,-6.5)
        #north
        dd=gsu.dms_to_dd(6, 30, 0, 'N')
        self.assertEqual(dd,6.5)
      
    def test_filter_exif_gps_data(self):
        #tuple
        dd=gsu.parse_exif_gps_data((6, 30, 0), reference= 'N') 
        self.assertEqual(dd,6.5)

        #float
        dd=gsu.parse_exif_gps_data(6.5, reference= 'S') 
        self.assertEqual(dd,-6.5)
        
if __name__ == '__main__':
    unittest.main()