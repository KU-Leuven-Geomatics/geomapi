import copy
import math
import os
import sys
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import itertools

import cv2
import ifcopenshell
import numpy as np
import open3d as o3d
import pye57
import ifcopenshell.util.selector

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.cadutils as cadu

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from data_loader_railway import DATALOADERRAILWAYINSTANCE 
from geomapi.utils import GEOMAPI_PREFIXES


################################## SETUP/TEARDOWN MODULE ######################

# def setUpModule():
#     #execute once before the module 
#     print('-----------------Setup Module----------------------')

# def tearDownModule():
#     #execute once after the module 
#     print('-----------------TearDown Module----------------------')



class TestCadutils(unittest.TestCase):





################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        
        cls.dataLoaderParking = DATALOADERPARKINGINSTANCE
        cls.dataLoaderRoad = DATALOADERROADINSTANCE
        cls.dataLoaderRailway = DATALOADERRAILWAYINSTANCE

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
        
        

################################## FIXTURES ######################
    # # @pytest.fixture(scope='module')
    # # @pytest.fixture
    # def test_data(*args):
    #     here = os.path.split(__file__)[0]
    #     return os.path.join(here, "testfiles", *args)

    # @pytest.fixture
    # def e57Path1():
    #     return test_data("pointcloud.e57")

    # @pytest.fixture
    # def ifcData():
    #     ifcPath=os.path.join(os.getcwd(),"testfiles", "ifcfile.ifc")
    #     classes= '.IfcBeam | .IfcColumn | .IfcWall | .IfcSlab'
    #     ifc = ifcopenshell.open(ifcPath)   
    #     selector = Selector()
    #     dataList=[]
    #     for ifcElement in selector.parse(ifc, classes): 
    #         dataList.append(ifcElement)
    #     return dataList
    
    
    

################################## TEST FUNCTIONS ######################

    def test_ezdxf_to_o3d(self):
        #mesh_to_arrays
        dxf = self.dataLoaderRailway.dxf
        geometry_groups,layer_groups=cadu.ezdxf_to_o3d(dxf,explode_blocks=False,join_geometries=False)
        self.assertEqual(len(geometry_groups), 14)
        self.assertEqual(len(layer_groups), 14)
        self.assertEqual(len(list(itertools.chain(*geometry_groups))), 282)
        


if __name__ == '__main__':
    unittest.main()
