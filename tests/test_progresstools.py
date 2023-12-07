import copy
import os
from pathlib import Path
import shutil
import time
import unittest
import ifcopenshell
import open3d as o3d
import sys

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from geomapi.nodes import *
# import geomapi.utils.geometryutils as gmu
import geomapi.tools.progresstools as pt

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 

class TestProgressTools(unittest.TestCase):


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


    def test_determine_percentage_of_coverage(self):
        reference=self.dataLoaderParking.pcd
        sources=[self.dataLoaderParking.slabMesh,
                 self.dataLoaderParking.wallMesh,
                 self.dataLoaderParking.columnMesh,
                 self.dataLoaderParking.beamMesh]
        percentages=pt.determine_percentage_of_coverage(sources=sources,reference=reference)
        self.assertEqual(len(percentages), len(sources))
        self.assertLess(percentages[0],0.005)
        self.assertLess(percentages[1],0.001)
        self.assertLess(percentages[2],0.5)
        self.assertLess(percentages[3],0.001)
