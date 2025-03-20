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
import geomapi.tools.validationtools as vt

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from geomapi.utils import GEOMAPI_PREFIXES

class TestValidationTools(unittest.TestCase):


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


    def test_decode_depthmap(self):
        vt.decode_depthmap()

    def test_plot_pano_positions(self):
        vt.plot_pano_positions()

    def test_get_heading(self):
        vt.get_heading()

    def test_get_zenit(self):
        vt.get_zenit()

    def test_navvis_csv_to_nodes(self):
        vt.navvis_csv_to_nodes()

    def test_get_boundingbox_of_list_of_geometries(self):
        vt.get_boundingbox_of_list_of_geometries()

    def test_match_BIM_points(self):
        vt.match_BIM_points()

    def test_compute_LOA(self):
        vt.compute_LOA()

    def test_plot_histogram(self):
        vt.plot_histogram()

    def test_color_point_cloud_by_LOA(self):
        vt.color_point_cloud_by_LOA()

    def test_color_point_cloud_by_distance(self):
        vt.color_point_cloud_by_distance()

    def test_csv_by_LOA(self):
        vt.csv_by_LOA()

    def test_excel_by_LOA(self):
        vt.excel_by_LOA()

    def test_color_BIMNode(self):
        vt.color_BIMNode()

    def test_cad_show_lines(self):
        vt.cad_show_lines()

    def test_sample_pcd_from_linesets(self):
        vt.sample_pcd_from_linesets()

    def test_get_linesets_inliers_in_box(self):
        vt.get_linesets_inliers_in_box()

    def test_create_selection_box_from_image_boundary_points(self):
        vt.create_selection_box_from_image_boundary_points()
