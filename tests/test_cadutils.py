import os
import sys
import time
import unittest
import itertools

import open3d as o3d


#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.cadutils as cu

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from data_loader_railway import DATALOADERRAILWAYINSTANCE 

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

################################## TEST FUNCTIONS ######################

    def test_ezdxf_to_o3d(self):
        #mesh_to_arrays
        dxf = self.dataLoaderRailway.dxf
        geometry_groups,layer_groups=cu.ezdxf_to_o3d(dxf,explode_blocks=False,join_geometries=False)
        self.assertEqual(len(geometry_groups), 14)
        self.assertEqual(len(layer_groups), 14)
        self.assertEqual(len(list(itertools.chain(*geometry_groups))), 282)
        
    def test_calculate_angle_between_lines(self):
        point1 = [0,0]
        point2 = [1,0]
        point3= [1,1]
        angle = 45
        self.assertEqual(angle, cu.calculate_angle_between_lines([point1, point2],[point1, point3]))

    def test_calculate_perpendicular_distance(self):
        point1 = [0,0]
        point2 = [1,0]
        point3 = [0,1]
        point4 = [1,1]
        distance = 1
        self.assertEqual(distance, cu.calculate_perpendicular_distance([point1, point2],[point3, point4]))
        point1 = [0,0]
        point2 = [1,1]
        point3 = [0,2]
        point4 = [1,2]
        distance = 2
        self.assertEqual(distance, cu.calculate_perpendicular_distance([point1, point2],[point3, point4]))

    def test_sample_pcd_from_linesets(self):
        points = o3d.utility.Vector3dVector([[0,0,0],[1.0,1,1]])
        lines = o3d.utility.Vector2iVector([[0,1]])
        lineset = o3d.geometry.LineSet(points = points, lines = lines)
        pcd,id = cu.sample_pcd_from_linesets([lineset])
        self.assertEqual(len(pcd.points), 18)

    def test_get_rgb_from_aci(self):
        self.assertEqual(cu.get_rgb_from_aci(0), (0,0,0))
        self.assertEqual(cu.get_rgb_from_aci(7), (255,255,255))

if __name__ == '__main__':
    unittest.main()
