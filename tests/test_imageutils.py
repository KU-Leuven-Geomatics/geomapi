import unittest
import numpy as np
import open3d as o3d
import os
import sys
import time

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.imageutils as iu

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 


 ################################## SETUP/TEARDOWN MODULE ######################

# def setUpModule():
#     #execute once before the module 
#     print('-----------------Setup Module----------------------')

# def tearDownModule():
#     #execute once after the module 
#     print('-----------------TearDown Module----------------------')



class TestCompletionTools(unittest.TestCase):
    
    

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

    # Testing the Utils

    def test_create_transformation_matrix(self):
        R = np.array([[1,2,3],[4,5,6],[7,8,9]])
        t = np.array([10,11,12])
        T = iu.create_transformation_matrix(R,t)
        test_T = np.array([[1,2,3,10],[4,5,6,11],[7,8,9,12],[0,0,0,1]])
        self.assertEqual (T.all(),test_T.all())

    def test_split_transformation_matrix(self):
        T = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        test_R = np.array([[1,2,3],[5,6,7],[9,10,11]])
        test_t = np.array([4,8,12])
        R,t = iu.split_transformation_matrix(T)
        self.assertEqual (test_R.all(),R.all())
        self.assertEqual (test_t.all(),t.all())
    
    



if __name__ == '__main__':
    unittest.main()
