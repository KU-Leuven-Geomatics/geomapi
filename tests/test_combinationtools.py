from pathlib import Path
import unittest
import numpy as np
import open3d as o3d
import os
import time

import geomapi.tools.combinationtools as ct

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
        cls.path= Path.cwd() / "tests" / "testfiles" 
       
        # Import 3d models
        cls.old_state = o3d.io.read_triangle_mesh(str(cls.path / "mesh" / "old_state.obj"))
        cls.new_state = o3d.io.read_triangle_mesh(str(cls.path / "mesh" / "new_state.obj"))
   
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

    def test_combine_geometry(self):
        newGeo = ct.combine_geometry(self.old_state, self.new_state)

if __name__ == '__main__':
    unittest.main()
