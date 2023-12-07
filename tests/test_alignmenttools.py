import unittest
import numpy as np
import open3d as o3d
import os
import time

import geomapi.tools.alignmenttools as at


 ################################## SETUP/TEARDOWN MODULE ######################

# def setUpModule():
#     #execute once before the module 
#     print('-----------------Setup Module----------------------')

# def tearDownModule():
#     #execute once after the module 
#     print('-----------------TearDown Module----------------------')

class TestAlignmentTools(unittest.TestCase):

 ################################## SETUP/TEARDOWN CLASS ######################
  
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        cls.path=os.path.join(os.getcwd(), "tests","testfiles") #os.pardir, 
       
        # Import 3d models
   
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

    def test_tools(self):
        self.assertEqual(1,1)



if __name__ == '__main__':
    unittest.main()