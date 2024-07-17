import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import sys
import cv2
import numpy as np
import open3d as o3d
import rdflib
from rdflib import RDF, RDFS, Graph, Literal, URIRef

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.geometryutils as gmu
from geomapi.nodes import *

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 

class TestNode(unittest.TestCase):



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

    def test_Node_creation_from_subject(self):
        #subject
        subject='myNode'
        node= GeometryNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'file:///'+subject)

    def test_Node_creation_with_cartesianBounds(self):
        #None
        cartesianBounds=None
        node= GeometryNode(cartesianBounds=cartesianBounds)
        self.assertIsNone(node.cartesianBounds)

        #np.array(6x1)
        cartesianBounds=np.array([1,2,3,4,5,6])
        node= GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.cartesianBounds.size,6)   

        #list
        cartesianBounds=[1,2,3,4,5,6]
        node= GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.cartesianBounds.size,6)   

        #np.array(other)
        cartesianBounds=np.array([[1,2],[3,4],[5,6]])
        node= GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.cartesianBounds.size,6)   

        #orientedBoundingBox
        node= GeometryNode(cartesianBounds=self.dataLoaderRoad.mesh.get_oriented_bounding_box())
        self.assertEqual(node.cartesianBounds.size,6)   

        #Vector3dVector
        cartesianBounds=gmu.get_oriented_bounds(np.array([1,2,3,4,5,6]) )
        node= GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.cartesianBounds.size,6)   

        #orientedBounds
        cartesianBounds=np.asarray(gmu.get_oriented_bounds(np.array([1,2,3,4,5,6]) ))
        node= GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.cartesianBounds.size,6)   

        #error
        self.assertRaises(ValueError,GeometryNode,cartesianBounds='thisisnotacartesianbound')


    def test_Node_creation_with_orientedBounds(self):
        #None
        orientedBounds=None
        node= GeometryNode(orientedBounds=orientedBounds)
        self.assertIsNone(node.orientedBounds)

        #np.array(8x3)
        orientedBounds=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]])
        node= GeometryNode(orientedBounds=orientedBounds)
        self.assertEqual(node.orientedBounds.size,24)   

        #list
        orientedBounds=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        node= GeometryNode(orientedBounds=orientedBounds)
        self.assertEqual(node.orientedBounds.size,24)   

        #np.array(other)
        orientedBounds=np.array([[1,4,7,10,13,16,19,22],
                                  [2,5,8,11,14,17,20,23],
                                  [3,6,9,12,15,18,21,24]])
        node= GeometryNode(orientedBounds=orientedBounds)
        self.assertEqual(node.orientedBounds.size,24)   

        #orientedBoundingBox
        node= GeometryNode(orientedBounds=self.dataLoaderRoad.mesh.get_oriented_bounding_box())
        self.assertEqual(node.orientedBounds.size,24)   

        #Vector3dVector
        points=o3d.utility.Vector3dVector(np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]]))   
        node= GeometryNode(orientedBounds=points)
        self.assertEqual(node.orientedBounds.size,24)   

        #error
        self.assertRaises(ValueError,GeometryNode,orientedBounds='thisisnotaorientedbound')


    def test_Node_creation_with_orientedBoundingBox(self):
        #None
        orientedBoundingBox=None
        node= GeometryNode(orientedBoundingBox=orientedBoundingBox)
        self.assertIsNone(node.orientedBoundingBox)

        #box
        node= GeometryNode(orientedBoundingBox=self.dataLoaderRoad.mesh.get_oriented_bounding_box())
        self.assertAlmostEqual(node.orientedBoundingBox.get_min_bound()[0],self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_min_bound()[0],delta=0.01)   

        #np.array(nx3)
        orientedBoundingBox=self.dataLoaderParking.mesh.get_oriented_bounding_box()
        points=np.asarray(orientedBoundingBox.get_box_points())
        node= GeometryNode(orientedBoundingBox=points)
        self.assertAlmostEqual(node.orientedBoundingBox.get_min_bound()[0],orientedBoundingBox.get_min_bound()[0],delta=0.01)   

        #geometry
        node= GeometryNode(orientedBoundingBox=self.dataLoaderParking.mesh)
        self.assertAlmostEqual(node.orientedBoundingBox.get_min_bound()[0],orientedBoundingBox.get_min_bound()[0],delta=0.01)   

        #Vector3dVector
        orientedBoundingBox=self.dataLoaderParking.mesh.get_oriented_bounding_box()
        points=orientedBoundingBox.get_box_points()
        node= GeometryNode(orientedBoundingBox=points)
        self.assertAlmostEqual(node.orientedBoundingBox.get_min_bound()[0],orientedBoundingBox.get_min_bound()[0],delta=0.01)   

    def test_Node_creation_with_cartesian_transform(self):
        #None
        cartesianTransform=None
        node= GeometryNode(cartesianTransform=cartesianTransform)
        self.assertIsNone(node.cartesianTransform)

        #center np.array(1x3)        
        center=self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_center()
        node= GeometryNode(cartesianTransform=center)
        self.assertAlmostEqual(node.cartesianTransform[0,3],center[0],delta=0.01)   

        # #cartesianBounds np.array(1x6)
        # cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderRoad.mesh.get_oriented_bounding_box())
        # node= GeometryNode(cartesianTransform=cartesianBounds)
        # self.assertAlmostEqual(node.cartesianTransform[0,3],(cartesianBounds[0]+cartesianBounds[1])/2,delta=0.01)   

        # #orientedBounds np.array(8x3)
        # box=self.dataLoaderRoad.mesh.get_oriented_bounding_box()
        # orientedBounds=np.asarray(box.get_box_points())
        # node= GeometryNode(cartesianTransform=orientedBounds)
        # self.assertAlmostEqual(node.cartesianTransform[0,3],center[0],delta=0.01)   

        # #np.ndarray(nx3)
        # array=np.concatenate((orientedBounds,orientedBounds),0)
        # node= GeometryNode(cartesianTransform=array)
        # self.assertAlmostEqual(node.cartesianTransform[0,3],center[0],delta=0.01)   

        #geometry
        node= GeometryNode(cartesianTransform=self.dataLoaderRoad.mesh)
        self.assertAlmostEqual(node.cartesianTransform[0,3],self.dataLoaderRoad.mesh.get_center()[0],delta=0.01)   

        #Vector3dVector
        points=self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_box_points()
        node= GeometryNode(cartesianTransform=points)
        self.assertAlmostEqual(node.cartesianTransform[0,3],center[0],delta=0.01)   

    def test_get_oriented_bounding_box(self):

        #empty
        node=GeometryNode()        
        self.assertIsNone(node.get_oriented_bounding_box())

        #orientedBoundingBox
        box=self.dataLoaderParking.mesh.get_oriented_bounding_box()
        node= GeometryNode(orientedBoundingBox=box)
        self.assertAlmostEqual(node.get_oriented_bounding_box().get_min_bound()[0],box.get_min_bound()[0],delta=0.01)   

        #cartesianBounds
        cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderParking.mesh)
        node= GeometryNode(cartesianBounds=cartesianBounds)
        self.assertAlmostEqual(node.get_oriented_bounding_box().get_min_bound()[0],self.dataLoaderParking.mesh.get_axis_aligned_bounding_box().get_min_bound()[0],delta=0.01)   

        #orientedBounds
        orientedBounds=gmu.get_oriented_bounds(cartesianBounds)
        node= GeometryNode(orientedBounds=orientedBounds)
        self.assertAlmostEqual(node.get_oriented_bounding_box().get_min_bound()[0],self.dataLoaderParking.mesh.get_axis_aligned_bounding_box().get_min_bound()[0],delta=0.01)   

    def test_get_cartesian_transform(self):

        #emtpy
        node=GeometryNode()        
        self.assertIsNone(node.get_cartesian_transform())

        #cartesianTransform
        node=GeometryNode(cartesianTransform=self.dataLoaderParking.imageCartesianTransform1)
        self.assertEqual(node.get_cartesian_transform()[0,3], self.dataLoaderParking.imageCartesianTransform1[0,3])

        # #cartesianBounds
        # cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderParking.mesh)
        # node=GeometryNode(cartesianBounds=cartesianBounds)
        # self.assertEqual(node.get_cartesian_transform()[0,3], (cartesianBounds[0]+cartesianBounds[1])/2)

        # #orientedBounds
        orientedBounds=np.asarray(self.dataLoaderParking.mesh.get_oriented_bounding_box().get_box_points())
        # node=GeometryNode(orientedBounds=orientedBounds)
        # self.assertEqual(node.get_cartesian_transform()[0,3], np.mean(orientedBounds,0)[0])

        #orientedBoundingBox
        node=GeometryNode(orientedBoundingBox=self.dataLoaderParking.mesh.get_oriented_bounding_box())
        self.assertEqual(node.get_cartesian_transform()[0,3], np.mean(orientedBounds,0)[0])

        #resource
        node=GeometryNode(resource=self.dataLoaderParking.mesh)
        self.assertEqual(node.get_cartesian_transform()[0,3],  self.dataLoaderParking.mesh.get_center()[0])

    def test_get_cartesian_bounds(self):
        #emtpy
        node=GeometryNode()        
        self.assertIsNone(node.get_cartesian_bounds())

        #cartesianBounds
        cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderParking.mesh)
        node=GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.get_cartesian_bounds()[0], cartesianBounds[0])

        #orientedBounds
        orientedBounds=np.asarray(self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_box_points())
        node=GeometryNode(orientedBounds=orientedBounds)
        self.assertAlmostEqual(node.get_cartesian_bounds()[0],self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_min_bound()[0],delta=0.01)

        #orientedBoundingBox
        node=GeometryNode(orientedBoundingBox=self.dataLoaderRoad.mesh.get_oriented_bounding_box())
        self.assertEqual(node.get_cartesian_bounds()[0],  self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_min_bound()[0])

        #resource
        node=GeometryNode(resource=self.dataLoaderRoad.mesh)
        self.assertEqual(node.get_cartesian_bounds()[0], self.dataLoaderRoad.mesh.get_min_bound()[0])


    def test_get_oriented_bounds(self):
        #emtpy
        node=GeometryNode()        
        self.assertIsNone(node.get_oriented_bounds())

        #cartesianBounds
        cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderRoad.mesh)
        node=GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.get_oriented_bounds()[0,0], cartesianBounds[0])

        #orientedBounds
        box=self.dataLoaderRoad.mesh.get_oriented_bounding_box()
        orientedBounds=np.asarray(box.get_box_points())
        node=GeometryNode(orientedBounds=orientedBounds)
        self.assertEqual(node.get_oriented_bounds()[2,2], orientedBounds[2,2])

        #orientedBoundingBox
        node=GeometryNode(orientedBoundingBox=self.dataLoaderRoad.mesh.get_oriented_bounding_box())
        self.assertEqual(node.get_oriented_bounds()[2,2],  np.asarray(self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_box_points())[2,2])

        #resource
        node=GeometryNode(resource=self.dataLoaderRoad.mesh)
        self.assertEqual(node.get_oriented_bounds()[2,2],  np.asarray(box.get_box_points())[2,2])

    def test_get_center(self):
        #emtpy
        node=GeometryNode()        
        self.assertIsNone(node.get_center())

        #cartesianBounds
        cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderParking.mesh)
        node=GeometryNode(cartesianBounds=cartesianBounds)
        self.assertEqual(node.get_center()[0], (cartesianBounds[0]+cartesianBounds[1])/2)

        #orientedBounds
        orientedBounds=np.asarray(self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_box_points())
        node=GeometryNode(orientedBounds=orientedBounds)
        self.assertEqual(node.get_center()[0], np.mean(orientedBounds,0)[0])

        #orientedBoundingBox
        node=GeometryNode(orientedBoundingBox=self.dataLoaderRoad.mesh.get_oriented_bounding_box())
        self.assertEqual(node.get_center()[0],  self.dataLoaderRoad.mesh.get_oriented_bounding_box().get_center()[0])

        #resource
        node=GeometryNode(resource=self.dataLoaderRoad.mesh)
        self.assertEqual(node.get_center()[0],  self.dataLoaderRoad.mesh.get_center()[0])

if __name__ == '__main__':
    unittest.main()
