import os
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import open3d as o3d
import rdflib
from rdflib import RDF, RDFS, Graph, Literal, URIRef
import sys
import numpy as np
#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from geomapi.nodes import *
from geomapi.tools import graph_to_nodes

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
# from data_loader_road import DATALOADERROADINSTANCE 
# from data_loader_railway import DATALOADERRAILWAYINSTANCE 

from geomapi.utils import GEOMAPI_PREFIXES
import geomapi.utils.geometryutils as gmu
import geomapi.utils as ut

class TestSetNode(unittest.TestCase):

################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        
        
        cls.dataLoaderParking = DATALOADERPARKINGINSTANCE
        
        #pointcloud
        cls.nodes=graph_to_nodes(cls.dataLoaderParking.resourceGraphPath,loadResource=True)
        
        cls.pcdNode=next(n for n in cls.nodes if isinstance(n,PointCloudNode))
        cls.meshNode=next(n for n in cls.nodes if isinstance(n,MeshNode))
        cls.imageNode=next(n for n in cls.nodes if isinstance(n,ImageNode))
        cls.bimNode=next(n for n in cls.nodes if isinstance(n,BIMNode))
        
        #big hull
        points=o3d.utility.Vector3dVector()
        for node in [cls.pcdNode.convexHull,cls.meshNode.convexHull,cls.bimNode.convexHull]:
            points.extend(node.vertices)
        pcd= o3d.geometry.PointCloud()
        pcd.points=points
        cls.big_hull, _ =pcd.compute_convex_hull()
        #big box
        cls.big_box =cls.big_hull.get_oriented_bounding_box()
        #big center
        cls.big_center = gmu.get_cartesian_transform(translation=cls.big_hull.get_center())
        
        

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

    def test_empty(self):
        node= SetNode()
        self.assertIsNotNone(node.subject)
        self.assertIsNone(node.linkedNodes)
        self.assertIsNone(node.linkedSubjects)
        self.assertTrue(np.allclose(node.cartesianTransform,np.array([[1,0,0,0],
                                                                                    [0,1,0,0],
                                                                                    [0,0,1,0],
                                                                                    [0,0,0,1]]),atol=0.001))
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,0,0]),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([0,0,0]),atol=0.001))
    
    def test_convexHull(self):

        node= SetNode(convexHull=self.pcdNode.convexHull)
        #check convexHull
        self.assertIsNotNone(node.convexHull)
        self.assertTrue(np.allclose(node.convexHull.get_center(),self.pcdNode.convexHull.get_center(),atol=0.001))
        #check orientedBoundingBox -> should be based on the convexHull
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),self.pcdNode.orientedBoundingBox.get_center(),atol=0.001))
        #check cartesianTransform -> should be center of convexHull
        self.assertTrue(np.allclose(node.get_center(),self.pcdNode.convexHull.get_center(),atol=0.001))
        
    def test_orientedBoundingBox(self):

        node= SetNode(orientedBoundingBox=self.meshNode.orientedBoundingBox)
        #check orientedBoundingBox
        self.assertIsNotNone(node.orientedBoundingBox)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),self.meshNode.orientedBoundingBox.get_center(),atol=0.001))
        #check convexHull -> should be based on the orientedBoundingBox
        self.assertTrue(np.allclose(node.convexHull.get_center(),self.meshNode.orientedBoundingBox.get_center(),atol=0.001))
        #check cartesianTransform -> should be center of orientedBoundingBox
        self.assertTrue(np.allclose(node.get_center(),self.meshNode.orientedBoundingBox.get_center(),atol=0.001))    

    def test_cartesianTransform(self):
        transform=np.array([[ 1,  0, 0,10],
                            [0, 1, 0,20],
                            [0,  0, 1,30],
                            [ 0,  0, 0,1]])
        node= SetNode(cartesianTransform=transform)
        #check cartesianTransform
        self.assertIsNotNone(node.cartesianTransform)
        self.assertTrue(np.allclose(node.get_center(),transform[:3,3],atol=0.001))
        #check resource -> should be the same as the cartesianTransform
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),transform[:3,3],atol=0.001))
        #check convexHull -> should be based on the cartesianTransform
        self.assertTrue(np.allclose(node.convexHull.get_center(),transform[:3,3],atol=0.001))
    


        
    def test_linked_nodes(self):
        
        
        #big hull
        points=o3d.utility.Vector3dVector()
        for node in [self.pcdNode.convexHull,self.meshNode.convexHull,self.bimNode.convexHull,self.imageNode.convexHull]:
            points.extend(node.vertices)
        pcd= o3d.geometry.PointCloud()
        pcd.points=points
        big_hull, _ =pcd.compute_convex_hull()
        #big box
        big_box =big_hull.get_oriented_bounding_box()
        #big center
        big_center = gmu.get_cartesian_transform(translation=big_hull.get_center())
        
        
        node= SetNode(linkedNodes=[self.meshNode,self.pcdNode,self.bimNode,self.imageNode])
        
        #check orientedBoundingBox -> should be based on the resource
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),big_box.get_center(),atol=0.001))    
        #check convexHull -> should be based on the resource
        self.assertTrue(np.allclose(node.convexHull.get_center(),big_hull.get_center(),atol=0.001))
        #check cartesianTransform -> should be unaffected as the center of the space
        self.assertTrue(np.allclose(node.cartesianTransform,np.eye(4),atol=0.001))
        #check linkedNodes
        self.assertEqual(len(node.linkedNodes),4)
        self.assertEqual(len(node.linkedSubjects),4)
        self.assertTrue(all(True for n in node.linkedNodes if n.resource is not None))
        
    def test_get_graph(self):
        node= SetNode(linkedNodes=self.nodes)
        g=node.get_graph(addLinkedNodes=False)
        self.dataLoaderParking.setGraph
        
        #compare the graphs
        self.assertEqual(len(g),len(self.dataLoaderParking.setGraph))
        self.assertTrue(all(True for s in g.subjects(RDF.type) if s in self.dataLoaderParking.setGraph.subjects(RDF.type)))
        self.assertTrue(all(True for p in g.predicates() if p in self.dataLoaderParking.setGraph.predicates()))
        self.assertTrue(all(True for o in g.objects() if o in self.dataLoaderParking.setGraph.objects()))


    def test_save_linked_resources(self):  
        node= SetNode(linkedNodes=self.nodes)
        node.save_linked_resources(self.dataLoaderParking.resourcePath)

    def test_transform(self):
        node= SetNode(linkedNodes=self.nodes)
        transformation = np.array([[0, 0, 1, 1],
                                   [0, 1, 0, 2],
                                   [1, 0, 0, 3],
                                   [0, 0, 0, 1]])
        initialLinkedNodeTransform = node.linkedNodes[0].cartesianTransform
        
        node.transform(transformation)
        np.testing.assert_almost_equal(transformation, node.cartesianTransform)
        np.testing.assert_almost_equal(node.linkedNodes[0].cartesianTransform, transformation @ initialLinkedNodeTransform)
         

    # def test_get_linked_resources(self):
    #     node= SetNode(graphPath=self.combinedGraphPath)
    #     resources=node.get_linked_resources()
    #     self.assertEqual(len(resources),len(node.linkedNodes))

    # def test_get_linked_resources_multiprocessing(self):
    #     node= SetNode(graphPath=self.combinedGraphPath)
    #     resources=node.get_linked_resources_multiprocessing()
    #     self.assertEqual(len(resources),len(node.linkedNodes))



if __name__ == '__main__':
    unittest.main()
