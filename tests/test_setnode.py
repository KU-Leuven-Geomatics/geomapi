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
from geomapi.tools import graph_path_to_nodes

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
        cls.nodes=graph_path_to_nodes(cls.dataLoaderParking.resourceGraphPath,getResource=True)
        
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
        self.assertEqual(len(node.linkedNodes),0)
        self.assertEqual(len(node.linkedSubjects),0)
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
    
    def test_resources(self):
        #1 resource
        resource = o3d.geometry.TriangleMesh.create_box(width=10.0, height=10.0, depth=10.0)
        
        node= SetNode(resource=resource)
        #check resource
        self.assertIsNotNone(node.resource)
        self.assertTrue(np.allclose(node.resource.get_center(),resource.get_center(),atol=0.001))
        #check orientedBoundingBox -> should be based on the resource
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),resource.get_center(),atol=0.001))
        #check convexHull -> should be based on the resource
        self.assertTrue(np.allclose(node.convexHull.get_center(),resource.get_center(),atol=0.001))
        #check cartesianTransform -> should be center of resource
        self.assertTrue(np.allclose(node.get_center(),resource.get_center(),atol=0.001))
        
        

        
        node= SetNode(resource=[self.pcdNode.resource,self.meshNode.resource,self.bimNode.resource])
        
        #check resource -> convexhull of all geometries
        self.assertIsNotNone(node.resource)
        self.assertTrue(np.allclose(node.resource.get_center(),self.big_hull.get_center(),atol=0.001))
        #check orientedBoundingBox -> should be based on the resource
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),self.big_box.get_center(),atol=0.001))
        #check convexHull -> should be based on the resource
        self.assertTrue(np.allclose(node.convexHull.get_center(),self.big_hull.get_center(),atol=0.001))
        #check cartesianTransform -> should be center of resource
        self.assertTrue(np.allclose(node.get_center(),self.big_hull.get_center(),atol=0.001))
        #check linkedNodes
        self.assertEqual(len(node.linkedNodes),3)
        self.assertEqual(len(node.linkedSubjects),3)
        self.assertTrue(all(True for n in node.linkedNodes if n.resource is not None))
        #check if first linkedNode is a MeshNode
        self.assertTrue(isinstance(node.linkedNodes[0],PointCloudNode))
        #check if second linkedNode is a PointCloudNode
        self.assertTrue(isinstance(node.linkedNodes[1],MeshNode))
        

        #test invalid resource
        self.assertRaises(ValueError,SetNode,resource='dfsgsdfgsd')

        
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
        #check cartesianTransform -> should be center of resource
        self.assertTrue(np.allclose(node.cartesianTransform,big_center,atol=0.001))
        #check linkedNodes
        self.assertEqual(len(node.linkedNodes),4)
        self.assertEqual(len(node.linkedSubjects),4)
        self.assertTrue(all(True for n in node.linkedNodes if n.resource is not None))
        
    def test_get_graph(self):
        node= SetNode(linkedNodes=self.nodes)
        g=node.get_graph()
        self.dataLoaderParking.setGraph
        
        #compare the graphs
        self.assertEqual(len(g),len(self.dataLoaderParking.setGraph))
        self.assertTrue(all(True for s in g.subjects(RDF.type) if s in self.dataLoaderParking.setGraph.subjects(RDF.type)))
        self.assertTrue(all(True for p in g.predicates() if p in self.dataLoaderParking.setGraph.predicates()))
        self.assertTrue(all(True for o in g.objects() if o in self.dataLoaderParking.setGraph.objects()))
        
    def test_set_graph(self):
        node= SetNode(graph=self.dataLoaderParking.setGraph)
        self.assertEqual(len(node.linkedSubjects),6)
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderParking.setGraph.triples((None, None, None)):
            if 'cartesianTransform' in p.toPython():
                matrix=ut.literal_to_matrix(o)
                #check if matrix elements are the same as the node cartesianTransform
                self.assertTrue(np.allclose(matrix,node.cartesianTransform,atol=0.001))
            if 'orientedBoundingBox' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                node_param=gmu.get_oriented_bounding_box_parameters(node.orientedBoundingBox)
                self.assertTrue(np.allclose(graph_param,node_param,atol=0.001))
            if 'convexHull' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                graph_volume=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(graph_param)).compute_convex_hull()[0].get_volume()
                node_volume=node.convexHull.get_volume()
                self.assertAlmostEqual(graph_volume,node_volume,delta=0.01)
            if 'focalLength35mm' in p.toPython():
                self.assertEqual(float(o),node.focalLength35mm)
            if 'principalPointU' in p.toPython():
                self.assertEqual(float(o),node.principalPointU)
            if 'principalPointV' in p.toPython():
                self.assertEqual(float(o),node.principalPointV)
            if 'imageWidth' in p.toPython():
                self.assertEqual(float(o),node.imageWidth)
            if 'imageLength' in p.toPython():
                self.assertEqual(float(o),node.imageHeight)
            if 'intrinsicMatrix' in p.toPython():
                matrix=ut.literal_to_matrix(o)
                self.assertTrue(np.allclose(matrix,node.intrinsicMatrix,atol=0.001))
       
       
    def test_resources_graph(self):        
        #big hull
        points=o3d.utility.Vector3dVector()
        for node in self.nodes:
            points.extend(node.convexHull.vertices)
        pcd= o3d.geometry.PointCloud()
        pcd.points=points
        big_hull, _ =pcd.compute_convex_hull()
        #big box
        big_box =big_hull.get_oriented_bounding_box()
        #big center
        big_center = gmu.get_cartesian_transform(translation=big_hull.get_center())
        
        
        node= SetNode(graph=self.dataLoaderParking.resourceGraph)
        self.assertEqual(len(node.linkedSubjects),6)
        self.assertEqual(len(node.linkedNodes),6)
        self.assertTrue(all(True for n in node.linkedNodes if n.resource is not None))
        
        self.assertTrue(np.allclose(node.resource.get_center(),big_hull.get_center(),atol=0.001))
        #check orientedBoundingBox -> should be based on the resource
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),big_box.get_center(),atol=0.001))    
        #check convexHull -> should be based on the resource
        self.assertTrue(np.allclose(node.convexHull.get_center(),big_hull.get_center(),atol=0.001))
        #check cartesianTransform -> should be center of resource
        self.assertTrue(np.allclose(node.cartesianTransform,big_center,atol=0.001))

    def test_combined_graph(self):
        g=self.dataLoaderParking.setGraph+self.dataLoaderParking.resourceGraph
        node= SetNode(graph=g)
        self.assertEqual(node.subject,next(self.dataLoaderParking.setGraph.subjects(RDF.type)))
        self.assertEqual(len(node.linkedNodes),len(list(self.dataLoaderParking.resourceGraph.subjects(RDF.type))))

    def test_set_graph_path(self):
        subject=next(self.dataLoaderParking.setGraph.subjects(RDF.type))
        node= SetNode(graphPath=self.dataLoaderParking.setGraphPath, subject=subject)
        self.assertEqual(node.subject,subject)
        
    def test_get_linked_nodes_from_self_linked_subjects(self):
        node= SetNode(graph=self.dataLoaderParking.setGraph)
        self.assertEqual(len(node.linkedSubjects),6)
        self.assertEqual(len(node.linkedNodes),0)
        node.get_linked_nodes(self.dataLoaderParking.resourceGraph)
        self.assertEqual(len(node.linkedNodes),6)
        
    def test_get_linked_nodes_from_other_linked_subjects(self):
        node= SetNode(graph=self.dataLoaderParking.setGraph)
        self.assertEqual(len(node.linkedSubjects),6)
        self.assertEqual(len(node.linkedNodes),0)
        node.get_linked_nodes(self.dataLoaderParking.resourceGraph)
        self.assertEqual(len(node.linkedNodes),6)

    def test_add_linked_nodes(self):
        combinedGraph=self.dataLoaderParking.setGraph+self.dataLoaderParking.resourceGraph
        node= SetNode(graph=combinedGraph)
        node.set_linked_nodes(Node())
        self.assertEqual(len(node.linkedNodes),7)
        self.assertEqual(len(node.linkedSubjects),7)

    def test_add_linked_nodes_with_doubles(self):
        #don't add the same node twice
        node= SetNode(graph=self.dataLoaderParking.resourceGraph)
        node.set_linked_nodes(self.pcdNode)
        self.assertEqual(len(node.linkedNodes),6)


    def test_save_linked_resources(self):  
        node= SetNode(graph=self.dataLoaderParking.resourceGraph)
        node.save_linked_resources(self.dataLoaderParking.resourcePath)

    # def test_get_linked_resources(self):
    #     node= SetNode(graphPath=self.combinedGraphPath)
    #     resources=node.get_linked_resources()
    #     self.assertEqual(len(resources),len(node.linkedNodes))

    # def test_get_linked_resources_multiprocessing(self):
    #     node= SetNode(graphPath=self.combinedGraphPath)
    #     resources=node.get_linked_resources_multiprocessing()
    #     self.assertEqual(len(resources),len(node.linkedNodes))

    def test_linked_nodes_to_graph(self):  
        combinedGraph=self.dataLoaderParking.resourceGraph
        node= SetNode(graph=combinedGraph)
        graph=node.linked_nodes_to_graph(os.path.join(self.dataLoaderParking.resourcePath,'linkednodesGraph.ttl'))
        #check if all linkedNodes are in the graph
        for n in node.linkedNodes:
            self.assertTrue(n.subject in graph.subjects(RDF.type))
    
    def test_set_to_graph(self):   
        combinedGraph=self.dataLoaderParking.resourceGraph
     
        node= SetNode(graph=combinedGraph)
        graph=node.set_to_graph(os.path.join(self.dataLoaderParking.resourcePath,'combinedGraph.ttl'))
        #check if all linkedNodes are in the graph
        for n in node.linkedNodes:
            self.assertTrue(n.subject in graph.subjects(RDF.type))
        #check if the node is in the graph
        self.assertTrue(node.subject in graph.subjects(RDF.type))

if __name__ == '__main__':
    unittest.main()
