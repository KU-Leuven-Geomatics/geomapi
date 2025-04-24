import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import numpy as np
import cv2
import open3d as o3d
import pye57
import rdflib
from rdflib import RDF, RDFS, Graph, Literal, URIRef
import sys

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
from geomapi.nodes import PointCloudNode

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from geomapi.utils import GEOMAPI_PREFIXES

class TestPointcloudNode(unittest.TestCase):



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

    def test_node_creation_from_graph(self):
        node= PointCloudNode(graph=self.dataLoaderParking.pcdGraph, subject= self.dataLoaderParking.pcdSubject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderParking.pcdGraph.triples(( self.dataLoaderParking.pcdSubject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual(Path(o.toPython()).resolve(),os.getcwd()/node.path) #not sure
            if 'cartesianTransform' in p.toPython():
                matrix=ut.literal_to_matrix(o)
                #check if matrix elements are the same as the node cartesianTransform
                self.assertTrue(np.allclose(matrix,node.cartesianTransform))
            if 'orientedBoundingBox' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                node_param=gmu.get_oriented_bounding_box_parameters(node.orientedBoundingBox)
                self.assertTrue(np.allclose(graph_param,node_param))
            if 'convexHull' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                graph_volume=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(graph_param)).compute_convex_hull()[0].get_volume()
                node_volume=node.convexHull.get_volume()
                self.assertAlmostEqual(graph_volume,node_volume,delta=0.01)
        
    def test_PointCloudNode_creation_from_graph_path(self):
        node= PointCloudNode(graphPath=self.dataLoaderParking.pcdGraphPath, subject=self.dataLoaderParking.pcdSubject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderParking.pcdGraph.triples(( self.dataLoaderParking.pcdSubject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderParking.pcdGraphPath.parent/Path(o.toPython())).resolve(),node.path) #not sure
            if 'cartesianTransform' in p.toPython():
                matrix=ut.literal_to_matrix(o)
                #check if matrix elements are the same as the node cartesianTransform
                self.assertTrue(np.allclose(matrix,node.cartesianTransform))

        
    def test_PointCloudNode_creation_from_path(self):
        #path1 without loadResource
        node= PointCloudNode(path=self.dataLoaderRoad.pcdPath)
        self.assertEqual(node.name,self.dataLoaderRoad.pcdPath.stem)

        #path2 with loadResource
        node= PointCloudNode(path=self.dataLoaderRoad.pcdPath,loadResource=True)        
        self.assertEqual(node.name,self.dataLoaderRoad.pcdPath.stem)
        self.assertEqual(node.pointCount,len(self.dataLoaderRoad.pcd.points))

        #path3 
        node= PointCloudNode(path=self.dataLoaderRoad.e57Path,loadResource=True)
        self.assertEqual(node.pointCount,self.dataLoaderRoad.e57.get_header(0).point_count)

    def test_PointCloudNode_creation_from_resource(self):
        #pcd
        node= PointCloudNode(resource=self.dataLoaderRoad.pcd)
        self.assertEqual(node.pointCount,len(self.dataLoaderRoad.pcd.points))
        self.assertTrue(isinstance(node.resource,o3d.geometry.PointCloud))        

        #e57 with header
        node= PointCloudNode(resource=self.dataLoaderParking.e572)
        self.assertEqual(node.pointCount,self.dataLoaderParking.e572.get_header(0).point_count)
        self.assertTrue(isinstance(node.resource,o3d.geometry.PointCloud))
        
        #e57 without header
        node= PointCloudNode(resource=self.dataLoaderRoad.e57)
        self.assertEqual(node.pointCount,self.dataLoaderRoad.e57.get_header(0).point_count)
        self.assertTrue(isinstance(node.resource,o3d.geometry.PointCloud))
        
        #e57 dict
        node= PointCloudNode(resource=self.dataLoaderParking.e572Data)
        self.assertEqual(node.pointCount,self.dataLoaderParking.e572.get_header(0).point_count)
        self.assertTrue(isinstance(node.resource,o3d.geometry.PointCloud))
        
        #las
        node= PointCloudNode(resource=self.dataLoaderParking.las)
        self.assertEqual(node.pointCount,len(self.dataLoaderParking.las.xyz))
        self.assertTrue(isinstance(node.resource,o3d.geometry.PointCloud))
        

    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=self.dataLoaderParking.pcdSubject
        node= PointCloudNode(subject=subject,
                             graph=self.dataLoaderParking.pcdGraph,
                             graphPath=self.dataLoaderParking.pcdGraphPath)

        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderParking.pcdGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderParking.pcdGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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

    def test_creation_from_subject_and_path(self):        
        node= PointCloudNode(subject=self.dataLoaderParking.pcdSubject,
                             path=self.dataLoaderParking.pcdPath,
                             loadResource=True)
        self.assertEqual(node.subject.toPython(),self.dataLoaderParking.pcdSubject.toPython())

    def test_creation_from_subject_and_path_and_graph(self):        
        node= PointCloudNode(subject=self.dataLoaderRoad.pcdSubject,
                             path=self.dataLoaderRoad.pcdPath,
                             graph=self.dataLoaderRoad.pcdGraph)
        self.assertEqual(node.subject.toPython(),self.dataLoaderRoad.pcdSubject.toPython())
        node.get_graph()
        initialGraph=ut.get_subject_graph(self.dataLoaderRoad.pcdGraph,subject=self.dataLoaderRoad.pcdSubject)
        self.assertEqual(len(node.graph),len(initialGraph)) 

    def test_creation_from_resource_and_path(self):        
        node= PointCloudNode(resource=self.dataLoaderParking.pcd,
                            path=self.dataLoaderParking.pcdPath)
        self.assertEqual(node.subject.toPython(),'http://'+ut.validate_string(Path(self.dataLoaderParking.pcdPath).stem) )

    def test_creation_from_subject_resource_and_path(self):        
        node= PointCloudNode(subject=self.dataLoaderRoad.pcdSubject,
                             resource=self.dataLoaderRoad.pcd,
                             path=self.dataLoaderRoad.pcdPath)
        self.assertEqual(node.subject.toPython(),self.dataLoaderRoad.pcdSubject.toPython() )
        
    def test_creation_from_subject_resource_and_path_and_graph(self):        
        node= PointCloudNode(subject=self.dataLoaderRoad.pcdSubject,
                             resource=self.dataLoaderRoad.pcd,
                             path=self.dataLoaderRoad.pcdPath,
                             graph=self.dataLoaderRoad.pcdGraph)
        self.assertEqual(node.subject.toPython(),self.dataLoaderRoad.pcdSubject.toPython() )
        node.get_graph()
        object=node.graph.value(node.subject,GEOMAPI_PREFIXES['geomapi'].path)
        self.assertEqual(Path(object.toPython()),Path(self.dataLoaderRoad.pcdPath) )

    def test_node_creation_with_get_resource(self):
        #pcd
        node= PointCloudNode(path=self.dataLoaderRoad.pcdPath,loadResource=True)
        self.assertEqual(node.pointCount,len(self.dataLoaderRoad.pcd.points))
        
        #e57
        node= PointCloudNode(path=self.dataLoaderRoad.e57Path,loadResource=True)
        self.assertEqual(node.pointCount,self.dataLoaderRoad.e57.get_header(0).point_count)
        
        #las
        node= PointCloudNode(path=self.dataLoaderParking.lasPath,loadResource=True)
        self.assertEqual(node.pointCount,len(self.dataLoaderParking.las.xyz))

        #graphPath with get resource
        node= PointCloudNode(subject=self.dataLoaderParking.pcdSubject,
                             graphPath=self.dataLoaderParking.pcdGraphPath,
                             loadResource=True)
        self.assertIsNotNone(node.resource)

    def test_delete_resource(self):
        #pcd
        node= PointCloudNode(resource=self.dataLoaderParking.pcd)
        self.assertIsNotNone(node.resource)
        del node.resource
        self.assertIsNone(node.resource)

    def test_save_resource(self):
        #no pcd -> False
        node= PointCloudNode()
        self.assertFalse(node.save_resource())

        #directory
        node= PointCloudNode(resource=self.dataLoaderParking.pcd)
        self.assertIsNotNone(node.resource)
        self.assertTrue(node.save_resource(self.dataLoaderParking.resourcePath))

        # #graphPath        
        # node= PointCloudNode(resource=self.pcd2,graphPath=self.graphPath)
        # self.assertTrue(node.save_resource())

        # #no path or graphPath
        # node= PointCloudNode(resource=self.pcd2)        
        # self.assertTrue(node.save_resource())

        ##invalid extension -> error
        #node= PointCloudNode(resource=self.pcd1)
        #self.assertRaises(ValueError,node.save_resource,self.resourcePath,'.kjhgfdfg')
#
        ##.pcd 
        #node= PointCloudNode(resource=self.pcd2)
        #self.assertTrue(node.save_resource(self.resourcePath,'.pcd'))
        #self.assertEqual(node.path,os.path.join(self.resourcePath,node.name+'.pcd'))
#
        ##.ply 
        #node= PointCloudNode(resource=self.pcd3)
        #self.assertTrue(node.save_resource(self.resourcePath,'.ply'))
        #self.assertEqual(node.path,os.path.join(self.resourcePath,node.name+'.ply'))
        #
        ##.e57 
        #node= PointCloudNode(resource=self.e57_2)
        #self.assertTrue(node.save_resource(self.resourcePath,'.e57'))
        #self.assertEqual(node.path,os.path.join(self.resourcePath,node.name+'.e57'))
        #
        ##path -> new name
        #node= PointCloudNode(subject=URIRef('mypcd'),path=self.path2,loadResource=True)
        #self.assertTrue(node.save_resource())
        #
        ##graphPath with directory
        #node=PointCloudNode(subject=self.subject2,graphPath=self.graphPath, resource=self.pcd3)
        #self.assertTrue(node.save_resource(self.resourcePath))

        # #graph with new subject
        # node=PointCloudNode(subject=self.subject3,grap=self.graph, resource=self.pcd3)
        # node.name='mypcd'
        # self.assertTrue(node.save_resource())

    def test_get_resource(self):
        #pcd
        node=PointCloudNode(resource=self.dataLoaderParking.e571)  
        self.assertIsNotNone(node.load_resource())

        #no pcd
        del node.resource
        self.assertIsNone(node.load_resource())

        #graphPath with loadResource
        node=PointCloudNode(graphPath=self.dataLoaderParking.pcdGraphPath,
                            subject=self.dataLoaderParking.pcdSubject,
                            loadResource=True)
        self.assertIsNotNone(node.load_resource())


if __name__ == '__main__':
    unittest.main()
