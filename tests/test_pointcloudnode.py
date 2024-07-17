import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value

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

    def test_PointCloudNode_creation_from_subject(self):
        #subject
        subject='myNode'
        node= PointCloudNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'file:///'+subject)

        #http://
        subject='http://session_2022_05_20'
        node= PointCloudNode(subject=subject)
        self.assertEqual(node.subject.toPython(),subject)
        
        #erroneous char       
        subject='[[http://ses>sion_2022_<05_20]]'
        node= PointCloudNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://__ses_sion_2022__05_20__')

    def test_PointCloudNode_creation_from_graph(self):
        node= PointCloudNode(graph=self.dataLoaderParking.pcdGraph, subject=self.dataLoaderParking.pcdSubject)
        self.assertEqual(node.subject.toPython(),self.dataLoaderParking.pcdSubject.toPython())
        object=self.dataLoaderParking.pcdGraph.value(self.dataLoaderParking.pcdSubject,self.dataLoaderParking.e57['pointCount'])
        self.assertEqual(node.pointCount,object.toPython())
        
    def test_PointCloudNode_creation_from_graph_path(self):
        node= PointCloudNode(graphPath=self.dataLoaderParking.pcdGraphPath, subject=self.dataLoaderParking.pcdSubject)
        self.assertEqual(node.subject.toPython(),self.dataLoaderParking.pcdSubject.toPython())
        object=self.dataLoaderParking.pcdGraph.value(self.dataLoaderParking.pcdSubject,self.dataLoaderParking.e57['pointCount'])
        self.assertEqual(node.pointCount,object.toPython())
        
    def test_PointCloudNode_creation_from_path(self):
        #path1 without getResource
        node= PointCloudNode(path=self.dataLoaderRoad.pcdPath)
        self.assertEqual(node.name,ut.get_filename(self.dataLoaderRoad.pcdPath))

        #path2 with getResource
        node= PointCloudNode(path=self.dataLoaderRoad.pcdPath,getResource=True)        
        self.assertEqual(node.name,ut.get_filename(self.dataLoaderRoad.pcdPath))
        self.assertEqual(node.pointCount,len(self.dataLoaderRoad.pcd.points))

        #path3 
        node= PointCloudNode(path=self.dataLoaderRoad.e57Path,getResource=True)
        self.assertEqual(node.pointCount,self.dataLoaderRoad.e57.get_header(0).point_count)

    def test_PointCloudNode_creation_from_resource(self):
        #pcd1
        node= PointCloudNode(resource=self.dataLoaderRoad.pcd)
        self.assertEqual(node.pointCount,len(self.dataLoaderRoad.pcd.points))

        #pcd2 -> e57 file
        node= PointCloudNode(resource=self.dataLoaderParking.e572)
        self.assertEqual(node.pointCount,self.dataLoaderParking.e572.get_header(0).point_count)

    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=self.dataLoaderParking.pcdSubject
        node= PointCloudNode(subject=subject,
                             graph=self.dataLoaderParking.pcdGraph,
                             graphPath=self.dataLoaderParking.pcdGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        node.to_graph()
        self.assertTrue((subject, self.dataLoaderParking.e57['pointCount'], Literal(node.pointCount)) in self.dataLoaderParking.pcdGraph)

    def test_creation_from_subject_and_path(self):        
        node= PointCloudNode(subject=self.dataLoaderParking.pcdSubject,
                             path=self.dataLoaderParking.pcdPath,
                             getResource=True)
        self.assertEqual(node.subject.toPython(),self.dataLoaderParking.pcdSubject.toPython())
        #box= self.pcd2.get_oriented_bounding_box()
        #min=np.asarray(box.get_box_points())
        #self.assertAlmostEqual(node.orientedBounds[0,0],min[0,0],delta=0.01)

    def test_creation_from_subject_and_path_and_graph(self):        
        node= PointCloudNode(subject=self.dataLoaderParking.pcdSubject,
                             path=self.dataLoaderParking.pcdPath,
                             graph=self.dataLoaderParking.pcdGraph,
                             getResource=True)
        self.assertEqual(node.subject.toPython(),self.dataLoaderParking.pcdSubject.toPython())
        node.to_graph()
        initialGraph=ut.get_subject_graph(self.dataLoaderParking.pcdGraph,subject=self.dataLoaderParking.pcdSubject)
        self.assertEqual(len(node.graph),len(initialGraph)) 

    def test_creation_from_resource_and_path(self):        
        node= PointCloudNode(resource=self.dataLoaderParking.pcd,
                            path=self.dataLoaderParking.pcdPath)
        self.assertEqual(node.subject.toPython(),'file:///'+ut.validate_string(ut.get_filename(self.dataLoaderParking.pcdPath)) )

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
        node.to_graph()
        object=node.graph.value(node.subject,self.dataLoaderRoad.v4d['path'])
        self.assertEqual(ut.parse_path(object.toPython()) ,'../pcd/road.pcd' )

    def test_node_creation_with_get_resource(self):
        #pcd
        node= PointCloudNode(resource=self.dataLoaderRoad.pcd)
        self.assertIsNotNone(node.resource)

        #path without getResource
        node= PointCloudNode(path=self.dataLoaderRoad.pcdPath)
        self.assertIsNone(node._resource)

        #path with getResource
        node= PointCloudNode(path=self.dataLoaderRoad.pcdPath,getResource=True)
        self.assertIsNotNone(node.resource)

        #graph with get resource
        node= PointCloudNode(subject=self.dataLoaderParking.pcdSubject,
                             graph=self.dataLoaderParking.pcdGraph,
                             getResource=True)
        self.assertIsNone(node.resource)
        
        #graphPath with get resource
        node= PointCloudNode(subject=self.dataLoaderParking.pcdSubject,
                             graphPath=self.dataLoaderParking.pcdGraphPath,
                             getResource=True)
        self.assertIsNotNone(node.resource)

    def test_delete_resource(self):
        #pcd
        node= PointCloudNode(resource=self.dataLoaderParking.pcd)
        self.assertIsNotNone(node._resource)
        del node.resource
        self.assertIsNone(node._resource)

    def test_save_resource(self):
        #no pcd -> False
        node= PointCloudNode()
        self.assertFalse(node.save_resource())

        #directory
        node= PointCloudNode(resource=self.dataLoaderParking.pcd)
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
        #node= PointCloudNode(subject=URIRef('mypcd'),path=self.path2,getResource=True)
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
        self.assertIsNotNone(node.get_resource())

        #no pcd
        del node.resource
        self.assertIsNone(node.get_resource())

        #graphPath with getResource
        node=PointCloudNode(graphPath=self.dataLoaderParking.pcdGraphPath,
                            subject=self.dataLoaderParking.pcdSubject,
                            getResource=True)
        self.assertIsNotNone(node.get_resource())

    def test_get_metadata_from_resource(self):
        #pcd
        node=PointCloudNode(resource=self.dataLoaderParking.e572)  
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.pointCount)

        #graphPath
        node=PointCloudNode(graphPath=self.dataLoaderParking.pcdGraphPath,
                            subject=self.dataLoaderParking.pcdSubject,
                            getResource=True)
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.pointCount)

if __name__ == '__main__':
    unittest.main()
