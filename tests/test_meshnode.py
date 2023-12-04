#LIBRARIES
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
from geomapi.nodes import *
from rdflib import RDF, RDFS, Graph, Literal, URIRef

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut

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

    def test_meshnode_creation_from_subject(self):
        #subject
        subject='myNode'
        node= MeshNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'file:///'+subject)

        #http://
        subject='http://session_2022_05_20'
        node= MeshNode(subject=subject)
        self.assertEqual(node.subject.toPython(),subject)
        
        #erroneous char       
        subject='[[http://ses>sion_2022_<05_20]]'
        node= MeshNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://__ses_sion_2022__05_20__')

    def test_meshnode_creation_from_graph(self):
        subject=next(self.dataLoaderParking.meshGraph.subjects(RDF.type))
        node= MeshNode(graph=self.dataLoaderParking.meshGraph, subject=subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        object=self.dataLoaderParking.meshGraph.value(subject,self.dataLoaderParking.v4d['faceCount'])
        self.assertEqual(node.faceCount,object.toPython())
        
    def test_meshnode_creation_from_graph_path(self):
        subject=next(self.dataLoaderRoad.meshGraph.subjects(RDF.type))
        node= MeshNode(graphPath=self.dataLoaderRoad.meshGraphPath, subject=subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        object=self.dataLoaderRoad.meshGraph.value(subject,self.dataLoaderRoad.v4d['faceCount'])
        self.assertEqual(node.faceCount,object.toPython())
        
    def test_meshnode_creation_from_path(self):
        #path1 without getResource
        node= MeshNode(path=self.dataLoaderParking.meshPath)
        self.assertEqual(node.name,ut.get_filename(self.dataLoaderParking.meshPath))
        #path2 with getResource
        node= MeshNode(path=self.dataLoaderParking.meshPath,getResource=True)        
        self.assertEqual(node.name,ut.get_filename(self.dataLoaderParking.meshPath))
        self.assertEqual(node.faceCount,len(self.dataLoaderParking.mesh.triangles))
        #path3 
        node= MeshNode(path=self.dataLoaderRoad.meshPath,getResource=True)
        self.assertEqual(node.name,ut.get_filename(self.dataLoaderRoad.meshPath))
        self.assertEqual(node.faceCount,len(self.dataLoaderRoad.mesh.triangles))

    def test_meshnode_creation_from_resource(self):
        #mesh1
        node= MeshNode(resource=self.dataLoaderParking.mesh)
        self.assertEqual(node.faceCount,len(self.dataLoaderParking.mesh.triangles))
        #mesh2
        node= MeshNode(resource=self.dataLoaderParking.slabMesh)
        self.assertEqual(node.faceCount,len(self.dataLoaderParking.slabMesh.triangles))
        #mesh3
        node= MeshNode(resource=self.dataLoaderRoad.pipeMesh)
        self.assertEqual(node.faceCount,len(self.dataLoaderRoad.pipeMesh.triangles))

    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=next(self.dataLoaderParking.meshGraph.subjects(RDF.type))
        node= MeshNode(subject=subject,
                       graph=self.dataLoaderParking.meshGraph,
                       graphPath=self.dataLoaderParking.meshGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        node.to_graph()
        self.assertTrue((subject, self.dataLoaderParking.v4d['faceCount'], Literal(node.faceCount)) in self.dataLoaderParking.meshGraph)

    def test_creation_from_subject_and_path(self):        
        node= MeshNode(subject='myMesh',
                       path=self.dataLoaderParking.meshPath,
                       getResource=True)
        self.assertEqual(node.subject.toPython(),'file:///myMesh')
        node.to_graph()
        box= self.dataLoaderParking.mesh.get_oriented_bounding_box()
        min=box.get_min_bound()
        self.assertAlmostEqual(np.asarray(node.orientedBounds)[0,0],min[0],delta=0.01)

    def test_creation_from_subject_and_path_and_graph(self):        
        node= MeshNode(subject=self.dataLoaderRoad.meshSubject,
                       path=self.dataLoaderRoad.meshPath,
                       graph=self.dataLoaderRoad.meshGraph,
                       getResource=True)
        self.assertEqual(node.subject,self.dataLoaderRoad.meshSubject)
        node.to_graph()
        initialGraph=ut.get_subject_graph(self.dataLoaderRoad.meshGraph,subject=self.dataLoaderRoad.meshSubject)
        self.assertEqual(len(node.graph),len(initialGraph))
        box= self.dataLoaderRoad.mesh.get_axis_aligned_bounding_box()
        min=box.get_min_bound()
        self.assertAlmostEqual(node.cartesianBounds[0],min[0],delta=0.01)

    def test_creation_from_resource_and_path(self):        
        node= MeshNode(resource=self.dataLoaderParking.mesh,path=self.dataLoaderParking.meshPath)
        self.assertEqual(node.subject.toPython(),'file:///'+ut.get_filename(self.dataLoaderParking.meshPath) )

    def test_creation_from_subject_resource_and_path(self):        
        node= MeshNode(subject='file:///road',resource=self.dataLoaderRoad.mesh,path=self.dataLoaderRoad.meshPath)
        self.assertEqual(node.subject.toPython(),'file:///road' )
        
    def test_creation_from_subject_resource_and_path_and_graph(self):        
        node= MeshNode(subject=self.dataLoaderRoad.meshSubject,
                       resource=self.dataLoaderRoad.mesh,
                       path=self.dataLoaderRoad.meshPath, 
                       graph=self.dataLoaderRoad.meshGraph)
        self.assertEqual(node.subject,self.dataLoaderRoad.meshSubject)
        node.to_graph()
        object=node.graph.value(node.subject,self.dataLoaderRoad.v4d['path'])
        self.assertEqual(ut.parse_path(object.toPython()),'../mesh/road.ply')

    def test_node_creation_with_get_resource(self):
        #mesh
        node= MeshNode(resource=self.dataLoaderParking.mesh)
        self.assertIsNotNone(node._resource)

        #path without getResource
        node= MeshNode(path=self.dataLoaderParking.meshPath)
        self.assertIsNone(node._resource)

        #path with getResource
        node= MeshNode(path=self.dataLoaderParking.meshPath,getResource=True)
        self.assertIsNotNone(node._resource)

        #graph with get resource
        node= MeshNode(subject='file:///parking',graph=self.dataLoaderParking.meshGraph,getResource=True)
        self.assertIsNone(node._resource)

        #graphPath with get resource
        node= MeshNode(subject='file:///parking',graphPath=self.dataLoaderParking.meshGraphPath,getResource=True)
        self.assertIsNotNone(node._resource)

    def test_delete_resource(self):
        node= MeshNode(resource=self.dataLoaderRoad.mesh)
        self.assertIsNotNone(node._resource)
        del node.resource
        self.assertIsNone(node._resource)

    # def test_save_resource(self):
    #     #no mesh -> False
    #     node= MeshNode()
    #     self.assertFalse(node.save_resource())

    #     #directory
    #     node= MeshNode(resource=self.dataLoaderRoad.foundationMesh)
    #     self.assertTrue(node.save_resource(self.dataLoaderParking.resourcePath))

    #     # #graphPath        
    #     # node= MeshNode(resource=self.mesh2,graphPath=self.meshGraphPath)
    #     # self.assertTrue(node.save_resource())

    #     #no path or graphPath
    #     node= MeshNode(resource=self.dataLoaderRoad.pipeMesh)        
    #     self.assertTrue(node.save_resource())

    #     #invalid extension -> error
    #     node= MeshNode(resource=self.dataLoaderRoad.roadMesh)
    #     self.assertRaises(ValueError,node.save_resource,os.path.join(self.dataLoaderRoad.path,'resources'),'.kjhgfdfg')

    #     #.ply -> ok
    #     node= MeshNode(resource=self.dataLoaderRoad.collectorMesh)
    #     self.assertTrue(node.save_resource(os.path.join(self.dataLoaderRoad.path,'resources'),'.ply'))
    #     self.assertEqual(node.path,(self.dataLoaderRoad.path / 'resources' / (node.name+'.ply')).as_posix())

    #     #.obj -> ok
    #     node= MeshNode(resource=self.dataLoaderParking.beamMesh)
    #     self.assertTrue(node.save_resource(os.path.join(self.dataLoaderParking.path,'resources'),'.obj'))
    #     self.assertEqual(node.path,(self.dataLoaderParking.path / 'resources' / (node.name+'.obj')).as_posix())

        
    #     #path -> new name
    #     node= MeshNode(subject=URIRef('myMesh'),path=self.dataLoaderParking.meshPath,getResource=True)
    #     self.assertTrue(node.save_resource())
        
    #     #graphPath with directory
    #     node=MeshNode(subject='file:///road',
    #                   graphPath=self.dataLoaderRoad.meshGraphPath, 
    #                   resource=self.dataLoaderRoad.mesh)
    #     self.assertTrue(node.save_resource(os.path.join(self.dataLoaderRoad.path,'resources')))

    #     #graph with new subject
    #     node=MeshNode(subject='file:///road',graph=self.dataLoaderRoad.meshGraph, resource=self.dataLoaderRoad.mesh)
    #     node.name='myMesh'
    #     self.assertTrue(node.save_resource())

    def test_get_resource(self):
        #mesh
        node=MeshNode(resource=self.dataLoaderRoad.mesh)  
        self.assertIsNotNone(node.get_resource())

        #no mesh
        node=MeshNode()
        self.assertIsNone(node.get_resource())

        #graphPath with getResource
        node=MeshNode(graphPath=self.dataLoaderParking.meshGraphPath,subject='file:///parking',getResource=True)
        self.assertIsNotNone(node.get_resource())

    def test_get_metadata_from_resource(self):
        #mesh
        node=MeshNode(resource=self.dataLoaderParking.mesh)  
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.faceCount)
        self.assertIsNotNone(node.pointCount)

        #graphPath
        node=MeshNode(graphPath=self.dataLoaderParking.meshGraphPath,subject='file:///parking',getResource=True)
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.faceCount)
        self.assertIsNotNone(node.pointCount)

if __name__ == '__main__':
    unittest.main()
