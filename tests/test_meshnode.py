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
from geomapi.nodes import MeshNode
from rdflib import RDF, RDFS, Graph, Literal, URIRef
import trimesh

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from geomapi.utils import GEOMAPI_PREFIXES



class TestMeshNode(unittest.TestCase):




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
        self.assertEqual(node.subject.toPython(),'http://'+subject)

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
        object=self.dataLoaderParking.meshGraph.value(subject,GEOMAPI_PREFIXES['geomapi'].faceCount)
        self.assertEqual(node.faceCount,object.toPython())
        
    def test_meshnode_creation_from_graph_path(self):
        subject=next(self.dataLoaderRoad.meshGraph.subjects(RDF.type))
        node= MeshNode(graphPath=self.dataLoaderRoad.meshGraphPath, subject=subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        object=self.dataLoaderRoad.meshGraph.value(subject,GEOMAPI_PREFIXES['geomapi'].faceCount)
        self.assertEqual(node.faceCount,object.toPython())
        
    def test_meshnode_creation_from_path(self):
        #path1 without loadResource
        node= MeshNode(path=self.dataLoaderParking.meshPath)
        self.assertEqual(node.name,self.dataLoaderParking.meshPath.stem)
        #path2 with loadResource
        node= MeshNode(path=self.dataLoaderParking.meshPath,loadResource=True)        
        self.assertEqual(node.name,self.dataLoaderParking.meshPath.stem)
        self.assertEqual(node.faceCount,len(self.dataLoaderParking.mesh.triangles))
        #path3 
        node= MeshNode(path=self.dataLoaderRoad.meshPath,loadResource=True)
        self.assertEqual(node.name,self.dataLoaderRoad.meshPath.stem)
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
        
        #trimesh
        mesh = trimesh.load(self.dataLoaderParking.meshPath)
        node= MeshNode(resource=    mesh)
        self.assertEqual(node.faceCount,len(np.asarray(self.dataLoaderRoad.mesh.triangles)))


    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=next(self.dataLoaderParking.meshGraph.subjects(RDF.type))
        node= MeshNode(subject=subject,
                       graph=self.dataLoaderParking.meshGraph,
                       graphPath=self.dataLoaderParking.meshGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        node.get_graph()
        self.assertTrue((subject, GEOMAPI_PREFIXES['geomapi'].faceCount, Literal(node.faceCount)) in self.dataLoaderParking.meshGraph)

    def test_creation_from_subject_and_path(self):        
        node= MeshNode(subject='myMesh',
                       path=self.dataLoaderParking.meshPath,
                       loadResource=True)
        self.assertEqual(node.subject.toPython(),'http://myMesh')
        node.get_graph()
        
        for s, p, o in node.get_graph().triples((None, None, None)):
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
                
    def test_creation_from_subject_and_path_and_graph(self):        
        node= MeshNode(subject=self.dataLoaderRoad.meshSubject,
                       path=self.dataLoaderRoad.meshPath,
                       graph=self.dataLoaderRoad.meshGraph,
                       loadResource=True)
        self.assertEqual(node.subject,self.dataLoaderRoad.meshSubject)
        node.get_graph()
        initialGraph=ut.get_subject_graph(self.dataLoaderRoad.meshGraph,subject=self.dataLoaderRoad.meshSubject)
        self.assertEqual(len(node.graph),len(initialGraph))
        

    def test_creation_from_resource_and_path(self):        
        node= MeshNode(resource=self.dataLoaderParking.mesh,path=self.dataLoaderParking.meshPath)
        node.get_graph(base='http://meshes#')
        self.assertEqual(node.subject,self.dataLoaderParking.meshSubject )

    def test_creation_from_subject_resource_and_path(self):        
        node= MeshNode(subject=self.dataLoaderRoad.meshSubject,resource=self.dataLoaderRoad.mesh,path=self.dataLoaderRoad.meshPath)
        self.assertEqual(node.subject,self.dataLoaderRoad.meshSubject )
        
    def test_creation_from_subject_resource_and_path_and_graph(self):        
        node= MeshNode(subject=self.dataLoaderRoad.meshSubject,
                       resource=self.dataLoaderRoad.mesh,
                       path=self.dataLoaderRoad.meshPath, 
                       graph=self.dataLoaderRoad.meshGraph)
        self.assertEqual(node.subject,self.dataLoaderRoad.meshSubject)
        node.get_graph()
        object=node.graph.value(node.subject,GEOMAPI_PREFIXES['geomapi'].path)
        self.assertEqual(Path(object.toPython()),Path(self.dataLoaderRoad.meshPath))

    def test_node_creation_with_get_resource(self):
        #mesh
        node= MeshNode(resource=self.dataLoaderParking.mesh)
        self.assertIsNotNone(node._resource)

        #path without loadResource
        node= MeshNode(path=self.dataLoaderParking.meshPath)
        self.assertIsNone(node._resource)

        #path with loadResource
        node= MeshNode(path=self.dataLoaderParking.meshPath,loadResource=True)
        self.assertIsNotNone(node._resource)

        #graph with get resource
        node= MeshNode(subject=self.dataLoaderParking.meshSubject,graph=self.dataLoaderParking.meshGraph,loadResource=True)
        self.assertIsNone(node._resource)

        #graphPath with get resource
        node= MeshNode(subject=self.dataLoaderParking.meshSubject,graphPath=self.dataLoaderParking.meshGraphPath,loadResource=True)
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
    #     node= MeshNode(subject=URIRef('myMesh'),path=self.dataLoaderParking.meshPath,loadResource=True)
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
        self.assertIsNone(node.load_resource()) # no path is given, cannot reload resource

        #no mesh
        node=MeshNode()
        self.assertIsNone(node.load_resource())

        #graphPath with loadResource
        node=MeshNode(graphPath=self.dataLoaderParking.meshGraphPath,subject=self.dataLoaderParking.meshSubject,loadResource=True)
        self.assertIsNotNone(node.load_resource())

    def test_get_metadata_from_resource(self):
        #mesh
        node=MeshNode(resource=self.dataLoaderParking.mesh)  
        for s, p, o in self.dataLoaderParking.meshGraph.triples((self.dataLoaderParking.meshSubject, None, None)):
            if 'cartesianTransform' in p.toPython():
                    matrix=ut.literal_to_matrix(o)
                    #check if matrix elements are the same as the node cartesianTransform
                    self.assertTrue(np.allclose(matrix,node.cartesianTransform,atol=0.001))
            if 'orientedBoundingBox' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                node_param=gmu.get_oriented_bounding_box_parameters(node.orientedBoundingBox)
                np.testing.assert_array_almost_equal(graph_param,node_param,2)
            if 'convexHull' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                graph_volume=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(graph_param)).compute_convex_hull()[0].get_volume()
                node_volume=node.convexHull.get_volume()
                self.assertAlmostEqual(graph_volume,node_volume,delta=0.01)

        #graphPath
        node=MeshNode(graphPath=self.dataLoaderParking.meshGraphPath,subject=self.dataLoaderParking.meshSubject,loadResource=True)
        for s, p, o in self.dataLoaderParking.meshGraph.triples((self.dataLoaderParking.meshSubject, None, None)):
            if 'cartesianTransform' in p.toPython():
                    matrix=ut.literal_to_matrix(o)
                    #check if matrix elements are the same as the node cartesianTransform
                    self.assertTrue(np.allclose(matrix,node.cartesianTransform,atol=0.001))
            if 'orientedBoundingBox' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                node_param=gmu.get_oriented_bounding_box_parameters(node.orientedBoundingBox)
                np.testing.assert_array_almost_equal(graph_param[:6],node_param[:6],2)
            if 'convexHull' in p.toPython():
                graph_param=ut.literal_to_matrix(o)
                graph_volume=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(graph_param)).compute_convex_hull()[0].get_volume()
                node_volume=node.convexHull.get_volume()
                self.assertAlmostEqual(graph_volume,node_volume,delta=0.01)

if __name__ == '__main__':
    unittest.main()
