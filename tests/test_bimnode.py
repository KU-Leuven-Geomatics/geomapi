import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import sys
import cv2
import ifcopenshell
import numpy as np
import rdflib
import ifcopenshell.util.selector 
from rdflib import RDF, RDFS, Graph, Literal, URIRef
import open3d as o3d
#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
from geomapi.nodes import BIMNode

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from geomapi.utils import GEOMAPI_PREFIXES

class TestBIMNode(unittest.TestCase):



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

    def test_empty_node(self):
        node= BIMNode()
        self.assertIsNotNone(node.subject)
        self.assertIsNotNone(node.name)
        #self.assertEqual(node.className,'IfcBuildingElement')
        
    def test_subject(self):
        #subject
        subject='myNode'
        node= BIMNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://'+subject)
    
    def test_name(self):
        node= BIMNode(name='name')
        self.assertEqual(node.name,'name')
        self.assertEqual(node.subject.toPython(),'http://name')
    
    def test_globalId(self):
        globalId='myGlobalId'
        node= BIMNode(globalId=globalId)
        self.assertEqual(node.globalId,globalId)
        self.assertNotIn(globalId,node.subject.toPython())

    def test_className(self):
        className='myClassName'
        node= BIMNode(className=className)
        self.assertEqual(node.className,className)
        self.assertNotIn(className,node.subject.toPython())
        
    def test_objectType(self):
        objectType='myObjectType'
        node= BIMNode(objectType=objectType)
        self.assertEqual(node.objectType,objectType)
        self.assertNotIn(objectType,node.subject.toPython())
        
    def test_ifc(self):
        #valid path
        node= BIMNode(resource=self.dataLoaderParking.ifcBeam)        
        #assert that node.globalId is in ifc
        self.assertIsNotNone(self.dataLoaderParking.ifc.by_guid(node.globalId))
        #check if name is correct
        self.assertEqual(node.name,self.dataLoaderParking.ifc.by_guid(node.globalId).Name)
        #check if className is correct
        self.assertEqual(node.className,self.dataLoaderParking.ifc.by_guid(node.globalId).is_a())
        #check if object type is correct
        self.assertEqual(node.objectType,self.dataLoaderParking.ifc.by_guid(node.globalId).ObjectType)
        #check if resource is None
        self.assertIsNotNone(node.resource)
        
    def test_path(self):
        #valid path
        path=self.dataLoaderParking.ifcWallPath
        globalId=path.stem.split('_')[-1]
        name=path.stem
          
        node= BIMNode(path=path)        
        self.assertEqual(node.path,path)
        self.assertEqual(node.subject.toPython(),'http://'+path.stem)
        self.assertEqual(node.name,name)
        self.assertIsNone(node.resource)
        
        #invalid path
        path='qsffqsdf.dwg'
        self.assertRaises(ValueError,BIMNode,path=path)
     

    def test_resource(self):
        #triangle mesh
        resource=self.dataLoaderParking.columnMesh
        node= BIMNode(resource=resource, loadResource=True)
        self.assertEqual(node.resource,resource)        
        
        #ifcopenshell entity
        resource=self.dataLoaderParking.ifcBeam
        mesh=self.dataLoaderParking.beamMesh
        node= BIMNode(resource=resource, loadResource=True)
        self.assertEqual(len(node.resource.vertices),len(mesh.vertices))
        self.assertEqual(node.globalId,resource.GlobalId)
        self.assertEqual(node.name,resource.Name)
        self.assertEqual(node.className,resource.is_a())
        self.assertEqual(node.objectType,resource.ObjectType)
        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],mesh.get_center()))
        self.assertEqual(node.faceCount,len(mesh.triangles)) #maybe np.asarray
        self.assertEqual(node.pointCount,len(mesh.vertices))
        self.assertAlmostEqual(node.convexHull.get_volume(),mesh.compute_convex_hull()[0].get_volume())
        
        #invalid resource
        resource='qsffqsdf'
        self.assertRaises(ValueError,BIMNode,resource=resource)

    def test_get_resource(self):
        #path+getResource
        mesh=self.dataLoaderParking.wallMesh
        node= BIMNode(path=self.dataLoaderParking.ifcWallPath,loadResource=True)
        self.assertTrue(isinstance(node.resource,o3d.geometry.TriangleMesh))
        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],mesh.get_center()))
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),mesh.get_oriented_bounding_box().get_center(),3)
        np.testing.assert_array_almost_equal(len(np.asarray(node.convexHull.vertices)),len(np.asarray(mesh.compute_convex_hull()[0].vertices)))
        self.assertEqual(node.faceCount,len(mesh.triangles)) 
        self.assertEqual(node.pointCount,len(mesh.vertices))
        
        #path unexisting
        path='qsffqsdf.obj'
        node= BIMNode(path=path,loadResource=True)
        self.assertIsNone(node.resource)
        

    def test_graphPath(self):
        node=BIMNode(graphPath=self.dataLoaderParking.ifcGraphPath)
        self.assertEqual(node.graphPath,self.dataLoaderParking.ifcGraphPath)
        self.assertTrue(node.subject in self.dataLoaderParking.ifcGraph.subjects())
        self.assertIsNotNone(node.globalId)
        self.assertIsNotNone(node.objectType)        
        
        
    def test_graphPath_with_subject(self):
        subject=next(self.dataLoaderParking.ifcGraph.subjects(RDF.type))
        node=BIMNode(graphPath=self.dataLoaderParking.ifcGraphPath,subject=subject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderParking.pcdGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderParking.ifcGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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

        #raise error when subject is not in graph
        self.assertRaises(ValueError,BIMNode,graphPath=self.dataLoaderParking.ifcGraphPath,subject=URIRef('mySubject'))
    
    def test_graph(self):
        subject=next(self.dataLoaderParking.ifcGraph.subjects(RDF.type))
        node=BIMNode(graphPath=self.dataLoaderParking.ifcGraphPath,subject=subject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderParking.pcdGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderParking.ifcGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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


    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=next(self.dataLoaderRoad.ifcGraph.subjects(RDF.type))
        node= BIMNode(subject=subject,graph=self.dataLoaderRoad.ifcGraph,graphPath=self.dataLoaderRoad.ifcGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())


    def test_clear_resource(self):
        #mesh
        node= BIMNode(resource=self.dataLoaderRoad.pipeMesh)
        self.assertIsNotNone(node.resource)
        del node.resource
        self.assertIsNone(node.resource)


if __name__ == '__main__':
    unittest.main()
