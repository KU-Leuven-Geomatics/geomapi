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
import geomapi.utils.cadutils as cadu
from geomapi.nodes import LineSetNode

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from data_loader_railway import DATALOADERRAILWAYINSTANCE 

from geomapi.utils import GEOMAPI_PREFIXES

class TestLinesetNode(unittest.TestCase):



################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        
        cls.dataLoaderParking = DATALOADERPARKINGINSTANCE
        cls.dataLoaderRoad = DATALOADERROADINSTANCE
        cls.dataLoaderRailway = DATALOADERRAILWAYINSTANCE

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
        node= LineSetNode()
        self.assertIsNotNone(node.subject)
        self.assertIsNotNone(node.name)
        
    def test_subject(self):
        #subject
        subject='myNode'
        node= LineSetNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://'+subject)
    
    def test_name(self):
        node= LineSetNode(name='name')
        self.assertEqual(node.name,'name')
        self.assertEqual(node.subject.toPython(),'http://name')
    
    def test_handle(self):
        handle='myHandle'
        node= LineSetNode(handle=handle)
        self.assertEqual(node.handle,handle)
        self.assertNotIn(handle,node.subject.toPython())

    def test_dxfType(self):
        dxfType='mydxfType'
        node= LineSetNode(dxfType=dxfType)
        self.assertEqual(node.dxfType,dxfType)
        self.assertNotIn(dxfType,node.subject.toPython())
        
    def test_layer(self):
        layer='layer'
        node= LineSetNode(layer=layer)
        self.assertEqual(node.layer,layer)
        self.assertNotIn(layer,node.subject.toPython())
        
    #def test_dxf_path(self):
    #    #valid path
    #    dxfPath=self.dataLoaderRailway.dxfPath
    #    node= LineSetNode(dxfPath=dxfPath)
    #    self.assertIsNotNone(self.dataLoaderRailway.dxf.entitydb.get(node.handle))
    #    self.assertEqual(node.layer,getattr(self.dataLoaderRailway.dxf.entitydb.get(node.handle).dxf,'layer'))
    #    self.assertIsNotNone(node.resource)
    #    
    #    #dxfPath with handle
    #    handle=self.dataLoaderRoad.entity.dxf.handle
    #    node= LineSetNode(dxfPath=dxfPath,handle=handle)
    #    self.assertEqual(node.handle,handle)
    #    
    #    #test if metadata is correctly parsed        
    #    entity = next(entity for entity in self.dataLoaderRailway.dxf.modelspace().query("LINE"))
    #    g=cadu.ezdxf_entity_to_o3d(entity)
    #    node=LineSetNode(dxfPath=self.dataLoaderRailway.dxfPath)
    #    self.assertEqual(node.dxfType,'LINE')
    #    self.assertEqual(node.lineCount,len(g.lines))
    #    self.assertEqual(node.pointCount,len(g.points))
    #    self.assertTrue(np.allclose(node.cartesianTransform[:3,3],g.get_center()))
    #    self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),gmu.get_oriented_bounding_box(g).get_center(),atol=0.1))
    #    self.assertTrue(np.allclose(node.convexHull.get_center(),g.get_center(),atol=0.1))
    #    #check layer
    #    self.assertEqual(node.layer,entity.dxf.layer)
    #    #check if the handle is correctly parsed
    #    self.assertEqual(node.handle,entity.dxf.handle)
    #    #check if resource is colored
    #    self.assertTrue(node.resource.has_colors())
    #    
    #   
    #    #invalid path
    #    dxfPath='qsffqsdf.dwg'
    #    self.assertRaises(ValueError,LineSetNode,dxfPath=dxfPath)
 
    def test_resource(self):
        #line
        resource=self.dataLoaderRoad.line
        node= LineSetNode(resource=resource)
        self.assertEqual(node.resource,resource)        
        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],resource.get_center()))
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),resource.get_center(),atol=0.1))
        self.assertTrue(np.allclose(node.convexHull.get_center(),resource.get_center(),atol=0.1))
        
        #ezdxf.entities.dxfentity.DXFEntity
        resource=self.dataLoaderRoad.entity
        line=self.dataLoaderRoad.insert
        node= LineSetNode(resource=resource)
        self.assertEqual(len(node.resource.points),len(line.points))
        self.assertEqual(node.handle,resource.dxf.handle)
        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],line.get_center()))
        self.assertEqual(node.lineCount,len(line.lines)) #maybe np.asarray
        self.assertEqual(node.pointCount,len(line.points))
        
        #lines and polylines
        dxf=self.dataLoaderRailway.dxf
        geometry_groups,layer_groups=cadu.ezdxf_to_o3d(dxf,explode_blocks=False,join_geometries=False,dtypes=['LINE','POLYLINE','LWPOLYLINE'])
        for listg,layer in zip(geometry_groups,layer_groups):
            for g in listg:
                if isinstance(g,o3d.geometry.LineSet) and len(g.lines) >=1:
                    node=LineSetNode(resource=g,layer=layer)
                    self.assertIsNotNone(node.resource)
                    self.assertEqual(node.layer,layer)
        
        
        #invalid resource
        resource='qsffqsdf'
        self.assertRaises(ValueError,LineSetNode,resource=resource)
       
    def test_path(self):
        #valid path
        path=self.dataLoaderRoad.cadPath
        name=path.stem
          
        node= LineSetNode(path=path)        
        self.assertEqual(node.path,path)
        self.assertEqual(node.subject.toPython(),'http://'+path.stem)
        self.assertEqual(node.name,name) #no globalID
        self.assertIsNone(node.resource)
        
        #invalid path
        path='qsffqsdf.dwg'
        self.assertRaises(ValueError,LineSetNode,path=path)
     
    def test_get_resource(self):
        #path+getResource
        resource=self.dataLoaderRoad.line
        node= LineSetNode(path=self.dataLoaderRoad.cadPath,loadResource=True)
        self.assertIsNotNone(node.resource)
        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],resource.get_center()))
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),gmu.get_oriented_bounding_box(resource).get_center(),atol=0.1))
        self.assertTrue(np.allclose(node.convexHull.get_center() ,resource.get_center(),atol=0.1))
        self.assertEqual(node.lineCount,len(resource.lines))
        self.assertEqual(node.pointCount,len(resource.points))
        
        #path unexisting
        path='qsffqsdf.dxf'
        node= LineSetNode(path=path,getResource=True)
        self.assertIsNone(node.resource)
        
    def test_graphPath(self):
        node=LineSetNode(graphPath=self.dataLoaderRoad.cadGraphPath)
        self.assertEqual(node.graphPath,self.dataLoaderRoad.cadGraphPath)
        self.assertIsNotNone(node.handle)
        self.assertIsNotNone(node.dxfType)        
        
        
    def test_graphPath_with_subject(self):
        subject=next(self.dataLoaderRoad.cadGraph.subjects(RDF.type))
        node=LineSetNode(graphPath=self.dataLoaderRoad.cadGraphPath,subject=subject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderRoad.pcdGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderRoad.cadGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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
        self.assertRaises(ValueError,LineSetNode,graphPath=self.dataLoaderRoad.cadGraphPath,subject=URIRef('mySubject'))
    
    def test_graph(self):
        subject=next(self.dataLoaderRoad.cadGraph.subjects(RDF.type))
        node=LineSetNode(graphPath=self.dataLoaderRoad.cadGraphPath,subject=subject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderRoad.cadGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderRoad.cadGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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
        subject=next(self.dataLoaderRoad.cadGraph.subjects(RDF.type))
        node= LineSetNode(subject=subject,graph=self.dataLoaderRoad.cadGraph,graphPath=self.dataLoaderRoad.cadGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        node.get_graph()
        self.assertTrue((subject,GEOMAPI_PREFIXES['geomapi'].lineCount, Literal(node.lineCount)) in self.dataLoaderRoad.cadGraph)


    def test_clear_resource(self):
        #mesh
        node= LineSetNode(resource=self.dataLoaderRoad.line)
        self.assertIsNotNone(node.resource)
        del node.resource
        self.assertIsNone(node.resource)

    # def test_save_resource(self):
    #     #no mesh -> False
    #     node= LineSetNode()
    #     self.assertFalse(node.export_resource())

    #     #directory
    #     node= LineSetNode(mesh=self.mesh2)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources')))

    #     #graphPath        
    #     node= LineSetNode(mesh=self.mesh2,graphPath=self.bimGraphPath2)
    #     self.assertTrue(node.export_resource())

    #     #no path or graphPath
    #     node= LineSetNode(mesh=self.mesh4)        
    #     self.assertTrue(node.export_resource())

    #     #invalid extension -> error
    #     node= LineSetNode(mesh=self.mesh3)
    #     self.assertRaises(ValueError,node.export_resource,os.path.join(self.path,'resources'),'.kjhgfdfg')

    #     #.ply -> ok
    #     node= LineSetNode(mesh=self.mesh2)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources'),'.ply'))
    #     self.assertEqual(node.path,os.path.join(self.path,'resources',node.name+'.ply'))

    #     #.obj -> ok
    #     node= LineSetNode(mesh=self.mesh3)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources'),'.obj'))
    #     self.assertEqual(node.path,os.path.join(self.path,'resources',node.name+'.obj'))
        
    #     #path -> new name
    #     node= LineSetNode(subject=URIRef('myMesh'),path=self.path2,getResource=True)
    #     self.assertTrue(node.export_resource())
    #     files=ut.get_list_of_files(ut.get_folder(node.path))
    #     self.assertTrue( node.path in files)
        
    #     #graphPath with directory
    #     node=LineSetNode(subject=self.subject2,graphPath=self.bimGraphPath2, mesh=self.mesh3)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources')))
    #     files=ut.get_list_of_files(ut.get_folder(node.path))
    #     self.assertTrue(node.path in files)

    #     #graph with new subject
    #     node=LineSetNode(subject=self.subject4,grap=self.bimGraph4, mesh=self.mesh4)
    #     node.name='myMesh'
    #     self.assertTrue(node.export_resource())
    #     files=ut.get_list_of_files(ut.get_folder(node.path))
    #     self.assertTrue(node.path in files)


if __name__ == '__main__':
    unittest.main()
