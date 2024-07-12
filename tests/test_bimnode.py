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

    def test_bimnode_creation_from_subject(self):
        #subject
        subject='myNode'
        node= BIMNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'file:///'+subject)

        #http://
        subject='http://session_2022_05_20'
        node= BIMNode(subject=subject)
        self.assertEqual(node.subject.toPython(),subject)
        
        #erroneous char       
        subject='[[http://ses>sion_2022_<05_20]]'
        node= BIMNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://__ses_sion_2022__05_20__')

    def test_bimnode_creation_from_graph(self):
        subject=next(self.dataLoaderRoad.ifcGraph.subjects(RDF.type))
        node= BIMNode(graph=self.dataLoaderRoad.ifcGraph, subject=subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        object=self.dataLoaderRoad.ifcGraph.value(subject,self.dataLoaderRoad.v4d['faceCount'])
        self.assertEqual(node.faceCount,object.toPython())
        
    def test_bimnode_creation_from_graph_path(self):
        subject=next(self.dataLoaderParking.ifcGraph.subjects(RDF.type))
        node= BIMNode(graphPath=self.dataLoaderParking.ifcGraphPath, subject=subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        object=self.dataLoaderParking.ifcGraph.value(subject,self.dataLoaderParking.v4d['faceCount'])
        self.assertEqual(node.faceCount,object.toPython())
                
#NOTE not all nodes have resources
    def test_bimnode_creation_from_ifcpath(self):
        #path1 without extra info -> take first IfcElement
        node= BIMNode(ifcPath=self.dataLoaderRoad.ifcPath)
        self.assertIsNotNone(node.name)

        #path2 with getResource without extra info -> take first IfcElement
        node= BIMNode(ifcPath=self.dataLoaderParking.ifcPath,getResource=True)        
        self.assertIsNotNone(node.name)
        self.assertIsNotNone(node.resource)

 
    def test_bimnode_creation_from_ifcElement(self):        
        #ifcElement
        node= BIMNode(resource=self.dataLoaderRoad.ifcRoad)
        self.assertEqual(node.subject.toPython(),'file:///BT1_Soort_Bedekking_WSV11_2PpmXyFnz1_QUVtw0_xe4L')

        #ifcElement + getResource
        node= BIMNode(resource=self.dataLoaderParking.ifcWall,ifcPath=self.dataLoaderParking.ifcPath)
        self.assertIsNotNone(node.resource)
        self.assertIsNotNone(node.ifcPath)

    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=next(self.dataLoaderRoad.ifcGraph.subjects(RDF.type))
        node= BIMNode(subject=subject,graph=self.dataLoaderRoad.ifcGraph,graphPath=self.dataLoaderRoad.ifcGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        node.to_graph()
        self.assertTrue((subject, self.dataLoaderRoad.v4d['faceCount'], Literal(node.faceCount)) in self.dataLoaderRoad.ifcGraph)
    
    # def test_creation_from_subject_and_mesh_and_graph(self):  
    #     subject=next(s for s in self.dataLoaderRoad.ifcGraph.subjects(RDF.type) if '2PpmXyFnz1_QUVtw0_xe4L' in s)    
    #     node= BIMNode(subject=subject,resource=self.dataLoaderRoad.roadMesh,graph=self.dataLoaderRoad.ifcGraph)
    #     self.assertEqual(node.subject,subject)
    #     node.to_graph()
    #     object=node.graph.value(node.subject,self.dataLoaderRoad.v4d['path'])
    #     self.assertEqual(ut.parse_path(object.toPython()),(Path("BIM") / (ut.get_filename(self.path3) + '.ply')).as_posix() )

    # def test_creation_from_subject_mesh_ifcPath_graph(self):     
    #     subject=next(s for s in self.dataLoaderRoad.ifcGraph.subjects(RDF.type) if '2sFWN4Spr5fegnupyER9kl' in s)    
    #     node= BIMNode(subject=self.subject4,resource=self.mesh4,path=self.path4, ifcPath=self.ifcPath4, graph=self.bimGraph4)
    #     self.assertEqual(node.subject,self.subject4 )
    #     node.to_graph()
    #     object=node.graph.value(node.subject,self.v4d['path'])
    #     self.assertEqual(ut.parse_path(object.toPython()),(Path("BIM") / (ut.get_filename(self.path4) + '.ply')).as_posix() )

    # def test_creation_from_subject_mesh_path_ifcPath_globalId_graph(self):      
    #     node= BIMNode(subject=self.subject1,resource=self.mesh1,path=self.path1, ifcPath=self.ifcPath1, globalId=self.ifcElement1.GlobalId, graph=self.bimGraph1)
    #     self.assertEqual(node.subject,self.subject1)
    #     node.to_graph()
    #     object=node.graph.value(node.subject,self.v4d['path'])
    #     self.assertEqual(ut.parse_path(object.toPython()),(Path("BIM") / (ut.get_filename(self.path1) + '.ply')).as_posix() )

    def test_node_creation_with_get_resource(self):
        #mesh
        node= BIMNode(resource=self.dataLoaderRoad.roadMesh)
        self.assertIsNotNone(node._resource)

        # #path without getResource
        # node= BIMNode(path=self.path2)
        # self.assertIsNone(node._resource)

        # #path with getResource
        # node= BIMNode(path=self.path3,getResource=True)
        # self.assertIsNotNone(node._resource)

        #graph with get resource
        subject=next(self.dataLoaderRoad.ifcGraph.subjects(RDF.type))

        node= BIMNode(subject=subject,graph=self.dataLoaderRoad.ifcGraph,getResource=True)
        self.assertIsNone(node._resource)
        
        # #graphPath with get resource
        # node= BIMNode(subject=subject,graphPath=self.dataLoaderRoad.ifcGraphPath,getResource=True)
        # self.assertIsNotNone(node._resource)

    def test_clear_resource(self):
        #mesh
        node= BIMNode(resource=self.dataLoaderRoad.pipeMesh)
        self.assertIsNotNone(node.resource)
        del node.resource
        self.assertIsNone(node.resource)

    # def test_save_resource(self):
    #     #no mesh -> False
    #     node= BIMNode()
    #     self.assertFalse(node.export_resource())

    #     #directory
    #     node= BIMNode(mesh=self.mesh2)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources')))

    #     #graphPath        
    #     node= BIMNode(mesh=self.mesh2,graphPath=self.bimGraphPath2)
    #     self.assertTrue(node.export_resource())

    #     #no path or graphPath
    #     node= BIMNode(mesh=self.mesh4)        
    #     self.assertTrue(node.export_resource())

    #     #invalid extension -> error
    #     node= BIMNode(mesh=self.mesh3)
    #     self.assertRaises(ValueError,node.export_resource,os.path.join(self.path,'resources'),'.kjhgfdfg')

    #     #.ply -> ok
    #     node= BIMNode(mesh=self.mesh2)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources'),'.ply'))
    #     self.assertEqual(node.path,os.path.join(self.path,'resources',node.name+'.ply'))

    #     #.obj -> ok
    #     node= BIMNode(mesh=self.mesh3)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources'),'.obj'))
    #     self.assertEqual(node.path,os.path.join(self.path,'resources',node.name+'.obj'))
        
    #     #path -> new name
    #     node= BIMNode(subject=URIRef('myMesh'),path=self.path2,getResource=True)
    #     self.assertTrue(node.export_resource())
    #     files=ut.get_list_of_files(ut.get_folder(node.path))
    #     self.assertTrue( node.path in files)
        
    #     #graphPath with directory
    #     node=BIMNode(subject=self.subject2,graphPath=self.bimGraphPath2, mesh=self.mesh3)
    #     self.assertTrue(node.export_resource(os.path.join(self.path,'resources')))
    #     files=ut.get_list_of_files(ut.get_folder(node.path))
    #     self.assertTrue(node.path in files)

    #     #graph with new subject
    #     node=BIMNode(subject=self.subject4,grap=self.bimGraph4, mesh=self.mesh4)
    #     node.name='myMesh'
    #     self.assertTrue(node.export_resource())
    #     files=ut.get_list_of_files(ut.get_folder(node.path))
    #     self.assertTrue(node.path in files)

    def test_get_resource(self):
        #mesh
        node=BIMNode(resource=self.dataLoaderRoad.collectorMesh)  
        self.assertIsNotNone(node.get_resource())

        #no mesh
        del node.resource
        self.assertIsNone(node.get_resource())

        # #graphPath with getResource
        # subject=next(self.dataLoaderRoad.ifcGraph.subjects(RDF.type))
        # node=BIMNode(graphPath=self.dataLoaderRoad.ifcGraphPath,subject=subject,getResource=True)
        # self.assertIsNotNone(node.get_resource())

    def test_get_metadata_from_resource(self):
        #mesh
        node=BIMNode(resource=self.dataLoaderParking.columnMesh)  
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.faceCount)
        self.assertIsNotNone(node.pointCount)

        #ifcElement
        node=BIMNode(resource=self.dataLoaderParking.ifcColumn)  
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.faceCount)
        self.assertIsNotNone(node.pointCount)

        #ifcPath and globalId
        node=BIMNode(ifcPath=self.dataLoaderParking.ifcPath,globalId=self.dataLoaderParking.ifcSlab.GlobalId, getResource=True)  
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.faceCount)
        self.assertIsNotNone(node.pointCount)

        #graphPath
        subject=next(self.dataLoaderParking.ifcGraph.subjects(RDF.type))
        node=BIMNode(graphPath=self.dataLoaderParking.ifcGraphPath,subject=subject,getResource=True)
        self.assertIsNotNone(node.orientedBounds)
        self.assertIsNotNone(node.cartesianBounds)
        self.assertIsNotNone(node.cartesianTransform)
        self.assertIsNotNone(node.faceCount)
        self.assertIsNotNone(node.pointCount)

    # def test_set_path(self):
    #     #valid path
    #     node=BIMNode()
    #     node.path= self.path1
    #     self.assertEqual(node.path,self.path1.as_posix())

    #     #preexisting
    #     node=BIMNode(path=self.path4)
    #     self.assertEqual(node.path,self.path4.as_posix())

    #     #graphPath & name
    #     node=BIMNode(subject=self.subject4,graphPath=self.bimGraphPath4)
    #     node.get_path()
    #     self.assertEqual(node.path,self.path4.as_posix())

    def test_set_ifc_path(self):
        #valid path
        node=BIMNode(ifcPath=self.dataLoaderRoad.ifcPath)
        self.assertEqual(node.ifcPath,self.dataLoaderRoad.ifcPath.as_posix())

        #invalid
        self.assertRaises(ValueError,BIMNode,ifcPath='qsffqsdf.dwg')

    def get_metadata_from_ifc_path(self):
        a=0
        elements=ifcopenshell.util.selector.filter_elements(self.dataLoaderParking.ifc, 'IfcElement')
        for ifcElement in elements:
            a+=1
            node=BIMNode(ifcPath=self.dataLoaderParking.ifcPath,globalId=ifcElement.GlobalId)            
            self.assertEqual(node.className,ifcElement.is_a())
            if a==100:
                break
        
        #check error no global id in ifc
        node= BIMNode()
        self.assertRaises(ValueError,BIMNode,ifcPath=self.dataLoaderParking.ifcPath,globalId='kjhgfdfg')

    def test_get_metadata_from_ifcElement(self):
        #ifc3
        a=0
        elements=ifcopenshell.util.selector.filter_elements(self.dataLoaderParking.ifc, 'IfcElement')
        for ifcElement in elements:
            a+=1
            node=BIMNode(resource=ifcElement)
            self.assertEqual(node.className,ifcElement.is_a())
            if a==100:
                break

if __name__ == '__main__':
    unittest.main()
