import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import sys
import cv2
import rdflib
from geomapi.nodes import *
from PIL import Image
from rdflib import RDF, RDFS, Graph, Literal, URIRef


#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.utils.imageutils as iu
from geomapi.nodes import *

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 

class TestImageNode(unittest.TestCase):



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

    def test_ImageNode_creation_from_subject(self):
        #subject
        subject='myNode'
        node= ImageNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'file:///'+subject)

        #http://
        subject='http://session_2022_05_20'
        node= ImageNode(subject=subject)
        self.assertEqual(node.subject.toPython(),subject)
        
        #erroneous char       
        subject='[[http://ses>sion_2022_<05_20]]'
        node= ImageNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://__ses_sion_2022__05_20__')

    def test_ImageNode_creation_from_graph(self):
        subject=next(self.dataLoaderParking.imgGraph.subjects(RDF.type))
        node= ImageNode(graph=self.dataLoaderParking.imgGraph, subject=subject)
        self.assertEqual(node.subject,subject)
        
    def test_ImageNode_creation_from_graph_path(self):
        subject=next(self.dataLoaderParking.imgGraph.subjects(RDF.type))
        node= ImageNode(graphPath=self.dataLoaderParking.imgGraphPath, subject=subject)
        self.assertEqual(node.subject,subject)
        
    def test_ImageNode_creation_from_path(self):
        #path1 without getResource
        node= ImageNode(path=self.dataLoaderParking.imagePath2)
        self.assertEqual(node.name,ut.get_filename(self.dataLoaderParking.imagePath2))
        self.assertIsNone(node._resource)

        #path2 with getResource
        node= ImageNode(path=self.dataLoaderRoad.imagePath1,getResource=True)        
        self.assertEqual(node.name,ut.get_filename(self.dataLoaderRoad.imagePath1))
        self.assertEqual(node.imageHeight,self.dataLoaderRoad.image1.shape[0])

    def test_ImageNode_creation_from_xmpPath(self):
        #path without extra info 
        node= ImageNode(xmpPath=self.dataLoaderParking.imageXmpPath1)
        self.assertIsNotNone(node.name)

        #path with getResource without extra info 
        node= ImageNode(xmpPath=self.dataLoaderParking.imageXmpPath2,getResource=True)        
        self.assertIsNotNone(node.name)
        self.assertIsNotNone(node.resource)


    def test_ImageNode_creation_from_xmlPath(self):
        
        
        #path without extra info 
        node= ImageNode(xmlPath=self.dataLoaderRoad.imageXmlPath,
                        subject='101_0367_0007')
        self.assertIsNotNone(node.name)

        #path with subject
        node= ImageNode(xmlPath=self.dataLoaderRoad.imageXmlPath,
                        subject='101_0367_0007',
                        getResource=True)        
        self.assertIsNotNone(node.name)
        self.assertIsNotNone(node.resource)

    def test_ImageNode_creation_from_resource(self):
        #img1
        node= ImageNode(resource=self.dataLoaderRoad.image1)
        self.assertEqual(node.imageHeight,self.dataLoaderRoad.image1.shape[0])
        #img2
        node= ImageNode(resource=self.dataLoaderRoad.image2)
        self.assertEqual(node.imageHeight,self.dataLoaderRoad.image2.shape[0])

    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=next(self.dataLoaderRoad.imgGraph.subjects(RDF.type))
        node= ImageNode(subject=subject,graph=self.dataLoaderRoad.imgGraph,graphPath=self.dataLoaderRoad.imgGraphPath)
        self.assertEqual(node.subject,subject)
        
    def test_creation_from_subject_and_path(self):   
        subject=next(self.dataLoaderRoad.ifcGraph.subjects(RDF.type))
     
        node= ImageNode(subject=subject,path=self.dataLoaderRoad.imagePath1,getResource=True)
        self.assertEqual(node.subject,subject)
        node.to_graph()
        self.assertEqual(node.imageHeight,self.dataLoaderRoad.image1.shape[0])
        
    def test_creation_from_subject_and_path_and_graph(self): 
        subject=self.dataLoaderParking.imageSubject1

        node= ImageNode(subject=subject,
                        path=self.dataLoaderParking.imagePath1,
                        graph=self.dataLoaderParking.imgGraph,
                        getResource=True)
        self.assertEqual(node.subject,subject)
        node.to_graph()
        initialGraph=ut.get_subject_graph(self.dataLoaderParking.imgGraph,subject=subject)
        self.assertEqual(len(node.graph),len(initialGraph))

    def test_creation_from_resource_and_path (self):      
        node= ImageNode(resource=self.dataLoaderParking.image1,
                        path=self.dataLoaderParking.imagePath1)
        self.assertEqual(node.subject,next(self.dataLoaderParking.imgGraph.subjects(RDF.type)))

    def test_creation_from_subject_and_resource_and_path_and_graph(self):      
        node= ImageNode(subject=self.dataLoaderRoad.imageSubject1,
                        resource=self.dataLoaderRoad.image1,
                        path=str(self.dataLoaderRoad.imagePath1), 
                        graph=self.dataLoaderRoad.imgGraph)
        self.assertEqual(node.subject,self.dataLoaderRoad.imageSubject1)

    def test_creation_from_subject_resource_path_xmpPath_graph(self):      
        node= ImageNode(subject=self.dataLoaderParking.imageSubject1,
                        resource=self.dataLoaderParking.image1,
                        path=self.dataLoaderParking.imagePath1, 
                        xmpPath=self.dataLoaderParking.imageXmpPath1,
                        graph=self.dataLoaderParking.imgGraph)
        self.assertEqual(node.subject,self.dataLoaderParking.imageSubject1)

    def test_node_creation_with_get_resource(self):
        #mesh
        node= ImageNode(resource=self.dataLoaderParking.image1)
        self.assertIsNotNone(node._resource)

        #path without getResource
        node= ImageNode(path=self.dataLoaderParking.imagePath2)
        self.assertIsNone(node._resource)

        #path with getResource
        node= ImageNode(path=self.dataLoaderParking.imagePath1,getResource=True)
        self.assertIsNotNone(node._resource)

        #graph with get resource
        node= ImageNode(subject=self.dataLoaderRoad.imageSubject1,
                        graph=self.dataLoaderRoad.imgGraph,
                        getResource=True)
        self.assertIsNone(node._resource)
        
        #graphPath with get resource
        node= ImageNode(subject=self.dataLoaderParking.imageSubject2,
                        graphPath=self.dataLoaderParking.imgGraphPath,
                        getResource=True)
        self.assertIsNone(node._resource)

    def test_clear_resource(self):
        #mesh
        node= ImageNode(resource=self.dataLoaderRoad.image1)
        self.assertIsNotNone(node._resource)
        del node.resource
        self.assertIsNone(node._resource)

    def test_save_resource(self):
        #no mesh -> False
        node= ImageNode()
        self.assertFalse(node.save_resource())

        #directory
        node= ImageNode(resource=self.dataLoaderRoad.image2)
        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))

        #graphPath        
        node= ImageNode(resource=self.dataLoaderParking.image1,
                        graphPath=self.dataLoaderParking.imgGraphPath)
        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))

        #no path or graphPath
        node= ImageNode(resource=self.dataLoaderRoad.image2)        
        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))

        #path -> new name
        node= ImageNode(subject=URIRef('myImg'),
                        path=self.dataLoaderRoad.imagePath2,
                        getResource=True)
        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))
        
        #graphPath with directory
        node=ImageNode(subject=self.dataLoaderParking.imageSubject1,
                       graphPath=self.dataLoaderParking.imgGraphPath,
                       resource=self.dataLoaderParking.image1)
        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))

        #graph with new subject
        node=ImageNode(subject=self.dataLoaderRoad.imageSubject2,
                       graph=self.dataLoaderRoad.imgGraph,
                       resource=self.dataLoaderRoad.image2)
        node.subject='myImg'
        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))

    def test_get_resource(self):
        #mesh
        node=ImageNode(resource=self.dataLoaderParking.image2)  
        self.assertIsNotNone(node.get_resource())

        #no mesh
        del node.resource
        self.assertIsNone(node.get_resource())

        #graphPath with getResource
        node=ImageNode(graphPath=str(self.dataLoaderParking.imgGraphPath),
                       subject=self.dataLoaderParking.imageSubject1,
                       getResource=True)
        self.assertIsNone(node.get_resource())

    def test_set_path(self):
        #valid path
        node=ImageNode()
        node.path= str(self.dataLoaderParking.imagePath1)
        self.assertEqual(node.path,self.dataLoaderParking.imagePath1.as_posix())

        #preexisting
        node=ImageNode(path=self.dataLoaderParking.imagePath2)
        self.assertEqual(node.path,self.dataLoaderParking.imagePath2.as_posix())

        #graphPath & name
        node=ImageNode(subject=self.dataLoaderParking.imageSubject1,
                       graphPath=self.dataLoaderParking.imgGraphPath)
        node.get_path()
        # self.assertEqual(node.path,str(self.dataLoaderParking.imagePath1))

    def test_set_xmp_path(self):
        #valid path
        node=ImageNode(xmpPath=str(self.dataLoaderParking.imageXmpPath1))
        self.assertEqual(node.xmpPath,self.dataLoaderParking.imageXmpPath1.as_posix())

        #invalid
        self.assertRaises(ValueError,ImageNode,xmpPath='qsffqsdf.dwg')

    def test_set_xml_path(self):
        #valid path
        node=ImageNode()
        node.xmlPath=self.dataLoaderRoad.imageXmlPath
        self.assertEqual(node.xmlPath,self.dataLoaderRoad.imageXmlPath.as_posix())

        #invalid
        self.assertRaises(ValueError,ImageNode,xmlPath='qsffqsdf.dwg')

if __name__ == '__main__':
    unittest.main()
