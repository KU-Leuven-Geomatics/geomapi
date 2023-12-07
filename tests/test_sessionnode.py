import os
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import open3d as o3d
import rdflib
from rdflib import RDF, RDFS, Graph, Literal, URIRef
import sys

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
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
        cls.tempNode= SessionNode(graph=cls.dataLoaderParking.resourceGraph)


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

    def test_SessionNode_creation_from_subject(self):
        #subject
        subject='myNode'
        node= SessionNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'file:///'+subject)

        #http://
        subject='http://session_2022_05_20'
        node= SessionNode(subject=subject)
        self.assertEqual(node.subject.toPython(),subject)
        
        #erroneous char       
        subject='[[http://ses>sion_2022_<05_20]]'
        node= SessionNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://__ses_sion_2022__05_20__')

    def test_SessionNode_creation_from_single_graph(self):
        node= SessionNode(graph=self.dataLoaderParking.sessionGraph)
        self.assertEqual(node.subject.toPython(),'file:///c8c9c891-9454-11ee-8f1b-c8f75043ce59')
        
    def test_SessionNode_creation_from_data_graph(self):        
        node= SessionNode(graph=self.dataLoaderParking.sessionGraph)
        self.assertEqual(len(node.linkedSubjects),6)

    # def test_SessionNode_creation_from_combined_graph(self):
    #     node= SessionNode(graph=self.data)
    #     self.assertEqual(node.subject,self.subject)
    #     self.assertEqual(len(node.linkedNodes),len(self.linkedSubjects))

    def test_SessionNode_creation_from_graph_path(self):
        subject=next(self.dataLoaderParking.sessionGraph.subjects(RDF.type))
        node= SessionNode(graphPath=self.dataLoaderParking.sessionGraphPath, subject=subject)
        self.assertEqual(node.subject,subject)
        
    def test_get_linked_nodes_from_self_linked_subjects(self):
        node= SessionNode(graph=self.dataLoaderParking.sessionGraph)
        self.assertEqual(len(node.linkedSubjects),6)
        self.assertEqual(len(node.linkedNodes),0)
        node.get_linked_nodes(self.dataLoaderParking.resourceGraph)
        self.assertEqual(len(node.linkedNodes),6)
        
    def test_get_linked_nodes_from_other_linked_subjects(self):
        node= SessionNode(graph=self.dataLoaderParking.sessionGraph)
        self.assertEqual(len(node.linkedSubjects),6)
        self.assertEqual(len(node.linkedNodes),0)
        node.get_linked_nodes(self.dataLoaderParking.resourceGraph)
        self.assertEqual(len(node.linkedNodes),6)

    def test_add_linked_nodes(self):
        combinedGraph=self.dataLoaderParking.sessionGraph+self.dataLoaderParking.resourceGraph
        node= SessionNode(graph=combinedGraph)
        node.linkedNodes.append(Node())
        self.assertEqual(len(node.linkedNodes),7)
        self.assertEqual(len(node.linkedSubjects),7)

    def test_SessionNode_creation_from_linked_nodes(self):
        node= SessionNode(linkedNodes=self.tempNode.linkedNodes)
        self.assertEqual(len(node.linkedNodes),len(self.tempNode.linkedNodes))

    def test_add_linked_nodes(self):
        node= SessionNode(linkedNodes=self.tempNode.linkedNodes)
        nodelist2=[MeshNode(),Node()]
        node.add_linked_nodes(nodelist2)
        self.assertEqual(len(node.linkedNodes),len(self.tempNode.linkedSubjects)+2)
    
    def test_add_linked_nodes_with_doubles(self):
        node= SessionNode(linkedNodes=self.tempNode.linkedNodes)
        nodelist2=[MeshNode(),self.tempNode.linkedNodes[0]]
        node.add_linked_nodes(nodelist2)
        self.assertEqual(len(node.linkedNodes),len(self.tempNode.linkedSubjects)+1)

    # def test_SessionNode_creation_from_resources(self):
    #     resources=[n.resource for n in self.tempNode.linkedNodes]
    #     node= SessionNode(linkedResources=resources)       
    #     self.assertTrue(all(True for n in node.linkedNodes if n.resource is not None))
    #     self.assertGreater(len(node.linkedNodes),0)

    def test_save_linked_resources(self):  
        combinedGraph=self.dataLoaderParking.sessionGraph+self.dataLoaderParking.resourceGraph
        node= SessionNode(graph=combinedGraph)
        node.save_linked_resources(self.dataLoaderParking.resourcePath)

    # def test_get_linked_resources(self):
    #     node= SessionNode(graphPath=self.combinedGraphPath)
    #     resources=node.get_linked_resources()
    #     self.assertEqual(len(resources),len(node.linkedNodes))

    # def test_get_linked_resources_multiprocessing(self):
    #     node= SessionNode(graphPath=self.combinedGraphPath)
    #     resources=node.get_linked_resources_multiprocessing()
    #     self.assertEqual(len(resources),len(node.linkedNodes))

    def test_get_metadata_from_linked_nodes(self):
        node= SessionNode(graph=self.dataLoaderParking.resourceGraph)
        node.get_metadata_from_linked_nodes()
        self.assertIsNotNone(node.orientedBoundingBox)

    def test_creation_from_subject_and_graph_and_graphPath(self):        
        subject=next(self.dataLoaderParking.resourceGraph.subjects(RDF.type))
        node= SessionNode(subject=subject,
                          graph=self.dataLoaderParking.resourceGraph,
                          graphPath=self.dataLoaderParking.resourceGraphPath)
        self.assertEqual(node.subject,subject)

    def test_linked_nodes_to_graph(self):  
        combinedGraph=self.dataLoaderParking.sessionGraph+self.dataLoaderParking.resourceGraph
        node= SessionNode(graph=combinedGraph)
        node.linked_nodes_to_graph(os.path.join(self.dataLoaderParking.resourcePath,'linkednodesGraph.ttl'))

    
    def test_session_to_graph(self):   
        combinedGraph=self.dataLoaderParking.sessionGraph+self.dataLoaderParking.resourceGraph
     
        node= SessionNode(graph=combinedGraph)
        node.session_to_graph(os.path.join(self.dataLoaderParking.resourcePath,'combinedGraph.ttl'))

if __name__ == '__main__':
    unittest.main()
