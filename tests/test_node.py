#LIBRARIES
import datetime
import sys
import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import cv2
import numpy as np
import open3d as o3d
import rdflib
from rdflib import RDF, RDFS, Graph, Literal, URIRef



#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.tools as tl 
from geomapi.nodes import *

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 

# from data_loader_parking import DataLoaderParking
# from data_loader_road import DataLoaderRoad



class TestNode(unittest.TestCase):
    
    
    

################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        
        cls.dataLoader = DATALOADERPARKINGINSTANCE

        #TIME TRACKING           
        et = time.time()
        print("startup time: "+str(et - st))
        print('{:50s} {:5s} '.format('tests','time'))
        print('------------------------------------------------------')

    @classmethod
    def tearDownClass(cls):
        #execute once after all tests
        if os.path.exists(cls.dataLoader.resourcePath):
            shutil.rmtree(cls.dataLoader.resourcePath)  
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
    def test_node_creation_with_subject(self):
        node=Node()
        node.subject=None
        self.assertIsNotNone(node.subject)
        node=Node('http://pointcloud2_0')
        self.assertEqual(node.subject.toPython(),'http://pointcloud2_0')
        node=Node('http://po/int/cloud2_ 0:')
        self.assertEqual(node.subject.toPython(),'http://po_int_cloud2__0_')
        node=Node('pointc&:lo ud2_0')
        self.assertEqual(node.subject.toPython(),'file:///pointc&_lo_ud2_0')
        node=Node('file:///pointc&:lo ud2_0')
        self.assertEqual(node.subject.toPython(),'file:///pointc&_lo_ud2_0')
        node=Node('file:///pointc&:lo /ud2_0')
        self.assertEqual(node.subject.toPython(),'file:///pointc&_lo__ud2_0')
        node=Node('4499de21-f13f-11ec-a70d-c8f75043ce59')
        self.assertEqual(node.subject.toPython(),'file:///4499de21-f13f-11ec-a70d-c8f75043ce59')
        node=Node('[this<has$to^change]')
        self.assertEqual(node.subject.toPython(),'file:///_this_has_to_change_')
        
    def test_node_creation_with_kwargs(self):
        #string
        attribute='myAttribute'
        node= Node('node',attribute=attribute)
        self.assertEqual(node.attribute,attribute)

        #float
        attribute=0.5
        node= Node('node',attribute=attribute)
        self.assertEqual(node.attribute,attribute)
        
        #list
        attribute=['myAttribute1','myAttribute2']
        node= Node('node',attribute=attribute)
        self.assertEqual(node.attribute,attribute)

        #URIRef
        attribute=URIRef('myAttribute1')
        node= Node('node',attribute=attribute)
        self.assertEqual(node.attribute,attribute)
        

    def test_node_creation_from_different_graphs(self):  
        
        #graph1
        graph=self.dataLoader.resourceGraph
        subject=next(graph.subjects(RDF.type))
        graph=ut.get_subject_graph(graph,subject)
        node= Node(graph=graph,graphPath=self.dataLoader.resourceGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        
        #graph2
        graph=self.dataLoader.meshGraph
        subject=next(graph.subjects(RDF.type))
        graph=ut.get_subject_graph(graph,subject)
        node= Node(graph=graph,graphPath=self.dataLoader.meshGraphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())

    def test_node_creation_from_graphs_with_wrong_subject(self):   
        self.assertRaises(ValueError,Node,subject='myNode',graph=self.dataLoader.pcdGraph)

    def test_node_creation_from_graphPaths(self):    
        #normal graphPath
        node= Node(graphPath=self.dataLoader.imgGraphPath)
        self.assertTrue(node.graphPath in self.dataLoader.files)

        #path nonexisting
        node= Node(graphPath=os.path.join(self.dataLoader.path,'qsdf.ttl'))
        self.assertIsNotNone(node.graphPath)

        #invalid path
        self.assertRaises(ValueError,Node,graphPath=os.path.join(self.dataLoader.path,'qsdf.qdf'))

        #graph4
        graph=self.dataLoader.resourceGraph
        graphPath=self.dataLoader.resourceGraphPath
        subject=next(graph.subjects(RDF.type))        
        node= Node(graphPath=graphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        
        #graph2
        graph=self.dataLoader.pcdGraph
        graphPath=self.dataLoader.pcdGraphPath
        subject=next(graph.subjects(RDF.type))
        node= Node(graphPath=graphPath,subject=subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())

    def test_get_metadata_from_graph(self):  
        #single graph
        subject=next(self.dataLoader.meshGraph.subjects(RDF.type))
        graph=ut.get_subject_graph(self.dataLoader.meshGraph,subject)    
        node= Node(graph=graph,graphPath=self.dataLoader.meshGraphPath)
        node.get_metadata_from_graph(node.graph,node.subject)
        if getattr(node,'cartesianBounds',None) is not None:
            self.assertEqual(node.subject.toPython(),subject.toPython())
        self.assertEqual(node.cartesianBounds.size,6)
        if getattr(node,'cartesianTransform',None) is not None:
            self.assertEqual(node.cartesianTransform.size,16)
        if getattr(node,'orientedBounds',None) is not None:        
            self.assertEqual(node.orientedBounds.size,24)
        if getattr(node,'geospatialTransform',None) is not None:        
            self.assertEqual(node.geospatialTransform, None)
        if getattr(node,'orientedBoundingBox',None) is not None:        
            self.assertIsInstance(node.orientedBoundingBox,o3d.geometry.OrientedBoundingBox)

    def test_get_name(self):  
        #empty
        node=Node()
        self.assertIsNotNone(node.name)

        #name
        node= Node(name='myN<<<ame')
        self.assertEqual(node.get_name(),'myN<<<ame')

        #path
        node= Node(path=self.dataLoader.pcdPath)
        self.assertEqual(node.get_name(),ut.get_filename(self.dataLoader.pcdPath))

        #subject
        node= Node('node')
        self.assertEqual(node.get_name(),'node')

        #http://
        node= Node('http://session_2022_05_20')
        self.assertEqual(node.get_name(),'session_2022_05_20')
        
        #file:///
        node= Node('file:///101_0366_0036')
        self.assertEqual(node.get_name(),'101_0366_0036')      

        #None
        node=Node()
        self.assertEqual(len(node.get_name()),36)

    def test_get_graph(self):
        #empty
        node=Node()
        self.assertIsNone(node.graph)

        #graph
        node=Node(graph=self.dataLoader.pcdGraph)
        self.assertLess(len(node.get_graph()),len(self.dataLoader.pcdGraph))

        #real graphPath
        node=Node(graphPath=self.dataLoader.pcdGraphPath)
        self.assertLess(len(node.get_graph()),len(self.dataLoader.pcdGraph))

        #graphPath non existent
        node=Node(graphPath='myNewGraphPath.ttl')
        self.assertIsNone(node.get_graph())

    def test_get_timestamp(self):
        #empty
        node=Node()
        time=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.assertEqual(node.get_timestamp(),time )

        #timestamp
        node=Node(timestamp=time)
        self.assertEqual(node.get_timestamp(),time)

        #path
        node=Node(path=self.dataLoader.pcdPath)
        self.assertEqual(node.get_timestamp(),ut.get_timestamp(self.dataLoader.pcdPath))

        #graphPath
        node=Node(graphPath=self.dataLoader.meshGraphPath)
        object=self.dataLoader.meshGraph.value(next(s for s in self.dataLoader.meshGraph.subjects(RDF.type)),self.dataLoader.openlabel['timestamp'])
        self.assertEqual(node.get_timestamp(),object.toPython())

    def test_get_subject(self):
        #empty -> uuid
        node=Node()
        self.assertEqual(len(node.subject),44)

        #subject
        node=Node('node')
        self.assertEqual(node.subject,URIRef('file:///node'))

        
        #subject
        node=Node(name='my:Nod:e')
        self.assertEqual(node.subject,URIRef('file:///my_Nod_e'))

        #graph
        s=next(self.dataLoader.imgGraph.subjects(RDF.type))
        node=Node(graph=self.dataLoader.imgGraph)
        self.assertEqual(node.get_subject(),s)

        #path
        node=Node(path=self.dataLoader.pcdPath)
        self.assertEqual(node.get_subject(),URIRef('file:///'+ut.validate_string(ut.get_filename(self.dataLoader.pcdPath))))

    def test_get_path(self): 
        #empty
        node=Node()
        self.assertIsNone(node.path)

        #path  
        node=Node(path=self.dataLoader.pcdPath)
        self.assertEqual(node.path,self.dataLoader.pcdPath.as_posix())

        #no path  -> cwd()
        node=Node()
        self.assertIsNone(node.get_path())

        #graphPath 
        node=Node(graphPath=self.dataLoader.resourceGraphPath)
        testPath=node.graph.value(node.subject,self.dataLoader.v4d['path'].toPython())
        self.assertIsNotNone(node.get_path(), testPath)

    def test_to_graph(self):
        #empty
        node=Node()
        graph=node.to_graph()
        subject=next(graph.subjects(RDF.type))
        self.assertEqual(subject,node.subject)

        #subject and kwargs
        attribute='myAttribute'
        # v4d=rdflib.Namespace('https://w3id.org/v4d/core#')
        node=Node(subject=URIRef('node'),attribute=attribute)
        graph=node.to_graph()
        subject=next(graph.subjects(RDF.type))
        self.assertEqual(subject,node.subject)
        object=next(node.graph.objects(subject,self.dataLoader.v4d['attribute']))
        self.assertEqual(object.toPython(),attribute)

    def test_to_graph_with_paths(self):
        #graphPath should be None
        node=Node(graphPath=self.dataLoader.pcdGraphPath)
        node.to_graph()
        testPath=node.graph.value(node.subject,self.dataLoader.v4d['graphPath'])
        self.assertIsNone(testPath)

        #paths should be shortened
        resourcePath=os.path.join(self.dataLoader.path,'resources','parking.obj')
        node=Node(graphPath=self.dataLoader.meshGraphPath)
        node.path=resourcePath
        node.to_graph()
        testPath=node.graph.value(node.subject,self.dataLoader.v4d['path']).toPython()
        self.assertEqual(testPath,os.path.join('..','resources','parking.obj'))

    def test_to_graph_with_save(self):
        #test save
        testPath=os.path.join(self.dataLoader.resourcePath,'graph3.ttl')
        node=Node(graph=self.dataLoader.imgGraph)
        node.to_graph(graphPath=testPath,save=True)        
        self.assertTrue(os.path.exists(testPath))
        newGraph=Graph().parse(testPath)
        self.assertEqual(len(newGraph),len(node.graph))

        #test invalid save
        testPath=os.path.join(self.dataLoader.path,'resources','graph3.sdfqlkbjdqsf')
        node=Node(graph=self.dataLoader.imgGraph)
        self.assertRaises(ValueError,node.to_graph,testPath,save=True)

    def test_save_graph(self):
        tempGraphPath=os.path.join(self.dataLoader.resourcePath,'node.ttl')

        #node with only subject
        subject='node'
        node= Node(subject=subject)
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())

        #node with a graph and some kwargs
        node= Node(graph=self.dataLoader.resourceGraph,myNewRemark='myNewRemark')        
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())

        #node with a graph and a subject and some change    
        subject=next(self.dataLoader.pcdGraph.subjects(RDF.type))
        node= Node(graph=self.dataLoader.pcdGraph,graphPath=self.dataLoader.pcdGraphPath,subject=subject)  
        node.pointCount=1000
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())
        self.assertEqual(node.pointCount,testnode.pointCount)

        #node with a graph, a graphPath and a subject and another change
        subject=next(self.dataLoader.meshGraph.subjects(RDF.type))
        node= Node(graph=self.dataLoader.meshGraph,graphPath=self.dataLoader.meshGraphPath,subject=subject)  
        node.cartesianTransform=np.array([[1,2,3,4],
                                        [1,2,3,4],
                                        [1,2,3,4],
                                        [1,2,3,4],])
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())
    
if __name__ == '__main__':
    unittest.main()
