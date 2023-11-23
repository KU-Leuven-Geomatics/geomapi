import datetime
import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value

import cv2
import geomapi.utils as ut
import numpy as np
import open3d as o3d
import rdflib
from geomapi.nodes import *
from rdflib import RDF, RDFS, Graph, Literal, URIRef


class TestNode(unittest.TestCase):

################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        cls.path= Path.cwd() / "test" / "testfiles"  

        #ONTOLOGIES
        cls.exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns#')
        cls.geo=rdflib.Namespace('http://www.opengis.net/ont/geosparql#') #coordinate system information
        cls.gom=rdflib.Namespace('https://w3id.org/gom#') # geometry representations => this is from mathias
        cls.omg=rdflib.Namespace('https://w3id.org/omg#') # geometry relations
        cls.fog=rdflib.Namespace('https://w3id.org/fog#')
        cls.v4d=rdflib.Namespace('https://w3id.org/v4d/core#')
        cls.openlabel=rdflib.Namespace('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#')
        cls.e57=rdflib.Namespace('http://libe57.org#')
        cls.xcr=rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
        cls.ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')

        #GRAPH 1
        cls.graphPath1=cls.path / 'bimGraph1.ttl'
        cls.graph1=Graph().parse(cls.graphPath1)

        #GRAPH 2
        cls.graphPath2=cls.path / 'resourceGraph.ttl'
        cls.graph2=Graph().parse(cls.graphPath2)

        #GRAPH 3
        cls.graphPath3=cls.path / 'pcdGraph.ttl'
        cls.graph3=Graph().parse(cls.graphPath3)

        #GRAPH 4
        cls.graphPath4=cls.path / 'meshGraph.ttl'
        cls.graph4=Graph().parse(cls.graphPath4)
        
        #GRAPH 5
        cls.graphPath5=cls.path / 'imgGraph.ttl'
        cls.graph5=Graph().parse(cls.graphPath5)
                
        #POINTCLOUD
        cls.pcdPath=cls.path / 'PCD'/"academiestraat week 22 a 20.pcd"
        cls.e57Path=cls.path / 'PCD'/"week22 photogrammetry - Cloud.e57"
        
        #MESH
        cls.meshPath=cls.path / 'MESH'/"week22.obj"
    
        #IMG
        cls.image1Path=cls.path / "IMG"/"IMG_2173.JPG"  
        cls.image2Path=cls.path / "IMG"/"IMG_2174.JPG"  

        #FILES
        cls.files=ut.get_list_of_files(os.getcwd())
        cls.files+=ut.get_list_of_files(cls.path)

        #RESOURCES
        cls.resourcePath=cls.path / "resources"
        if not os.path.exists(cls.resourcePath):
            os.mkdir(cls.resourcePath)

        #TIME TRACKING           
        et = time.time()
        print("startup time: "+str(et - st))
        print('{:50s} {:5s} '.format('tests','time'))
        print('------------------------------------------------------')

    @classmethod
    def tearDownClass(cls):
        #execute once after all tests
        print('-----------------TearDown Class----------------------')
        shutil.rmtree(cls.resourcePath)      
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
        
        #graph3
        graph=self.graph3
        subject=next(graph.subjects(RDF.type))
        graph=ut.get_subject_graph(graph,subject)
        node= Node(graph=graph,graphPath=self.graphPath3)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        
        #graph2
        graph=self.graph2
        subject=next(graph.subjects(RDF.type))
        graph=ut.get_subject_graph(graph,subject)
        node= Node(graph=graph,graphPath=self.graphPath2)
        self.assertEqual(node.subject.toPython(),subject.toPython())

    def test_node_creation_from_graphs_with_wrong_subject(self):   
        self.assertRaises(ValueError,Node,subject='myNode',graph=self.graph5)

    def test_node_creation_from_graphpath_with_wrong_subject(self): 
        self.assertRaises(ValueError,Node,subject='qsdsfqsdfq',graphPath=self.graphPath3)

    def test_node_creation_from_graphPaths(self):    
        #normal graphPath
        node= Node(graphPath=self.graphPath1)
        self.assertTrue(node.graphPath in self.files)

        #path nonexisting
        node= Node(graphPath=os.path.join(self.path,'qsdf.ttl'))
        self.assertIsNotNone(node.graphPath)

        #invalid path
        self.assertRaises(ValueError,Node,graphPath=os.path.join(self.path,'qsdf.qdf'))

        #graph4
        graph=self.graph4
        graphPath=self.graphPath4
        subject=next(graph.subjects(RDF.type))        
        node= Node(graphPath=graphPath)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        
        #graph2
        graph=self.graph2
        graphPath=self.graphPath2
        subject=next(graph.subjects(RDF.type))
        node= Node(graphPath=graphPath,subject=subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())

    def test_get_metadata_from_graph(self):  
        #single graph
        subject=next(self.graph2.subjects(RDF.type))
        graph=ut.get_subject_graph(self.graph2,subject)    
        node= Node(graph=graph,graphPath=self.graphPath2)
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

        #big graph
        subject=next(self.graph3.subjects(RDF.type))
        node= Node(graph=self.graph3, subject=subject,graphPath=self.graphPath3)
        node.get_metadata_from_graph(node.graph,node.subject)
        self.assertEqual(node.subject.toPython(),subject.toPython())
        if getattr(node,'cartesianBounds',None) is not None:
            self.assertEqual(node.cartesianBounds.size,6)
        if getattr(node,'cartesianTransform',None) is not None:
            self.assertEqual(node.cartesianTransform.size,16)
        if getattr(node,'orientedBounds',None) is not None:        
            self.assertEqual(node.orientedBounds.size,24)
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
        node= Node(path=self.pcdPath)
        self.assertEqual(node.get_name(),ut.get_filename(self.pcdPath))

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
        node=Node(graph=self.graph3)
        self.assertLess(len(node.get_graph()),len(self.graph3))

        #real graphPath
        node=Node(graphPath=self.graphPath3)
        self.assertLess(len(node.get_graph()),len(self.graph3))

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
        node=Node(path=self.pcdPath)
        self.assertEqual(node.get_timestamp(),ut.get_timestamp(self.pcdPath))

        #graphPath
        node=Node(graphPath=self.graphPath2)
        object=self.graph2.value(next(s for s in self.graph2.subjects(RDF.type)),self.openlabel['timestamp'])
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
        s=next(self.graph2.subjects(RDF.type))
        node=Node(graph=self.graph2)
        self.assertEqual(node.get_subject(),s)

        #path
        node=Node(path=self.pcdPath)
        self.assertEqual(node.get_subject(),URIRef('file:///'+ut.validate_string(ut.get_filename(self.pcdPath))))

    def test_get_path(self): 
        #empty
        node=Node()
        self.assertIsNone(node.path)

        #path  
        node=Node(path=self.pcdPath)
        self.assertEqual(node.path,self.pcdPath.as_posix())

        #no path  -> cwd()
        node=Node()
        self.assertIsNone(node.get_path())

        #graphPath 
        node=Node(graphPath=self.graphPath2)
        testPath=node.graph.value(node.subject,self.v4d['path'].toPython())
        self.assertIsNotNone(node.get_path(), testPath)

    def test_to_graph(self):
        #empty
        node=Node()
        graph=node.to_graph()
        subject=next(graph.subjects(RDF.type))
        self.assertEqual(subject,node.subject)

        #subject and kwargs
        attribute='myAttribute'
        v4d=rdflib.Namespace('https://w3id.org/v4d/core#')
        node=Node(subject=URIRef('node'),attribute=attribute)
        graph=node.to_graph()
        subject=next(graph.subjects(RDF.type))
        self.assertEqual(subject,node.subject)
        object=next(node.graph.objects(subject,v4d['attribute']))
        self.assertEqual(object.toPython(),attribute)


    def test_to_graph_with_paths(self):
        #graphPath should be None
        node=Node(graphPath=self.graphPath1)
        node.to_graph()
        testPath=node.graph.value(node.subject,self.v4d['graphPath'])
        self.assertIsNone(testPath)

        #paths should be shortened
        resourcePath=os.path.join(self.path,'resources','week22.obj')
        node=Node(graphPath=self.graphPath3)
        node.path=resourcePath
        node.to_graph()
        testPath=node.graph.value(node.subject,self.v4d['path']).toPython()
        self.assertEqual(testPath,os.path.join('resources','week22.obj'))

    def test_to_graph_with_save(self):
        #test save
        testPath=os.path.join(self.resourcePath,'graph3.ttl')
        node=Node(graph=self.graph3)
        node.to_graph(graphPath=testPath,save=True)        
        self.assertTrue(os.path.exists(testPath))
        newGraph=Graph().parse(testPath)
        self.assertEqual(len(newGraph),len(node.graph))

        #test invalid save
        testPath=os.path.join(self.path,'resources','graph3.sdfqlkbjdqsf')
        node=Node(graph=self.graph3)
        self.assertRaises(ValueError,node.to_graph,testPath,save=True)

    def test_save_graph(self):
        tempGraphPath=os.path.join(self.resourcePath,'node.ttl')

        #node with only subject
        subject='node'
        node= Node(subject=subject)
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())

        #node with a graph and some kwargs
        node= Node(graph=self.graph3,myNewRemark='myNewRemark')        
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())

        #node with a graph and a subject and some change    
        subject=next(self.graph2.subjects(RDF.type))
        node= Node(graph=self.graph2,graphPath=self.graphPath2,subject=subject)  
        node.pointCount=1000
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())
        self.assertEqual(node.pointCount,testnode.pointCount)

        #node with a graph, a graphPath and a subject and another change
        subject=next(self.graph2.subjects(RDF.type))
        node= Node(graph=self.graph2,graphPath=self.graphPath2,subject=subject)  
        node.cartesianTransform=np.array([[1,2,3,4],
                                        [1,2,3,4],
                                        [1,2,3,4],
                                        [1,2,3,4],])
        node.to_graph(tempGraphPath,save=True)
        testnode=Node(graphPath=tempGraphPath)
        self.assertEqual(node.subject.toPython(),testnode.subject.toPython())
    
if __name__ == '__main__':
    unittest.main()
