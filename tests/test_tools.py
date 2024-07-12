import os
from pathlib import Path
import shutil
import time
import unittest
import sys
import cv2
from rdflib import RDF, RDFS, Graph, Literal, URIRef

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from geomapi.nodes import *
import geomapi.tools as tl

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 

class TestTools(unittest.TestCase):


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

    def test_e57xml_to_nodes(self):
        # e57XML
        nodes=tl.e57xml_to_nodes(path=self.dataLoaderParking.e57XmlPath)
        self.assertEqual(len(nodes),2)
        #0
        self.assertAlmostEqual(nodes[0].cartesianTransform[0,3],0.379,delta=0.01)
        self.assertLess(nodes[0].cartesianTransform[0,0],0)
        self.assertAlmostEqual(nodes[0].cartesianBounds[0],-4.351,delta=0.01)
        #1
        self.assertEqual(nodes[1].cartesianTransform[0,3],0.0)
        self.assertEqual(nodes[1].cartesianTransform[0,0],-1)
        self.assertAlmostEqual(nodes[1].cartesianBounds[0],-5.69,delta=0.01)
        
        #with kwargs
        nodes=tl.e57xml_to_nodes(path=self.dataLoaderParking.e57XmlPath,myattrib=1)
        self.assertEqual(nodes[0].myattrib,1)

        # with getResource 
        # nodes=tl.e57xml_to_nodes(e57XmlPath=self.e57XmlPath,getResource=True)
        # geometries=[n.resource for n in nodes]
        # self.assertEqual(len(geometries),2)

    def test_img_xml_to_nodes(self):

        self.assertEqual(len(tl.img_xml_to_nodes(self.dataLoaderRoad.imageXmlPath,filterByFolder=False) ),78)
        self.assertEqual(len(tl.img_xml_to_nodes(self.dataLoaderRoad.imageXmlPath,filterByFolder=True) ),2)
        nodes=tl.img_xml_to_nodes(self.dataLoaderRoad.imageXmlPath,filterByFolder=True,getResource=True) 
        self.assertEqual(len(nodes),2)
        self.assertEqual(len([n.resource for n in nodes]),2)
        
        self.assertEqual(len(tl.img_xml_to_nodes(self.dataLoaderRoad.imageXmlPath,skip=10,filterByFolder=False) ),8)
                        
    def test_e57header_to_nodes(self):

        #E571
        nodes=tl.e57header_to_nodes(path=self.dataLoaderParking.e57Path1)
        self.assertEqual(len(nodes),2)
        self.assertIsNone(nodes[0]._resource)
        #0
        self.assertAlmostEqual(nodes[0].cartesianTransform[0,3],4.52,delta=0.01)
        #1
        self.assertAlmostEqual(nodes[1].cartesianTransform[0,3],0.51,delta=0.01)
        
        #with kwargs
        nodes=tl.e57xml_to_nodes(path=self.dataLoaderParking.e57XmlPath,myattrib=1)
        self.assertEqual(nodes[0].myattrib,1)

        # # with getResource 
        # nodes=tl.e57xml_to_nodes(e57Path=self.e57XmlPath,getResource=True)
        # geometries=[n.resource for n in nodes]
        # self.assertEqual(len(geometries),2)

        #E572
        tl.e57header_to_nodes(path=self.dataLoaderParking.e57Path2)
        self.assertAlmostEqual(nodes[0].cartesianTransform[0,3],0.379,delta=0.01)
        self.assertLess(nodes[0].cartesianTransform[0,0],0)
        self.assertAlmostEqual(nodes[0].cartesianBounds[0],-4.351,delta=0.01)

   
    def test_ifc_to_nodes(self):
        #IFC1
        nodes=tl.ifc_to_nodes(path=self.dataLoaderParking.ifcPath,classes='IfcColumn')
        self.assertEqual(len(nodes), 125)


    def test_nodes_to_graph(self):
        nodes=tl.ifc_to_nodes(path=self.dataLoaderRoad.ifcPath)
        graph=tl.nodes_to_graph(nodes)
        subjects=[s for s in graph.subjects(RDF.type)]
        self.assertEqual(len(nodes), len(subjects))

    def test_graph_to_nodes(self):
        graph=Graph().parse(self.dataLoaderParking.resourceGraphPath)
        nodes=tl.graph_to_nodes(graph=graph)
        subjects=[s for s in graph.subjects(RDF.type)]
        self.assertEqual(len(nodes),len(subjects))

        #with getResource
        nodes=tl.graph_to_nodes(graph=graph,getResource=True)
        geometries=[n.resource for n in nodes]
        self.assertEqual(len(nodes),len(geometries))

    def test_graphpath_to_nodes(self):
        nodes=tl.graph_path_to_nodes(path=self.dataLoaderParking.resourceGraphPath)
        graph=Graph().parse(self.dataLoaderParking.resourceGraphPath)
        subjects=[s for s in graph.subjects(RDF.type)]
        self.assertEqual(len(nodes),len(subjects))

        #with getResource
        nodes=tl.graph_path_to_nodes(path=self.dataLoaderParking.resourceGraphPath,getResource=True)
        geometries=[n.resource for n in nodes]
        self.assertEqual(len(nodes),len(geometries))

    # def test_select_k_nearest_nodes(self):
    #     nodes=tl.graph_path_to_nodes(graphPath=self.bimGraphPath1,getResource=True)
    #     list,distances=tl.select_k_nearest_nodes(nodes[0],nodes)
    #     self.assertEqual(len(list),4)

    #     #error k<=0
    #     self.assertRaises(ValueError,tl.select_k_nearest_nodes,node=nodes[0],nodelist=nodes,k=-5 )

    # def test_select_nodes_with_centers_in_radius(self):
    #     nodes=tl.graph_path_to_nodes(graphPath=self.bimGraphPath2,getResource=True)
    #     list,distances=tl.select_nodes_with_centers_in_radius(nodes[0],nodes,r=10)
    #     self.assertEqual(len(list),1)
    #     for d in distances:
    #         self.assertLess(d,10)

    #     #error r<=0
    #     self.assertRaises(ValueError,tl.select_nodes_with_centers_in_radius,nodes[0],nodes,r=0 )
    #     self.assertRaises(ValueError,tl.select_nodes_with_centers_in_radius,nodes[0],nodes,r=-5 )

    def test_select_nodes_with_centers_in_bounding_box(self): 
        nodes=tl.graph_path_to_nodes(path=self.dataLoaderRoad.ifcGraphPath,getResource=True)
        list=tl.select_nodes_with_centers_in_bounding_box(nodes[0],nodes,u=5,v=5,w=5)
        self.assertEqual(len(list),31)

    def test_select_nodes_with_intersecting_bounding_box(self): 
        nodes=tl.graph_path_to_nodes(path=self.dataLoaderRoad.ifcGraphPath,getResource=True)
        list=tl.select_nodes_with_intersecting_bounding_box(nodes[0],nodes,u=5,v=5,w=5)
        self.assertGreater(len(list),0)

    def test_select_nodes_with_intersecting_resources(self):

        pcdNode = PointCloudNode(path=self.dataLoaderRoad.pcdPath,getResource=True)
        nodes=tl.ifc_to_nodes_multiprocessing(path=self.dataLoaderRoad.ifcPath,getResource=True)
        self.assertEqual(len(nodes),64)

        list= tl.select_nodes_with_intersecting_resources(pcdNode,nodes)
        print(list)
        self.assertEqual(len(list),64)

    def test_get_mesh_representation(self):
        nodes=tl.graph_path_to_nodes(path=self.dataLoaderParking.resourceGraphPath,getResource=True)        
        for n in nodes:
            geometry=tl.get_mesh_representation(n)
            if(geometry is not None):
                self.assertTrue('TriangleMesh' in str(type(geometry)))


if __name__ == '__main__':
    unittest.main()
