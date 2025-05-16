import os
from pathlib import Path
import shutil
import time
import unittest
import sys
import cv2
import numpy as np
from rdflib import RDF, RDFS, Graph, Literal, URIRef

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from geomapi.nodes import *
import geomapi.tools as tl
from geomapi.utils import geometryutils as gmu

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from data_loader_railway import DATALOADERRAILWAYINSTANCE
from data_loader_indoor_site import DATALOADERINDOORSITEINSTANCE

class TestTools(unittest.TestCase):


################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        
        cls.dataLoaderParking = DATALOADERPARKINGINSTANCE
        cls.dataLoaderRoad = DATALOADERROADINSTANCE
        cls.dataLoaderRailway = DATALOADERRAILWAYINSTANCE
        cls.dataLoaderIndoorSite = DATALOADERINDOORSITEINSTANCE

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

    def test_graph_to_nodes(self):
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderParking.resourceGraphPath)
        graph=Graph().parse(self.dataLoaderParking.resourceGraphPath)
        subjects=[s for s in graph.subjects(RDF.type)]
        self.assertEqual(len(nodes),len(subjects))

        #with getResource
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderParking.resourceGraphPath,loadResource=True)
        geometries=[n.resource for n in nodes]
        self.assertEqual(len(nodes),len(geometries))

        graph=Graph().parse(self.dataLoaderParking.resourceGraphPath)
        nodes=tl.graph_to_nodes(graph=graph)
        subjects=[s for s in graph.subjects(RDF.type)]
        self.assertEqual(len(nodes),len(subjects))

        #with getResource
        nodes=tl.graph_to_nodes(graph=graph,loadResource=True)
        geometries=[n.resource for n in nodes]
        self.assertEqual(len(nodes),len(geometries))

    def test_e57xml_to_pointcloud_nodes(self):
        # e57XML
        nodes=tl.e57xml_to_pointcloud_nodes(xmlPath=self.dataLoaderParking.e57XmlPath)
        self.assertEqual(len(nodes),2)
        #0
        self.assertAlmostEqual(nodes[0].cartesianTransform[0,3],0.379,delta=0.01)
        self.assertLess(nodes[0].cartesianTransform[0,0],0)
        #1
        self.assertEqual(nodes[1].cartesianTransform[0,3],0.0)
        self.assertEqual(nodes[1].cartesianTransform[0,0],-1)
        
        #with kwargs
        nodes=tl.e57xml_to_pointcloud_nodes(xmlPath=self.dataLoaderParking.e57XmlPath,myattrib=1)
        self.assertEqual(nodes[0].myattrib,1)

        # with getResource 
        # nodes=tl.e57xml_to_nodes(e57XmlPath=self.e57XmlPath,getResource=True)
        # geometries=[n.resource for n in nodes]
        # self.assertEqual(len(geometries),2)

    def test_xml_to_image_nodes(self):

        self.assertEqual(len(tl.xml_to_image_nodes(self.dataLoaderRoad.imageXmlPath,filterByFolder=False) ),78)
        self.assertEqual(len(tl.xml_to_image_nodes(self.dataLoaderRoad.imageXmlPath,filterByFolder=True) ),2)
        nodes=tl.xml_to_image_nodes(self.dataLoaderRoad.imageXmlPath,filterByFolder=True,getResource=True) 
        self.assertEqual(len(nodes),2)
        self.assertEqual(len([n.resource for n in nodes]),2)
        
        self.assertEqual(len(tl.xml_to_image_nodes(self.dataLoaderRoad.imageXmlPath,skip=10,filterByFolder=False) ),8)
                        
    def test_e57_to_pointcloud_nodes(self):

        #E57 1
        nodes=tl.e57_to_pointcloud_nodes(e57Path=self.dataLoaderParking.e57Path1)
        self.assertEqual(len(nodes),2)
        self.assertIsNone(nodes[0]._resource)
        #0
        self.assertAlmostEqual(nodes[0].cartesianTransform[0,3],4.52,delta=0.01)
        #1
        self.assertAlmostEqual(nodes[1].cartesianTransform[0,3],0.51,delta=0.01)
        
        #with kwargs
        nodes=tl.e57xml_to_pointcloud_nodes(xmlPath=self.dataLoaderParking.e57XmlPath,myattrib=1)
        self.assertEqual(nodes[0].myattrib,1)

        # # with getResource 
        # nodes=tl.e57xml_to_nodes(e57Path=self.e57XmlPath,getResource=True)
        # geometries=[n.resource for n in nodes]
        # self.assertEqual(len(geometries),2)

        #E572
        tl.e57_to_pointcloud_nodes(e57Path=self.dataLoaderParking.e57Path2)
        self.assertAlmostEqual(nodes[0].cartesianTransform[0,3],0.379,delta=0.01)
        self.assertLess(nodes[0].cartesianTransform[0,0],0)

    def test_dxf_to_lineset_nodes(self):
        tl.dxf_to_lineset_nodes(dxfPath=self.dataLoaderRailway.dxfPath)

    def test_dxf_to_ortho_nodes(self):
        nodes = tl.dxf_to_ortho_nodes(dxfPath=self.dataLoaderRailway.orthoDxfPath2,height=self.dataLoaderRailway.orthoHeight)
        #dxfPath and name + height-> offset in y and z
        node=nodes[0] #OrthoNode(dxfPath=self.dataLoaderRailway.orthoDxfPath2,name='railway-0-0',height=self.dataLoaderRailway.orthoHeight)
        self.assertEqual(node.dxfPath,self.dataLoaderRailway.orthoDxfPath2)
        #check cartesianTransform
        np.testing.assert_array_almost_equal(node.cartesianTransform,self.dataLoaderRailway.orthoCartesianTransform,3)
        #check orientedBoundingBox. height default 5
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),np.array([263379.5193, 151089.1667 ,self.dataLoaderRailway.orthoHeight-5]),3)
        #check convexHull
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),np.array([263379.5193, 151089.1667 ,self.dataLoaderRailway.orthoHeight-5]),3)


    def test_ifc_to_bim_nodes(self):
        #IFC1
        nodes=tl.ifc_to_bim_nodes(path=self.dataLoaderParking.ifcPath,classes='IfcColumn')
        self.assertEqual(len(nodes), 125)

    def test_navvis_csv_to_pano_nodes(self):
        nodes=tl.navvis_csv_to_pano_nodes(csvPath =self.dataLoaderIndoorSite.csvPath)
        self.assertEqual(len(nodes), 30)

    def select_nodes_k_nearest_neighbors(self):
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderRoad.ifcGraphPath,loadResource=True)
        center = gmu.get_translation(nodes[0].cartesianTransform)
        list,distances=tl.select_nodes_k_nearest_neighbors(nodes, center)
        self.assertEqual(len(list),4)

        #error k<=0
        self.assertRaises(ValueError,tl.select_nodes_k_nearest_neighbors,node=nodes[0],nodelist=nodes,k=-5 )

    def test_select_nodes_within_radius(self):
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderRoad.ifcGraphPath,loadResource=True)
        center = gmu.get_translation(nodes[0].cartesianTransform)
        list,distances=tl.select_nodes_within_radius(nodes,center,radius=10)
        self.assertEqual(len(list),1)
        for d in distances:
            self.assertLess(d,10)

        #error r<=0
        self.assertRaises(ValueError,tl.select_nodes_within_radius,nodes,center,radius=0 )
        self.assertRaises(ValueError,tl.select_nodes_within_radius,nodes,center,radius=-5 )

    def test_select_nodes_within_bounding_box(self): 
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderRoad.ifcGraphPath,loadResource=True)
        bbox = nodes[0].orientedBoundingBox
        list=tl.select_nodes_within_bounding_box(nodes,bbox,margin = [5,5,5])
        self.assertEqual(len(list),4)

    def test_select_nodes_within_convex_hull(self): 
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderRoad.ifcGraphPath,loadResource=True)
        hull = nodes[0].convexHull
        list=tl.select_nodes_within_convex_hull(nodes,hull)
        self.assertEqual(len(list),1)

    def test_select_nodes_intersecting_bounding_box(self): 
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderRoad.ifcGraphPath,loadResource=True)
        bbox = nodes[0].orientedBoundingBox
        list=tl.select_nodes_within_bounding_box(nodes,bbox,margin = [5,5,5])
        self.assertGreater(len(list),0)

    def test_select_nodes_intersecting_convex_hull(self): 
        nodes=tl.graph_to_nodes(graphPath=self.dataLoaderRoad.ifcGraphPath,loadResource=True)
        hull = nodes[0].convexHull
        list=tl.select_nodes_intersecting_convex_hull(nodes,hull)
        self.assertGreater(len(list),0)


    def test_nodes_to_graph(self):
        nodes=tl.ifc_to_bim_nodes(path=self.dataLoaderRoad.ifcPath)
        graph=tl.nodes_to_graph(nodes)
        subjects=[s for s in graph.subjects(RDF.type)]
        self.assertEqual(len(nodes), len(subjects))
        

if __name__ == '__main__':
    unittest.main()

        