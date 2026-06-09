import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import sys
import cv2
import rdflib
from geomapi.nodes import ImageNode,LineSetNode
from PIL import Image
from rdflib import RDF, RDFS, Graph, Literal, URIRef
import numpy as np
import open3d as o3d
import copy

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.utils.imageutils as iu
import geomapi.utils.geometryutils as gmu
from geomapi.nodes import ImageNode

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from data_loader_railway import DATALOADERRAILWAYINSTANCE 

from geomapi.utils import GEOMAPI_PREFIXES

class TestImageNode(unittest.TestCase):



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
        node= ImageNode()
        self.assertIsNotNone(node.subject)
        self.assertIsNotNone(node.name)
        self.assertIsNone(node.resource)
        self.assertEqual(node.imageWidth,640)
        self.assertEqual(node.imageHeight,480)
        self.assertEqual(node.focalLength35mm,35)
        self.assertEqual(node.principalPointU ,0)
        self.assertEqual(node.principalPointV ,0)
        self.assertIsNotNone(node.timestamp)
        
    def test_subject(self):
        #subject
        subject='myNode'
        node= ImageNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://'+subject)
        
    def test_name(self):
        node= ImageNode(name='name')
        self.assertEqual(node.name,'name')
        self.assertEqual(node.subject.toPython(),'http://name')    
        
    def test_imageWidth(self):
        node= ImageNode(imageWidth=100)
        self.assertEqual(node.imageWidth,100)
        self.assertEqual(node.principalPointU,0)
        #raise error when text
        self.assertRaises(ValueError,ImageNode,imageWidth='qsdf')
    
    def test_imageHeight(self):
        node= ImageNode(imageHeight=100)
        self.assertEqual(node.imageHeight,100)
        self.assertEqual(node.principalPointV,0)
        #raise error when text
        self.assertRaises(ValueError,ImageNode,imageHeight='qsdf')
    def test_depth(self):
        node= ImageNode(depth=100)
        self.assertEqual(node.depth,100)
        np.testing.assert_array_equal(node.cartesianTransform[:3,3],np.array([0,0,0]))
        #check oriented bounding box
        np.testing.assert_array_equal(node.orientedBoundingBox.get_center(),np.array([0,0,50]))
        #check convex hull -> note that convexhull is mean of all the vertices. because its a pyramid, the center is more towards the base
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),np.array([0,0,80]))
    
    def test_cartesianTransform(self):
        #create a convex hull in the shape of a pyramid
        points=np.array([[0,0,0], #top pyramid
                         [-10,-10,50], #base pyramid
                         [10,-10,50],
                         [-10,10,50],
                         [10,10,50]])
        base_hull=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).compute_convex_hull()[0]
        base_hull_center=np.array([0,0,40])
        base_hull_box_center=np.array([0,0,25])
        
        #base hull
        node= ImageNode(convexHull=base_hull)
        expectedCartesianTransform=np.eye(4)
        np.testing.assert_array_almost_equal(node.cartesianTransform,expectedCartesianTransform)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),base_hull_center)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),base_hull_box_center)
        
        #translation
        translated_hull=copy.deepcopy(base_hull)
        translated_hull.translate([1,0,0])
        translated_hull_center=np.array([1,0,40])
        translated_hull_box_center=np.array([1,0,25])
        node= ImageNode(convexHull=translated_hull)
        expectedCartesianTransform=np.array([[1,0,0,1], 
                                            [0,1,0,0],
                                            [0,0,1,0],
                                            [0,0,0,1]])
        np.testing.assert_array_almost_equal(node.cartesianTransform,expectedCartesianTransform,3)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),translated_hull_center,3)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),translated_hull_box_center,3)
        
        #rotation
        rotated_hull_90_x=copy.deepcopy(base_hull)
        expectedCartesianTransform=np.array([[0,0,1,0], 
                                            [0,1,0,0],
                                            [-1,0,0,0],
                                            [0,0,0,1]])
        rotated_hull_90_x.rotate(expectedCartesianTransform[:3,:3]  ,center=[0,0,0])
        rotated_hull_center=np.array([40,0,0])
        rotated_hull_box_center=np.array([25,0,0])
        node= ImageNode(convexHull=rotated_hull_90_x)
        
        np.testing.assert_array_almost_equal(node.cartesianTransform,expectedCartesianTransform,3)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),rotated_hull_center,3)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),rotated_hull_box_center,3)
        
        #rotation an translation
        rotated_translated_hull=copy.deepcopy(base_hull)
        expectedCartesianTransform=np.array([[0,0,1,1], 
                                            [0,1,0,0],
                                            [-1,0,0,0],
                                            [0,0,0,1]])
        rotated_translated_hull.rotate(expectedCartesianTransform[:3,:3] ,center=[0,0,0])
        rotated_translated_hull.translate(expectedCartesianTransform[:3,3])
        rotated_translated_center=np.array([41,0,0])
        rotated_translated_box_center=np.array([26,0,0])
        node= ImageNode(convexHull=rotated_translated_hull)
        np.testing.assert_array_almost_equal(node.cartesianTransform,expectedCartesianTransform,3)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),rotated_translated_center,3)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),rotated_translated_box_center,3)
        
    def test_orientedBoundingBox_and_convex_hull(self):
        #translation
        cartesianTransform=np.array([[1,0,0,1],
                                     [0,1,0,0],
                                     [0,0,1,0],
                                     [0,0,0,1]])
        node= ImageNode(cartesianTransform=cartesianTransform)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),np.array([1,0,0.5]),3)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),np.array([1,0,0.8]),3)
        
        #90° rotation around z-axis
        rotation_matrix_90_z=   np.array( [[ 0, -1 , 0. ,0       ],
                                        [ 1,  0,  0.   ,0     ],
                                        [ 0.   ,       0.    ,      1.    ,0    ],
                                        [0,0,0,1]])  
        node= ImageNode(cartesianTransform=rotation_matrix_90_z)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),np.array([0,0,0.5]),3)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),np.array([0,0,0.8]),3)
        
        #90° rotation around x-axis
        rotation_matrix_90_x=   np.array( [[ 1, 0 , 0. ,0       ],
                                        [ 0,  0,  -1   ,0     ],
                                        [ 0.   ,       1    ,      0    ,0    ],
                                        [0,0,0,1]])  
        node= ImageNode(cartesianTransform=rotation_matrix_90_x)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),np.array([0,-0.5,0]),3)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),np.array([0,-0.8,0]),3)
        
        #90° rotation around x-axis + translation
        rotation_matrix_90_x=   np.array( [[ 1, 0 , 0. ,1       ],
                                        [ 0,  0,  -1   ,0     ],
                                        [ 0.   ,       1    ,      0    ,0    ],
                                        [0,0,0,1]])  
        node= ImageNode(cartesianTransform=rotation_matrix_90_x)
        np.testing.assert_array_almost_equal(node.orientedBoundingBox.get_center(),np.array([1,-0.5,0]),3)
        np.testing.assert_array_almost_equal(node.convexHull.get_center(),np.array([1,-0.8,0]),3)
    
    def test_focalLength35mm(self):
        node= ImageNode(focalLength35mm=100)
        self.assertEqual(node.focalLength35mm,100)
        
        #raise error when text
        self.assertRaises(ValueError,ImageNode,focalLength35mm='qsdf')
        
    def test_principalPointU(self):
        node= ImageNode(principalPointU=100)
        self.assertEqual(node.principalPointU,100)
        self.assertEqual(node.intrinsicMatrix[0,2],node.imageWidth/2+100)
        
        #raise error when text
        self.assertRaises(ValueError,ImageNode,principalPointU='qsdf')
        
    def test_principalPointV(self):
        node= ImageNode(principalPointV=100)
        self.assertEqual(node.principalPointV,100)
        self.assertEqual(node.intrinsicMatrix[1,2],node.imageHeight/2+100)
        
        #raise error when text
        self.assertRaises(ValueError,ImageNode,principalPointV='qsdf')
                    
    def test_path(self):
        #path1 without loadResource
        node= ImageNode(path=self.dataLoaderParking.imagePath2)
        self.assertEqual(node.name,self.dataLoaderParking.imagePath2.stem)
        self.assertIsNone(node._resource)

        #path2 with loadResource
        node= ImageNode(path=self.dataLoaderRoad.imagePath1,loadResource=True)        
        self.assertEqual(node.name,self.dataLoaderRoad.imagePath1.stem)
        self.assertEqual(node.imageHeight,self.dataLoaderRoad.image1.shape[0])
        self.assertIsNotNone(node._resource)
        
        #raise error when wrong path
        self.assertRaises(ValueError,ImageNode,path='dfsgsdfgsd')
        
    def test_xmpPath(self):
        #path without extra info 
        node= ImageNode(xmpPath=self.dataLoaderParking.imageXmpPath1)
        self.assertEqual(node.xmpPath,self.dataLoaderParking.imageXmpPath1)        
        self.assertEqual(node.name,self.dataLoaderParking.imageXmpPath1.stem)
        self.assertEqual(node.focalLength35mm,self.dataLoaderParking.focalLength35mm)
        self.assertEqual(node.principalPointU,self.dataLoaderParking.principalPointU)
        self.assertEqual(node.principalPointV,self.dataLoaderParking.principalPointV)
        np.testing.assert_array_almost_equal(node.cartesianTransform,self.dataLoaderParking.imageCartesianTransform1,3)
        self.assertIsNone(node.resource)
        
        
        #path with loadResource without extra info 
        node= ImageNode(xmpPath=self.dataLoaderParking.imageXmpPath2,path=self.dataLoaderParking.imagePath2, loadResource=True)        
        self.assertEqual(node.xmpPath,self.dataLoaderParking.imageXmpPath2)        
        self.assertEqual(node.name,self.dataLoaderParking.imageXmpPath2.stem)
        self.assertEqual(node.imageWidth,np.asarray(self.dataLoaderParking.image2).shape[1])
        self.assertEqual(node.imageHeight,np.asarray(self.dataLoaderParking.image2).shape[0])     
        np.testing.assert_array_almost_equal(node.cartesianTransform,self.dataLoaderParking.imageCartesianTransform2,3)
        self.assertIsNotNone(node.resource)
        
        #raise error when wrong path
        self.assertRaises(ValueError,ImageNode,principalPointU='xmpPath')
        
    def test_resource(self):
        #cv2 image
        node= ImageNode(resource=self.dataLoaderRoad.image1)
        self.assertEqual(node.imageHeight,self.dataLoaderRoad.image1.shape[0])
        self.assertEqual(node.resource.shape[0],self.dataLoaderRoad.image1.shape[0])
        
        #PIL image
        node= ImageNode(resource=self.dataLoaderRoad.image2)
        self.assertEqual(node.imageHeight,np.asarray(self.dataLoaderRoad.image2).shape[0])
        self.assertEqual(node.resource.shape[0],np.asarray(self.dataLoaderRoad.image2).shape[0])
        
        #open3d image
        node= ImageNode(resource=self.dataLoaderParking.image2)
        self.assertEqual(node.imageHeight,np.asarray(self.dataLoaderParking.image2).shape[0])
        self.assertEqual(node.resource.shape[0],np.asarray(self.dataLoaderParking.image2).shape[0])
        
        
    
    def test_graphPath(self):
        node=ImageNode(graphPath=self.dataLoaderRailway.imgGraphPath)
        self.assertEqual(node.graphPath,self.dataLoaderRailway.imgGraphPath)
        self.assertTrue(node.subject in self.dataLoaderRailway.imgGraph.subjects())
        self.assertIsNotNone(node.imageHeight)
        self.assertIsNotNone(node.imageWidth)        
        
        
    def test_graphPath_with_subject(self):
        subject=next(self.dataLoaderRailway.imgGraph.subjects(RDF.type))
        node=ImageNode(graphPath=self.dataLoaderRailway.imgGraphPath,subject=subject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderRailway.imgGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderRailway.imgGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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
            if 'focalLength35mm' in p.toPython():
                self.assertEqual(float(o),node.focalLength35mm)
            if 'principalPointU' in p.toPython():
                self.assertEqual(float(o),node.principalPointU)
            if 'principalPointV' in p.toPython():
                self.assertEqual(float(o),node.principalPointV)
            if 'imageWidth' in p.toPython():
                self.assertEqual(float(o),node.imageWidth)
            if 'imageLength' in p.toPython():
                self.assertEqual(float(o),node.imageHeight)
            if 'intrinsicMatrix' in p.toPython():
                matrix=ut.literal_to_matrix(o)
                self.assertTrue(np.allclose(matrix,node.intrinsicMatrix,atol=0.001))

        #raise error when subject is not in graph
        self.assertRaises(ValueError,LineSetNode,graphPath=self.dataLoaderRailway.imgGraphPath,subject=URIRef('mySubject'))
    
    def test_graph(self):
        subject=next(self.dataLoaderRailway.imgGraph.subjects(RDF.type))
        node=ImageNode(graphPath=self.dataLoaderRailway.imgGraphPath,subject=subject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderRailway.imgGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderRailway.imgGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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
            if 'focalLength35mm' in p.toPython():
                self.assertEqual(float(o),node.focalLength35mm)
            if 'principalPointU' in p.toPython():
                self.assertEqual(float(o),node.principalPointU)
            if 'principalPointV' in p.toPython():
                self.assertEqual(float(o),node.principalPointV)
            if 'imageWidth' in p.toPython():
                self.assertEqual(float(o),node.imageWidth)
            if 'imageLength' in p.toPython():
                self.assertEqual(float(o),node.imageHeight)
            if 'intrinsicMatrix' in p.toPython():
                matrix=ut.literal_to_matrix(o)
                self.assertTrue(np.allclose(matrix,node.intrinsicMatrix,atol=0.001))

 
    def test_node_creation_with_load_resource(self):
        #mesh
        node= ImageNode(resource=self.dataLoaderParking.image1)
        self.assertIsNotNone(node._resource)

        #path without loadResource
        node= ImageNode(path=self.dataLoaderParking.imagePath2)
        self.assertIsNone(node._resource)

        #path with loadResource
        node= ImageNode(path=self.dataLoaderParking.imagePath1,loadResource=True)
        self.assertIsNotNone(node._resource)

        #graph with get resource
        node= ImageNode(subject=self.dataLoaderRoad.imageSubject1,
                        graph=self.dataLoaderRoad.imgGraph,
                        loadResource=True)
        self.assertIsNone(node._resource)
        
        #graphPath with get resource
        node= ImageNode(subject=self.dataLoaderParking.imageSubject2,
                        graphPath=self.dataLoaderParking.imgGraphPath,
                        loadResource=True)
        self.assertIsNotNone(node._resource)

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
                        loadResource=True)
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

    def test_load_resource(self):
        #mesh
        node=ImageNode(resource=self.dataLoaderParking.image2)  
        self.assertIsNone(node.load_resource())

        #graphPath with loadResource
        node=ImageNode(graphPath=str(self.dataLoaderParking.imgGraphPath),
                       subject=self.dataLoaderParking.imageSubject1,
                       loadResource=True)
        self.assertIsNotNone(node.load_resource())

    def test_set_path(self):
        #valid path
        node=ImageNode()
        node.path= str(self.dataLoaderParking.imagePath1)
        self.assertEqual(node.path,self.dataLoaderParking.imagePath1)

        #preexisting
        node=ImageNode(path=self.dataLoaderParking.imagePath2)
        self.assertEqual(node.path,self.dataLoaderParking.imagePath2)

        #graphPath & name
        node=ImageNode(subject=self.dataLoaderParking.imageSubject1,
                       graphPath=self.dataLoaderParking.imgGraphPath)
        self.assertEqual(node.path,self.dataLoaderParking.imagePath1)

    def test_set_xmp_path(self):
        #valid path
        node=ImageNode(xmpPath=str(self.dataLoaderParking.imageXmpPath1))
        self.assertEqual(node.xmpPath,self.dataLoaderParking.imageXmpPath1)

        #invalid
        self.assertRaises(ValueError,ImageNode,xmpPath='qsffqsdf.dwg')

    def test_set_xml_path(self):
        #valid path
        node=ImageNode()
        node.xmlPath=self.dataLoaderRoad.imageXmlPath
        self.assertEqual(node.xmlPath,self.dataLoaderRoad.imageXmlPath)

        #invalid
        self.assertRaises(ValueError,ImageNode,xmlPath='qsffqsdf.dwg')
        
    def test_pixel_to_world_coordinates(self):
        node=ImageNode(graph=self.dataLoaderRailway.imgGraph)
        worldCoordinate = node.pixel_to_world_coordinates(self.dataLoaderRailway.imgCoordinate,self.dataLoaderRailway.distance)
        pixel, depth=node.world_to_pixel_coordinates(worldCoordinate)
        np.testing.assert_array_almost_equal(self.dataLoaderRailway.imgCoordinate.flatten(),pixel.flatten(),1)

    def test_transform_translation(self):
        #empty node
        node = ImageNode()        
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        translation = [1,0,0]
        node.transform(translation=translation)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.cartesianTransform,gmu.get_cartesian_transform(translation=translation),atol=0.001))
        
        
        #real node
        node=ImageNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject=self.dataLoaderRailway.imageSubject1)
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        cartesianTransform = node.cartesianTransform
        translation = [1,0,0]
        node.transform(translation=translation)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.cartesianTransform[:3,3].flatten(),cartesianTransform[:3,3].flatten()+translation,atol=0.001))
        
    def test_transform_rotation(self):
        
        #90° rotation around z-axis
        rotation_euler = [0,0,90]
        rotation_matrix=   np.array( [[ 0, -1 , 0.        ],
                                        [ 1,  0,  0.        ],
                                        [ 0.   ,       0.    ,      1.        ]])  
        
        #(0,0) node with rotation matrix     
        node = ImageNode()          
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        node.transform(rotation=rotation_matrix)
        self.assertAlmostEqual(node.cartesianTransform[0,3],0,delta=0.01 )
        self.assertAlmostEqual(node.cartesianTransform[0,0],0,delta=0.01 )
        self.assertAlmostEqual(node.cartesianTransform[0,1],-1,delta=0.01 )
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
   
    def test_create_rays(self):
        node=ImageNode(graph=self.dataLoaderRailway.imgGraph)
        
        #list of points
        node=ImageNode()
        rays=node.create_rays([[100,100], [200,200], [300,300], [400,400]])
        self.assertTrue(np.allclose(rays[0][:3],node.cartesianTransform[:3,3],atol=0.001))
        self.assertEqual(rays.shape,(4,6))
        
        #check centerpont shooting in Z direction
        node=ImageNode()
        rays=node.create_rays([node.imageWidth/2.0,node.imageHeight/2.0],50)
        startpoint,endpoint=gmu.rays_to_points(rays)
        self.assertEqual(np.linalg.norm(startpoint-node.get_center()),0)
        self.assertEqual(np.linalg.norm(endpoint-node.get_center()),1) 
        self.assertEqual(np.linalg.norm(endpoint-np.array([0,0,1])),0) 
        
        #different input formats single inputs
        inputs = [
            [1681.93211083, 10003.48867638],
            np.array([1681.93211083, 10003.48867638]),
            np.array([[1681.93211083, 10003.48867638]])
        ]
        results = [node.create_rays(input_data) for input_data in inputs]
        for result in results:
            np.testing.assert_array_almost_equal(result[0][:3],node.cartesianTransform[:3,3],1)
            self.assertEqual(result.shape,(1,6))


    def test_get_image_features(self):
        imageNode=ImageNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',resource=self.dataLoaderRailway.image1)
        keypoints,descriptors=imageNode.get_image_features()
        #check if the number of keypoints is the same as the number of descriptors
        self.assertEqual(len(keypoints),len(descriptors))
        
    def test_draw_keypoints_on_image(self):
        imageNode=ImageNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',resource=self.dataLoaderRailway.image1)
        image=imageNode.draw_keypoints_on_image(keypoint_size=300)

if __name__ == '__main__':
    unittest.main()
