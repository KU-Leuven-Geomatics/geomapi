#import os
#from pathlib import Path
#import shutil
#import time
#import unittest
#from multiprocessing.sharedctypes import Value
#import sys
#import cv2
#import rdflib
#from geomapi.nodes import PanoNode,LineSetNode
#from PIL import Image
#from rdflib import RDF, RDFS, Graph, Literal, URIRef
#import numpy as np
#import open3d as o3d
#import copy
#import pandas as pd
#
##GEOMAPI
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
#sys.path.append(parent_dir)
#import geomapi.utils as ut
#import geomapi.utils.imageutils as iu
#import geomapi.utils.geometryutils as gmu
#from geomapi.nodes import PanoNode
#
##DATA
#sys.path.append(current_dir)
#
#from geomapi.utils import GEOMAPI_PREFIXES
#
#class TestPanoNode(unittest.TestCase):
#
#
#
################################### SETUP/TEARDOWN CLASS ######################
#    @classmethod
#    def setUpClass(cls):
#        #execute once before all tests
#        print('-----------------Setup Class----------------------')
#        st = time.time()
#        
#        cls.path= Path.cwd() / "tests" / "testfiles"
#        
#        #PANO
#        cls.panoPath = cls.path / 'pano'/ "00000-pano.jpg"
#        cls.pano = Image.open(cls.panoPath)
#        cls.depthPath = cls.path / 'pano'/ "00000-pano_depth.png"
#        cls.depthMap = Image.open(cls.depthPath)
#        cls.csvPath = cls.path / 'pano'/ "pano-poses.csv"
#        cls.csv = pd.read_csv(cls.csvPath)
#        cls.jsonPath = cls.path / 'pano'/ "00000-info.json"
#        cls.json = pd.read_json(cls.jsonPath)
#        print(f'    loaded {cls.pano}')
#
#        #TIME TRACKING 
#        et = time.time()
#        print("startup time: "+str(et - st))
#        print('{:50s} {:5s} '.format('tests','time'))
#        print('------------------------------------------------------')
#
#    @classmethod
#    def tearDownClass(cls):
#        #execute once after all tests
#        # if os.path.exists(cls.dataLoaderParking.resourcePath):
#        #     shutil.rmtree(cls.dataLoaderParking.resourcePath)  
#        print('-----------------TearDown Class----------------------')   
# 
# 
#
################################### SETUP/TEARDOWN ######################
#
#    def setUp(self):
#        #execute before every test
#        self.startTime = time.time()   
#
#    def tearDown(self):
#        #execute after every test
#        t = time.time() - self.startTime
#        print('{:50s} {:5s} '.format(self._testMethodName,str(t)))
################################### TEST FUNCTIONS ######################
#    def test_empty_node(self):
#        node= PanoNode()
#        self.assertIsNotNone(node.subject)
#        self.assertIsNotNone(node.name)
#        # self.assertEqual(node.imageWidth,640)
#        # self.assertEqual(node.imageHeight,480)
#        self.assertIsNotNone(node.timestamp)
#        np.testing.assert_array_almost_equal(node.cartesianTransform,np.eye(4),3)
#        np.testing.assert_array_almost_equal(node.orientedBoundingBox.extent,np.array([2,2,2]),3)
#        np.testing.assert_array_almost_equal(node.convexHull.get_center(),np.array([0,0,0]),3)
#        
#        
#    def test_imageWidth(self):
#        node= PanoNode(imageWidth=100)
#        self.assertEqual(node.imageWidth,100)
#        #raise error when text
#        self.assertRaises(ValueError,PanoNode,imageWidth='qsdf')
#    
#    def test_imageHeight(self):
#        node= PanoNode(imageHeight=100)
#        self.assertEqual(node.imageHeight,100)
#        #raise error when text
#        self.assertRaises(ValueError,PanoNode,imageHeight='qsdf')
#        
#  
#    def test_cartesianTransform(self):
#        ball=o3d.geometry.TriangleMesh.create_sphere(radius=0.5) #this gets Obox with x,y,z and no rotation
#        ball.translate([0,0,-1])
#        box=o3d.geometry.TriangleMesh.create_box(1,1,1)
#        box.translate([0,0,1])
#        
#        #base hull
#        node= PanoNode(convexHull=ball)
#        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],np.array([ 0,0,-1]),atol=0.001))
#        
#        #oriented box + convex hull -> convex hull  has priority
#        node= PanoNode(orientedBoundingBox=box,
#                       convexHull=ball)
#        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],np.array([ 0,0,-1]),atol=0.001))
#
#        #depth + convex hull + oriented box -> depth  has priority
#        node= PanoNode(depthPath=self.depthPath,convexHull=ball,orientedBoundingBox=box,getResource=True)
#        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],np.array([ 6.38405617, -1.74535209,  0.29007973]),atol=0.001))
#
#        #json
#        node= PanoNode(jsonPath=self.jsonPath)
#        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],np.array([0.0896231, -0.189631, -1.05647]),atol=0.001))
#
#        
#        #json + depth -> json has priority
#        node= PanoNode(jsonPath=self.jsonPath,depthPath=self.depthPath,getResource=True)
#        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],np.array([0.0896231, -0.189631, -1.05647]),atol=0.001))
#        
#        
#    def test_transformations(self):
#        #translation
#        cartesianTransform=np.array([[1,0,0,1],
#                                     [0,1,0,0],
#                                     [0,0,1,0],
#                                     [0,0,0,1]])
#        node= PanoNode(cartesianTransform=cartesianTransform)
#        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([1,0,25]),atol=0.001))
#        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([1,0,39.534867912]),atol=0.001))
#        
#        #90° rotation around z-axis
#        rotation_matrix_90_z=   np.array( [[ 0, -1 , 0. ,0       ],
#                                        [ 1,  0,  0.   ,0     ],
#                                        [ 0.   ,       0.    ,      1.    ,0    ],
#                                        [0,0,0,1]])  
#        node= PanoNode(cartesianTransform=rotation_matrix_90_z)
#        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,0,25]),atol=0.001))
#        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([0,0,39.534867912]),atol=0.001))
#        
#        #90° rotation around x-axis
#        rotation_matrix_90_x=   np.array( [[ 1, 0 , 0. ,0       ],
#                                        [ 0,  0,  -1   ,0     ],
#                                        [ 0.   ,       1    ,      0    ,0    ],
#                                        [0,0,0,1]])  
#        node= PanoNode(cartesianTransform=rotation_matrix_90_x)
#        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,-25,0]),atol=0.001))
#        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([0,-39.534867912,0]),atol=0.001))
#        
#        #90° rotation around x-axis + translation
#        rotation_matrix_90_x=   np.array( [[ 1, 0 , 0. ,1       ],
#                                        [ 0,  0,  -1   ,0     ],
#                                        [ 0.   ,       1    ,      0    ,0    ],
#                                        [0,0,0,1]])  
#        node= PanoNode(cartesianTransform=rotation_matrix_90_x)
#        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([1,-25,0]),atol=0.001))
#        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([1,-39.534867912,0]),atol=0.001))
#    
#    def test_path(self):
#        #path1 without getResource
#        node= PanoNode(path=self.panoPath)
#        self.assertEqual(node.name,self.panoPath.stem)
#        self.assertIsNone(node._resource)
#
#        #path2 with getResource
#        node= PanoNode(path=self.panoPath,getResource=True)        
#        self.assertEqual(node.name,self.panoPath)
#        self.assertEqual(node.imageHeight,self.pano.shape[0])
#        self.assertIsNotNone(node._resource)
#        
#        #raise error when wrong path
#        self.assertRaises(ValueError,PanoNode,path='dfsgsdfgsd')
#        
#    def test_json_path(self):
#        #path without extra info 
#        node= PanoNode(jsonPath=self.jsonPath)
#        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],np.array([0.0896231, -0.189631, -1.05647])))
#        self.assertEqual(node.timestamp,'2022-06-03T14:27:23')      
#        
#        #raise error when wrong path
#        self.assertRaises(ValueError,PanoNode,jsonPath='sdfqsfd')
#        
#    
#    def test_resource(self):
#        #PIL image
#        node= PanoNode(resource=self.pano)
#        self.assertEqual(node.imageHeight,self.pano.shape[0])
#        self.assertEqual(node.resource.shape[0],self.pano.shape[0])
#        
#    def test_graphPath(self):
#        node=PanoNode(graphPath=self.panoGraphPath)
#        self.assertEqual(node.graphPath,self.panoGraphPath)
#        self.assertTrue(node.subject in self.panoGraph.subjects())
#        self.assertIsNotNone(node.imageHeight)
#        self.assertIsNotNone(node.imageWidth)        
#        
#        
#    def test_graphPath_with_subject(self):
#        subject=next(self.panoGraph.subjects(RDF.type))
#        node=PanoNode(graphPath=self.panoGraphPath,subject=subject)
#        
#        #check if the graph is correctly parsed
#        for s, p, o in self.panoGraph.triples((subject, None, None)):
#            if 'path' in p.toPython():
#                self.assertEqual((self.panoGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
#            if 'cartesianTransform' in p.toPython():
#                matrix=ut.literal_to_matrix(o)
#                #check if matrix elements are the same as the node cartesianTransform
#                self.assertTrue(np.allclose(matrix,node.cartesianTransform,atol=0.001))
#            if 'orientedBoundingBox' in p.toPython():
#                graph_param=ut.literal_to_matrix(o)
#                node_param=gmu.get_oriented_bounding_box_parameters(node.orientedBoundingBox)
#                self.assertTrue(np.allclose(graph_param,node_param,atol=0.001))
#            if 'convexHull' in p.toPython():
#                graph_param=ut.literal_to_matrix(o)
#                graph_volume=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(graph_param)).compute_convex_hull()[0].get_volume()
#                node_volume=node.convexHull.get_volume()
#                self.assertAlmostEqual(graph_volume,node_volume,delta=0.01)
#            if 'depthPath' in p.toPython():
#                self.assertEqual((self.panoGraphPath.parent/Path(o.toPython())).resolve(),node.depthPath) 
#            if 'imageWidth' in p.toPython():
#                self.assertEqual(float(o),node.imageWidth)
#            if 'imageLength' in p.toPython():
#                self.assertEqual(float(o),node.imageHeight)
#            if 'intrinsicMatrix' in p.toPython():
#                matrix=ut.literal_to_matrix(o)
#                self.assertTrue(np.allclose(matrix,node.intrinsicMatrix,atol=0.001))
#
#        #raise error when subject is not in graph
#        self.assertRaises(ValueError,LineSetNode,graphPath=self.dataLoaderRailway.imgGraphPath,subject=URIRef('mySubject'))
#    
#    def test_graph(self):
#        subject=next(self.dataLoaderRailway.imgGraph.subjects(RDF.type))
#        node=PanoNode(graphPath=self.dataLoaderRailway.imgGraphPath,subject=subject)
#        
#        #check if the graph is correctly parsed
#        for s, p, o in self.dataLoaderRailway.imgGraph.triples((subject, None, None)):
#            if 'path' in p.toPython():
#                self.assertEqual((self.dataLoaderRailway.imgGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
#            if 'cartesianTransform' in p.toPython():
#                matrix=ut.literal_to_matrix(o)
#                #check if matrix elements are the same as the node cartesianTransform
#                self.assertTrue(np.allclose(matrix,node.cartesianTransform,atol=0.001))
#            if 'orientedBoundingBox' in p.toPython():
#                graph_param=ut.literal_to_matrix(o)
#                node_param=gmu.get_oriented_bounding_box_parameters(node.orientedBoundingBox)
#                self.assertTrue(np.allclose(graph_param,node_param,atol=0.001))
#            if 'convexHull' in p.toPython():
#                graph_param=ut.literal_to_matrix(o)
#                graph_volume=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(graph_param)).compute_convex_hull()[0].get_volume()
#                node_volume=node.convexHull.get_volume()
#                self.assertAlmostEqual(graph_volume,node_volume,delta=0.01)
#            if 'focalLength35mm' in p.toPython():
#                self.assertEqual(float(o),node.focalLength35mm)
#            if 'principalPointU' in p.toPython():
#                self.assertEqual(float(o),node.principalPointU)
#            if 'principalPointV' in p.toPython():
#                self.assertEqual(float(o),node.principalPointV)
#            if 'imageWidth' in p.toPython():
#                self.assertEqual(float(o),node.imageWidth)
#            if 'imageLength' in p.toPython():
#                self.assertEqual(float(o),node.imageHeight)
#            if 'intrinsicMatrix' in p.toPython():
#                matrix=ut.literal_to_matrix(o)
#                self.assertTrue(np.allclose(matrix,node.intrinsicMatrix,atol=0.001))
#
# 
#    def test_node_creation_with_get_resource(self):
#        #mesh
#        node= PanoNode(resource=self.dataLoaderParking.image1)
#        self.assertIsNotNone(node._resource)
#
#        #path without getResource
#        node= PanoNode(path=self.dataLoaderParking.imagePath2)
#        self.assertIsNone(node._resource)
#
#        #path with getResource
#        node= PanoNode(path=self.dataLoaderParking.imagePath1,getResource=True)
#        self.assertIsNotNone(node._resource)
#
#        #graph with get resource
#        node= PanoNode(subject=self.dataLoaderRoad.imageSubject1,
#                        graph=self.dataLoaderRoad.imgGraph,
#                        getResource=True)
#        self.assertIsNone(node._resource)
#        
#        #graphPath with get resource
#        node= PanoNode(subject=self.dataLoaderParking.imageSubject2,
#                        graphPath=self.dataLoaderParking.imgGraphPath,
#                        getResource=True)
#        self.assertIsNotNone(node._resource)
#
#    def test_clear_resource(self):
#        #mesh
#        node= PanoNode(resource=self.dataLoaderRoad.image1)
#        self.assertIsNotNone(node._resource)
#        del node.resource
#        self.assertIsNone(node._resource)
#
#    def test_save_resource(self):
#        #no mesh -> False
#        node= PanoNode()
#        self.assertFalse(node.save_resource())
#
#        #directory
#        node= PanoNode(resource=self.dataLoaderRoad.image2)
#        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))
#
#        #graphPath        
#        node= PanoNode(resource=self.dataLoaderParking.image1,
#                        graphPath=self.dataLoaderParking.imgGraphPath)
#        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))
#
#        #no path or graphPath
#        node= PanoNode(resource=self.dataLoaderRoad.image2)        
#        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))
#
#        #path -> new name
#        node= PanoNode(subject=URIRef('myImg'),
#                        path=self.dataLoaderRoad.imagePath2,
#                        getResource=True)
#        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))
#        
#        #graphPath with directory
#        node=PanoNode(subject=self.dataLoaderParking.imageSubject1,
#                       graphPath=self.dataLoaderParking.imgGraphPath,
#                       resource=self.dataLoaderParking.image1)
#        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))
#
#        #graph with new subject
#        node=PanoNode(subject=self.dataLoaderRoad.imageSubject2,
#                       graph=self.dataLoaderRoad.imgGraph,
#                       resource=self.dataLoaderRoad.image2)
#        node.subject='myImg'
#        self.assertTrue(node.save_resource(self.dataLoaderRoad.resourcePath))
#
#    def test_get_resource(self):
#        #mesh
#        node=PanoNode(resource=self.dataLoaderParking.image2)  
#        self.assertIsNotNone(node.get_resource())
#
#        #no mesh
#        del node.resource
#        self.assertIsNone(node.get_resource())
#
#        #graphPath with getResource
#        node=PanoNode(graphPath=str(self.dataLoaderParking.imgGraphPath),
#                       subject=self.dataLoaderParking.imageSubject1,
#                       getResource=True)
#        self.assertIsNotNone(node.get_resource())
#
#    def test_set_path(self):
#        #valid path
#        node=PanoNode()
#        node.path= str(self.dataLoaderParking.imagePath1)
#        self.assertEqual(node.path,self.dataLoaderParking.imagePath1)
#
#        #preexisting
#        node=PanoNode(path=self.dataLoaderParking.imagePath2)
#        self.assertEqual(node.path,self.dataLoaderParking.imagePath2)
#
#        #graphPath & name
#        node=PanoNode(subject=self.dataLoaderParking.imageSubject1,
#                       graphPath=self.dataLoaderParking.imgGraphPath)
#        node.get_path()
#        self.assertEqual(node.path,self.dataLoaderParking.imagePath1)
#
#    def test_set_xmp_path(self):
#        #valid path
#        node=PanoNode(xmpPath=str(self.dataLoaderParking.imageXmpPath1))
#        self.assertEqual(node.xmpPath,self.dataLoaderParking.imageXmpPath1)
#
#        #invalid
#        self.assertRaises(ValueError,PanoNode,xmpPath='qsffqsdf.dwg')
#
#    def test_set_xml_path(self):
#        #valid path
#        node=PanoNode()
#        node.xmlPath=self.dataLoaderRoad.imageXmlPath
#        self.assertEqual(node.xmlPath,self.dataLoaderRoad.imageXmlPath)
#
#        #invalid
#        self.assertRaises(ValueError,PanoNode,xmlPath='qsffqsdf.dwg')
#    
#    def test_get_pcd_from_depth_map(self):
#        
#        node=PanoNode(depth=self.depth,resource=self.pano)
#        pcd=node.get_pcd_from_depth_map()
#        points=np.asarray(pcd.points)
#        middle_point=points[int(points.shape[0]/2)]
#        #check if the middle point of the point cloud is the same as the middle point of the depth map
#        self.assertTrue(np.allclose(middle_point,np.array([8.161,0.575,10.292]),atol=0.001))
#        #check if number of points is the same as the number of pixels in the depth map
#        self.assertEqual(points.shape[0],self.depth.size)
#        #check if colors are the same as the colors of the image
#        colors=np.asarray(pcd.colors)
#        self.assertTrue(np.allclose(colors[0],self.pano[0]/255),atol=0.001)
#        
#    def test_world_to_pixel_coordinates(self):
#        #array(1,4)
#        node=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject=self.dataLoaderRailway.imageSubject1)
#        pixel=node.world_to_pixel_coordinates(self.dataLoaderRailway.worldCoordinate)
#        self.assertTrue(np.allclose(pixel,self.dataLoaderRailway.imgCoordinate,atol=10))
#        
#        #array(1,3)
#        pixel=node.world_to_pixel_coordinates(self.dataLoaderRailway.worldCoordinate[0][:3])
#        self.assertTrue(np.allclose(pixel,self.dataLoaderRailway.imgCoordinate,atol=10))
#        
#        # #different input formats single point
#        # inputs = [
#        #     [0,0,0],
#        #     [0,0,0,1],
#        #     np.array([0,0,0]),
#        #     np.array([[0,0,0,1]])
#        # ]
#        # results = [node.world_to_pixel_coordinates(input_data) for input_data in inputs]
#        # for result in results:
#        #     self.assertEqual(result.shape,(2,))
#            
#        #different input formats multiple points
#        inputs = [
#            [[0,0,0],[0,0,0]],
#            [[0,0,0,1],[0,0,0,1]],
#            np.array([[0,0,0],[0,0,0]]),
#            np.array([[0,0,0,1],[0,0,0,1]])
#        ]
#        results = [node.world_to_pixel_coordinates(input_data) for input_data in inputs]
#        for result in results:
#            self.assertEqual(result.shape,(2,2))
#
#    def test_transform_translation(self):
#        #empty node
#        node = PanoNode()        
#        box=copy.deepcopy(node.orientedBoundingBox)
#        hull=copy.deepcopy(node.convexHull)
#        translation = [1,0,0]
#        node.transform(translation=translation)
#        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.get_center()+translation,atol=0.001))
#        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.get_center()+translation,atol=0.001))
#        self.assertTrue(np.allclose(node.cartesianTransform,gmu.get_cartesian_transform(translation=translation),atol=0.001))
#        
#        
#        #real node
#        node=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject=self.dataLoaderRailway.imageSubject1)
#        box=copy.deepcopy(node.orientedBoundingBox)
#        hull=copy.deepcopy(node.convexHull)
#        translation = [1,0,0]
#        node.transform(translation=translation)
#        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.get_center()+translation,atol=0.001))
#        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.get_center()+translation,atol=0.001))
#        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],node.cartesianTransform[:3,3]+translation,atol=0.001))
#        
#    def test_transform_rotation(self):
#        
#        #90° rotation around z-axis
#        rotation_euler = [0,0,90]
#        rotation_matrix=   np.array( [[ 0, -1 , 0.        ],
#                                        [ 1,  0,  0.        ],
#                                        [ 0.   ,       0.    ,      1.        ]])  
#        
#        #(0,0) node with rotation matrix     
#        node = PanoNode()          
#        box=copy.deepcopy(node.orientedBoundingBox)
#        hull=copy.deepcopy(node.convexHull)
#        node.transform(rotation=rotation_matrix)
#        self.assertAlmostEqual(node.cartesianTransform[0,3],0,delta=0.01 )
#        self.assertAlmostEqual(node.cartesianTransform[0,0],0,delta=0.01 )
#        self.assertAlmostEqual(node.cartesianTransform[0,1],-1,delta=0.01 )
#        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
#        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
#   
#    def test_create_rays(self):
#        node=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',getResource=True)
#        
#        #no inputs
#        node=PanoNode()
#        rays=node.create_rays()
#        self.assertTrue(np.allclose(rays[0][:3],node.cartesianTransform[:3,3],atol=0.001))
#        self.assertEqual(rays.shape,(4,6)) #4 cornerpoints
#        
#        #check centerpont shooting in Z direction
#        node=PanoNode()
#        rays=node.create_rays([node.imageHeight/2,node.imageWidth/2],50)
#        startpoint,endpoint=gmu.rays_to_points(rays)
#
#        self.assertEqual(np.linalg.norm(startpoint-node.get_center()),0)
#        self.assertEqual(np.linalg.norm(endpoint-node.get_center()),50) 
#        self.assertEqual(np.linalg.norm(endpoint-np.array([0,0,50])),0) 
#        
#        #different input formats single inputs
#        inputs = [
#            [1681.93211083, 10003.48867638],
#            np.array([1681.93211083, 10003.48867638]),
#            np.array([[1681.93211083, 10003.48867638]])
#        ]
#        results = [node.create_rays(input_data) for input_data in inputs]
#        for result in results:
#            self.assertTrue(np.allclose(result[0][:3],node.cartesianTransform[:3,3],atol=0.01))
#            self.assertEqual(result.shape,(1,6))
#            
#        #check specific case
#        node=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',getResource=False)
#        rays=node.create_rays(self.dataLoaderRailway.imgCoordinate,self.dataLoaderRailway.distance)        
#        self.assertTrue(np.allclose(rays[0][:3],node.cartesianTransform[:3,3],atol=0.001))
#        _,endpoint=gmu.rays_to_points(rays)        
#        self.assertTrue(np.allclose(endpoint,self.dataLoaderRailway.worldCoordinate[0][:3],atol=0.01))
#    
#    def test_pixel_to_world_coordinates(self):
#        node=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',resource=self.dataLoaderRailway.image1)
#        worldCoordinate = node.pixel_to_world_coordinates(self.dataLoaderRailway.imgCoordinate,self.dataLoaderRailway.distance)
#        self.assertTrue(np.allclose(worldCoordinate,self.dataLoaderRailway.worldCoordinate[0][:3],atol=0.1))
#        
#        
#    def test_project_lineset_on_image(self):
#        PanoNode=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',resource=self.dataLoaderRailway.image1)
#        PanoNode.project_lineset_on_image(self.dataLoaderRailway.line)
#
#    def test_get_image_features(self):
#        PanoNode=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',resource=self.dataLoaderRailway.image1)
#        keypoints,descriptors=PanoNode.get_image_features()
#        #check if the number of keypoints is the same as the number of descriptors
#        self.assertEqual(len(keypoints),len(descriptors))
#        
#    def test_draw_keypoints_on_image(self):
#        PanoNode=PanoNode(xmlPath=self.dataLoaderRailway.imageXmlPath,subject='P0024688',resource=self.dataLoaderRailway.image1)
#        image=PanoNode.draw_keypoints_on_image(keypoint_size=300)

if __name__ == '__main__':
    unittest.main()

import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import sys
import cv2
import rdflib
from geomapi.nodes import PanoNode
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
from geomapi.nodes import PanoNode

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from data_loader_railway import DATALOADERRAILWAYINSTANCE 

from geomapi.utils import GEOMAPI_PREFIXES

class TestPanoNode(unittest.TestCase):



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
        # if os.path.exists(cls.dataLoaderRailway.resourcePath):
        #     shutil.rmtree(cls.dataLoaderRailway.resourcePath)  
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
        node= PanoNode()
        self.assertIsNotNone(node.subject)
        self.assertIsNotNone(node.name)
        self.assertEqual(node.imageWidth,2000)
        self.assertEqual(node.imageHeight,1000)
        self.assertEqual(node.depth,1)
        self.assertIsNone(node.jsonPath)        
        self.assertIsNotNone(node.timestamp)        
        
    def test_subject(self):
        #subject
        subject='myNode'
        node= PanoNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://'+subject)
        self.assertEqual(node.name,subject)
        self.assertEqual(node.depth,1)
        self.assertEqual(node.imageHeight,1000)
        self.assertEqual(node.imageWidth,2000)
        
        
    def test_name(self):
        node= PanoNode(name='name')
        self.assertEqual(node.name,'name')
        self.assertEqual(node.subject.toPython(),'http://name')    
        
    def test_imageWidth(self):
        node= PanoNode(imageWidth=100)
        self.assertEqual(node.imageWidth,100)
        self.assertEqual(node.orientedBoundingBox.extent[0],2)
        
        #raise error when text
        self.assertRaises(ValueError,PanoNode,imageWidth='qsdf')
    
    def test_imageHeight(self):
        node= PanoNode(imageHeight=100)
        self.assertEqual(node.imageHeight,100)
        self.assertEqual(node.orientedBoundingBox.extent[1],2)
        
        #raise error when text
        self.assertRaises(ValueError,PanoNode,imageHeight='qsdf')
    
    def test_cartesianTransform(self):
        #create a convex hull in the shape of a box
        base_hull = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1) #this gets Obox with x,y,z and no rotation
        base_hull.translate((-0.5,-0.5,-0.5))
        #base hull
        node= PanoNode(cartesianTransform=base_hull.get_center())
        expectedCartesianTransform=np.eye(4)
        self.assertTrue(np.allclose(node.cartesianTransform,expectedCartesianTransform,atol=0.001))
        
        
    def test_orientedBoundingBox_and_convex_hull(self):
        #default box width should be 20, height 10m, and depth 10
              
        #translation
        cartesianTransform=np.array([[1,0,0,1],
                                     [0,1,0,0],
                                     [0,0,1,0],
                                     [0,0,0,1]])
        node= PanoNode(cartesianTransform=cartesianTransform)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,0,0]),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([0,0,0]),atol=0.001))
        
    
    def test_depth(self):
        node= PanoNode(depth=100)
        self.assertEqual(node.depth,100)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,0,0]),atol=0.001))
        
        #raise error when text
        self.assertRaises(ValueError,PanoNode,depth='qsdf')
        
        
    def test_path(self):
        #path1 without loadResource
        node= PanoNode(path=self.dataLoaderRailway.orthoPath2)
        self.assertEqual(node.name,self.dataLoaderRailway.orthoPath2.stem)
        self.assertIsNone(node._resource)

        #path2 with loadResource
        node= PanoNode(path=self.dataLoaderRailway.orthoPath2,loadResource=True)        
        self.assertEqual(node.name,self.dataLoaderRailway.orthoPath2.stem)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)        
        self.assertEqual(node.imageHeight,self.dataLoaderRailway.ortho2.shape[0])
        self.assertIsNotNone(node._resource)
        
        #raise error when wrong path
        self.assertRaises(ValueError,PanoNode,path='dfsgsdfgsd')
        
    
    def test_resource(self):
        #tiff
        node= PanoNode(resource=self.dataLoaderRailway.ortho2)
        self.assertEqual(node.imageHeight,self.dataLoaderRailway.ortho2.shape[0])
        self.assertEqual(node.resource.shape[0],self.dataLoaderRailway.ortho2.shape[0])
                
    def test_graphPath(self):
        node=PanoNode(graphPath=self.dataLoaderRailway.orthoGraphPath)
        self.assertEqual(node.graphPath,self.dataLoaderRailway.orthoGraphPath)
        self.assertTrue(node.subject in self.dataLoaderRailway.orthoGraph.subjects())
        self.assertIsNotNone(node.imageHeight)
        self.assertIsNotNone(node.imageWidth)    
        
    def test_graphPath_with_subject(self):
        subject=next(self.dataLoaderRailway.orthoGraph.subjects(RDF.type))
        node=PanoNode(graphPath=self.dataLoaderRailway.orthoGraphPath,subject=subject)
        
        self.assertEqual(len(node.adjacent),3)
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderRailway.orthoGraph.triples((subject, None, None)):
            if 'gsd' in p.toPython():
                self.assertEqual(float(o),node.gsd)
            if 'depth' in p.toPython():
                self.assertEqual(float(o),node.depth)
            if 'height' in p.toPython():
                self.assertEqual(float(o),node.height)
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderRailway.orthoGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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
            if 'principalPointU' in p.toPython():
                self.assertEqual(float(o),node.principalPointU)
            if 'principalPointV' in p.toPython():
                self.assertEqual(float(o),node.principalPointV)
            if 'imageWidth' in p.toPython():
                self.assertEqual(float(o),node.imageWidth)
            if 'imageLength' in p.toPython():
                self.assertEqual(float(o),node.imageHeight)

        #raise error when subject is not in graph
        self.assertRaises(ValueError,PanoNode,graphPath=self.dataLoaderRailway.orthoGraphPath,subject=URIRef('mySubject'))
    
    def test_graph(self):
        subject=next(self.dataLoaderRailway.orthoGraph.subjects(RDF.type))
        node=PanoNode(graphPath=self.dataLoaderRailway.orthoGraphPath,subject=subject)
        
        #check if the graph is correctly parsed
        for s, p, o in self.dataLoaderRailway.orthoGraph.triples((subject, None, None)):
            if 'path' in p.toPython():
                self.assertEqual((self.dataLoaderRailway.orthoGraphPath.parent/Path(o.toPython())).resolve(),node.path) 
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
            if 'principalPointU' in p.toPython():
                self.assertEqual(float(o),node.principalPointU)
            if 'principalPointV' in p.toPython():
                self.assertEqual(float(o),node.principalPointV)
            if 'imageWidth' in p.toPython():
                self.assertEqual(float(o),node.imageWidth)
            if 'imageLength' in p.toPython():
                self.assertEqual(float(o),node.imageHeight)

 
    def test_node_creation_with_get_resource(self):
        #mesh
        node= PanoNode(resource=self.dataLoaderRailway.ortho2)
        self.assertIsNotNone(node._resource)

        #path without loadResource
        node= PanoNode(path=self.dataLoaderRailway.orthoPath2)
        self.assertIsNone(node._resource)

        #path with loadResource
        node= PanoNode(path=self.dataLoaderRailway.orthoPath2,loadResource=True)
        self.assertIsNotNone(node._resource)

        #graph with get resource
        node= PanoNode(subject=self.dataLoaderRailway.orthoSubject,
                        graph=self.dataLoaderRailway.orthoGraph,
                        loadResource=True)
        self.assertIsNone(node._resource)
        
        #graphPath with get resource
        node= PanoNode(subject=self.dataLoaderRailway.orthoSubject,
                        graphPath=self.dataLoaderRailway.orthoGraphPath,
                        loadResource=True)
        self.assertIsNotNone(node._resource)

    def test_save_resource(self):
        #no mesh -> False
        node= PanoNode()
        self.assertFalse(node.save_resource())

        #directory
        node= PanoNode(resource=self.dataLoaderRailway.ortho2)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #graphPath        
        node= PanoNode(resource=self.dataLoaderRailway.ortho2,
                        graphPath=self.dataLoaderRailway.orthoGraphPath)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #no path or graphPath
        node= PanoNode(resource=self.dataLoaderRailway.ortho2)        
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #path -> new name
        node= PanoNode(subject=URIRef('myImg'),
                        path=self.dataLoaderRailway.orthoPath2,
                        loadResource=True)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))
        
        #graphPath with directory
        node=PanoNode(subject=self.dataLoaderRailway.orthoSubject,
                       graphPath=self.dataLoaderRailway.orthoGraphPath,
                       resource=self.dataLoaderRailway.ortho2)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #graph with new subject
        node=PanoNode(subject=self.dataLoaderRailway.orthoSubject,
                       graph=self.dataLoaderRailway.orthoGraph,
                       resource=self.dataLoaderRailway.ortho2)
        node.subject='myImg'
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

    def test_get_resource(self):
        #mesh
        node=PanoNode(resource=self.dataLoaderRailway.ortho2)  
        self.assertIsNone(node.load_resource())

        #graphPath with loadResource
        node=PanoNode(graphPath=str(self.dataLoaderRailway.orthoGraphPath),
                       subject=self.dataLoaderRailway.orthoSubject,
                       loadResource=True)
        self.assertIsNotNone(node.load_resource())

    def test_set_path(self):
        #valid path
        node=PanoNode()
        node.path= str(self.dataLoaderRailway.orthoPath2)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)

        #preexisting
        node=PanoNode(path=self.dataLoaderRailway.orthoPath2)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)

        #graphPath & name
        node=PanoNode(subject=self.dataLoaderRailway.orthoSubject,
                       graphPath=self.dataLoaderRailway.orthoGraphPath)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)

    def test_transform_translation(self):
        #empty node
        node = PanoNode()        
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        translation = [1,0,0]
        node.transform(translation=translation)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.cartesianTransform,gmu.get_cartesian_transform(translation=translation),atol=0.001))
        
        #real node
        node=PanoNode(dxfPath=self.dataLoaderRailway.orthoDxfPath2,subject=self.dataLoaderRailway.orthoSubject)
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        translation = [1,0,0]
        transform = node.cartesianTransform
        node.transform(translation=translation)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.cartesianTransform[:3,3],transform[:3,3]+translation,atol=0.001))
        
    def test_transform_rotation(self):
        
        #90° rotation around z-axis
        rotation_euler = [0,0,90]
        rotation_matrix=   np.array( [[ 0, -1 , 0.        ],
                                        [ 1,  0,  0.        ],
                                        [ 0.   ,       0.    ,      1.        ]])  
        
        #(0,0) node with rotation matrix     
        node = PanoNode()          
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        node.transform(rotation=rotation_matrix)
        self.assertAlmostEqual(node.cartesianTransform[0,3],0,delta=0.01 )
        self.assertAlmostEqual(node.cartesianTransform[0,0],0,delta=0.01 )
        self.assertAlmostEqual(node.cartesianTransform[0,1],-1,delta=0.01 )
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
    
    def test_create_rays(self):
        node = PanoNode(imageHeight=500, imageWidth=1000, depth=50.0)
        rays = node.create_rays()
        self.assertIsInstance(rays, np.ndarray)
        self.assertEqual(rays.shape[1], 6)  # [origin (3), direction (3)]
        self.assertEqual(rays.shape[0], 6)  # default: 6 directions

        points = np.array([[0, 0], [100, 200]])
        depths = np.array([30, 60])
        rays = node.create_rays(imagePoints=points, depths=depths)
        self.assertEqual(rays.shape, (2, 6))

    def test_world_to_pixel_coordinates(self):
        node = PanoNode(imageHeight=500, imageWidth=1000, depth=50.0)
        # Input a known direction (unit vector pointing forward)
        world_coords = np.array([
            [1, 0, 0],   # right
            [0, 1, 0],   # up
            [0, 0, 1],   # forward
        ])
        output = node.world_to_pixel_coordinates(world_coords)
        self.assertEqual(output.shape, (3, 2))
        self.assertTrue((output >= 0).all())
        self.assertTrue((output[:, 0] <= node.imageWidth).all())
        self.assertTrue((output[:, 1] <= node.imageHeight).all())

    def test_pixel_to_world_coordinates(self):
        node = PanoNode(imageHeight=500, imageWidth=1000, depth=50.0)

        pixels = np.array([[0, 0], [100, 200], [200, 300]])
        depths = np.array([10, 20, 30])
        coords = node.pixel_to_world_coordinates(pixels, depths)
        self.assertEqual(coords.shape, (3, 3))
        
    def test_project_lineset_on_image(self):
        node=PanoNode(dxfPath=self.dataLoaderRailway.orthoDxfPath2,
               tfwPath=self.dataLoaderRailway.orthoTfwPath2,
               path=self.dataLoaderRailway.orthoPath2,
               loadResource=True,
               height=300,
               depth=50)
        self.dataLoaderRailway.line        
        node.project_lineset_on_image(self.dataLoaderRailway.line)


if __name__ == '__main__':
    unittest.main()
