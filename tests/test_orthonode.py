import os
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value
import sys
import cv2
import rdflib
from geomapi.nodes import OrthoNode
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
from geomapi.nodes import OrthoNode

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from data_loader_railway import DATALOADERRAILWAYINSTANCE 

from geomapi.utils import GEOMAPI_PREFIXES

class TestOrthoNode(unittest.TestCase):



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
        node= OrthoNode()
        self.assertIsNotNone(node.subject)
        self.assertIsNotNone(node.name)
        self.assertEqual(node.imageWidth,2000)
        self.assertEqual(node.imageHeight,1000)
        self.assertEqual(node.depth,10)
        self.assertEqual(node.gsd,0.01)        
        self.assertIsNone(node.dxfPath)        
        self.assertIsNotNone(node.timestamp)        
        
    def test_subject(self):
        #subject
        subject='myNode'
        node= OrthoNode(subject=subject)
        self.assertEqual(node.subject.toPython(),'http://'+subject)
        self.assertEqual(node.name,subject)
        self.assertEqual(node.gsd,0.01)
        self.assertEqual(node.imageHeight,1000)
        self.assertEqual(node.imageWidth,2000)
        
        
    def test_name(self):
        node= OrthoNode(name='name')
        self.assertEqual(node.name,'name')
        self.assertEqual(node.subject.toPython(),'http://name')    
        
    def test_imageWidth(self):
        node= OrthoNode(imageWidth=100)
        self.assertEqual(node.imageWidth,100)
        self.assertEqual(node.orientedBoundingBox.extent[0],1)
        
        #raise error when text
        self.assertRaises(ValueError,OrthoNode,imageWidth='qsdf')
    
    def test_imageHeight(self):
        node= OrthoNode(imageHeight=100)
        self.assertEqual(node.imageHeight,100)
        self.assertEqual(node.orientedBoundingBox.extent[1],1)
        
        #raise error when text
        self.assertRaises(ValueError,OrthoNode,imageHeight='qsdf')
    
    def test_cartesianTransform(self):
        #create a convex hull in the shape of a box
        base_hull = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1) #this gets Obox with x,y,z and no rotation
        base_hull.translate((-0.5,-0.5,-0.5))
        #base hull
        node= OrthoNode(cartesianTransform=base_hull.get_center())
        expectedCartesianTransform=np.eye(4)
        self.assertTrue(np.allclose(node.cartesianTransform,expectedCartesianTransform,atol=0.001))
        
        
    def test_orientedBoundingBox_and_convex_hull(self):
        #default box width should be 20, height 10m, and depth 10
              
        #translation
        cartesianTransform=np.array([[1,0,0,1],
                                     [0,1,0,0],
                                     [0,0,1,0],
                                     [0,0,0,1]])
        node= OrthoNode(cartesianTransform=cartesianTransform)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([1,0,5]),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([1,0,5]),atol=0.001))
        
        #90° rotation around z-axis
        rotation_matrix_90_z=   np.array( [[ 0, -1 , 0. ,0       ],
                                        [ 1,  0,  0.   ,0     ],
                                        [ 0.   ,       0.    ,      1.    ,0    ],
                                        [0,0,0,1]])  
        node= OrthoNode(cartesianTransform=rotation_matrix_90_z)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,0,5]),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([0,0,5]),atol=0.001))
        
        #90° rotation around x-axis
        rotation_matrix_90_x=   np.array( [[ 1, 0 , 0. ,0       ],
                                        [ 0,  0,  -1   ,0     ],
                                        [ 0.   ,       1    ,      0    ,0    ],
                                        [0,0,0,1]])  
        node= OrthoNode(cartesianTransform=rotation_matrix_90_x)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,-5,0]),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([0,-5,0]),atol=0.001))
        
        #90° rotation around x-axis + translation
        rotation_matrix_90_x=   np.array( [[ 1, 0 , 0. ,1       ],
                                        [ 0,  0,  -1   ,0     ],
                                        [ 0.   ,       1    ,      0    ,0    ],
                                        [0,0,0,1]])  
        node= OrthoNode(cartesianTransform=rotation_matrix_90_x)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([1,-5,0]),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([1,-5,0]),atol=0.001))
    
    def test_depth(self):
        node= OrthoNode(depth=100)
        self.assertEqual(node.depth,100)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([0,0,50]),atol=0.001))
        
        #raise error when text
        self.assertRaises(ValueError,OrthoNode,depth='qsdf')
        
        
    def test_path(self):
        #path1 without loadResource
        node= OrthoNode(path=self.dataLoaderRailway.orthoPath2)
        self.assertEqual(node.name,self.dataLoaderRailway.orthoPath2.stem)
        self.assertIsNone(node._resource)

        #path2 with loadResource
        node= OrthoNode(path=self.dataLoaderRailway.orthoPath2,loadResource=True)        
        self.assertEqual(node.name,self.dataLoaderRailway.orthoPath2.stem)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)        
        self.assertEqual(node.imageHeight,self.dataLoaderRailway.ortho2.shape[0])
        self.assertIsNotNone(node._resource)
        
        #raise error when wrong path
        self.assertRaises(ValueError,OrthoNode,path='dfsgsdfgsd')
        
    
    def test_resource(self):
        #tiff
        node= OrthoNode(resource=self.dataLoaderRailway.ortho2)
        self.assertEqual(node.imageHeight,self.dataLoaderRailway.ortho2.shape[0])
        self.assertEqual(node.resource.shape[0],self.dataLoaderRailway.ortho2.shape[0])
                
    def test_graphPath(self):
        node=OrthoNode(graphPath=self.dataLoaderRailway.orthoGraphPath)
        self.assertEqual(node.graphPath,self.dataLoaderRailway.orthoGraphPath)
        self.assertTrue(node.subject in self.dataLoaderRailway.orthoGraph.subjects())
        self.assertIsNotNone(node.imageHeight)
        self.assertIsNotNone(node.imageWidth)    
        
    def test_graphPath_with_subject(self):
        subject=next(self.dataLoaderRailway.orthoGraph.subjects(RDF.type))
        node=OrthoNode(graphPath=self.dataLoaderRailway.orthoGraphPath,subject=subject)
        
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
        self.assertRaises(ValueError,OrthoNode,graphPath=self.dataLoaderRailway.orthoGraphPath,subject=URIRef('mySubject'))
    
    def test_graph(self):
        subject=next(self.dataLoaderRailway.orthoGraph.subjects(RDF.type))
        node=OrthoNode(graphPath=self.dataLoaderRailway.orthoGraphPath,subject=subject)
        
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
        node= OrthoNode(resource=self.dataLoaderRailway.ortho2)
        self.assertIsNotNone(node._resource)

        #path without loadResource
        node= OrthoNode(path=self.dataLoaderRailway.orthoPath2)
        self.assertIsNone(node._resource)

        #path with loadResource
        node= OrthoNode(path=self.dataLoaderRailway.orthoPath2,loadResource=True)
        self.assertIsNotNone(node._resource)

        #graph with get resource
        node= OrthoNode(subject=self.dataLoaderRailway.orthoSubject,
                        graph=self.dataLoaderRailway.orthoGraph,
                        loadResource=True)
        self.assertIsNone(node._resource)
        
        #graphPath with get resource
        node= OrthoNode(subject=self.dataLoaderRailway.orthoSubject,
                        graphPath=self.dataLoaderRailway.orthoGraphPath,
                        loadResource=True)
        self.assertIsNotNone(node._resource)

    def test_save_resource(self):
        #no mesh -> False
        node= OrthoNode()
        self.assertFalse(node.save_resource())

        #directory
        node= OrthoNode(resource=self.dataLoaderRailway.ortho2)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #graphPath        
        node= OrthoNode(resource=self.dataLoaderRailway.ortho2,
                        graphPath=self.dataLoaderRailway.orthoGraphPath)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #no path or graphPath
        node= OrthoNode(resource=self.dataLoaderRailway.ortho2)        
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #path -> new name
        node= OrthoNode(subject=URIRef('myImg'),
                        path=self.dataLoaderRailway.orthoPath2,
                        loadResource=True)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))
        
        #graphPath with directory
        node=OrthoNode(subject=self.dataLoaderRailway.orthoSubject,
                       graphPath=self.dataLoaderRailway.orthoGraphPath,
                       resource=self.dataLoaderRailway.ortho2)
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

        #graph with new subject
        node=OrthoNode(subject=self.dataLoaderRailway.orthoSubject,
                       graph=self.dataLoaderRailway.orthoGraph,
                       resource=self.dataLoaderRailway.ortho2)
        node.subject='myImg'
        self.assertTrue(node.save_resource(self.dataLoaderRailway.resourcePath))

    def test_get_resource(self):
        #mesh
        node=OrthoNode(resource=self.dataLoaderRailway.ortho2)  
        self.assertIsNone(node.load_resource())

        #graphPath with loadResource
        node=OrthoNode(graphPath=str(self.dataLoaderRailway.orthoGraphPath),
                       subject=self.dataLoaderRailway.orthoSubject,
                       loadResource=True)
        self.assertIsNotNone(node.load_resource())

    def test_set_path(self):
        #valid path
        node=OrthoNode()
        node.path= str(self.dataLoaderRailway.orthoPath2)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)

        #preexisting
        node=OrthoNode(path=self.dataLoaderRailway.orthoPath2)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)

        #graphPath & name
        node=OrthoNode(subject=self.dataLoaderRailway.orthoSubject,
                       graphPath=self.dataLoaderRailway.orthoGraphPath)
        self.assertEqual(node.path,self.dataLoaderRailway.orthoPath2)

    def test_transform_translation(self):
        #empty node
        node = OrthoNode()        
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        translation = [1,0,0]
        node.transform(translation=translation)
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.get_center()+translation,atol=0.001))
        self.assertTrue(np.allclose(node.cartesianTransform,gmu.get_cartesian_transform(translation=translation),atol=0.001))
        
        #real node
        node=OrthoNode(dxfPath=self.dataLoaderRailway.orthoDxfPath2,subject=self.dataLoaderRailway.orthoSubject)
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
        node = OrthoNode()          
        box=copy.deepcopy(node.orientedBoundingBox)
        hull=copy.deepcopy(node.convexHull)
        node.transform(rotation=rotation_matrix)
        self.assertAlmostEqual(node.cartesianTransform[0,3],0,delta=0.01 )
        self.assertAlmostEqual(node.cartesianTransform[0,0],0,delta=0.01 )
        self.assertAlmostEqual(node.cartesianTransform[0,1],-1,delta=0.01 )
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),box.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
        self.assertTrue(np.allclose(node.convexHull.get_center(),hull.rotate(rotation_matrix,center=node.get_center()).get_center(),atol=0.001))
    
    def test_create_rays(self):        
        #no inputs
        node=OrthoNode()
        rays=node.create_rays()
        result=np.array([[-10.,   5.,   0.,   0.,   0., 100.],
                [ 10.,   5.,   0.,   0.,   0., 100.],
                [-10.,  -5.,   0.,   0.,   0., 100.],
                [ 10.,  -5.,   0.,   0.,   0., 100.]])
        self.assertTrue(np.allclose(rays,result,atol=0.001))
        
           
    def test_world_to_pixel_coordinates(self):
        node=OrthoNode(cartesianTransform=np.array([[ 1,  0, 0,0],
                                                    [0, -1, 0,0],
                                                    [0,  0, -1,0],
                                                    [ 0,  0, 0,1]]))
        pixel=node.world_to_pixel_coordinates(np.array([[0,0,10], #center
                        [-10,5,100], #top left
                        [10,5,100], #top right
                        [-10,-5,100], #bottom left
                        [10,-5,100] #bottom right
                        ]))
        self.assertTrue(np.allclose(pixel,np.array([[500,1000], #center
                                    [0,0], #top left
                                    [0,2000], #top right
                                    [1000,0], #bottom left
                                    [1000,2000]])#bottom right
                                    ,atol=10))  
            
    def test_pixel_to_world_coordinates(self):
        node=OrthoNode()
        pixels=node.pixel_to_world_coordinates(np.array([[500,1000], #center
                                                        [0,0], #top left
                                                        [0,2000], #top right
                                                        [1000,0], #bottom left
                                                        [1000,2000]]),depths=50)#bottom right
        result=np.array([[0,0,50], #center
                        [-10,5,50], #top left
                        [10,5,50], #top right
                        [-10,-5,50], #bottom left
                        [10,-5,50]]) #bottom right
        
        self.assertTrue(np.allclose(pixels,result,atol=0.01))  
        
    def test_project_lineset_on_image(self):
        node=OrthoNode(dxfPath=self.dataLoaderRailway.orthoDxfPath2,
               tfwPath=self.dataLoaderRailway.orthoTfwPath2,
               path=self.dataLoaderRailway.orthoPath2,
               loadResource=True,
               height=300,
               depth=50)
        self.dataLoaderRailway.line        
        node.project_lineset_on_image(self.dataLoaderRailway.line)

        
    def test_tfw_path(self):
       
        #tfwPath and name + height-> offset in y and z
        node=OrthoNode(tfwPath=self.dataLoaderRailway.orthoTfwPath2,
                       name='railway-0-0',
                       height=self.dataLoaderRailway.orthoHeight,
                       resource=self.dataLoaderRailway.ortho2)
        self.assertEqual(node.tfwPath,self.dataLoaderRailway.orthoTfwPath2)
        #check cartesianTransform
        self.assertTrue(np.allclose(node.cartesianTransform,self.dataLoaderRailway.orthoCartesianTransform,atol=0.001))
        #check orientedBoundingBox. height default 5
        self.assertTrue(np.allclose(node.orientedBoundingBox.get_center(),np.array([263379.5193, 151089.1667 ,self.dataLoaderRailway.orthoHeight-5]),atol=0.001))
        #check convexHull
        self.assertTrue(np.allclose(node.convexHull.get_center(),np.array([263379.5193, 151089.1667 ,self.dataLoaderRailway.orthoHeight-5]),atol=0.001))
                
        #raise error when wrong path
        self.assertRaises(ValueError,OrthoNode,tfwPath='dfsgsdfgsd')
        
        
    def test_height(self):
        #default height should be 0
        node=OrthoNode()
        self.assertEqual(node.height,0)
        
        #height
        node=OrthoNode(height=10)
        self.assertEqual(node.height,10)
        
        #raise error when text
        self.assertRaises(ValueError,OrthoNode,height='qsdf')

if __name__ == '__main__':
    unittest.main()
