import copy
import math
import os
import sys
from pathlib import Path
import shutil
import time
import unittest
from multiprocessing.sharedctypes import Value

import cv2
import ifcopenshell
import numpy as np
import open3d as o3d
import pye57
import ifcopenshell.util.selector
import trimesh

#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.geometryutils as gmu

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 
from geomapi.utils import GEOMAPI_PREFIXES


################################## SETUP/TEARDOWN MODULE ######################

# def setUpModule():
#     #execute once before the module 
#     print('-----------------Setup Module----------------------')

# def tearDownModule():
#     #execute once after the module 
#     print('-----------------TearDown Module----------------------')



class TestGeometryutils(unittest.TestCase):





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
        
        

################################## FIXTURES ######################
    # # @pytest.fixture(scope='module')
    # # @pytest.fixture
    # def test_data(*args):
    #     here = os.path.split(__file__)[0]
    #     return os.path.join(here, "testfiles", *args)

    # @pytest.fixture
    # def e57Path1():
    #     return test_data("pointcloud.e57")

    # @pytest.fixture
    # def ifcData():
    #     ifcPath=os.path.join(os.getcwd(),"testfiles", "ifcfile.ifc")
    #     classes= '.IfcBeam | .IfcColumn | .IfcWall | .IfcSlab'
    #     ifc = ifcopenshell.open(ifcPath)   
    #     selector = Selector()
    #     dataList=[]
    #     for ifcElement in selector.parse(ifc, classes): 
    #         dataList.append(ifcElement)
    #     return dataList
    
    
    

################################## TEST FUNCTIONS ######################

    def test_get_rotation_matrix_from_forward_up(self):
        forward = np.array([0,0,1])
        up = np.array([0,1,0])
        rot_matrix = np.array([[ 1.,  0.,  0.],[ 0.,  1.,  0.],[0.,  0.,  1.]])
        rotated_vector = np.around(gmu.get_rotation_matrix_from_forward_up(forward, up),0)
        np.testing.assert_array_equal(rot_matrix,rotated_vector)

    def test_convert_to_homogeneous_3d_coordinates(self):
        # Regular 3d vector
        vector = [1,0,0]
        homo_vector = [[1,0,0,1]]
        np.testing.assert_array_equal(gmu.convert_to_homogeneous_3d_coordinates(vector),homo_vector)
        # Homogeneous 3d vector
        vector = [1,0,0,1]
        homo_vector = [[1,0,0,1]]
        np.testing.assert_array_equal(gmu.convert_to_homogeneous_3d_coordinates(vector),homo_vector)
        # Scaled Homogeneous 3d vector
        vector = [1,0,0,2]
        homo_vector = [[0.5,0,0,1]]
        np.testing.assert_array_equal(gmu.convert_to_homogeneous_3d_coordinates(vector),homo_vector)
        # N-d Homogeneous 3d vector
        vector = [[1,0,0], [0,1,0],[0,0,1]]
        homo_vector = [[1,0,0,1], [0,1,0,1],[0,0,1,1]]
        np.testing.assert_array_equal(gmu.convert_to_homogeneous_3d_coordinates(vector),homo_vector)

    def test_create_visible_point_cloud_from_meshes(self):
        referenceMesh1= copy.deepcopy(self.dataLoaderParking.slabMesh)
        referenceMesh1.translate([1,0,0])
        referenceMesh2= copy.deepcopy(self.dataLoaderParking.slabMesh)
        referenceMesh2.translate([0,1,0])
        
        # 1 geometry
        identityPointClouds1, percentages1=gmu.create_visible_point_cloud_from_meshes(geometries=self.dataLoaderParking.slabMesh,
                                                references=referenceMesh1)
        self.assertEqual(len(identityPointClouds1),1)
        self.assertGreater(len(identityPointClouds1[0].points),10)
        self.assertEqual(len(percentages1),1)        
        self.assertLess(percentages1[0],0.2)

        # multiple geometries 
        list=[self.dataLoaderParking.slabMesh,self.dataLoaderParking.wallMesh]
        references=[referenceMesh1,referenceMesh2]
        identityPointClouds2, percentages2=gmu.create_visible_point_cloud_from_meshes(geometries=list,
                                                references=references)
        self.assertEqual(len(identityPointClouds2),2)
        self.assertLess(len(identityPointClouds2[0].points),len(identityPointClouds1[0].points))
        self.assertEqual(len(percentages2),2)        
        self.assertLess(percentages2[0],percentages1[0])

    def test_mesh_to_trimesh(self):
        triMesh=gmu.mesh_to_trimesh(self.dataLoaderParking.slabMesh)
        vertices = o3d.utility.Vector3dVector(triMesh.vertices)
        triangles = o3d.utility.Vector3iVector(triMesh.faces)
        mesh_o3d = o3d.geometry.TriangleMesh(vertices, triangles)
        self.assertEqual(len(self.dataLoaderParking.slabMesh.triangles),len(mesh_o3d.triangles))

    def test_crop_mesh_by_convex_hull(self):
        """Test cropping inside a convex hull"""
        source_mesh = trimesh.creation.box(extents=[2, 2, 2])
        cutter_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.5)

        cropped_meshes = gmu.crop_mesh_by_convex_hull(source_mesh, [cutter_mesh], inside=True)

        assert isinstance(cropped_meshes, list)
        assert len(cropped_meshes) > 0
        for cropped in cropped_meshes:
            assert isinstance(cropped, trimesh.Trimesh)
            assert len(cropped.vertices) > 0  # Should retain some part of the mesh

    def test_sample_geometry(self):
        """Test sampling a TriangleMesh"""
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        resolution = 0.1
        sampled_pcd = gmu.sample_geometry([mesh], resolution)

        assert isinstance(sampled_pcd, list)
        assert isinstance(sampled_pcd[0], o3d.geometry.PointCloud)
        assert len(sampled_pcd[0].points) > 0  # Should contain points

    def test_get_points_and_normals(self):
        gmu.get_points_and_normals(self.dataLoaderParking.pcd)

    def test_compute_nearest_neighbors(self):
        # Create some reference and query points
        reference_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        query_points = np.array([[0.1, 0.1, 0.1], [2.1, 2.1, 2.1]])

        # Expected nearest neighbors for each query point
        expected_indices = np.array([0, 2])  # Closest reference point indices
        expected_distances = np.linalg.norm(reference_points[expected_indices] - query_points, axis=1)

        indices, distances = gmu.compute_nearest_neighbors(query_points, reference_points, n=1)

        # Check if the returned values match expectations
        np.testing.assert_array_equal(indices.flatten(), expected_indices)
        np.testing.assert_allclose(distances.flatten(), expected_distances, atol=1e-5)



    def test_arrays_to_mesh_and_mesh_to_arrays(self):
        #mesh_to_arrays
        tuple=gmu.mesh_to_arrays(self.dataLoaderParking.meshPath)
        self.assertEqual(len(tuple[0]),len(self.dataLoaderParking.mesh.vertices))
        self.assertEqual(len(tuple[1]),len(self.dataLoaderParking.mesh.triangles))
        self.assertEqual(len(tuple[2]),len(self.dataLoaderParking.mesh.vertex_colors))
        self.assertEqual(len(tuple[3]),len(self.dataLoaderParking.mesh.vertex_normals))
        self.assertEqual(tuple[4],0)

        #arrays_to_mesh
        mesh=gmu.arrays_to_mesh(tuple)
        self.assertEqual(len(mesh.vertices),len(self.dataLoaderParking.mesh.vertices))
        self.assertEqual(len(mesh.triangles),len(self.dataLoaderParking.mesh.triangles))
        self.assertEqual(len(mesh.vertex_colors),len(self.dataLoaderParking.mesh.vertex_colors))

        mesh=gmu.arrays_to_mesh(tuple[:2])

    def test_arrays_to_pcd(self):
        #pcd_to_arrays
        tuple=gmu.pcd_to_arrays(self.dataLoaderRoad.pcdPath)
        self.assertEqual(len(tuple[0]),len(self.dataLoaderRoad.pcd.points))
        self.assertEqual(len(tuple[1]),len(self.dataLoaderRoad.pcd.colors))
        self.assertEqual(len(tuple[1]),len(self.dataLoaderRoad.pcd.normals))
        self.assertEqual(tuple[3],0)
     
        #arrays_to_mesh
        pcd=gmu.e57_array_to_pcd(tuple)
        self.assertEqual(len(pcd.points),len(self.dataLoaderRoad.pcd.points))
        self.assertEqual(len(pcd.colors),len(self.dataLoaderRoad.pcd.colors))
        self.assertEqual(len(pcd.normals),len(self.dataLoaderRoad.pcd.normals))

    def test_create_identity_point_cloud(self):
        # 1 geometry
        identityPointCloud, indentityArray=gmu.create_identity_point_cloud(self.dataLoaderParking.slabMesh)
        self.assertEqual(len(identityPointCloud.points),len(indentityArray))

        #multiple geometries
        list=[self.dataLoaderParking.slabMesh, self.dataLoaderParking.wallMesh]
        identityPointCloud2, indentityArray2=gmu.create_identity_point_cloud(list)
        self.assertEqual(len(identityPointCloud2.points),len(indentityArray2))

    def test_cap_mesh(self):
        print('test_cap_mesh NOT IMPLEMENTED')
        self.assertEqual(0,0)
        
    def test_box_to_mesh(self):
        box=o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        boundingBox=box.get_oriented_bounding_box()
        mesh =gmu.box_to_mesh(boundingBox) 
        self.assertIsInstance(mesh,o3d.geometry.TriangleMesh)
       
    def test_ifc_to_mesh(self):
        classes= 'IfcBeam,IfcColumn,IfcWall,IfcSlab'
        ifcCounter=0
        meshCounter =0
        
        for ifcElement in ifcopenshell.util.selector.filter_elements(self.dataLoaderParking.ifc,classes): 
            ifcCounter+=1
            mesh=gmu.ifc_to_mesh(ifcElement)
            self.assertIsInstance(mesh,o3d.geometry.TriangleMesh)
            if len(mesh.vertices) !=0:
                meshCounter +=1
            if ifcCounter==20:
                break
        self.assertEqual(meshCounter,ifcCounter)
      
    def test_get_oriented_bounding_box(self):     
        #orientedBounds
        myBox=self.dataLoaderRoad.mesh.get_oriented_bounding_box()
        boxPointsGt=np.asarray(myBox.get_box_points())
        box=gmu.get_oriented_bounding_box(boxPointsGt)
        boxPoints=np.asarray(box.get_box_points())
        
        for i in range(0,7):
            for j in range(0,2):
                self.assertAlmostEqual(boxPointsGt[i][j],boxPoints[i][j],delta=0.01)
                
    def test_get_oriented_bounding_box(self):
        # Test with cartesian bounds
        cartesian_bounds = np.array([-1, 1, -1, 1, -1, 1])
        obb = gmu.get_oriented_bounding_box(cartesian_bounds)
        self.assertIsInstance(obb, o3d.geometry.OrientedBoundingBox)
        
        # Test with 8 bounding points
        bounding_points = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ])
        obb = gmu.get_oriented_bounding_box(bounding_points)
        self.assertIsInstance(obb, o3d.geometry.OrientedBoundingBox)
        
        # Test with parameters (center, extent, euler_angles)
        parameters = np.array([
            [4, 5, 6],  # Center
            [1, 2, 3],  # Extent
            [np.pi, 0, 0]   # Euler angles (radians)
        ])
        obb = gmu.get_oriented_bounding_box(parameters)
        self.assertIsInstance(obb, o3d.geometry.OrientedBoundingBox)
        
        # Test with an array of 3D points
        points = np.random.rand(100, 3)  # Random 3D points
        obb = gmu.get_oriented_bounding_box(points)
        self.assertIsInstance(obb, o3d.geometry.OrientedBoundingBox)
        
        # Test with Open3D PointCloud
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        obb = gmu.get_oriented_bounding_box(pcd)
        self.assertIsInstance(obb, o3d.geometry.OrientedBoundingBox)

    def test_get_oriented_bounding_box_parameters(self):
        center = [0, 0, 0]
        extent = [1, 2, 3]
        euler_angles = [0, 0, 90]

        obb = gmu.get_oriented_bounding_box(np.array([center, extent, euler_angles]).flatten(), True)
        parameters = gmu.get_oriented_bounding_box_parameters(obb)

        expected_parameters = np.hstack((center, extent, euler_angles))

        np.testing.assert_array_almost_equal(parameters, expected_parameters, decimal=6)
    
      
    # def test_get_cartesian_transform(self):
    #     cartesianBounds=np.array([-1.0,1,-0.5,0.5,-5,-4])       
    #     translation=np.array([1, 2, 3])
    #     rotation=np.array([1,0,0,5,2,6,4,7,8])

    #     #no data
    #     cartesianTransform=gmu.get_cartesian_transform()
    #     self.assertEqual(cartesianTransform.shape[0],4)
    #     self.assertEqual(cartesianTransform.shape[1],4)
    #     self.assertEqual(cartesianTransform[1,1],1)
    #     self.assertEqual(cartesianTransform[2,3],0)

    #     #rotation + translation
    #     cartesianTransform=gmu.get_cartesian_transform(rotation=rotation,translation=translation)
    #     self.assertEqual(cartesianTransform[1,1],2)
    #     self.assertEqual(cartesianTransform[0,3],1)

    #     #cartesianBounds
    #     cartesianTransform=gmu.get_cartesian_transform(cartesianBounds=cartesianBounds)
    #     self.assertEqual(cartesianTransform[1,1],1)
    #     self.assertEqual(cartesianTransform[2,3],-4.5)
        
    
    def test_get_cartesian_transform(self):
        # Test cases
        test_cases = [
            {
                "name": "identity_transform",
                "rotation": None,
                "translation": None,
                "expected": np.eye(4)
            },
            {
                "name": "rotation_matrix", # Rotation of 90 degrees around the Z-axis
                "rotation": np.array([[0, -1, 0],
                                      [1, 0, 0],
                                      [0, 0, 1]]),
                "translation": None,
                "expected": np.array([[0, -1, 0, 0],
                                      [1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
            },
            {
                "name": "euler_angles",
                "rotation": (0,0,90),  # Rotation of 90 degrees around the Z-axis
                "translation": None,
                "expected": np.array([[0, -1, 0, 0],
                                      [1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
            },
            {
                "name": "translation_vector",
                "rotation": None,
                "translation": np.array([1, 2, 3]),
                "expected": np.array([[1, 0, 0, 1],
                                      [0, 1, 0, 2],
                                      [0, 0, 1, 3],
                                      [0, 0, 0, 1]])
            },
            {
                "name": "rotation_and_translation",
                "rotation": np.array([[0, -1, 0],
                                      [1, 0, 0],
                                      [0, 0, 1]]),
                "translation": np.array([1, 2, 3]),
                "expected": np.array([[0, -1, 0, 1],
                                      [1, 0, 0, 2],
                                      [0, 0, 1, 3],
                                      [0, 0, 0, 1]])
            }
        ]

        for case in test_cases:
            with self.subTest(case["name"]):
                result = gmu.get_cartesian_transform(rotation=case["rotation"], translation=case["translation"])
                np.testing.assert_array_almost_equal(result, case["expected"], decimal=6)
                
    def test_get_oriented_bounds(self):        
        box=self.dataLoaderRoad.mesh.get_axis_aligned_bounding_box()
        boxPoints=np.asarray(box.get_box_points())
        cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderRoad.mesh)
        boundingPoints=np.asarray(gmu.get_oriented_bounds(cartesianBounds)) 

        for i in range(0,7):
            for j in range(0,2):
                self.assertAlmostEqual(boundingPoints[i][j],boxPoints[i][j],delta=0.01)
        
    def test_get_box_inliers(self):
        mesh=self.dataLoaderParking.wallMesh.translate([0,0,3])
        wallBox=mesh.get_oriented_bounding_box()

        wallInliers= gmu.get_box_inliers(sourceBox=wallBox, testBoxes=self.dataLoaderParking.bimBoxes) 
        self.assertEqual(len(wallInliers),2)
        
    def test_get_box_intersections(self):
        mesh=self.dataLoaderParking.wallMesh.translate([0,0,3])
        wallBox=mesh.get_oriented_bounding_box()
        wallInliers= gmu.get_box_intersections(sourceBox=wallBox, testBoxes=self.dataLoaderParking.bimBoxes)
        self.assertEqual(len(wallInliers),2)
        
    def test_get_cartesian_bounds(self):
        #box
        box=self.dataLoaderRoad.mesh.get_oriented_bounding_box()
        minBounds=box.get_min_bound()
        maxBounds=box.get_max_bound()
        cartesianBounds=gmu.get_cartesian_bounds(box)
        self.assertEqual(minBounds[0],cartesianBounds[0])
        self.assertEqual(maxBounds[2],cartesianBounds[5])

        #mesh
        cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderRoad.mesh)
        minBounds=self.dataLoaderRoad.mesh.get_min_bound()
        maxBounds=self.dataLoaderRoad.mesh.get_max_bound()
        self.assertEqual(minBounds[0],cartesianBounds[0])
        self.assertEqual(maxBounds[2],cartesianBounds[5])

        #pointcloud
        cartesianBounds=gmu.get_cartesian_bounds(self.dataLoaderRoad.pcd)
        minBounds=self.dataLoaderRoad.pcd.get_min_bound()
        maxBounds=self.dataLoaderRoad.pcd.get_max_bound()
        self.assertEqual(minBounds[0],cartesianBounds[0])
        self.assertEqual(maxBounds[2],cartesianBounds[5])
        
    def test_get_triangles_center(self):
        mesh=self.dataLoaderRoad.mesh      
        triangleIndices=[0,1,2]    
        centers=gmu.get_triangles_center(mesh,triangleIndices)
        self.assertEqual(centers.size,9)
        self.assertAlmostEqual(centers[0][0],100632.19010416667,delta=0.01)
        self.assertAlmostEqual(centers[2][2],6.768118858337402,delta=0.01)
        
 
    def test_e57path_to_pcd(self):
        e57=pye57.E57(str(self.dataLoaderParking.e57Path1)) 
        header = e57.get_header(0)
        pcd=gmu.e57path_to_pcd(e57Path=self.dataLoaderParking.e57Path1, e57Index=0) 
        self.assertEqual(len(pcd.points) , header.point_count)

    def test_e57_to_arrays(self):
        e57=pye57.E57(str(self.dataLoaderParking.e57Path1)) 
        header = e57.get_header(0)
        tuple=gmu.e57_to_arrays(self.dataLoaderParking.e57Path1)
        self.assertEqual(len(tuple),5)
        self.assertEqual(len(tuple[0]),header.point_count)
        self.assertEqual(len(tuple[1]),header.point_count)
        #self.assertEqual(len(tuple[2]),header.point_count)

    def test_e57_to_pcd(self):
        e57=pye57.E57(str(self.dataLoaderParking.e57Path1)) 
        header = e57.get_header(0)
        pcd=gmu.e57_to_pcd(e57, percentage=0.5)
        self.assertEqual(len(pcd.points),int(header.point_count*0.5))

        e57=pye57.E57(str(self.dataLoaderParking.e57Path1)) 
        header = e57.get_header(1)
        pcd=gmu.e57_to_pcd(e57, percentage=0.5)
        self.assertAlmostEqual(len(pcd.points),int(header.point_count*0.5),delta=10000)

    def test_e57path_to_pcds_multiprocessing(self):
        e57=pye57.E57(str(self.dataLoaderParking.e57Path1)) 
        header1 = e57.get_header(0)
        pcds=gmu.e57path_to_pcds_multiprocessing(self.dataLoaderParking.e57Path1, percentage=0.5)
        self.assertEqual(len(pcds),2)        
        self.assertEqual(len(pcds[0].points),int(header1.point_count*0.5))

        #e57=pye57.E57(self.e57Path2) 
        #header1 = e57.get_header(1)
        #header2 = e57.get_header(2) # Header 2 is not found
        #pcds=gmu.e57path_to_pcds_multiprocessing(self.e57Path2, percentage=0.5)
        #self.assertEqual(len(pcds[0].points),int(header1.point_count*0.5))
        #self.assertEqual(len(pcds[1].points),int(header2.point_count*0.5))

    def test_pcd_to_arrays(self):
        tuple=gmu.pcd_to_arrays(self.dataLoaderRoad.pcdPath, percentage=0.5)
        self.assertEqual(len(tuple),4)
        self.assertEqual(len(tuple[0]),int(len(self.dataLoaderRoad.pcd.points)*0.5))
        self.assertEqual(len(tuple[1]),int(len(self.dataLoaderRoad.pcd.points)*0.5))
        self.assertEqual(len(tuple[2]),int(len(self.dataLoaderRoad.pcd.points)*0.5))        
        self.assertEqual(tuple[3],0)

    def test_mesh_to_arrays(self):
        tuple=gmu.mesh_to_arrays(self.dataLoaderRoad.meshPath)
        self.assertEqual(len(tuple),5)
        self.assertEqual(len(tuple[0]),len(self.dataLoaderRoad.mesh.vertices))
        self.assertEqual(len(tuple[1]),len(self.dataLoaderRoad.mesh.triangles))
        self.assertEqual(len(tuple[2]),len(self.dataLoaderRoad.mesh.vertex_colors))
        self.assertEqual(len(tuple[3]),len(self.dataLoaderRoad.mesh.vertex_normals))        
        self.assertEqual(tuple[4],0)

    def test_e57_get_colors(self):
        e57=pye57.E57(str(self.dataLoaderRoad.e57Path)) 
        gmu.e57_update_point_field(e57)
        raw_data = e57.read_scan_raw(0)  
        header = e57.get_header(0)
        self.assertEqual(len(raw_data["cartesianX"]) , header.point_count)

        colors=gmu.e57_get_colors(raw_data)
        self.assertEqual(len(colors),len(raw_data["cartesianX"]))
        self.assertEqual(len(colors),header.point_count)
        
    def test_crop_geometry_by_box(self):
        #test point cloud
        box=self.dataLoaderRoad.mesh.get_oriented_bounding_box()
        pcd=gmu.crop_geometry_by_box(self.dataLoaderRoad.pcd, box) 
        self.assertIsInstance(pcd,o3d.geometry.PointCloud)
        self.assertGreater(len(pcd.points),600000)

        #test mesh
        box=self.dataLoaderRoad.pcd.get_oriented_bounding_box()
        mesh=gmu.crop_geometry_by_box(self.dataLoaderRoad.mesh, box, subdivide = 0)
        self.assertIsInstance(mesh,o3d.geometry.TriangleMesh)
        self.assertGreater(len(mesh.vertices),10000)


    def test_get_mesh_inliers(self):
        #mesh
        sources=[self.dataLoaderParking.slabMesh,self.dataLoaderParking.wallMesh]
        indices=gmu.get_mesh_inliers(sources=sources,reference=self.dataLoaderParking.mesh)
        self.assertEqual(len(indices),1)
        
        #pcd
        sources=[self.dataLoaderParking.slabMesh,self.dataLoaderParking.wallMesh]
        indices=gmu.get_mesh_inliers(sources=sources,reference=self.dataLoaderParking.pcd)
        self.assertEqual(len(indices),1)

    def test_expand_box(self):
        cartesianBounds=np.array([0,10,7,8.3,-2,2])       
        box=gmu.get_oriented_bounding_box(cartesianBounds)
        #positive expansion
        expandedbox1=gmu.expand_box(box,u=5,v=3,w=1)
        self.assertEqual(expandedbox1.extent[0],box.extent[0]+5)
        self.assertEqual(expandedbox1.extent[1],box.extent[1]+3)
        self.assertEqual(expandedbox1.extent[2],box.extent[2]+1)
        
        #negate expansion
        expandedbox2=gmu.expand_box(box,u=-1,v=-1,w=-1)
        self.assertEqual(expandedbox2.extent[0],box.extent[0]-1)
        self.assertEqual(expandedbox2.extent[1],box.extent[1]-1)
        self.assertEqual(expandedbox2.extent[2],box.extent[2]-1)
            
    def test_join_geometries(self):
        #TriangleMesh
        mesh1=self.dataLoaderRoad.mesh
        mesh2=copy.deepcopy(mesh1)
        mesh2.translate([5,0,0])
        joinedMeshes= gmu.join_geometries([mesh1,mesh2]) 
        self.assertEqual(len(joinedMeshes.triangles), (len(mesh1.triangles)+len(mesh2.triangles)))
        self.assertTrue(joinedMeshes.has_vertex_colors())

        #PointCloud
        pcd1=self.dataLoaderRoad.pcd
        pcd2=copy.deepcopy(pcd1)
        pcd2.translate([5,0,0])
        joinedpcd= gmu.join_geometries([pcd1,pcd2]) 
        self.assertEqual(len(joinedpcd.points), (len(pcd1.points)+len(pcd2.points)))
        self.assertTrue(joinedpcd.has_colors())
        
    def test_crop_geometry_by_distance(self):
        sourcepcd=self.dataLoaderRoad.pcd
        sourceMesh=self.dataLoaderRoad.mesh
        cutterMeshes=self.dataLoaderRoad.bimMeshes
        # mesh + [mesh]
        result1=gmu.crop_geometry_by_distance(source=sourceMesh,reference=cutterMeshes)
        self.assertGreater(len(result1.vertices),30 )

        # pcd + [mesh]
        result2=gmu.crop_geometry_by_distance(source=sourcepcd,reference=cutterMeshes)
        self.assertGreater(len(result2.points),3000 ) 

        # mesh + pcd
        result3=gmu.crop_geometry_by_distance(source=sourceMesh,reference=sourcepcd)
        self.assertGreater(len(result3.vertices),30 ) 

        # pcd + mesh
        result4=gmu.crop_geometry_by_distance(source=sourcepcd,reference=sourceMesh)
        self.assertLess(len(result4.points),1200000 ) 
    
        
    # def test_crop_geometry_by_convex_hull(self):
    #     sourceMesh=gmu.mesh_to_trimesh(self.dataLoaderParking.mesh)
    #     cutters=[gmu.mesh_to_trimesh(mesh) for mesh in self.dataLoaderParking.bimMeshes]

    #     innerCrop=gmu.crop_mesh_by_convex_hull(source=sourceMesh, cutters=cutters, inside = True )
    #     self.assertEqual(len(innerCrop),1)
    #     self.assertGreater(len(innerCrop[0].vertices),7600)

    #     outerCrop=gmu.crop_mesh_by_convex_hull(source=sourceMesh, cutters=cutters[0], inside = False ) 
    #     self.assertGreater(len(outerCrop[0].vertices),29000)         
        
    def test_get_translation(self):
        box=self.dataLoaderRoad.mesh.get_oriented_bounding_box()
        centerGt=box.get_center()

        #cartesianBounds
        cartesianBounds=gmu.get_cartesian_bounds(box)
        center=gmu.get_translation(cartesianBounds)
        self.assertAlmostEqual(math.dist(centerGt,center),0,delta=0.01)

        #orientedBounds
        orientedBounds= np.asarray(box.get_box_points())
        center=gmu.get_translation(orientedBounds)
        self.assertAlmostEqual(math.dist(centerGt,center),0,delta=0.01)

        #cartesianTransform
        center=gmu.get_translation(self.dataLoaderRoad.imageCartesianTransform2)
        self.assertAlmostEqual(self.dataLoaderRoad.imageCartesianTransform2[0][3],center[0],delta=0.01)
       
    # def test_get_mesh_collisions_trimesh(self):
    #     inliers=gmu.get_mesh_collisions_trimesh(sourceMesh=self.dataLoaderParking.mesh ,
    #                                             geometries =self.dataLoaderParking.bimMeshes) 
    #     self.assertEqual(len(inliers),1 )

    # def test_get_pcd_collisions(self):
    #     inliers=gmu.get_pcd_collisions(sourcePcd=self.dataLoaderParking.pcd, 
    #                                    geometries =self.dataLoaderParking.bimMeshes)
    #     self.assertLess(len(inliers),150 )

    def test_get_rotation_matrix(self):        
  

        #cartesianTransform
        r1=gmu.get_rotation_matrix(self.dataLoaderRoad.imageCartesianTransform1)
        self.assertAlmostEqual(r1[0][0],self.dataLoaderRoad.imageCartesianTransform1[0][0], delta=0.01)

        # #Euler angles
        # r2=gmu.get_rotation_matrix(np.array([-121.22356551,   83.97341873,   -7.21021157]))
        # self.assertAlmostEqual(r2[0][0],self.image2CartesianTransform[0][0], delta=0.01)
        
        # #quaternion
        # r3=gmu.get_rotation_matrix(np.array([ 0.60465535, -0.28690085 , 0.66700836 ,-0.32738304]))
        # self.assertAlmostEqual(r3[0][0],self.image2CartesianTransform[0][0], delta=0.01)

        # #orientedBounds
        # box=self.dataLoaderRoad.mesh.get_oriented_bounding_box()
        # r4=gmu.get_rotation_matrix(np.asarray(box.get_box_points()))
        # self.assertAlmostEqual(r4[0][0],rotationGt[0][0], delta=0.01)

    

if __name__ == '__main__':
    unittest.main()
