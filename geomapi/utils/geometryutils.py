"""
Geometryutils - a Python library for processing mesh and point cloud data.
"""
import concurrent.futures
import copy
import math
from typing import List, Tuple
import pandas as pd
import sys
import itertools
from pathlib import Path
from sklearn.neighbors import NearestNeighbors # to compute nearest neighbors
import laspy # this is to process las point clouds
import cv2
import geomapi.utils as ut
from pathlib import Path 

import geomapi.utils.imageutils as iu #! this might be a problem

import ifcopenshell
import ifcopenshell.geom as geom
import numpy as np
import open3d as o3d
import pye57
import trimesh
from scipy.spatial.transform import Rotation as R

pye57.e57.SUPPORTED_POINT_FIELDS.update({'nor:normalX' : 'd','nor:normalY': 'd','nor:normalZ': 'd'})

def array_to_colors(array: np.array, colors:np.array=None) -> np.array:
    """Map colors according to the unique values in the array.\n

    Args:
        1. array (np.array(n,1)): array with scalars e.g. predictions.\n
        2. colors (np.array(n,3)): e.g. np.array([[1,0,0],[0,1,0]]). colors are automatically mapped from [0,1].\n

    Returns:
        np.array(n,3)
    """
    values=np.unique(array)

    #validate inputs
    colors=colors if colors is not None else np.array([ut.get_random_color() for v in range(len(values))])
    assert colors.shape[1] == 3, f'colors.shape[1] != 3, got {colors.shape[1]}'
    assert colors.shape[0] == len(values),f'colors.shape[1] != values.shape[1], got {values.shape[1]}'
    colors=np.c_[colors/255]  if np.amax(colors)>1 else colors
        
    #map colors
    colorArray=np.empty((array.shape[0],3))
    for v,c in zip(values,colors):    
        colorArray[array==v]=c
    return colorArray

def array_subsample(array:np.ndarray,percentage:float=0.1)->np.ndarray:
    """subsample rows of np.array

    Args:
        array (np.ndarray): 
        percentage (float, optional): downsampling percentage. Defaults to 0.1.

    Returns:
        np.ndarray: output array
    """
    #create mask of True and False
    size=math.ceil(array.shape[0]*percentage)
    choice = np.random.choice(range(array.shape[0]), size=(size,), replace=False)    
    ind = np.zeros(array.shape[0], dtype=bool)
    ind[choice] = True
    
    #mask the array
    return array[ind,:] 

def arrays_to_mesh(tuple) -> o3d.geometry.TriangleMesh:
    """Returns TriangleMesh from arrays.\n

    Args:
        tuple (Tuple): \n
            1. vertexArray:np.array \n
            2. triangleArray:np.array \n
            3. (optional) colorArray:np.array \n
            4. (optional) normalArray:np.array \n

    Returns:
        o3d.geometry.PointCloud
    """ 
    if (len(tuple) < 2):
        raise ValueError("The tuple should contain at least verteces and triangles")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tuple[0])
    mesh.triangles = o3d.utility.Vector3iVector(tuple[1])

    if len(tuple) > 2:
        mesh.vertex_colors = o3d.utility.Vector3dVector(tuple[2])
    if len(tuple) > 3:
        mesh.vertex_normals = o3d.utility.Vector3dVector(tuple[3])
    return mesh

def box_to_mesh(box:o3d.geometry) ->o3d.geometry.TriangleMesh:
    """Returns o3d.geometry.TriangleMesh of an OrientedBoundingBox or AxisAlignedBoundingBox. \n

    Args:
        box (o3d.geometry.OrientedBoundingBox or AxisAlignedBoundingBox).

    Returns:
        o3d.geometry.TriangleMesh
    """
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=box.get_box_points()
    #triangles rotate counterclockwise
    mesh.triangles= o3d.utility.Vector3iVector(np.array([[0,2,1],
                        [0,1,3],
                        [0,3,2],
                        [1,6,3],
                        [1,7,6],
                        [1,2,7],
                        [2,3,5],
                        [2,5,4],
                        [2,4,7],
                        [3,4,5],
                        [3,6,4],
                        [4,6,7]])) 
    return mesh 
                  

def color_by_intensity(pcd:o3d.geometry.PointCloud, intensities:np.array) -> o3d.geometry.PointCloud:
    """ Colorize a o3d.geometry.PointCloud with a numpy array of intesities.
    The intensties are assumed to have a maximum value of 65535.
    
    Args:
        1. pcd (o3d.geometry.PointCloud): Point Cloud to colorize.
        2. intensities (ndarray): (mx1) values [0-65535].
        
    Returns
        o3d.geometry.PointCloud     
    """
    assert intensities.shape[0] == np.asarray(pcd.points).shape[0], f'length intensities ({intensities.shape[0]}) differs from pcd ({np.asarray(pcd.points).shape[0]})'
    
    # If the intensity array is RGB encoded, first normalize it using opencv to 16-bit precision
    intensities_norm = cv2.normalize(intensities, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

    # Using numpy.column_stack() provide equal values to RGB values and then assign to 'colors' property 
    # of the point cloud.
    # Since Open3D point cloud's 'colors' property is float64 array of shape (num_points, 3), range [0, 1],
    # we have to normalize the intensity array by dividing it by 65535
    colors = np.column_stack((intensities_norm, intensities_norm, intensities_norm)) / 65535

    # Now colors array is grayscale array (r = g = b)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def compute_nearest_neighbors(query_points:np.ndarray,
                              reference_points:np.ndarray,
                              query_normals:np.ndarray = None, 
                              reference_normals:np.ndarray = None, 
                              n:int=5,
                              distanceThreshold=None)->Tuple[np.ndarray,np.ndarray]:
    """Compute index and distance to nearest neighboring point in the reference dataset.\n
    if the normals are given, it uses them to apply a normal filtering
    For the normal filtering, the n closest neighbors are considered of which the correspondence with the best matching normal is retained. \n

    **NOTE**:  The index of outliers is set to -1 if distanceTreshold is not None.\n
    
    Args:
        1. query_points (np.array[n,3]): points to evaluate.\n
        2. query_normals (np.array[n,3]): normals to evaluate.\n
        3. reference_points (np.array[n,3]): reference points.\n
        4. reference_normals (np.array[n,3]): reference normals.\n
        5. n (int, optional): number of neighbors.\n
        5. distanceTreshold (_type_, optional): Distance threshold for the nearest neighbors.. Defaults to None.

    Returns:
        Tuple[np.array,np.array]: indices, distances 
    """
    #compute nearest neighbors 
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(reference_points)
    distances,indices, = nbrs.kneighbors(query_points)
    # apply normal filtering if the normals are given
    if(query_normals is not None and reference_normals is not None):
        #compute dotproduct
        dotproducts=np.empty((indices.shape))
        for i,ind in enumerate(np.hsplit(indices,indices.shape[1])):
            dotproducts[:, i]= np.einsum('ij,ij->i', np.take(reference_normals, ind.flatten().T, axis=0), query_normals)
        #select index with highest dotproduct
        ind=np.argmax(np.absolute(dotproducts), axis=1)
        indices = indices[np.arange(indices.shape[0]), ind]
        distances = distances[np.arange(distances.shape[0]), ind]        
        
        #filter distances   
        if distanceThreshold is not None:
            assert distanceThreshold>0 ,f'distanceTreshold should be positive, got {distanceThreshold}'
            indices=np.where(distances>distanceThreshold,-1,indices)
            distances=np.where(distances>distanceThreshold,-1,distances)  
        
    return indices,distances

def compute_raycasting_collisions(geometries:List[o3d.geometry.Geometry],rays:np.array)->Tuple[np.array,np.array]:
    """Compute the collisions between a set of Open3D geometries and rays.\n

    .. image:: ../../../docs/pics/Raycasting_1.PNG
    .. image:: ../../../docs/pics/boxes1.PNG
    
    Args:
        1.geometries (List[o3d.geometry.Geometry]): A set of Point clouds or meshes. \n
        2.rays (np.array[m,n,6]): Set of rays (1D or 2D) with 6 columns as the last dimension. A ray consists of a startpoint np.array[n,0:3] and a direction np.array[n,3:6]\n

    Returns:
        Tuple[np.array,np.array]: distances (inf if not hit), geometry_ids (4294967295 if not hit)
    """
    #validate rays
    if 'Tensor' in str(type(rays)):
        assert rays.shape[-1]==6, f'rays.shape[-1] should be 6, got {rays.shape[1]}.'
    if 'array' in str(type(rays)):        
        assert rays.shape[-1]==6, f'rays.shape[-1] should be 6, got {rays.shape[1]}.'
        rays = o3d.core.Tensor(rays,dtype=o3d.core.Dtype.Float32)  
      
    #create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()

    #validate geometries and add them to the scene
    for g in ut.item_to_list(geometries):
        reference=o3d.t.geometry.TriangleMesh.from_legacy(g)
        scene.add_triangles(reference)
    
    #compute raycasting
    ans = scene.cast_rays(rays)
    return ans['t_hit'].numpy(),ans['geometry_ids'].numpy()

def convert_to_homogeneous_3d_coordinates(input_data: list | np.ndarray) -> np.ndarray:
    """
    Converts 3D Cartesian coordinates into homogeneous coordinates or normalizes 
    existing homogeneous coordinates.

    Args:
        input_data (list or numpy.ndarray): The input data representing 3D coordinates.
                                            Each row should have either 3 (Cartesian) 
                                            or 4 (homogeneous) elements.

    Returns:
        numpy.ndarray: A 2D NumPy array where:
            - If input has 3 columns, a fourth column of ones is added.
            - If input has 4 columns, all elements are normalized by the last column.
            - Otherwise, a ValueError is raised.
    """
    # Convert to 2D array
    input_data = ut.map_to_2d_array(input_data)

    # Convert Cartesian coordinates to homogeneous
    if input_data.shape[1] == 3:
        homogeneous_column = np.ones((input_data.shape[0], 1))
        input_data = np.hstack((input_data, homogeneous_column))
    elif input_data.shape[1] == 4:
        # Normalize by the last coordinate
        input_data = input_data / input_data[:, -1][:, np.newaxis]
    else:
        raise ValueError("Each coordinate should have either 3 or 4 elements.")

    return input_data

def create_camera_frustum_mesh(transform, focal_length_35mm, depth=5.0):
    """
    Creates a 3D camera frustum mesh as a triangular pyramid using 35mm-equivalent focal length.

    Parameters:
    -----------
    transform : np.ndarray, shape (4, 4)
        A 4x4 transformation matrix that places the camera frustum in world coordinates.

    focal_length_35mm : float
        Camera focal length in millimeters assuming a standard full-frame (36mm x 24mm) sensor.
        Used to compute the field of view of the frustum.

    depth : float, optional (default=5.0)
        The distance from the camera origin to the far image plane along the Z-axis.

    Returns:
    --------
    mesh : o3d.geometry.TriangleMesh
        A triangular mesh representing the camera frustum as a 3D pyramid.

    Notes:
    ------
    - The frustum is constructed with its apex at the origin [0, 0, 0] and a rectangular base at the given depth.
    - The field of view is calculated from the sensor size and focal length.
    - The mesh includes triangular sides and a back face (the image plane), and is colored uniformly.
    - The result is useful for visualizing camera poses and orientations in 3D.

    Example:
    --------
    frustum_mesh = create_camera_frustum_mesh(np.eye(4), focal_length_35mm=50.0)
    o3d.visualization.draw_geometries([frustum_mesh])
    """
    # Standard full-frame sensor size (in mm)
    sensor_width = 36.0
    sensor_height = 24.0

    # Compute field of view based on 35mm focal length
    fov_x = 2 * math.atan((sensor_width / 2) / focal_length_35mm)
    fov_y = 2 * math.atan((sensor_height / 2) / focal_length_35mm)

    # Calculate half-width and half-height at far plane
    half_width = depth * math.tan(fov_x / 2)
    half_height = depth * math.tan(fov_y / 2)

    # Define vertices of the frustum pyramid
    vertices = np.array([
        [0, 0, 0],  # Camera origin

        [-half_width, -half_height, depth],  # Far bottom-left
        [ half_width, -half_height, depth],  # Far bottom-right
        [ half_width,  half_height, depth],  # Far top-right
        [-half_width,  half_height, depth],  # Far top-left
    ])

    # Define triangles (faces of the pyramid)
    triangles = [
        [0, 1, 2],  # Bottom face
        [0, 2, 3],  # Right face
        [0, 3, 4],  # Top face
        [0, 4, 1],  # Left face
        [1, 2, 3], [1, 3, 4],  # Back (far) face (optional)
    ]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Optional: set color
    mesh.paint_uniform_color([0.8, 0.2, 0.2])

    # Optional: compute normals
    mesh.compute_vertex_normals()

    # Transform to world space
    mesh.transform(transform)

    return mesh

def create_camera_frustum_mesh_with_image(transform, image_width, image_height, focal_length_35mm, depth=5.0, image_cv2=None):
    """
    Creates a 3D camera frustum as an Open3D LineSet, and optionally a textured image plane at the frustum's far end.

    Parameters:
    -----------
    transform : np.ndarray, shape (4, 4)
        A 4x4 transformation matrix representing the camera's pose in world coordinates.

    image_width : int
        Width of the image in pixels. Used to determine the aspect ratio of the camera's sensor.

    image_height : int
        Height of the image in pixels. Used to determine the aspect ratio of the camera's sensor.

    focal_length_35mm : float
        Camera focal length in millimeters, assuming a 35mm-equivalent sensor width of 36mm.
        This is used to compute the field of view and shape of the frustum.

    depth : float, optional (default=5.0)
        Distance from the camera origin to the far/image plane along the Z-axis (i.e., frustum depth).

    image_cv2 : np.ndarray, optional
        An OpenCV (NumPy) RGB or BGR image. If provided, a textured mesh is created at the far end
        of the frustum using the image dimensions for accurate aspect ratio.

    Returns:
    --------
    line_set : o3d.geometry.LineSet
        A LineSet representing the frustum's edges.

    image_mesh : o3d.geometry.TriangleMesh or None
        A textured TriangleMesh showing the input image at the far end of the frustum.
        Returns None if `image_cv2` is not provided.

    Notes:
    ------
    - The frustum assumes a perspective camera with a virtual sensor width of 36mm.
    - The image plane is placed at `depth` units away from the camera origin along the local Z-axis.
    - This function is useful for visualizing camera poses and projected views in 3D scenes.

    Example:
    --------
    frustum, img_plane = create_camera_frustum_mesh(transform, w, h, 50.0, depth=5.0, image_cv2=cv2_img)
    o3d.visualization.draw_geometries([frustum, img_plane] if img_plane else [frustum])
    """
    
    # Use 36mm width as the base (standard full-frame width)
    sensor_width_mm = 36.0
    sensor_aspect = image_height / image_width
    sensor_height_mm = sensor_width_mm * sensor_aspect

    # Compute field of view in radians
    fov_x = 2 * math.atan((sensor_width_mm / 2) / focal_length_35mm)
    fov_y = 2 * math.atan((sensor_height_mm / 2) / focal_length_35mm)

    # Compute half width and height at the given depth
    half_width = depth * math.tan(fov_x / 2)
    half_height = depth * math.tan(fov_y / 2)

    # Define frustum vertices
    vertices = np.array([
        [0, 0, 0],  # Camera origin

        [-half_width, -half_height, depth],  # Far bottom-left
        [ half_width, -half_height, depth],  # Far bottom-right
        [ half_width,  half_height, depth],  # Far top-right
        [-half_width,  half_height, depth],  # Far top-left
    ])

    # Frustum wireframe connections
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])  # Blue

    # Apply transformation
    line_set.transform(transform)

    image_mesh = None
    if image_cv2 is not None:
        # Get image dimensions and aspect ratio
        img_height, img_width = image_cv2.shape[:2]
        aspect_ratio = img_width / img_height

        # Match the frustum's field of view while preserving aspect ratio
        img_plane_half_height = half_height
        img_plane_half_width = img_plane_half_height * aspect_ratio

        # Image plane vertices (quad)
        plane_vertices = np.array([
            [img_plane_half_width, -img_plane_half_height, depth],  # Bottom-left
            [ -img_plane_half_width, -img_plane_half_height, depth],  # Bottom-right
            [ -img_plane_half_width,  img_plane_half_height, depth],  # Top-right
            [img_plane_half_width,  img_plane_half_height, depth],  # Top-left
        ])

        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ])

        # create the uv coordinates
        v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                 [0, 1], [1, 0], [0, 0]])

        # Convert OpenCV image (BGR to RGB if needed)
        o3d_image = o3d.geometry.Image(image_cv2)

        image_mesh = o3d.geometry.TriangleMesh()
        image_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
        image_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        image_mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
        image_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))
        image_mesh.textures = [o3d_image]
        image_mesh.compute_vertex_normals()
        image_mesh.transform(transform)

    return line_set, image_mesh

def create_ellipsoid_mesh(radii: np.ndarray, transformation: np.ndarray, resolution: int = 30):
    """
    Create an Open3D TriangleMesh of an ellipsoid with a given set of radii and transformation matrix.

    .. image:: ../../../docs/pics/ellipsoid.jpg

    Args:
        - radii: A numpy array [a,b,c] representing the radii of the ellipsoid along the primary, secondary and tertiary axes.
        - transformation: A 4x4 transformation matrix (numpy array) to apply to the ellipsoid mesh.
        - resolution: The resolution of the mesh (default 30).
    
    Returns
        - An Open3D TriangleMesh object representing the ellipsoid.
    """
    # Create a parametric grid for the ellipsoid
    u = np.linspace(0, 2 * np.pi, resolution)  # azimuth angle
    v = np.linspace(0, np.pi, resolution)      # polar angle
    u, v = np.meshgrid(u, v)

    # Parametric equations of an ellipsoid
    a = radii[0] * np.sin(v) * np.cos(u)
    b = radii[1] * np.sin(v) * np.sin(u)
    c = radii[2] * np.cos(v)

    # Flatten the arrays and stack them into Nx3 shape for vertices
    vertices = np.stack((a.flatten(), b.flatten(), c.flatten()), axis=-1)

    # Create faces (triangles) using the mesh grid
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = idx1 + resolution
            faces.append([idx1, idx1 + 1, idx2])
            faces.append([idx2, idx1 + 1, idx2 + 1])
    faces = np.asarray(faces)

    # Create the Open3D TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()

    # Apply the transformation matrix to the vertices
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    transformed_vertices = (transformation @ vertices_homogeneous.T).T[:, :3]  # Apply transformation

    # Set the vertices and triangles (faces)
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute the normals and update the mesh
    mesh.compute_vertex_normals()

    return mesh

def create_identity_point_cloud(geometries: o3d.geometry.PointCloud, resolution:float = 0.1, getNormals=False) -> Tuple[o3d.geometry.PointCloud, np.array]:
    """Returns a sampled point cloud colorized per object of a set of objects along with an array with an identifier for each point.

    TODO: MB also store normals 
    
    Args:
        1.geometries (o3d.geometry.PointCloud or o3d.geometry.TriangleMesh) \n
        2.resolution (float, optional): (down)sampling resolution for the point cloud. Defaults to 0.1.\n

    Raises:
        ValueError: Geometries must be o3d.geometry (PointCloud or TriangleMesh)

    Returns:
        Tuple[colorized point cloud (o3d.geometry.PointCloud), identityArray(np.array)] per geometry
    """
    geometries=ut.item_to_list(geometries)    
    colorArray=np.random.random((len(geometries),3))
    indentityArray=None
    identityPointCloud=o3d.geometry.PointCloud()

    for i,geometry in enumerate(geometries):
        if 'PointCloud' in str(type(geometry)) :
            pcd=geometry.voxel_down_sample(resolution)
            get_points_and_normals(pcd,getNormals=getNormals) if getNormals else None
            indentityArray=np.vstack((indentityArray,np.full((len(pcd.points), 1), i)))
            pcd.paint_uniform_color(colorArray[i])
            identityPointCloud +=pcd
            # np.concatenate((np.asarray(identityPointCloud.points),np.asarray(identityPointCloud.points)),axis=0)
        elif 'TriangleMesh' in str(type(geometry)):
            area=geometry.get_surface_area()
            count=int(area/(resolution*resolution))
            if count>0:
                count=count if count>0 else len(np.asarray(geometry.vertices))
                pcd=geometry.sample_points_uniformly(number_of_points=count, use_triangle_normal=getNormals)
            else:
                pcd=o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(np.array([geometry.get_center()]))
                
            indentityArray=np.vstack((indentityArray,np.full((len(pcd.points), 1), i)))
            pcd.paint_uniform_color(colorArray[i])
            identityPointCloud +=pcd
        else:
            print(f'{geometry} is invalid')
            continue
    indentityArray=indentityArray.flatten()
    indentityArray=np.delete(indentityArray,0)
    return identityPointCloud, indentityArray

def create_obb_from_orthophoto(cartesian_transform, image_width, image_height, gsd, depth):
    """
    Creates an oriented bounding box (OBB) for an orthographic image,
    with the origin at the center of the back face (z = 0) of the box.

    Parameters:
        cartesian_transform (np.ndarray): 4x4 transformation matrix.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        gsd (float): Ground sampling distance (meters per pixel).
        depth (float): Depth (thickness) of the image in meters.

    Returns:
        o3d.geometry.OrientedBoundingBox: The computed OBB.
    """
    # Convert image size from pixels to meters
    width_m = image_width * gsd
    height_m = image_height * gsd

    # Define center of back face (at z = 0)
    local_center = np.array([0,0, depth / 2])

    # Create box centered at that point
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = (cartesian_transform @ np.append(local_center, 1))[:3]
    obb.extent = np.array([width_m, height_m, depth])
    obb.R = cartesian_transform[:3, :3]  # Extract rotation

    return obb

def create_transform_from_pyramid_points(points, tol=1e-6):
    """
    Given 5 points of a pyramid (4 base + 1 tip, unordered),
    find the tip and compute a transform with:
    - origin at tip
    - forward toward base center
    - up aligned as closely as possible to world_up
    """
    assert points.shape == (5, 3), "Input must be a 5x3 array of 3D points."
    world_up = np.array([0, 1, 0])

    # Step 1: Find 4 coplanar points
    def are_coplanar(p1, p2, p3, p4):
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p4 - p1
        volume = np.abs(np.dot(np.cross(v1, v2), v3))
        return volume < tol

    coplanar_idxs = None
    for idxs in itertools.combinations(range(5), 4):
        if are_coplanar(*points[list(idxs)]):
            coplanar_idxs = list(idxs)
            break
    if coplanar_idxs is None:
        raise ValueError("No 4 coplanar points found.")

    # Step 2: Identify the 5th point
    all_idxs = set(range(5))
    fifth_idx = list(all_idxs - set(coplanar_idxs))[0]
    origin = points[fifth_idx]
    rect = points[coplanar_idxs]

    # Step 3: Find the edge closest to world up → use as 'up'
    best_up = None
    best_alignment = -np.inf
    for i in range(4):
        a = rect[i]
        b = rect[(i + 1) % 4]
        edge = b - a
        edge_norm = edge / np.linalg.norm(edge)
        alignment = np.abs(np.dot(edge_norm, world_up))
        if alignment > best_alignment:
            best_alignment = alignment
            best_up = edge_norm
    up = best_up

    # Step 4: Compute forward vector (toward center of plane)
    center = np.mean(rect, axis=0)
    forward = center - origin
    forward /= np.linalg.norm(forward)

    # Step 5: Compute right and re-orthogonalized up
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    # Step 6: Assemble transform matrix
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = forward
    transform[:3, 3] = origin

    return transform

def create_xyz_grid(bounds:List[float], resolutions:List[float])->np.ndarray:
    """Generate a xyz grid. If only a single value is needed, set the boundaries equal e.g. xMin=xMax and the resolution to 1 e.g. dx=1.\n

    Args:
        1.bounds (List[float]): [xMin,xMax,yMin,yMax,zMin,zMax]\n
        2.resolutions (List[float]):[dx,dy,dz]\n

    Returns:
        np.array(x,y,z)
    """
    assert(len(bounds) == 2*len(resolutions))

    values=[]
    for i,res in enumerate(resolutions):
        #fix single values
        if bounds[2*i]==bounds[2*i+1]:
            bounds[2*i+1]+=1
            res+=1
        #generate values       
        values.append( np.arange(bounds[2*i], bounds[2*i+1], res )) 
    grid  = np.meshgrid(values[0],values[1],values[2])
    return np.stack(grid)
def crop_dataframe_from_meshes(df: pd.DataFrame,meshes:List[o3d.geometry.TriangleMesh]) -> List[pd.DataFrame]:
    """Crop point cloud and divide the inliers per waterthight mesh. \n

    Args:
        1. dataFrame (pd.DataFrame): Pandas dataframe with first three columns as [X,Y,Z].\n
        2. meshes (o3d.geometry.TriangleMesh): meshes to test the inliers.\n

    Returns:
        List[o3d.geometry.PointCloud] 
    """
    meshes=ut.item_to_list(meshes)
    assert df.shape[0] >0
    assert 'TriangleMesh' in str(type(meshes[0]))    

    pcd=dataframe_to_pcd(df,pointFields=['x', 'y', 'z'])
    _,indices=crop_point_cloud_from_meshes(pcd,meshes)
    newDataFrames=[None]*len(meshes)
    for i,list in enumerate(indices):
        if len(list) != 0:
            newDataFrames[i]=df.loc[list]
    return newDataFrames
def crop_geometry_by_box(geometry:o3d.geometry, box:o3d.geometry.OrientedBoundingBox, subdivide:int = 0) ->o3d.geometry:
    """Crop portion of a mesh/pointcloud that lies within an OrientedBoundingBox.\n

    .. image:: ../../../docs/pics/selection_BB_mesh2.PNG

    Args:
        1. geometry (o3d.geometry.TriangleMesh or o3d.geometry.PointCloud): Geometry to be cropped\n
        2. box (o3d.geometry.OrientedBoundingBox): bouding cutter geometry\n
        3. subdivide (int): number of interations to increase the density of the mesh (1=x4, 2=x16, etc.)\n
        
    Returns:
        o3d.geometry.TriangleMesh or None
    """
    # transform box to axis aligned box 
    r=box.R
    t=box.center
    transformedbox=copy.deepcopy(box)
    transformedbox=transformedbox.translate(-t)
    transformedbox=transformedbox.rotate(r.transpose(),center=(0, 0, 0))
    
    # transform geometry to coordinate system of the box
    transformedGeometry=copy.deepcopy(geometry)
    transformedGeometry=transformedGeometry.translate(-t)
    transformedGeometry=transformedGeometry.rotate(r.transpose(),center=(0, 0, 0))

    # convert to pcd if geometry is a mesh (crop fails with mesh geometry)
    if type(geometry) is o3d.geometry.PointCloud:
        croppedGeometry=transformedGeometry.crop(transformedbox)             
    elif type(geometry) is o3d.geometry.TriangleMesh:
        if subdivide!=0:
            transformedGeometry=transformedGeometry.subdivide_midpoint(subdivide)
        indices=transformedbox.get_point_indices_within_bounding_box(transformedGeometry.vertices) # this is empty
        if len(indices) !=0:
            croppedGeometry=transformedGeometry.select_by_index(indices,cleanup=True)
        else:
            return None

    # return croppedGeometry to original position
    if croppedGeometry is not None:
        croppedGeometry=croppedGeometry.rotate(r,center=(0, 0, 0))
        croppedGeometry=croppedGeometry.translate(t)
        return croppedGeometry
    else:
        return None

def crop_geometry_by_distance(source: o3d.geometry.Geometry, reference:List[o3d.geometry.Geometry], threshold : float =0.1) -> o3d.geometry.PointCloud:
    """Returns the portion of a pointcloud that lies within a range of another mesh/point cloud.\n
    
    .. image:: ../../../docs/pics/crop_by_distance2.PNG

    Args:
        1. source (o3d.geometry.TriangleMesh or o3d.geometry.PointCloud) : point cloud to filter \n
        2. cutters (o3d.geometry.TriangleMesh or o3d.geometry.PointCloud): list of reference data \n
        3. threshold (float, optional): threshold Euclidean distance for the filtering.Defaults to 0.1m. \n

    Returns:
        o3d.geometry (TriangleMesh or PointCloud)
    """
    #validate inputs
    reference=join_geometries(ut.item_to_list(reference))
    assert threshold>0 ,f'threshold>0 expected, got: {threshold}'

    #sample reference if a mesh
    reference=reference.sample_points_uniformly(number_of_points=1000000) if 'TriangleMesh' in str(type(reference)) else reference
    #sample source if a mesh
    if type(source) is o3d.geometry.PointCloud:
        sourcePcd=source    
        #compute distance
        distances=sourcePcd.compute_point_cloud_distance(reference)
        #remove vertices > threshold
        ind=np.where(np.asarray(distances) <= threshold)[0]
        selectedPcd=source.select_by_index(ind) if ind.size >0 else None
    else:
        sourcePcd=o3d.geometry.PointCloud()
        sourcePcd.points=o3d.utility.Vector3dVector(np.asarray(source.vertices))
        #compute distance
        distances=sourcePcd.compute_point_cloud_distance(reference)
        #remove vertices > threshold
        ind=np.where(np.asarray(distances) <= threshold)[0]    
        selectedPcd=source.select_by_index(ind, cleanup=True)  if ind.size >0 else None
    return selectedPcd

def crop_geometry_by_raycasting(source:o3d.geometry.TriangleMesh, cutter: o3d.geometry.TriangleMesh, inside : bool = True,strict : bool = True ) -> o3d.geometry.TriangleMesh:
    """Select portion of a mesh that lies within a mesh shape (if not closed, convexhull is called).\n
    
    inside=True
    .. image:: ../../../docs/pics/crop_by_ray_casting1.PNG

    inside=False 
    .. image:: ../../../docs/pics/crop_by_ray_casting2.PNG

    Args:
        1. source (o3d.geometry.TriangleMesh) : mesh to cut \n
        2. cutter (o3d.geometry.TriangleMesh) : mesh that cuts \n
        3. inside (bool): 'True' to retain inside. 'False' to retain outside \n
        4. strict (bool): 'True' if no face vertex is allowed outside the bounds, 'False' allows 1 vertex to lie outside \n

    Returns:
        outputmesh(o3d.geometry.TriangleMesh)
    """
    #check if cutter is closed 
    cutter,_=cutter.compute_convex_hull() if not cutter.is_watertight() else (cutter,None)

    #raycast the scene to determine which points are inside
    query_points=o3d.core.Tensor(np.asarray(source.vertices),dtype=o3d.core.Dtype.Float32 )
    cpuMesh = o3d.t.geometry.TriangleMesh.from_legacy(cutter)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cpuMesh)
    ans=scene.compute_occupancy(query_points)
    
    #create a mask of the results and process the mesh
    occupancyList = (ans==0) if inside else (ans>0)  
    outputmesh=copy.deepcopy(source)
    if strict:
        outputmesh.remove_vertices_by_mask(occupancyList)
        outputmesh.remove_degenerate_triangles()
        outputmesh.remove_unreferenced_vertices
    else:
        triangles=copy.deepcopy(np.asarray(outputmesh.triangles)) #can we remove this one?
        indices= [i for i, x in enumerate(occupancyList) if x == True]
        #mark all unwanted points as -1
        triangles[~np.isin(triangles,indices)] = -1
        # if 2/3 vertices are outside, flag the face
        occupancyList=np.ones ((triangles.shape[0],1), dtype=bool)

        for idx,row in enumerate(triangles):
            if (row[0] ==-1 and row[1]==-1) or (row[0] ==-1 and row[2]==-1) or (row[1] ==-1 and row[2]==-1):
                occupancyList[idx]=False
        outputmesh.remove_triangles_by_mask(occupancyList)
        outputmesh.remove_unreferenced_vertices()
    return outputmesh
def crop_mesh_by_convex_hull(source:trimesh.Trimesh, cutters: List[trimesh.Trimesh], inside : bool = True ) -> trimesh.Trimesh:
    """Cut a portion of a mesh that lies within the convex hull of another mesh.
    
    .. image:: ../../../docs/pics/crop_by_convex_hull.PNG

    Args:
        1. source (trimesh.Trimesh):   mesh that will be cut \n
        2. cutter (trimesh.Trimesh):   mesh of which the faces are used for the cuts. Face normals should point outwards (positive side) \n
        3. strict (bool):           True if source faces can only be part of a single submesh\n
        4. inside (bool):           True if retain the inside of the intersection\n
        
    Returns:
        mesh (trimesh.Trimesh) or None 
    """
    #validate list
    cutters=ut.item_to_list(cutters)

    submeshes=[]
    for cutter in cutters:
        submesh=None
        #compute faces and centers
        convexhull=cutter.convex_hull
        plane_normals=convexhull.face_normals
        plane_origins=convexhull.triangles_center

        if inside: # retain inside
            submesh=source.slice_plane(plane_origins, -1*plane_normals)
            if len(submesh.vertices)!=0:
                submeshes.append(submesh)
        else:# retain outside
            #cut source mesh for every slicing plane on the box
            meshes=[]
            for n, o in zip(plane_normals, plane_origins):
                tempMesh= source.slice_plane(o, n)
                if not tempMesh.is_empty:
                    meshes.append(tempMesh)
            if len(meshes) !=0: # gather pieces
                combined = trimesh.util.concatenate( [ meshes ] )
                combined.merge_vertices(merge_tex =True,merge_norm =True )
                combined.remove_duplicate_faces()
                submesh=combined
                submeshes.append(submesh)
    return submeshes

def crop_point_cloud_from_meshes(pcd: o3d.geometry.PointCloud,meshes:List[o3d.geometry.TriangleMesh]) -> List[o3d.geometry.PointCloud]:
    """Crop point cloud and divide the inliers per waterthight mesh. \n

    Args:
        1. pcd (o3d.geometry.PointCloud): point cloud to be cropped.\n
        2. meshes (o3d.geometry.TriangleMesh). cutter objects.\n 

    Returns:
        List[o3d.geometry.PointCloud] 
    """
    assert len(pcd.points) !=0
    assert 'TriangleMesh' in str(type(meshes[0]))

    meshes=ut.item_to_list(meshes)
    croppedPcds=[None]*len(meshes)
    indices=[None]*len(meshes)
    pcdRemaining=pcd
    for i,m in enumerate(meshes):
        #check if cutter is closed         
        m,_=m.compute_convex_hull() if not m.is_watertight() else (m,None)

        #create raycasting scene
        scene = o3d.t.geometry.RaycastingScene()
        cpuReference = o3d.t.geometry.TriangleMesh.from_legacy(m)
        _ = scene.add_triangles(cpuReference)

        # determine occupancy 
        query_points = o3d.core.Tensor(np.asarray(pcdRemaining.points), dtype=o3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query_points)
        indices[i]=np.where(occupancy.numpy() ==1 )[0]  
        nonIndices=np.where(occupancy.numpy() ==0 )[0] 

        # crop point cloud
        if len(indices[i]) !=0:
            croppedPcds[i]= pcd.select_by_index(indices[i])
            pcdRemaining=pcd.select_by_index(nonIndices) 
    return croppedPcds, indices

def dataframe_to_las(dataframe: pd.DataFrame,xyz:List[int]=[0,1,2],rgb:List[int]=None,dtypes:List[str]=None) -> laspy.lasdata:
    """Convert a dataframe representing a point cloud to a las point cloud file.
    View laspy dimension and type formatting at https://laspy.readthedocs.io/en/latest/lessbasic.html.\n
    
    E.g.: las=dataframe_to_las(dataframe,rgb=[3,4,5])    

    Args:
        1.dataframe (pd.DataFrame): data frame with a number of columns such as xyz, rgb and some scalar fields (conform numpy)
        2.xyz (List[int], optional): Indices of the xyz coordinates in the dataframe. Defaults to [0,1,2].
        3.rgb (List[int], optional): Indices of the color information in the dataframe e.g. [3,4,5]. Defaults to None.
        4.dtypes (List[str], optional): types of the scalar fields that will be added e.g. ['float32','uint8']. Defaults to [float32] equal to the length of the scalar fields.

    Returns:
        laspy.lasdata: output las file 
    """
    #0.get xyz data
    xyz=dataframe.iloc[:,:3].to_numpy()
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(xyz, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # 2. Create a Las from xyz
    las = laspy.LasData(header)

    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    
    # 3. add rgb if present
    if rgb:
        rgb=dataframe.iloc[:,rgb].to_numpy()
        las.red=rgb[:, 0]
        las.green=rgb[:, 1]
        las.blue=rgb[:, 2]
        
    # 4. Create extra dims for scalar fields
    names=dataframe.columns[3:] if rgb is None else dataframe.columns[[t for t in np.arange(3,len(dataframe.columns)) if t not in rgb]] #! there might be a problem here
    dtypes=dtypes if dtypes else ['float32' for n in names]
    extraBytesParams=[laspy.ExtraBytesParams(name=name, type=dtype) for name,dtype in zip(names,dtypes)]
    las.add_extra_dims(extraBytesParams)   
    [setattr(las,name,dataframe[name].to_numpy()) for i,name in enumerate(names)]
    
    return las

def dataframe_to_pcd(df:pd.DataFrame,xyz=[0,1,2],rgb=[3,4,5],n=None,transform:np.array=None)->o3d.geometry.PointCloud:
    """Convert Pandas dataframe to o3d.geometry.PointCloud.\n

    **NOTE**: this is slow. Ignoring color and normals speeds up the process by about 30%. More efficient method needed.\n

    Args:
        1. df (pd.DataFrame): Dataframe with named columns ['x', 'y', 'z'] and optional ['R', 'G', 'B'] and ['Nx', 'Ny', 'Nz'].\n
        2. pointFields (List[str]): optional column names. defaults to ['x', 'y', 'z','R', 'G', 'B','Nx', 'Ny', 'Nz']\n

    Raises:
        ValueError: No valid xyz data. Make sure column headers are names X,Y,Z.

    Returns:
        o3d.geometry.PointCloud 
    """
    #validate transform
    if transform is not None:
        assert transform.shape[0]==4
        assert transform.shape[1]==4

    # #validate pointfields    
    # if pointFields == None:
    #     pointFields=['x', 'y', 'z','R', 'G', 'B','Nx', 'Ny', 'Nz']
    # fields=[s.casefold() for s in pointFields]

    #create point cloud
    pcd=o3d.geometry.PointCloud()
    # if (all(elem.casefold() in fields for elem in ['X', 'Y', 'Z'])):
        # xyz=df.get([pointFields[0], pointFields[1], pointFields[2]])
    points=df.iloc[:,xyz].to_numpy()
    points=transform_points( points,transform) if transform is not None else points
    pcd.points=o3d.utility.Vector3dVector(points)
    # else:
    #     raise ValueError('No valid xyz data.')

    # if (all(elem.casefold() in fields for elem in ['R', 'G', 'B'])): 
        # rgb=df.get(['R', 'G', 'B'])
    colors=df.iloc[:,rgb].to_numpy()        
    colors=colors/255    if np.amax(colors)>1 else colors
    pcd.colors=o3d.utility.Vector3dVector(colors)

    # if (all(elem.casefold() in pointFields for elem in ['Nx', 'Ny', 'Nz'])): 
    # nxyz=df.get(['Nx', 'Ny', 'Nz'])
    if n:
        normals=df.iloc[:,n].to_numpy()
        normals=transform_points( normals,transform) if transform is not None else normals
        pcd.normals=o3d.utility.Vector3dVector(normals)

        
    # newnxyz=transform_points( nxyz.to_numpy(),transform) if transform is not None else nxyz.to_numpy()
    # pcd.normals=o3d.utility.Vector3dVector(newnxyz)
    
    return pcd

def describe_element(name:str, df):
    """ Takes the columns of a dataframe and builds a ply-like description.\n
    
    Args:
        1. name: str\n
        2. df: pandas DataFrame\n
        
    Returns:
        element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int', 'b': 'bool'}
    element = ['element ' + name + ' ' + str(len(df))]
    if name == 'face':
        element.append("property list uchar int vertex_indices")
    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])
    return element

def divide_box_in_boxes(box: o3d.geometry.Geometry,size:List[float]=None, parts:List[int]=None)->Tuple[List[o3d.geometry.Geometry],List[str]]:
    """Subdivide an open3d OrientedBoundingBox or AxisAlignedBoundingBox into a set of smaller boxes (either by size of number of parts).
    
    .. image:: ../../../docs/pics/subselection1.PNG

    Args:
        box (o3d.geometry.OrientedBoundingBox or AxisAlignedBoundingBox): box to divide
        size (list[float], optional): X, Y and Z size of the subdivided boxes in meter e.g. [10,10,5].
        parts (list[int], optional): X, Y and Z number of parts to divide the box in e.g. [7,7,1].

    Returns:
        List[o3d.geometry.AxisAlignedBoundingBox]: list of boxes
    """
    #parse inputs
    if 'OrientedBoundingBox' in str(type(box)):
        # get bounds
        extent=box.extent()        
    elif 'AxisAlignedBoundingBox' in str(type(box)):
        extent=box.get_extent()
        
    # get bounds
    minBound=box.get_min_bound()
    maxBound=box.get_max_bound()   
    center=box.get_center() 
    
    # get xyz ranges within the box
    size=size if size is not None and parts is None else extent / np.array(parts)
    # if size > extent, take centerpoint
    xRange=np.arange(minBound[0]+size[0]/2,maxBound[0],size[0]) if size[0]<=extent[0] else np.array([center[0]])
    yRange=np.arange(minBound[1]+size[1]/2,maxBound[1],size[1]) if size[1]<=extent[1] else np.array([center[1]])
    zRange=np.arange(minBound[2]+size[2]/2,maxBound[2],size[2]) if size[2]<=extent[2] else np.array([center[2]])

    # create names
    xNames=np.arange(0,len(xRange))
    yNames=np.arange(0,len(yRange))
    zNames=np.arange(0,len(zRange))
    xn,yn,zn=np.meshgrid(xNames,yNames,zNames,indexing='xy')
    names = np.stack((xn,yn,zn), axis = -1)
    names=np.reshape(names,(-1,3))

    #create relative center points
    xx,yy,zz=np.meshgrid(xRange,yRange,zRange,indexing='xy')
    grid = np.stack((xx,yy,zz), axis = -1)
    grid_list=np.reshape(grid,(-1,3))
    #create box
    small_box=expand_box(box,u=-extent[0]+size[0],v=-extent[1]+size[1],w=-extent[2]+size[2])

    boxes=[]
    for p in grid_list:
        box=copy.deepcopy(small_box)
        box.translate(p,False)
        boxes.append(box)
    return boxes,names


def divide_pcd_per_height(heights:List[float], pointCloud:o3d.geometry.PointCloud)->List[o3d.geometry.PointCloud]:
    """Devides a point cloud based on a set of heights.\n

    Args:
        1. heights (List[float]): heights along which to split the point cloud.\n
        2. pointCloud (o3d.geometry.PointCloud): PointCloud to split.\n

    Returns:
        List[o3d.geometry.PointCloud] is ascending order.
    """
    heights=ut.item_to_list(heights)
    #sort based on z
    indices = np.argsort(np.asarray(pointCloud.points)[:,3])
    sortedz=np.sort(np.asarray(pointCloud.points)[:,3])
    #get splitting indices
    splittingIndices=[np.find_nearest(sortedz, value) for value in heights[1:-1]] 
    #split arrays and fetch data
    indexArrays=np.split(indices, splittingIndices)[0]
    pcds=[pointCloud.select_by_index(r.toList()) for r in indexArrays if r.size !=0]
    return pcds

def e57_to_arrays(e57Path:str,e57Index : int = 0,percentage:float=1.0,tasknr:int=0)->Tuple[np.array,np.array,np.array,np.array,int]:
    """Convert a scan from a pye57.e57.E57 file to a tuple of 4 arrays.\n

    Features:
        0. ['cartesianX', 'cartesianY', 'cartesianZ'] \n
        1. ['colorRed', 'colorGreen', 'colorBlue'] \n
        2. ['nor:normalX', 'nor:normalY', 'nor:normalZ']\n
        3. cartesianTransform (np.array)
        4. tasknr (int): int to retrieve order in multiprocessing

    Args:
        1. e57 (pye57.e57.E57) \n
        2. e57Index (int,optional): index of the scan. Typically found in e57.scan_count \n
        3. tasknr (int): int to retrieve order in multiprocessing

    Returns:
        Tuple[pointArray(np.array),colorArray(np.array),normalArray(np.array),cartesianTransform(np.array),tasknr(int)]
    """
    e57 = pye57.E57(str(e57Path))
    e57_update_point_field(e57)
    
    #get transformation
    header = e57.get_header(e57Index)    
    cartesianTransform= e57_get_cartesian_transform(header)
    
    #get points    
    rawData = e57.read_scan_raw(e57Index)    
    if all(elem in header.point_fields  for elem in ['cartesianX', 'cartesianY', 'cartesianZ']):   
        pointArray=e57_get_xyz_from_raw_data(rawData)
    elif all(elem in header.point_fields  for elem in ['sphericalRange', 'sphericalAzimuth', 'sphericalElevation']):
        pointArray=e57_get_xyz_from_spherical_raw_data(rawData)

    #downnsample
    if percentage <1.0:
        indices=np.random.random_integers(0,len(pointArray)-1,int(len(pointArray)*percentage))
        pointArray=pointArray[indices]

    #get color or intensity
    colorArray=None
    if (all(elem in header.point_fields  for elem in ['colorRed', 'colorGreen', 'colorBlue'])
        or 'intensity' in header.point_fields ): 
        colorArray=e57_get_colors(rawData)
        if percentage <1.0:
            colorArray=colorArray[indices]

    #get normals
    normalArray=None
    if all(elem in header.point_fields  for elem in ['nor:normalX', 'nor:normalY', 'nor:normalZ']): 
        normalArray=e57_get_normals(rawData)
        if percentage <1.0:
            normalArray=normalArray[indices]

    return pointArray,colorArray,normalArray,cartesianTransform,tasknr

def e57_to_pcd(e57:pye57.e57.E57 , e57Index : int = 0,percentage:float=1.0)->o3d.geometry.PointCloud:
    """Convert a scan from a pye57.e57.E57 file to o3d.geometry.PointCloud.

    Args:
        1. e57 (pye57.e57.E57) 
        2. e57Index (int,optional) 
        3. percentage (float,optional): downsampling ratio. defaults to 1.0 (100%) 

    Returns:
        o3d.geometry.PointCloud
    """
    e57_update_point_field(e57)
    header = e57.get_header(e57Index)
    #get transformation
    cartesianTransform=e57_get_cartesian_transform(header)
    #get raw geometry (no transformation)
    rawData = e57.read_scan_raw(e57Index)    

        
    if all(elem in header.point_fields  for elem in ['cartesianX', 'cartesianY', 'cartesianZ']):   
        pointArray=e57_get_xyz_from_raw_data(rawData)
    elif all(elem in header.point_fields  for elem in ['sphericalRange', 'sphericalAzimuth', 'sphericalElevation']):   
        pointArray=e57_get_xyz_from_spherical_raw_data(rawData)
    else:
        raise ValueError('e57 rawData parsing failed.')

    #downnsample
    if percentage <1.0:
        indices=np.random.randint(0,len(pointArray)-1,int(len(pointArray)*percentage))
        pointArray=pointArray[indices]

    #create point cloud
    points = o3d.utility.Vector3dVector(pointArray)
    pcd=o3d.geometry.PointCloud(points)

    #get color or intensity
    if (all(elem in header.point_fields  for elem in ['colorRed', 'colorGreen', 'colorBlue'])
        or 'intensity' in header.point_fields ): 
        colors=e57_get_colors(rawData)
        if percentage <1.0:
            colors=colors[indices]
        pcd.colors=o3d.utility.Vector3dVector(colors) 

    #get normals
    if all(elem in header.point_fields  for elem in ['nor:normalX', 'nor:normalY', 'nor:normalZ']): 
        normals=e57_get_normals(rawData)
        if percentage <1.0:
            normals=normals[indices]
        pcd.normals=o3d.utility.Vector3dVector(normals)

    #return transformed data
    pcd.transform(cartesianTransform)    if cartesianTransform is not None else None
    return pcd
def get_e57_from_pcd (pcd:o3d.geometry.PointCloud ) ->dict:
    """Returns the data of an o3d.geometry.PointCloud as the data structure of an e57 file so it can be written to file. 

    Args:
        pcd (o3d.geometry.PointCloud)

    Returns:
        Data3D dictionary conform the E57 standard.
    """
    data3D={}
    if pcd.has_points():
        array=np.asarray(pcd.points)
        cartesianX=array[:,0]
        cartesianY=array[:,1]
        cartesianZ=array[:,2]
        data3D.update({'cartesianX' : cartesianX,'cartesianY':cartesianY,'cartesianZ':cartesianZ})

    if pcd.has_colors():
        array=np.asarray(pcd.colors)
        colorRed=array[:,0]*255
        colorGreen=array[:,1]*255
        colorBlue=array[:,2]*255
        data3D.update({'colorRed' : colorRed,'colorGreen':colorGreen,'colorBlue':colorBlue})
    
    if pcd.has_normals():
        array=np.asarray(pcd.normals)
        nx=array[:,0]
        ny=array[:,1]
        nz=array[:,2]
        data3D.update({'nor:normalX' : nx,'nor:normalY':ny,'nor:normalZ':nz})
    return data3D

def e57_array_to_pcd(tuple) -> o3d.geometry.PointCloud:
    """Returns PointCloud from e57 arrays.\n

    Args:
        tuple (Tuple): \n
            1. pointArray:np.array \n
            2. colorArray:np.array \n
            3. normalArray:np.array \n
            4. cartesianTransform:np.array \n

    Returns:
        o3d.geometry.PointCloud
    """        
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(tuple[0])
    if tuple[1] is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(tuple[1])
    if tuple[2] is not None:
        pointcloud.normals = o3d.utility.Vector3dVector(tuple[2])
    if len(tuple)==5 and 'ndarray' in str(type(tuple[3])) :
        pointcloud.transform(tuple[3])  
    return pointcloud

def e57_dict_to_pcd(e57:dict,percentage:float=1.0)->o3d.geometry.PointCloud:
    """Convert a scan from a e57 dictionary (raw scandata) to o3d.geometry.PointCloud.

    Args:
        1. e57 dict
        2. e57Index (int,optional) 
        3. percentage (float,optional): downsampling ratio. defaults to 1.0 (100%) 

    Returns:
        o3d.geometry.PointCloud
    """
    if all(elem in e57.keys()  for elem in ['cartesianX', 'cartesianY', 'cartesianZ']):   
            pointArray=e57_get_xyz_from_raw_data(e57)
    elif all(elem in e57.keys()  for elem in ['sphericalRange', 'sphericalAzimuth', 'sphericalElevation']):   
        pointArray=e57_get_xyz_from_spherical_raw_data(e57)
    else:
        raise ValueError('e57 rawData parsing failed.')

    #downnsample
    if percentage <1.0:
        indices=np.random.randint(0,len(pointArray)-1,int(len(pointArray)*percentage))
        pointArray=pointArray[indices]

    #create point cloud
    points = o3d.utility.Vector3dVector(pointArray)
    pcd=o3d.geometry.PointCloud(points)

    #get color or intensity
    if (all(elem in e57.keys()  for elem in ['colorRed', 'colorGreen', 'colorBlue'])
        or 'intensity' in e57.keys() ): 
        colors=e57_get_colors(e57)
        if percentage <1.0:
            colors=colors[indices]
        pcd.colors=o3d.utility.Vector3dVector(colors) 

    #get normals
    if all(elem in e57.keys()  for elem in ['nor:normalX', 'nor:normalY', 'nor:normalZ']): 
        normals=e57_get_normals(e57)
        if percentage <1.0:
            normals=normals[indices]
        pcd.normals=o3d.utility.Vector3dVector(normals)

    #return transformed data
    return pcd

def e57_fix_rotation_order(rotation_matrix:np.array) -> np.array:
    """Switch the rotation from clockwise to counter-clockwise in e57 rotation matrix. See following url for more information:\n

    https://robotics.stackexchange.com/questions/10702/rotation-matrix-sign-convention-confusion\n
    
    Currently only changes the signs of elements [0,1,5,8].\n

    Args:
        rotation_matrix(np.array(3x3))
    
    Returns:
        transformed rotation_matrix(np.array(3x3))
    """
    r=rotation_matrix
    return np.array([[-r[0,0],-r[0,1],r[0,2]],
                    [r[1,0],r[1,1],-r[1,2]],
                    [r[2,0],r[2,1],-r[2,2]]])

def e57_get_cartesian_transform(header:dict)->np.ndarray:
    """Returns cartesianTransform (np.array(4x4)) from a e57 header.\n

    Args:
        header (dict): retrieved from pye57 e57.get_header(e57Index)

    Returns:
        cartesianTransform (np.array(4x4))
    """
    cartesianTransform=np.diag(np.diag(np.ones((4,4))))    
    if 'pose' in header.scan_fields:
        rotation_matrix=None
        translation=None
        if getattr(header,'rotation',None) is not None:
                rotation_matrix=header.rotation_matrix
        if getattr(header,'translation',None) is not None:
            translation=header.translation
        cartesianTransform=get_cartesian_transform(rotation=rotation_matrix,translation=translation)
    return  cartesianTransform

def e57_get_cartesian_transform(header)-> np.array:
    """Returns the cartesianTransform from an e57 header.\n

    Args:
        header (e57): rotation and translation should be present.

    Returns:
        np.array (4x4): transformation Matrix
    """
    if 'pose' in header.scan_fields:
        rotation_matrix=None
        translation=None
        if getattr(header,'rotation',None) is not None:
            rotation_matrix=header.rotation_matrix
        if getattr(header,'translation',None) is not None:
            translation=header.translation
        cartesianTransform=get_cartesian_transform(rotation=rotation_matrix,translation=translation)
        return cartesianTransform
    else:
        return None

def e57_get_colors(rawData: dict)->np.ndarray:
    """Extract color of intensity information from e57 raw data (3D data) and output Open3D o3d.utility.Vector3dVector(colors).\n

    Args:
        rawData(dict): e57 dictionary resulting from e57.read_scan_raw(e57Index)\n
    
    Returns:
        np.array(nx3): RGB or intensity color information. RGB is prioritised.
    """ 
    r,g,b,i=None,None,None,None
    for key in rawData.keys():
        if all(elem.casefold() in key.casefold() for elem in ['red']):
            r=rawData.get(key)
        if all(elem.casefold() in key.casefold() for elem in ['green']):
            g=rawData.get(key)
        if all(elem.casefold() in key.casefold() for elem in ['blue']):
            b=rawData.get(key)  
        if all(elem.casefold() in key.casefold() for elem in ['intensity']):
            i=rawData.get(key)  

    if all(n is not None for n in [r ,g, b]):
        if np.amax(r)<=1:
            colors=np.c_[r , g,b ]  
        elif np.amax(r) <=255:
            colors=np.c_[r/255 , g/255,b/255 ]  
        else:
            r=(r - np.min(r))/np.ptp(r)
            g=(g - np.min(g))/np.ptp(g)
            b=(b - np.min(b))/np.ptp(b)
            colors=np.c_[r , g,b ]  
        return np.reshape(colors.flatten('F'),(len(r),3))

    elif i is not None:
        if np.amax(i) <=1:
            colors=np.c_[i , i,i ]  
        else:
            i=(i - np.min(i))/np.ptp(i)
            colors=np.c_[i , i,i ]  
        return np.reshape(colors.flatten('F'),(len(i),3))

def e57_get_normals(rawData:dict)->np.ndarray:
    """Returns normal vectors from e57 rawData.\n

    Args:
        rawData (dict): e57 dictionary resulting from e57.read_scan_raw(e57Index).\n

    Returns:
         np.array(nx3): magnitude 1
    """
    nx,ny,nz = None,None,None
    for key in rawData.keys():
        if all(elem.casefold() in key.casefold() for elem in ['n','o' ,'x']):
            nx=rawData.get(key)
        if all(elem.casefold() in key.casefold() for elem in ['n', 'o' ,'y']):
            ny=rawData.get(key)
        if all(elem.casefold() in key.casefold() for elem in ['n','o' , 'z']):
            nz=rawData.get(key)
    return np.reshape(np.vstack(( nx,ny,nz)).flatten('F'),(len(nx),3)) if all(n is not None for n in [nx ,ny, nz]) else None
    

def e57_get_xyz_from_raw_data(rawData: dict)->np.ndarray:
    """Returns the xyz coordinates from e57 raw data.\n

    Args:
        rawData (e57 dict):  rawData = e57.read_scan_raw(e57Index).   
        
    Returns:
        np.array (nx3): XYZ cartesian coordinates np.array.
    """
    x=rawData.get('cartesianX')
    y=rawData.get('cartesianY')
    z=rawData.get('cartesianZ') 
    pointArray=np.reshape(np.vstack(( x,y,z)).flatten('F'),(len(x),3))
    return pointArray

def e57_get_xyz_from_spherical_raw_data(rawData: dict) -> np.array:
    """Converts spherical(rae) to cartesian(xyz), where rae = range, azimuth(theta), 
    elevation(phi). Where range is in meters and angles are in radians.\n
    
    Reference for formula: http://www.libe57.org/bestCoordinates.html \n

    Args:
        rawData (e57 dict):  rawData = e57.read_scan_raw(e57Index).  
        
    Returns:
        np.array (nx3): XYZ cartesian coordinates np.array.
    """
    range = rawData.get('sphericalRange')
    theta = rawData.get('sphericalAzimuth')
    phi =  rawData.get('sphericalElevation')
    range_cos_phi = range * np.cos(phi)
    pointArray=np.reshape(np.vstack( range_cos_phi * np.cos(theta),
                                    range_cos_phi * np.sin(theta),
                                    range * np.sin(phi)).flatten('F'),(len(range),3))
    return pointArray

def e57_update_point_field(e57:pye57.e57.E57):
    """Update e57 point fields with any point field in the file.

    Args:
        e57 (pye57.e57.E57):
    """
    header = e57.get_header(0)
    pointFields=header.point_fields
    for f in pointFields:
        pye57.e57.SUPPORTED_POINT_FIELDS.update({f : 'd'})

def e57path_to_pcd(e57Path:Path|str , e57Index : int = 0,percentage:float=1.0) ->o3d.geometry.PointCloud:
    """Load an e57 file and convert the data to o3d.geometry.PointCloud.\n

    Args:
        e57path

    Raises:
        ValueError: Invalid e57Path.

    Returns:
        o3d.geometry.PointCloud
    """
    e57 = pye57.E57(str(e57Path))
    e57_update_point_field(e57)
    pcd=e57_to_pcd(e57,e57Index,percentage)
    return pcd

def e57path_to_pcds_multiprocessing(e57Path:str,percentage:float=1.0) ->List[o3d.geometry.PointCloud]:
    """Load an e57 file and convert all data to a list of o3d.geometry.PointCloud objects.\n

    **NOTE**: Complex types cannot be pickled (serialized) by Windows. Therefore, a two step parsing is used where e57 data is first loaded as np.arrays with multi-processing.
    Next, the arrays are passed to o3d.geometry.PointClouds outside of the loop.\n  

    **NOTE**: starting parallel processing takes a bit of time. This method will start to outperform single-core import from 3+ pointclouds.\n

    Args:
        1. e57path(str): absolute path to .e57 file\n
        2. percentage(float,optional): percentage of points to load. Defaults to 1.0 (100%)\n

    Raises:
        ValueError: Invalid e57Path.

    Returns:
        o3d.geometry.PointCloud
    """   
    #update pointfields
    e57 = pye57.E57(str(e57Path))
    e57_update_point_field(e57)
    pcds=[None]*e57.scan_count
    #set up multi-processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # First convert all e57 data to np.arrays
        results=[executor.submit(e57_to_arrays,e57Path=e57Path,e57Index=s,percentage=percentage,tasknr=s) for s in range(e57.scan_count)]
        # Next, the arrays are assigned to point clouds outside the loop.
        for r in concurrent.futures.as_completed(results):
            tasknr=r.result()[-1]
            print(r.result()[-2])
            pcd=e57_array_to_pcd(r.result())
            pcds[tasknr]=pcd
    return pcds

def expand_box(box: o3d.geometry, u=5.0,v=5.0,w=5.0) -> o3d.geometry:
    """expand an o3d.geometry.BoundingBox in u(x), v(y) and w(z) direction with a certain offset.

    Args:
        1. box (o3d.geometry.OrientedBoundingBox, o3d.geometry.AxisAlignedBoundingBox,o3d.geometry.TriangleMesh box)
        2. u (float, optional): Offset in X. Defaults to 5.0m.
        3. v (float, optional): Offset in Y. Defaults to 5.0m.
        4. w (float, optional): Offset in Z. Defaults to 5.0m.

    Returns:
        o3d.geometry.OrientedBoundingBox or o3d.geometry.AxisAlignedBoundingBox
    """        
    if isinstance(box,o3d.geometry.TriangleMesh) and len(np.asarray(box.vertices))==8:
        vertices=np.array(box.vertices)
        #compute modifications
        xmin=-u/2
        xmax=u/2
        ymin=-v/2
        ymax=v/2
        zmin=-w/2
        zmax=w/2
        vertices=np.array([vertices[0,:]+[xmin,ymin,zmin],
                            vertices[0,:]+[xmax,ymin,zmin],
                            vertices[0,:]+[xmin,ymax,zmin],
                            vertices[0,:]+[xmax,ymax,zmin],
                            vertices[0,:]+[xmin,ymin,zmax],
                            vertices[0,:]+[xmax,ymin,zmax],
                            vertices[0,:]+[xmin,ymax,zmax],
                            vertices[0,:]+[xmax,ymax,zmax]])
        box.vertices = o3d.utility.Vector3dVector(vertices)
        return box
    elif isinstance(box,o3d.geometry.OrientedBoundingBox):
        center = box.get_center()
        orientation = box.R 
        extent = box.extent + [u,v,w] 
        return o3d.geometry.OrientedBoundingBox(center,orientation,extent) 
    elif isinstance(box,o3d.geometry.AxisAlignedBoundingBox):
        # assert ((abs(u)<= box.get_extent()[0]) and (abs(v)<=box.get_extent()[1]) and (abs(w)<=box.get_extent()[2])), f'cannot schrink more than the extent of the box.'
        minBound=box.get_min_bound()
        maxBound=box.get_max_bound()
        new_minBound=minBound-np.array([u,v,w])/2
        new_maxBound=maxBound+np.array([u,v,w])/2        
        return o3d.geometry.AxisAlignedBoundingBox(new_minBound,new_maxBound) 
    else:
        raise ValueError('Invalid input type')

def extract_points(geometry):
    """Extracts points from PointCloud, TriangleMesh, LineSet, Vector3dVector or NumPy array."""
    if isinstance(geometry, np.ndarray):
        if geometry.ndim == 1 and geometry.size == 6: # [xMin,xMax,yMin,yMax,zMin,zMax]
            x_min, x_max, y_min, y_max, z_min, z_max = geometry
            corners = np.array([
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ])
            return corners
        elif geometry.ndim == 1 and geometry.size == 9: #  [center(3),extent(3),euler_angles(3)]
            center=geometry[:3]
            extent=geometry[3:6]
            euler_angles=geometry[6:9]        
            rotation_matrix = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
            box = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
            return box
        if geometry.ndim != 2 or geometry.shape[1] != 3:
            raise ValueError("NumPy array must have shape (N, 3)")
        return geometry
    elif isinstance(geometry, (o3d.geometry.PointCloud,  o3d.geometry.LineSet)):
        return np.asarray(geometry.points)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        return np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.OrientedBoundingBox):
        return geometry
    elif isinstance(geometry ,o3d.utility.Vector3dVector):
        return np.asarray(geometry)
    else:
        raise TypeError("Unsupported geometry type. Use PointCloud, TriangleMesh, LineSet, Vector3dVector or NumPy array.")

def generate_virtual_images(geometries: List[o3d.geometry.Geometry],cartesianTransforms: List[np.array],width:int=640,height:int=480,f:float=400)-> List[o3d.geometry.Image]:
    """Generate a set of Open3D Images from cartesianTransforms and geometries. \n
    The same intrinsic camera parameters are used for all cartesianTransforms that are passed to the function.\n

    .. image:: ../../../docs/pics/rendering3.PNG

    Args:
        1. geometries (List[o3d.geometry]):o3d.geometry.PointCloud or o3d.geometry.TriangleMesh \n
        2. cartesianTransforms (List[np.array 4x4]): [Rt] \n
        3. width (int, optional): image width in pix. Defaults to 640pix.\n
        4. height (int, optional): image height in pix. Defaults to 480pix. \n
        5. f (float, optional): focal length in pix. Defaults to 400pix. \n

    Returns:
        List[o3d.geometry.Image]
    """
    #set renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width,height)
    mtl=o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # set internal camera orientation
    fx=f
    fy=f
    cx=width/2-0.5
    cy=width/2-0.5
    intrinsic=o3d.camera.PinholeCameraIntrinsic(width,height,fx,fy,cx,cy)
    
    #add geometries
    geometries=ut.item_to_list(geometries)
    for idx,geometry in enumerate(geometries):
        render.scene.add_geometry(str(idx),geometry,mtl) 

    #set cameras and generate images
    list=[]
    cartesianTransforms=ut.item_to_list(cartesianTransforms)
    for cartesianTransform in cartesianTransforms:
        extrinsic=np.linalg.inv(cartesianTransform)
        render.setup_camera(intrinsic,extrinsic)
        img = render.render_to_image()
        list.append(img)
    return list if len(list) !=0 else None

def get_backface_center_transform(obb: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    """
    Returns the full 4x4 Cartesian transform (position and rotation) of the center
    of the back face (along -Z in local space) of an oriented bounding box.

    Parameters:
        obb (o3d.geometry.OrientedBoundingBox): The oriented bounding box.

    Returns:
        np.ndarray: 4x4 transformation matrix for the center of the back face.
    """
    # Rotation (3x3)
    R = obb.R

    # Vector pointing from the center toward the back face in local coordinates
    back_offset_local = np.array([0, 0, -obb.extent[2] / 2])

    # Convert to world space using rotation
    back_offset_world = R @ back_offset_local

    # New origin = center of the back face
    backface_origin = obb.center + back_offset_world

    # Construct 4x4 transform
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = backface_origin

    return transform

def get_box_inliers(sourceBox:o3d.geometry.OrientedBoundingBox, testBoxes: List[o3d.geometry.OrientedBoundingBox],t_d:float=0.5)->List[int]:
    """Return the indices of the testBoxes of which the bounding points lie within the sourceBox.\n

    .. image:: ../../../docs/pics/get_box_inliers1.PNG

    Args:
        1. sourceBox (o3d.geometry.OrientedBoundingBox) \n
        2. testBoxes (o3d.geometry.OrientedBoundingBox)\n
        
    Returns:
        list (List[int]): Indices of testBoxes \n
    """
    sourceBox=expand_box(sourceBox,u=t_d,v=t_d,w=t_d)
    testBoxes=ut.item_to_list(testBoxes) 
    array= [False] * len(testBoxes)
    for idx,testbox in enumerate(testBoxes):
        if testbox is not None:
            points=testbox.get_box_points()
            indices=sourceBox.get_point_indices_within_bounding_box(points)        
            if len(indices) !=0:
                array[idx]= True
    list = [ i for i in range(len(array)) if array[i]]
    return list

def get_box_intersections(sourceBox:o3d.geometry.OrientedBoundingBox, testBoxes: List[o3d.geometry.OrientedBoundingBox])->List[int]:
    """Return indices of testBoxes of which the geometry intersects with the sourceBox.\n
    2 oriented bounding boxes (A,B) overlap if the projection from B in the coordinate system of A on all the axes overlap. \n
    The projection of B on the oriented axes of A is simply the coordinate range for that axis.\n

    .. image:: ../../../docs/pics/get_box_inliers1.PNG

    Args:
        1. sourceBox (o3d.geometry.OrientedBoundingBox):   box to test\n
        2. testBoxes (o3d.geometry.OrientedBoundingBox):   boxes to test\n

    Returns:
        list (List[int]):       indices of testBoxes
    """       
    #validate inputs
    testBoxes=ut.item_to_list(testBoxes)
    array= [False] * len(testBoxes)

    for idx,testbox in enumerate(testBoxes):
    # compute axes of box A
        if testbox is not None:
            #transform box to aligned coordinate system
            transformedboxA=copy.deepcopy(sourceBox)
            transformedboxA=transformedboxA.translate(-sourceBox.center)
            transformedboxA=transformedboxA.rotate(sourceBox.R.transpose(),center=(0, 0, 0))
            
            #transform testbox to aligned coordinate system
            transformedboxB=copy.deepcopy(testbox)
            transformedboxB=transformedboxB.translate(-sourceBox.center)
            transformedboxB=transformedboxB.rotate(sourceBox.R.transpose(),center=(0, 0, 0))

            # compute coordinates of bounding points of B in coordinate system of A
            minA=transformedboxA.get_min_bound()
            minB=transformedboxB.get_min_bound()
            maxA=transformedboxA.get_max_bound()
            maxB=transformedboxB.get_max_bound()

            if (maxB[0]>=minA[0] and minB[0]<=maxA[0]):
                if (maxB[1]>=minA[1] and minB[1]<=maxA[1]):
                    if (maxB[2]>=minA[2] and minB[2]<=maxA[2]):
                        array[idx]= True  
    # return index if B overlaps with A in all three axes u,v,w 
    list = [ i for i in range(len(array)) if array[i]]
    return list

def get_cartesian_bounds(geometry : o3d.geometry.Geometry) ->np.ndarray:
    """Get cartesian bounds from Open3D geometry.\n

    Args:
        geometry (o3d.geometry): Open3D geometry supertype (PointCloud, TriangleMesh, OrientedBoundingBox, etc.)\n

    Returns:
        np.array: [xMin,xMax,yMin,yMax,zMin,zMax]
    """
    max=geometry.get_max_bound()
    min=geometry.get_min_bound()
    return np.array([min[0],max[0],min[1],max[1],min[2],max[2]])

def get_cartesian_transform(translation: np.array = None,
                            rotation: np.array = None                            
                            ) -> np.ndarray:
    """Return cartesianTransform from rotation, translation or cartesianBounds inputs.

    Args:
        - translation (Optional[np.ndarray]): A 3-element translation vector
        - rotation (Optional[np.ndarray]): A 3x3 rotation matrix, Euler angles $(R_x,R_y,R_z)$ or a rotation quaternion $(q_x,q_y,q_z,q_w)$.

    Returns:
        cartesianTransform (np.ndarray): The 4x4 transformation matrix
    """   
    # Initialize identity rotation matrix and zero translation vector
    r = np.eye(3)
    t = np.zeros((3, 1))

    # Update rotation matrix if provided
    if rotation is not None:
        rotation=np.asarray(rotation)
        if rotation.size == 3: #Euler angles
            r = R.from_euler('xyz', rotation,degrees=True).as_matrix()
        elif rotation.size == 9: #rotation matrix
            r = np.reshape(np.asarray(rotation), (3, 3))
        elif rotation.size == 4: #quaternion
            r = R.from_quat(rotation).as_matrix()
        else:
            raise ValueError("Rotation must be either a 3x3 matrix or a tuple/list of three Euler angles.")

    # Update translation vector if provided
    if translation is not None:
        t = np.reshape(np.asarray(translation), (3, 1))

    # Create the last row of the transformation matrix
    h = np.array([[0, 0, 0, 1]])

    # Concatenate rotation and translation to form the 3x4 upper part of the transformation matrix
    transformation = np.hstack((r, t))

    # Add the last row to make it a 4x4 matrix
    transformation = np.vstack((transformation, h))

    return transformation

def get_convex_hull(geometry, thickness=1e-3, distanceTreshold = 10000):
    """Computes the Convex Hull for PointCloud, Mesh, LineSet, or NumPy array."""
    points = extract_points(geometry)
    # If input is already an OBB, extract box points
    if isinstance(points, o3d.geometry.OrientedBoundingBox):
        points = np.asarray(points.get_box_points())
    points = preprocess_points_for_geometry(points, thickness)

    # Center the points to improve numerical stability
    center = np.mean(points, axis=0)
    if(np.linalg.norm(center) > distanceTreshold): # points are really far away
        points_centered = points - center
    else: points_centered = points

    # Compute convex hull on centered points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_centered))
    hull, _ = pcd.compute_convex_hull()

    # Translate back to original space
    if(np.linalg.norm(center) > distanceTreshold): # points are really far away
        hull.translate(center)
    hull.paint_uniform_color([0.0, 1.0, 0.0])  # Green for visualization

    return hull

def get_geometry_from_path(path : str) -> o3d.geometry.Geometry:
    """Gets a open3d Geometry from a path.

    Args:
        path (str): The absulute path to the resource\n

    Returns:
        o3d.geometry: open3d.Pointcloud or open3d.Trianglemesh, depending on the extension
    """
    newGeometry = None
    if(path.endswith(tuple(ut.MESH_EXTENSION))):
        newGeometry = o3d.io.read_triangle_mesh(path)
        if(not newGeometry.has_triangles()):
            newGeometry = o3d.io.read_point_cloud(path)
        #if not newGeometry.has_vertex_normals():
        else:
            newGeometry.compute_vertex_normals()
    elif(path.endswith(tuple(ut.PCD_EXTENSION))):
        newGeometry = o3d.io.read_point_cloud(path)
    return newGeometry


def get_mesh_collisions_trimesh(sourceMesh: o3d.geometry.TriangleMesh, geometries: List[o3d.geometry.TriangleMesh]) -> List[int]:
    """Return indices of geometries that collide with the source.\n

    .. image:: ../../../docs/pics/collision_4.PNG

    Args:
        1. sourceMesh (o3d.geometry.TriangleMesh)\n
        2. geometries (List[o3d.geometry.TriangleMesh])

    Returns:
        List[int]: indices of inliers.
    """
    if 'TriangleMesh' in str(type(sourceMesh)) and len(sourceMesh.triangles) >0:
        myTrimesh=mesh_to_trimesh(sourceMesh)
        geometries=ut.item_to_list(geometries)
        # add all geometries to the collision manager
        collisionManager=trimesh.collision.CollisionManager()
        for idx,geometry in enumerate(geometries):
            if 'TriangleMesh' in str(type(geometry)) and len(geometry.triangles) >1:
                referenceTrimesh=mesh_to_trimesh(geometry)
                collisionManager.add_object(idx,referenceTrimesh)

        # report the collisions with the sourceMesh
        (is_collision, names ) = collisionManager.in_collision_single(myTrimesh, transform=None, return_names=True, return_data=False)    
        if is_collision:
            list=[int(name) for name in names]
            return list
    else:
        raise ValueError('condition not met: type(sourceMesh) is o3d.geometry.TriangleMesh and len(sourceMesh.triangles) >0')

def get_mesh_inliers(sources:List[o3d.geometry.TriangleMesh], reference:o3d.geometry.TriangleMesh) -> List[int]:
    """Returns the indices of the geometries that lie within a reference geometry. The watertightness of both source and reference geometries is determined after which the occupancy is computed for a numer of sampled points in the source geometries. \n

    **NOTE**: all source ponts/vertices should lie within the reference, else this is False.\n 

    .. image:: ../../../docs/pics/selection_BB_mesh5.PNG
    
    Args:
        1. sources (List[o3d.geometry.TriangleMesh or o3d.geometry.PointCloud]): geometries to test for collision\n
        2. reference (o3d.geometry.TriangleMesh or o3d.geometry.PointCloud): reference geometry \n

    Returns:
        List[int]: Indices of the inlier geometries
    """
    #validate inputs    
    sources=ut.item_to_list(sources)
    #compute watertight hull if necessary
    reference,_=(reference,None) if ('TriangleMesh' in str(type(reference)) and reference.is_watertight()) else reference.compute_convex_hull()
    
    #create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    reference = o3d.t.geometry.TriangleMesh.from_legacy(reference)
    _ = scene.add_triangles(reference)
    #compute inliers
    inliers=[None]*len(sources)
    for i,source in enumerate(sources):
        source,_=(source,None) if ('TriangleMesh' in str(type(source)) and source.is_watertight()) else source.compute_convex_hull()
        query_points = o3d.core.Tensor(np.asarray(source.vertices), dtype=o3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query_points)        
        inliers[i]=True if np.any(occupancy.numpy()) else False       
    #select indices    
    ind=np.where(np.asarray(inliers) ==True)[0]     
    return ind

def get_oriented_bounding_box(geometry, thickness=1e-3, distanceTreshold = 10000):
    """Computes the Oriented Bounding Box (OBB) for PointCloud, Mesh, LineSet, or NumPy array."""
    points = extract_points(geometry)
    # If already an OBB, just return it
    if isinstance(points, o3d.geometry.OrientedBoundingBox):
        return points
    points = preprocess_points_for_geometry(points, thickness)

    # Compute center and shift points
    center = np.mean(points, axis=0)
    if(np.linalg.norm(center) > distanceTreshold): # points are really far away
        points_centered = points - center
    else: points_centered = points

    # Convert to Open3D point cloud and compute OBB
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_centered))
    obb = pcd.get_oriented_bounding_box()

    # Translate OBB back to original location
    if(np.linalg.norm(center) > distanceTreshold): # points are really far away
        obb.translate(center)

    return obb



def get_oriented_bounding_box_parameters(orientedBoundingBox: o3d.geometry.OrientedBoundingBox)->np.ndarray:
    """
    Extract the center, extent, and Euler angles from an Open3D oriented bounding box.

    Parameters:
    obb (o3d.geometry.OrientedBoundingBox): The oriented bounding box from which to extract parameters.

    Returns:
    tuple: A tuple containing the center (list), extent (list), and Euler angles (list in degrees).
    """
    center = orientedBoundingBox.center
    extent = orientedBoundingBox.extent
    rotation_matrix = copy.deepcopy(orientedBoundingBox.R)
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    print("The euler angles are derived from the rotation matrix, please note that this representation has a number of disadvantages")
    return np.hstack((center, extent, euler_angles))

def get_oriented_bounds(cartesianBounds:np.array) -> List[o3d.utility.Vector3dVector]:
    """Get 8 bounding box from cartesianBounds.\n

    Args:
        cartesianBounds (np.array[6x1])

    Returns:
        List[o3d.utility.Vector3dVector]
    """    
    array=np.empty((8,3))
    xMin=cartesianBounds[0]
    xMax=cartesianBounds[1]
    yMin=cartesianBounds[2]
    yMax=cartesianBounds[3]
    zMin=cartesianBounds[4]
    zMax=cartesianBounds[5]

    #same order as Open3D
    array[0]=np.array([xMin,yMin,zMin])
    array[1]=np.array([xMax,yMin,zMin])
    array[2]=np.array([xMin,yMax,zMin])
    array[3]=np.array([xMin,yMin,zMax])
    array[4]=np.array([xMax,yMax,zMax])
    array[5]=np.array([xMin,yMax,zMax])
    array[6]=np.array([xMax,yMin,zMax])
    array[7]=np.array([xMax,yMax,zMin])

    return o3d.utility.Vector3dVector(array)
    

def get_pcd_collisions(sourcePcd: o3d.geometry.PointCloud, geometries: List[o3d.geometry.PointCloud]) -> List[int]:
    """Return indices of geometries that collide with the source. This detection is based on the convex hull of the geometries and the sourcePcd.\n

    Args:
        1. sourceMesh (o3d.geometry.TriangleMesh)\n
        2. geometries (List[o3d.geometry.TriangleMesh])\n

    Returns:
        List[int]: indices
    """
    geometries=ut.item_to_list(geometries)
    sourceHull, _ = sourcePcd.compute_convex_hull()
    hulls,_= map(list, zip(*[g.compute_convex_hull() for g in geometries]))   
    return get_mesh_collisions_trimesh(sourceMesh=sourceHull, geometries=hulls)

def get_pcd_from_depth_map(self)->o3d.geometry.PointCloud:
    """
    Convert a panoramic depth map and image colors (equirectangular) to a 3D point cloud.
    
    Args:
        - self._depthMap: 2D numpy array containing depth values (equirectangular depth map)
        - self._resource: 2D numpy array containing color values (equirectangular color image)
    
    Returns:
        - An Open3D point cloud object
    """
    if not isinstance(self._depthMap,np.ndarray):
        return None
    
    # Get the dimensions of the depth map
    height, width = self._depthMap.shape
    
    # Check if the color image and depth map have the same dimensions
    resource=cv2.resize(self._resource,(self._depthMap.shape[1],self._depthMap.shape[0])) if (isinstance(self._resource,np.ndarray) and not self._resource.shape[0] == self._depthMap.shape[0]) else self._resource

    # field of view in radians
    fov_horizontal_rad =  2*np.pi
    fov_vertical_rad = np.pi

    # Generate arrays for pixel coordinates
    # u: horizontal pixel coordinates (0 to width-1), v: vertical pixel coordinates (0 to height-1)
    u = np.linspace(0, width - 1, width)
    v = np.linspace(0, height - 1, height)#[::-1]  # Flip vertically (top to bottom)
    u, v = np.meshgrid(u, v)

    # Map pixels to spherical coordinates
    # Azimuth (longitude) theta is mapped from 0 to 2*pi across the width of the image
    theta = u / (width - 1) * fov_horizontal_rad - np.pi  # Map [0, width-1] to [-pi, pi]

    # Elevation (latitude) phi is mapped from 0 to pi across the height of the image
    phi = v / (height - 1) * fov_vertical_rad - (np.pi / 2)  # Map [0, height-1] to [-pi/2, pi/2]

    # Spherical to Cartesian conversion in camera coordinate system (z-forward, y-up)
    x = self._depthMap * np.cos(phi) * np.sin(theta)    # x-axis (left-right in pinhole model)
    y = self._depthMap * -np.sin(phi)                   # y-axis (up-down in pinhole model)
    z = self._depthMap * np.cos(phi) * np.cos(theta)    # z-axis (forward-backward in pinhole model)
    
    # Flatten the x, y, z arrays to create a list of points
    points = np.stack((x.flatten(order='C'), y.flatten(order='C'), z.flatten(order='C')), axis=-1)
    
    # Flatten the color image to correspond to the points
    colors=None
    if isinstance(resource,np.ndarray):
        colors = resource.reshape(-1, 3, order='C')  # Ensure row-major order (C-style)
        #divide by 255 if the colors are in the range 0-255
        colors = colors / 255 if np.max(colors) > 1 else colors
    
    # Create an Open3D point cloud from the points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if isinstance(colors,np.ndarray):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # #note that Z goes through the center of the image, so we need to rotate the point cloud 90° clockwise around the x-axis
    # r = R.from_euler('x', -90, degrees=True).as_matrix()
    # pcd.rotate(r,center = [0,0,0])
    
    #transform to the cartesianTransform
    pcd.transform(self._cartesianTransform) if self._cartesianTransform is not None else None
    return pcd

def get_points_and_normals(pcd,transform:np.ndarray=None,getNormals=False)-> Tuple[np.ndarray,np.ndarray]:
    """Extract points from different point cloud formats. Optionally extract or generate normals and apply a rigid body transformation.

    Args:
        1. pcd (_type_): point cloud \n
        2. transform (np.array[4,4], optional): Rigid body transformation. Defaults to None.\n
        3. getNormals (bool, optional): Defaults to False.\n

    Raises:
        ValueError: Only open3d, laspy and pandas data formats are currently supported.\n

    Returns:
        Tuple[np.array,np.array]: points, normals
    """
    if 'LasData' in str(type(pcd)):
        points= transform_points(pcd.xyz,transform) if transform is not None else pcd.xyz
        normals= las_get_normals(pcd,transform) if getNormals else None
    elif 'PointCloud' in str(type(pcd)):
        pcd.transform(transform) if transform is not None else None
        points=np.asarray(pcd.points) 
        normals=pcd_get_normals(pcd) if getNormals else None
    elif 'DataFrame' in str(type(pcd)):
        raise NotImplementedError("Dataframe Query_points is not implemented yet")
        # TODO: implement dataframe query_points and query points
    else:
        raise ValueError('type(pcd) == o3d.geometry.PointCloud, laspy point cloud or pandas dataframe')
    return points,normals

def get_rotation_matrix_from_forward_up(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix from a forward and an up vector. (right, up, forward)

    Args:
        forward (np.ndarray): A 3-element array representing the forward direction.
        up (np.ndarray): A 3-element array representing the up direction.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    # Normalize the vectors
    forward = forward / np.linalg.norm(forward)

    # Compute the right vector as the cross product of up and forward
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    
    # Recompute the up vector to ensure orthogonality
    up = np.cross(forward, right)
    
    # Construct the rotation matrix
    rotation_matrix = np.column_stack((right, up, forward))
    
    return rotation_matrix
def get_rotation_matrix(data:np.array) -> np.array:
    """Get rotation matrix from one of the following inputs.\n

    Args:
        1. cartesianTransform (np.array [4x4])\n
        2. Euler angles (np.array[3x1]): yaw, pitch, roll  (in degrees)\n
        3. quaternion (np.array[4x1]): w,x,y,z\n
        4. orientedBounds(np.array[8x3])\n

    Returns:
        rotationMatrix (np.array[3x3])
    """  
    if data.size ==16:
        return data[0:3,0:3]
    elif data.size == 4:
        r = R.from_quat(data)
        return r.as_matrix()
    elif data.size ==3:
        r = R.from_euler('zyx',data,degrees=True)
        return r.as_matrix()
    elif data.size == 24:
        boxPointsVector = o3d.utility.Vector3dVector(data)
        open3dOrientedBoundingBox = o3d.geometry.OrientedBoundingBox.create_from_points(boxPointsVector)
        return np.asarray(open3dOrientedBoundingBox.R)    
    else:
        raise ValueError('No suitable input array found')

def get_translation(data)-> np.array:
    """Get translation vector from various inputs.\n

    Args:
        0. cartesianTransform (np.array [4x4])\n
        1. cartesianBounds (np.array[6x1])\n
        2. orientedBounds(np.array[8x3])\n
        3. Open3D geometry\n

    Raises:
        ValueError: data.size !=6 (cartesianBounds), data.shape[1] !=3 (orientedBounds), data.size !=16 (cartesianTransform) or type != Open3D.geometry

    Returns:
        np.array[3x1]
    """
    if data.size ==6: #cartesianBounds
        x=(data[1]+data[0])/2
        y=(data[3]+data[2])/2
        z=(data[5]+data[4])/2
        return np.array([x,y,z])
    elif data.shape[1] ==3:   #orientedBounds
        return np.mean(data,axis=0)
    elif data.size ==16: #cartesianTransform
        return data[0:3,3]
    elif 'Open3D' in str(type(data)): #Open3D geometry
        return data.get_center()
    else:
        raise ValueError(" data.size !=6 (cartesianBounds), data.shape[1] !=3 (orientedBounds), data.size !=16 (cartesianTransform) or type != Open3D.geometry")

def get_triangles_center(mesh:o3d.geometry.TriangleMesh,triangleIndices: List[int]=None) -> np.array:
    """Get the centerpoints of a set of mesh triangles.\n

    Args:
        1. mesh (o3d.geometry.TriangleMesh)\n
        2. triangleIndices (List[int], optional): Indices to evaluate. Defaults to all triangles\n

    Raises:
        ValueError: len(triangleIndices)>len(mesh.triangles)\n
        ValueError: all(x > len(mesh.triangles) for x in triangleIndices)\n

    Returns:
        np.array[nx3] XYZ centers of triangles
    """
    #validate inputs
    triangleIndices=range(0,len(mesh.triangles)) if not triangleIndices else triangleIndices
    assert len(triangleIndices)<=len(mesh.triangles)
    
    #get triangles and compute center
    triangles=np.asarray([triangle for idx,triangle in enumerate(mesh.triangles) if idx in triangleIndices])
    centers=np.empty((len(triangles),3))
    for idx,row in enumerate(triangles):
        points=np.array([mesh.vertices[row[0]],mesh.vertices[row[1]],mesh.vertices[row[2]]])
        centers[idx]=np.mean(points,axis=0)
    return centers

def ifc_to_mesh(ifcElement:ifcopenshell.entity_instance)-> o3d.geometry.TriangleMesh: 
    """Convert an ifcOpenShell geometry to an Open3D TriangleMesh.\n

    Args:
        ifcElement (ifcopenshell.entity_instance): IfcOpenShell Element parsed from and .ifc file. See BIMNode for more documentation.

    Raises:
        ValueError: Geometry production error. This function throws an error if no geometry can be parsed for the ifcElement.  

    Returns:
        o3d.geometry.TriangleMesh: Open3D Mesh Geometry of the ifcElment boundary surface
    """    
    try:
        if ifcElement.get_info().get("Representation"): 
            # Set geometry settings and global coordinates as true
            settings = geom.settings() 
            settings.set(settings.USE_WORLD_COORDS, True) 
        
            # Extract vertices/faces of the IFC geometry
            shape = geom.create_shape(settings, ifcElement) 
            ifcVertices = shape.geometry.verts 
            ifcFaces = shape.geometry.faces 

            #Group the vertices and faces in a way they can be read by Open3D
            vertices = [[ifcVertices[i], ifcVertices[i + 1], ifcVertices[i + 2]] for i in range(0, len(ifcVertices), 3)]
            faces = [[ifcFaces[i], ifcFaces[i + 1], ifcFaces[i + 2]] for i in range(0, len(ifcFaces), 3)]

            #Convert grouped vertices/faces to Open3D objects 
            o3dVertices = o3d.utility.Vector3dVector(np.asarray(vertices))
            o3dTriangles = o3d.utility.Vector3iVector(np.asarray(faces))

            # Create the Open3D mesh object
            mesh=o3d.geometry.TriangleMesh(o3dVertices,o3dTriangles)
            if len(mesh.triangles)>1:
                return mesh
            else: 
                return None 
    except:
        print('Geometry production error')
        return None
def ifc_get_materials(ifcElements:List[ifcopenshell.entity_instance])-> List[str]: 
    """Get ifc materials from an ifcElement

    Args:
        ifcElements (List[ifcopenshell.entity_instance])

    Returns:
        List[str]: names of materials
    """
    material_list=[]
    for ifcElement in ut.item_to_list(ifcElements):
        if ifcElement.HasAssociations:
            for i in ifcElement.HasAssociations:
                if i.is_a('IfcRelAssociatesMaterial'):
                    if i.RelatingMaterial.is_a('IfcMaterial'):
                        material_list.append(i.RelatingMaterial.Name)

                    if i.RelatingMaterial.is_a('IfcMaterialList'):
                        for materials in i.RelatingMaterial.Materials:
                            material_list.append(materials.Name)

                    if i.RelatingMaterial.is_a('IfcMaterialLayerSetUsage'):
                        for materials in i.RelatingMaterial.ForLayerSet.MaterialLayers:
                            material_list.append(materials.Material.Name)
    return material_list

def img_to_arrays(path:str,tasknr:int=0)->Tuple[np.array,int]:
    """Convert an image from a file path to a tuple of 1 np.arrays and a tasknr (this function is used for multi-processing).\n

    Args:
        1. path (str): path to mesh file \n
        2. tasknr(int,optional): tasknr used to keep the order in multiprocessing.\n

    Returns:
        Tuple[img (np.array), tasknr (int)]
    """
    path = str(path)
    img=cv2.imread(path)
    return img, tasknr

def join_geometries(geometries:List[o3d.geometry.Geometry])->o3d.geometry.Geometry:
    """Join together a number of o3d geometries e.g. LineSet, PointCloud or o3d.TriangleMesh instances.

    **NOTE**: Only members of the same geometryType can be merged.
    
    **NOTE**: np.arrays can also be processed (these are processed as point clouds)

    Args:
        - geometries (List[o3d.geometry.Geometry]) : LineSet, PointCloud, OrientedBoundingBox, TriangleMesh or np.array[nx3]

    Returns:
        merged o3d.geometries
    """
    if not geometries:
        raise ValueError("The geometries list is empty.")
    
    geom_type = type(geometries[0])
    if not all(isinstance(g, geom_type) for g in geometries):
        raise TypeError("All geometries must be of the same type.")
    
    if geom_type == o3d.geometry.PointCloud:
        merged = o3d.geometry.PointCloud()
        for g in geometries:
            merged += g
        return merged
    
    elif geom_type == o3d.geometry.LineSet:
        merged = o3d.geometry.LineSet()
        for g in geometries:
            merged += g
        return merged
    
    elif geom_type == o3d.geometry.TriangleMesh:
        merged = o3d.geometry.TriangleMesh()
        for g in geometries:
            merged += g
        return merged
    
    elif isinstance(geometries[0], np.ndarray):
        merged = np.vstack(geometries)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged))
    
    else:
        raise TypeError("Unsupported geometry type for merging.")
        

def las_to_pcd(las:laspy.LasData, transform:np.array=None, getColors:bool=True,getNormals:bool=False)->o3d.geometry.PointCloud:
    """Converts a laspy point cloud to an open3d point cloud.\n

    Args:
        1. las (laspy.LasData): laspy point cloud.\n
        2. transform (np.array[4x4], optional): offset transform i.e. to remove global coordinates. Defaults to None.\n
        3. getColors (bool, optional): Defaults to True.\n
        4. getNormals (bool, optional): Defaults to False.\n

    Returns:
        o3d.geometry.PointCloud
    """
    #validate transform
    if transform is not None:
        assert transform.shape[0]==4
        assert transform.shape[1]==4
    
    #create point cloud
    pcd = o3d.geometry.PointCloud()    
    newxyz=transform_points( las.xyz,transform) if transform is not None else las.xyz
    pcd.points=o3d.utility.Vector3dVector(newxyz)
    
    #compute colors
    if (all(elem.casefold() in las.point_format.dimension_names for elem in ['red', 'green', 'blue'])):
        red = las['red']
        green = las['green']
        blue = las['blue']
        #if color is 32 bit, only keep 8 bit color
        if red.max()>255:
            red = las['red'] >> 8 & 0xFF
            green = las['green'] >> 8 & 0xFF
            blue = las['blue'] >> 8 & 0xFF
        # if colorspace is [0-255] -> remap to [0-1]
        if red.max() >1:
            red=red/255
            green=green/255
            blue=blue/255
        pcd.colors=o3d.utility.Vector3dVector(np.vstack((red,green,blue)).transpose())
        
    #compute normals
    if getNormals:
        if all(n in ['normalX','normalY','normalZ'] for n in list(las.point_format.dimension_names)):
            normals=np.hstack((las['normalX'],las['normalY'],las['normalZ']))
            newNormals=transform_points(normals,transform) if transform is not None else normals
            pcd.normals=o3d.utility.Vector3dVector(newNormals)
        else:
            pcd.estimate_normals()
    return pcd

def las_add_extra_dimensions(las:laspy.LasData, recordData:np.array, names:List[str]=['newField'], dtypes:List[str]=["uint8"] )-> laspy.LasData:
    """Add one or more columns of data to an existing las point cloud file.\n
    View laspy dimension and type formatting at https://laspy.readthedocs.io/en/latest/lessbasic.html.\n
    
    **NOTE**: to be tested
    
    Args:
        1. las (laspy.LasData): las point cloud (laspy API).\n
        2. recordData (tuple with np.array[len(pcd.points)] or pandas.DataFrame): array or dataframe with len(columns) == len(pcd.points).\n
        3. names (str, optional): dimension names. Defaults to ['newField'].\n
        4. dtypes (str, optional): types of the columns. Defaults to ["uint8"].\n

    Returns:
        laspy.LasData: _description_
    """
    #validate inputs
    names=ut.item_to_list(names)
    dtypes=ut.item_to_list(dtypes)
    assert 'las' in str(type(las))
    assert len(names)==len(dtypes)    
    if 'array' in str(type(recordData)):
        assert recordData.shape[0]==las.x.shape[0] ,f'one of the recordData contains entries != (len(las.xyz)).'
        assert recordData.shape[1]==len(names), f'recordData.shape[1] !=len(names).'
    elif 'dataframe' in str(type(recordData)):
        assert len(recordData)==las.x.shape[0],f'one of the recordData contains entries != (len(las.xyz)).'
        assert len(recordData.columns)==len(names), f'recordData.shape[1] !=len(names).'
    elif 'Tuple' in str(type(recordData)):
        assert len(recordData)==len(names), f'len(recordData) !=len(names).'
        
    #create dimensions
    extraBytesParams=[laspy.ExtraBytesParams(name=name, type=dtype) for name,dtype in zip(names,dtypes)]
    las.add_extra_dims(extraBytesParams)   

    #add data
    if 'array' in str(type(recordData[0])):
        [setattr(las,name,recordData[i]) for i,name in enumerate(names)]
    elif 'dataframe' in str(type(recordData)):
        [setattr(las,name,recordData[[i]]) for i,name in enumerate(names)]
    return las

def las_get_data(las,indices:np.ndarray=None,excludedList:List[str]=None)->np.ndarray:
    """Get all the relevant data from a las file i.e. the points, colors, intensity and user assigned values such as the classification or features.

    Args:
        las (laspy.Laspy): point cloud to extract the data from
        indices (np.ndarray): array with indices to extract
        excludedList (List[str], optional): List with point fields to exlude. []'X', 'Y', 'Z','red', 'green', 'blue'] should be removed as they are automatically assigned if present. Other values that are excluded are ['X', 'Y', 'Z','red', 'green', 'blue','return_number', 'number_of_returns', 'synthetic', 'key_point', 
            'withheld', 'overlap', 'scanner_channel', 'scan_direction_flag', 
            'edge_of_flight_line', 'user_data', 'scan_angle', 'point_source_id', 'gps_time'].

    Returns:
        np.array (points,colors, user_defined_values)
    """
    # get dimension names
    dimension_names=list(las.point_format.dimension_names)

    # remove unwanted dimension names
    excludedList=None
    excludedList=['X', 'Y', 'Z','red', 'green', 'blue','return_number', 'number_of_returns', 'synthetic', 'key_point', 
                'withheld', 'overlap', 'scanner_channel', 'scan_direction_flag', 
                'edge_of_flight_line', 'user_data', 'scan_angle', 'point_source_id', 'gps_time'] if excludedList is None else excludedList
    dimension_names=[dim for dim in dimension_names if dim not in excludedList]

    # gather points
    data=las.xyz 
    names=['X', 'Y', 'Z']

    # gather color if present
    if las['red'].T.shape[0]==data.shape[0]:
        data=np.hstack((data,np.array([las['red']]).T,np.array([las['red']]).T,np.array([las['red']]).T)) 
        names.extend(['red', 'green', 'blue'])

    # gather other fields if present
    for dim in dimension_names:
        if las[dim].T.shape[0]==data.shape[0] :
            data=np.hstack((data,np.array([las[dim]]).T)) 
            names.append(dim)
    
    #get indices if present
    data=data[indices] if indices is not None else data
    
    return data,names

def las_get_normals(las:laspy.LasData,transform:np.array=None) -> np.array:
    if all(n in ['normalX','normalY','normalZ'] for n in list(las.point_format.dimension_names)):
        normals=np.hstack((las['normalX'],las['normalY'],las['normalZ']))
        normals=transform_points(normals,transform) if transform is not None else normals
    else:
        normals=np.asarray(las_to_pcd(las,transform,getColors=False,getNormals=True).normals)    
    return normals

def las_subsample(las: laspy.LasData,percentage:float=0.1) -> laspy.LasData:
    """Subsample a las file given a percentage [0-1].
    
    The order is assumed to be ['X','Y','Z','red','green','blue','intensity','classification']
    
    **NOTE**: if a classification is present, its maximum value must be <32 (unit4) else the values are remapped
    
    Args:
        1.las ( laspy.LasData): las file
        2.percentage (float, optional): percentage to downsample [0-1].

    Returns:
        laspy.lasdata: output las file 
    """
    #get all las arrays
    array,names=las_get_data(las)
    
    #subsample array
    array=array_subsample(array,percentage)
        
    #create a new las header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(array[:,0:3], axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])
    new_las = laspy.LasData(header)
    
    #fill in las from array    
    new_las.x = array[:, 0]
    new_las.y = array[:, 1]
    new_las.z = array[:, 2]
    array=array[:,3:]
    
    #add rgb if present
    if any(n for n in names if n in ['red','green','blue']):
        new_las.red=array[:, 0]
        new_las.green=array[:, 1]
        new_las.blue=array[:, 2]
        array=array[:,3:]
    
    #add intensity if present
    if any(n for n in names if n in ['intensity']):
        new_las.intensity=array[:, 0]
        array=array[:,1:]
    
    #add classification if present (max 31)
    if any(n for n in names if n in ['classification']):        
        new_las.classification=array[:, 0] if array[:, 0].max()<=31 else array[:, 0]-array[:, 0].min()
        array=array[:,1:]
    
    #create extra dims for scalar fields
    names= [n for n in names if n not in ['X','Y','Z','red','green','blue','intensity','classification']]
    dtypes=['float32' for n in names]
    new_las=las_add_extra_dimensions(new_las,array,names=names,dtypes=dtypes)    
    
    return new_las

def mesh_to_arrays(path:str,tasknr:int=0)->Tuple[np.array,np.array,np.array,np.array,int]:
    """Convert a mesh from a file path to a tuple of 4 np.arrays.\n

    Features:
        0. vertexArray,triangleArray,colorArray,normalArray,tasknr. \n

    Args:
        1. path (str): path to mesh file. \n
        2. tasknr(int,optional): tasknr used to keep the order in multiprocessing.\n

    Returns:
        Tuple[np.array,np.array,np.array,int]: vertexArray,triangleArray,colorArray,normalArray,tasknr
    """
    mesh=o3d.io.read_triangle_mesh(str(path))   
    vertexArray=np.asarray(mesh.vertices)
    triangleArray=np.asarray(mesh.triangles)
    colorArray=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    normalArray=np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
    return vertexArray,triangleArray,colorArray,normalArray,tasknr
def create_visible_point_cloud_from_meshes (geometries: List[o3d.geometry.TriangleMesh], 
                                            references:List[o3d.geometry.TriangleMesh], 
                                            resolution:float = 0.1,
                                            getNormals:bool=False)-> Tuple[List[o3d.geometry.PointCloud], List[float]]:
    """Returns a set of point clouds sampled on the geometries. Each point cloud has its points filtered to not lie within or collide with any of the reference geometries. As such, this method returns the **visible** parts of a set of sampled point clouds. \n
    
    For every point cloud, the percentage of visibility is also reported. This method takes about 50s for 1000 geometries. \n
    \n
    E.g. The figure shows the points of the visible point cloud that were rejected due to their proximity to the other mesh geometries.

    .. image:: ../../../docs/pics/invisible_points.PNG

    Args:
        1. geometries (List[o3d.geometry.TriangleMesh]): Meshes that will be sampled up to the resolution. \n
        2. references (List[o3d.geometry.TriangleMesh]): reference meshes that are used to spatially filter the sampled point clouds so only 'visible' points are retained. If some targget \n
        3. resolution (float, optional): Spatial resolution to sample meshes. Defaults to 0.1m. \n

    Raises:
        ValueError: any('TriangleMesh' not in str(type(g)) for g in geometries )\n
        ValueError: any('TriangleMesh' not in str(type(g)) for g in references )\n

    Returns:
        Tuple[List[o3d.geometry.PointCloud], List[percentages [0-1.0]]] per geometry
    """
    geometries=ut.item_to_list(geometries)    
    references=ut.item_to_list(references)    

    #validate geometries
    if  any('TriangleMesh' not in str(type(g)) for g in geometries ):
        raise ValueError('Only submit o3d.geometry.TriangleMesh objects') 
    #validate geometries
    if  any('TriangleMesh' not in str(type(g)) for g in references ):
        raise ValueError('Only submit o3d.geometry.TriangleMesh objects') 

    colorArray=np.random.random((len(geometries),3))
    identityPointClouds=[]
    percentages=[]

    for i,geometry in enumerate(geometries):        
        # create a reference scene 
        referenceGeometries=[g for g in references if g !=geometry ]
        reference=join_geometries(referenceGeometries)
        scene = o3d.t.geometry.RaycastingScene()
        cpuReference = o3d.t.geometry.TriangleMesh.from_legacy(reference)
        _ = scene.add_triangles(cpuReference)

        # sample mesh (optional with normals)
        if getNormals and not geometry.has_triangle_normals():
            geometry.compute_triangle_normals()
        area=geometry.get_surface_area()
        count=int(area/(resolution*resolution))
        pcd=geometry.sample_points_uniformly(number_of_points=10*count,use_triangle_normal=getNormals)
        pcd=pcd.voxel_down_sample(resolution)

        # determine visibility from distance and occupancy querries
        query_points = o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32)
        unsigned_distance = scene.compute_distance(query_points)
        occupancy = scene.compute_occupancy(query_points)
        indices=np.where((unsigned_distance.numpy() >=0.5*resolution) & (occupancy.numpy() ==0) )[0]     

        # crop sampled point cloud to only the visible points
        pcdCropped = pcd.select_by_index(indices)
        pcdCropped.paint_uniform_color(colorArray[i])
        identityPointClouds.append(pcdCropped)

        #report percentage
        percentages.append((len(pcdCropped.points)/len(pcd.points)))
    return identityPointClouds, percentages

def mesh_to_trimesh(geometry: o3d.geometry.Geometry) -> trimesh.Trimesh:
    """Convert open3D.geometry.TriangleMesh to [trimesh.Trimesh](https://trimsh.org/trimesh.html). \n
    
    **NOTE**: Only vertex_colors are implemented instead of face_colors with textures.\n

    Args:
        geometry (Open3D.geometry): OrientedBoundingBox, AxisAlignedBoundingBox or TriangleMesh
    
    Returns:
        trimesh.Trimesh
    """
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        vertices = np.asarray(geometry.vertices)
        faces = np.asarray(geometry.triangles)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Optionally add vertex normals
        if geometry.has_vertex_normals():
            mesh.vertex_normals = np.asarray(geometry.vertex_normals)

        # Optionally add vertex colors
        if geometry.has_vertex_colors():
            colors = np.asarray(geometry.vertex_colors)
            if colors.max() <= 1.0:  # Open3D uses [0, 1], Trimesh expects [0, 255]
                colors = (colors * 255).astype(np.uint8)
            mesh.visual.vertex_colors = colors

        return mesh
    
    elif isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
        point_cloud = trimesh.points.PointCloud(points)

        # Add colors if present
        if geometry.has_colors():
            colors = np.asarray(geometry.colors)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            point_cloud.colors = colors

        # Add normals if present (stored as metadata, not native in Trimesh PointCloud)
        if geometry.has_normals():
            point_cloud.metadata['normals'] = np.asarray(geometry.normals)

        return point_cloud

    elif isinstance(geometry, o3d.geometry.LineSet):
        points = np.asarray(geometry.points)
        lines = np.asarray(geometry.lines)
        entities = [trimesh.path.entities.Line([start, end]) for start, end in lines]
        path = trimesh.path.Path3D(entities=entities, vertices=points)

        # Optional: attach colors if present
        if geometry.has_colors():
            colors = np.asarray(geometry.colors)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            path.metadata['colors'] = colors

        return path
    
    else:
        raise TypeError(f"Unsupported Open3D geometry type: {type(geometry)}")
    
def mesh_get_lineset(geometry: o3d.geometry.TriangleMesh, color: np.array = ut.get_random_color()) -> o3d.geometry.LineSet:
    """Returns a lineset representation of a mesh.\n

    Args:
        1. geometry (open3d.geometry.trianglemesh): The mesh to convert. \n
        2. color (np.array[3,], optional): The color to paint the lineset. Defaults to random color.\n

    Returns:
        open3d.geometry.LineSet: the lineset from the mesh.
    """
    assert len(color)==3 
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(geometry)
    ls.paint_uniform_color(color)
    return ls

def normalize_vectors(array:np.ndarray, axis:int=-1, order:int=2) -> np.ndarray:
    """Normalize an set of vectors np.array(:,3) to unit vectors len(1).\n

    Args:
        array (np.array(:,3))
        axis (int, optional): Defaults to -1.
        order (int, optional): Defaults to 2.

    Returns:
        np.array(:,3)
    """
    l2 = np.atleast_1d(np.linalg.norm(array, order, axis))
    l2[l2==0] = 1
    return array / np.expand_dims(l2, axis)

def octree_to_voxelmesh(octree:o3d.geometry.Octree)-> o3d.geometry.TriangleMesh:
    """Create a TriangleMesh from an octree.

    Args:
        octree (o3d.geometry.Octree): size of each node will be used to scale the voxels.

    Returns:
        o3d.geometry.TriangleMesh
    """

    def f_traverse_generate_mesh(node, node_info):
        """Callback function that changes the color to the dominant color in each leafnode.
        """
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):               
                cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                # paint it with the color of the current voxel
                cube.paint_uniform_color(node.color) 
                # scale the box using the size of the voxel
                cube.scale(node_info.size, center=cube.get_center())
                # translate the box to the center of the voxel
                cube.translate(node_info.origin, relative=False)
                # add the box to the TriangleMesh object
                cubes.append(cube) 
        return None
    cubes=[]
    octree.traverse(f_traverse_generate_mesh) 
    return join_geometries(cubes)
    

def pcd_to_arrays(path:str,percentage:float=1.0,tasknr:int=0)->Tuple[np.array,np.array,np.array,int]:
    """Convert a pcd from a pcd file to a tuple of 3 np.arrays.\n

    Features:
        1. ['cartesianX', 'cartesianY', 'cartesianZ'] \n
        2. ['colorRed', 'colorGreen', 'colorBlue'] \n
        3. ['normalX', 'normalY', 'normalZ']\n
        4. tasknr (int): int to retrieve order in multiprocessing.\n

    Args:
        1. path (str): path to .pcd file. \n
        2. percentage (float,optional): downsampling ratio. defaults to 1.0 (100%). \n
        3. tasknr (int): int to retrieve order in multiprocessing.\n

    Returns:
        Tuple[np.array,np.array,np.array,np.array]: pointArray,colorArray,normalArray,tasknr
    """
    pcd=o3d.io.read_point_cloud(str(path))   
    pointArray=np.asarray(pcd.points)
    colorArray=np.asarray(pcd.colors) if pcd.has_colors() else None
    normalArray=np.asarray(pcd.normals) if pcd.has_normals() else None    

    #downnsample
    if percentage <1.0:
        indices=np.random.randint(0,len(pcd.points)-1,int(len(pcd.points)*percentage))
        pointArray=pointArray[indices]
        if pcd.has_colors():
            colorArray=colorArray[indices]
        if pcd.has_normals():
            normalArray=normalArray[indices]
    return pointArray,colorArray,normalArray,tasknr

def pcd_to_las(pcd:o3d.geometry.PointCloud,**kwargs)->laspy.LasData:
    """Convert a dataframe representing a point cloud to a las point cloud file.
    View laspy dimension and type formatting at https://laspy.readthedocs.io/en/latest/lessbasic.html.\n
    
    E.g.: las=dataframe_to_las(dataframe,rgb=[3,4,5])    

    Args:
        1.dataframe (pd.DataFrame): data frame with a number of columns such as xyz, rgb and some scalar fields (conform numpy)
        2.xyz (List[int], optional): Indices of the xyz coordinates in the dataframe. Defaults to [0,1,2].
        3.rgb (List[int], optional): Indices of the color information in the dataframe e.g. [3,4,5]. Defaults to None.
        4.dtypes (List[str], optional): types of the scalar fields that will be added e.g. ['float32','uint8']. Defaults to [float32] equal to the length of the scalar fields.

    Returns:
        laspy.lasdata: output las file 
    """
    #0.get xyz data
    xyz=np.asarray(pcd.points)
    
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(xyz, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # 2. Create a Las from xyz
    las = laspy.LasData(header)

    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    
    # 3. add rgb if present
    if pcd.has_colors():
        rgb=np.asarray(pcd.colors)
        las.red=rgb[:, 0]
        las.green=rgb[:, 1]
        las.blue=rgb[:, 2]
        
    # 4. Create extra dims for scalar fields
    names = list(kwargs.keys())
    dtypes = [np.asarray(kwargs[name]).dtype for name in names]        
    extraBytesParams=[laspy.ExtraBytesParams(name=name, type=dtype) for name,dtype in zip(names,dtypes)]
    las.add_extra_dims(extraBytesParams)   
    
    # 5. set data      
    [setattr(las,name,np.asarray(kwargs[name])) for name in names]
    
    return las



def pcd_to_voxelmesh(pcd:o3d.geometry.PointCloud,voxel_size:float=0.4,colorUse:int=0 )-> o3d.geometry.TriangleMesh:
    """Create a TriangleMesh from a PointCloud.

    Args:
        pcd (o3d.geometry.PointCloud): 
        voxel_size (float, optional): size of the voxels. Defaults to 0.4m.
        colorUse (int, optional): If 0, the colors per voxel will be averaged. If 1, the dominant color per voxel will be retained (this is quite slow). Defaults to 0.

    Returns:
        o3d.geometry.TriangleMesh: _description_
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,  voxel_size)       
    colorArray=np.empty((len(voxel_grid.get_voxels()),3))
    
    if colorUse==1:
        colors=np.asarray(pcd.colors)
        graycolors,ind=np.unique(np.dot(colors[...,:3], [0.2989, 0.5870, 0.1140]),return_index=True)
        uniquecolors=colors[ind]
        
        for i,v in enumerate(voxel_grid.get_voxels()):
            c=np.dot(v.color[...,:3], [0.2989, 0.5870, 0.1140])
            idx=np.abs(graycolors - c).argmin()
            colorArray[i]=uniquecolors[idx]
        return voxelgrid_to_mesh(voxel_grid,voxel_size,colorArray)
    return voxelgrid_to_mesh(voxel_grid,voxel_size)                                           

def pcd_get_normals(pcd:o3d.geometry.PointCloud)->np.ndarray:
    """Compute open3d point cloud normals if not already present.\n

    Args:
        pcd (o3d.geometry.PointCloud)

    Returns:
        np.array:
    """
    pcd.estimate_normals() if not pcd.has_normals() else None
    return np.asarray(pcd.normals)

def preprocess_points_for_geometry(points, thickness=1e-3):
    """Handles special cases: single point, colinear, and coplanar points."""
    # Remove all the duplicate points
    points = np.unique(points, axis=0)
    
    if len(points) == 1:
        # Single point: Expand to a small cube
        offsets = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]) * thickness * 10
        return points + offsets
    
    elif len(points) == 2 or np.linalg.matrix_rank(points - points.mean(axis=0)) == 1:
        # Colinear case: Expand perpendicularly
        line_vec = points[-1] - points[0]
        line_vec = line_vec / np.linalg.norm(line_vec)

        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(line_vec, arbitrary_vec) or np.allclose(line_vec, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])

        perp1 = np.cross(line_vec, arbitrary_vec)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(line_vec, perp1)

        return np.vstack([points, points + perp1 * thickness, points - perp1 * thickness,
                                  points + perp2 * thickness, points - perp2 * thickness])

    elif np.linalg.matrix_rank(points - points.mean(axis=0)) == 2:
        # Coplanar case: Add minimal thickness
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]  # Smallest eigenvector
        return np.vstack([points, points + normal * thickness, points - normal * thickness])

    return points  # If already 3D, return as-is
def project_meshes_to_rgbd_images (meshes:List[o3d.geometry.TriangleMesh], extrinsics: List[np.array],intrinsics:List[np.array], scale:float=1.0, fill_black:int=0)->Tuple[List[np.array],List[np.array]]:
    """Project a set of meshes given camera parameters.

    .. image:: ../../../docs/pics/Raycasting_6.PNG

    Args:
        1.meshes (List[o3d.geometry.TriangleMesh]): set of TriangleMeshes.\n
        2.imgNodes (List[ImageNode]): should contain imageWidth,imageHeight,cartesianTransform and focalLength35mm\n
        3.scale (float, optional): scale to apply to imagery (typically for downscaling). Defaults to 1.\n
        4.fill_black (int, optional): Region to fill in black pixels. 5 is a good value.\n
        
    Returns:
        Tuple[List[np.array],List[np.array]]: colorImages,depthImages
    """
    #validate meshes
    mesh=join_geometries(ut.item_to_list(meshes))
    extrinsics=ut.item_to_list(extrinsics)
    intrinsics=ut.item_to_list(intrinsics)
    
    
    #create lists
    colorImages=[]
    depthImages=[]    
    
    #create raytracing scene
    scene = o3d.t.geometry.RaycastingScene()
    reference=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(reference)
    
    #create a colorArray from the mesh triangles (color of first vertex is taken)
    colors=np.asarray(mesh.vertex_colors)
    indices=np.asarray(mesh.triangles)[:,0]
    triangle_colors=colors[indices]
    #append black color at the end of the array for the invalid hits
    triangle_colors=np.vstack((triangle_colors,np.array([0,0,0])))
    
    #create rays  
    for e,i in zip(extrinsics,intrinsics):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                                        intrinsic_matrix =i,
                                        extrinsic_matrix =np.linalg.inv(e),
                                        width_px=math.ceil((i[0,2]+0.5)*2),
                                        height_px=math.ceil((i[1,2]+0.5)*2))
        
        #apply scale
        rays=rays.numpy()
        rays=o3d.core.Tensor(rays[::int(1/scale),::int(1/scale)])
        #cast rays
        ans = scene.cast_rays(rays) 
        
        #get triangle_ids that are hit per ray
        triangle_ids=ans['primitive_ids'].numpy() # triangles
        rows,columns=triangle_ids.shape        
        triangle_ids=triangle_ids.flatten()
        # replace invalid id's by last (which is the above added black color)
        np.put(triangle_ids,np.where(triangle_ids==scene.INVALID_ID),triangle_colors.shape[0]-1) 
        
        #select colors 
        colors=triangle_colors[triangle_ids]

        #reshape array back to normal
        colorImage=np.reshape(colors,(rows,columns,3))
                
        #fill black if necessary
        colorImage=iu.fill_black_pixels(colorImage,fill_black)         if fill_black !=0       else colorImage
        depthImage=iu.fill_black_pixels(ans['t_hit'].numpy(),fill_black)         if fill_black !=0       else ans['t_hit'].numpy()

        #add to list
        colorImages.append(colorImage)    
        depthImages.append(depthImage)

    return colorImages,depthImages

def rays_to_lineset(rays:np.ndarray,distances=None)->o3d.geometry.LineSet:
    """Convert an array or o3d.tensor to a lineset that can be visualized in open3d.\n
    
    .. image:: ../../../docs/pics/Raycasting_3.PNG

    Args:
        1.rays (np.array[n,6] or o3d.core.Tensor): ray consisting of a startpoint np.array[n,0:3] and a direction np.array[n,3:6]\n
        2.distances (float or np.array[n],Optional): distance/distances over which to cast each ray. Defaults to 1.0m. 

    Returns:
        o3d.geometry.LineSet
    """
    # Reshape rays and distances if necessary
    rays = np.asarray(rays) if not isinstance(rays,np.ndarray) else rays
    
    if rays.ndim == 1 and distances is None:
        distances=np.full((1,), 1)
    elif rays.ndim == 2 and distances is None:
        distances=np.full((1,1), 1)
    elif rays.ndim == 2 and distances is not None:
        distances=np.full((1,1), 1)
    elif distances is not None and distances.size != rays.shape[0]:
        raise ValueError("The size of distances array must match the number of rays")

    distances[distances == np.inf] = 50
        
    #get start and endpoints
    startpoints, endpoints=rays_to_points(rays,distances)
    points=np.vstack((startpoints,endpoints))
    
    #create lines
    lines=[]
    start=np.arange(start=0,stop=rays.shape[0] )[..., np.newaxis]
    end=np.arange(start=rays.shape[0],stop=points.shape[0] )[..., np.newaxis]  
    lines = np.hstack((start, end))
    
    #create lineset
    line_set = o3d.geometry.LineSet()
    # colors = [[1, 0, 0] for i in range(len(lines))]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

def rays_to_points(rays:np.ndarray,distances:np.ndarray=np.array([1.0])) -> Tuple[np.ndarray,np.ndarray]:
    """Converts a set of rays to start-and endpoints.\n

    Args:
        - rays (np.array[n,6] or o3d.core.Tensor): ray consisting of a startpoint np.array[n,0:3] and a direction np.array[n,3:6].\n
        - distances (np.array[n], optional): scalar or array with distances of the ray. Defaults to 1.0m.\n

    Returns:
        Tuple[np.array,np.array]: startpoints, endpoints
    """
    
    # Validate inputs
    if 'Tensor' in str(type(rays)):
        rays = rays.numpy()
    
    # Reshape rays if necessary
    rays = np.asarray(rays) if not isinstance(rays,np.ndarray) else rays

    if rays.ndim == 1:
        assert rays.shape[0] == 6, f'rays.shape[0] should be 6, got {rays.shape[0]}.'
        rays = np.reshape(rays, (1, 6))
    elif rays.ndim == 2:
        assert rays.shape[1] == 6, f'rays.shape[1] should be 6, got {rays.shape[1]}.'
    else:
        raise ValueError("Invalid rays shape. Expected shape (n, 6) or (6,).")
    
        
    # Ensure distances is a 1D array with the correct size
    distances=np.asarray(distances)if not isinstance(distances,np.ndarray) else distances
    if distances.size != rays.shape[0]:
        if distances.size == 1:
            distances = np.full((rays.shape[0],), distances[0])
        else:
            raise ValueError("The size of distances array must match the number of rays")

    
    #stack rays and distances
    rays=np.hstack((rays,np.reshape(distances,(rays.shape[0],1))))
    
    #compute endpoints
    def myfunction(x):
        return np.array([x[0] + x[3]*x[-1],
                         x[1] + x[4]*x[-1],
                         x[2] + x[5]*x[-1]])
    startpoints=rays[:,0:3]
    endpoints=np.apply_along_axis(myfunction, axis=1, arr=rays)    
    
    return startpoints, endpoints

    

def sample_geometry(geometries:List[o3d.geometry.Geometry],resolution:float=0.1)->List[o3d.geometry.PointCloud]:
    """Sample the surface, line or point cloud of an open3d object given a resolution.

    Args:
        geometries (List[o3d.geometry.Geometry]): o3d.Geometry.LineSet,o3d.Geometry.TriangleMesh or o3d.Geometry.PointCloud
        resolution (float, optional): spacing between sampled points. Defaults to 0.1m.

    Returns:
        List[o3d.geometry.PointCloud]
    """
    isList = isinstance(geometries, List) # if the input is a list, always return a list
    geometries=ut.item_to_list(geometries)
        
    point_clouds=[]
    for g in geometries:
        pcd=o3d.geometry.PointCloud()
        
        if 'TriangleMesh' in str(type(g)) and len(g.vertices) != 0:
            area=g.get_surface_area()
            count=int(area/(resolution*resolution))
            pcd=g.sample_points_uniformly(number_of_points=count)
            
        elif 'PointCloud' in str(type(g)) and len(g.points) != 0: 
            pcd=g.voxel_down_sample(resolution)
            
        elif 'ndarray' in str(type(g)):
            pcd.points=o3d.utility.Vector3dVector(g)
            pcd=pcd.voxel_down_sample(resolution)
            
        elif 'LineSet' in str(type(g)):    
            # Get line segments from the LineSet
            pointArray=np.asarray(g.points)
            points = []

            for line in np.asarray(g.lines): #! this is not efficient
                #get start and end
                start_point = pointArray[line[0]]
                end_point = pointArray[line[1]]
                #get direction and length
                direction = end_point - start_point
                length = np.linalg.norm(direction)
                #compute number of points
                num_points = int(length / resolution)
                if num_points > 0:
                    step = direction / num_points
                    p=[start_point + r * step for r in range(num_points + 1)]
                    points.extend(p)
            pcd.points=o3d.utility.Vector3dVector(points)  
                  
        point_clouds.append(pcd)
        
    return point_clouds if len(point_clouds)>1 or isList else point_clouds[0]

def save_dataframe_as_ply(filename, points=None, mesh=None, as_text=False, comments=None):
    """Write a PLY file populated with the given fields.\n
    
    Args:
        1. filename (str) :The created file will be named with this\n
        2. points (ndarray): \n
        3. mesh (ndarray): \n
        4. as_text (bool): Set the write mode of the file. Defaults to binary.\n
        5. comments: list of string\n
        
    Returns
        bool: True if no problems
    """
    if not filename.endswith('ply'):
        filename += '.ply'
    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']
        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')
            
        if comments:
            for comment in comments:
                header.append('comment ' + comment)

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))
        header.append('end_header')
        for line in header:
            ply.write("%s\n" % line)
    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')
    else:
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)
    return True

def save_view_point(geometry: o3d.geometry, filename:str) -> None:
    """Saves the viewpoint of a set of geometry in the given filename.\n
    
    **NOTE**: this is very inefficient -> join geometries first

    Args:
        1. geometry (o3d.geometry): geometries to visualize.\n
        2. filename (str): absolute filepath\n
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometry)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()

def segment_pcd_by_connected_component(pcd:o3d.geometry.PointCloud, eps:float=0.03, minPoints:int=10,printProgress:bool=False) -> List[o3d.geometry.PointCloud]:
    """Returns list of point clouds segmented by db_cluster DBSCAN algorithm Ester et al., ‘A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise’, 1996.\n

    Args:
        1. pcd (o3d.geometry.PointCloud) \n
        2. eps (float, optional): Density parameter that is used to find neighbouring points. Defaults to 0.03m.\n
        3. minPoints (int, optional): Minimum number of points to form a cluster. Defaults to 10.\n
        4. printProgress (bool)

    Raises:
        ValueError: len(pcd.points)<minPoints

    Returns:
        List[o3d.geometry.PointCloud]
    """
    # validate point cloud    
    assert len(pcd.points)<minPoints, f'len(pcd.points)<minPoints'
    
    pcds=[]
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=minPoints, print_progress=printProgress))

    labelList=np.unique(labels)    
    for l in range(0,labelList[-1]):
        indices = np.where(labels == l)[0]
        pcds.append(pcd.select_by_index(indices))
    return ut.item_to_list(pcds)
def show_geometries(geometries : 'List[o3d.geometry]', color : bool = False):
    """Displays different types of geometry in a scene

    **NOTE**: this is very inefficient -> join geometries first

    Args:
        geometries (List[open3d.geometry]): The list of geometries
        color (bool, optional): recolor the objects to have a unique color. Defaults to False.
    """
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    viewer.add_geometry(frame)
    for i, geometry in enumerate(geometries):
        if color:
            geometry.paint_uniform_color(ut.get_random_color())
            # geometry.paint_uniform_color(matplotlib.colors.hsv_to_rgb([float(i)/len(geometries),0.8,0.8]))
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([1,1,1])
    opt.light_on = True
    viewer.run()

def split_pcd_by_labels(point_cloud:o3d.geometry.PointCloud,labels:np.ndarray)-> Tuple[List[o3d.geometry.PointCloud],np.ndarray]:
    """Split a point cloud in parts to match a list of labels. The result is a set of point clouds, one for each unique label.

    Args:
        point_cloud (o3d.geometry.PointCloud):
        labels (np.ndarray): integer array with the same length as the point clouds 

    Returns:
        Tuple[List[o3d.geometry.PointCloud],np.ndarray]: point clouds, unique labels
    """
    pcdList=[]
    unique=np.unique(labels)
    for i in unique:    
        ind = np.where(labels==i)[0]
        pcdList.append(point_cloud.select_by_index(ind))
    return pcdList,unique

def split_quad_faces(faces:np.ndarray) ->np.ndarray:
    """Split an array of quad faces e.g. [[0,1,2,3]] into triangle faces e.g. [[0,1,2],[0,2,3]]

    Args:
        faces (np.ndarray[nx4]) 

    Returns:
        np.ndarray[2nx3]
    """
    newFaces=np.zeros((1,3),dtype=np.uint16)
    for f in faces:
        f=np.array(f)       
        newFaces=np.vstack((newFaces,np.array([f[0:3]]),np.array([f[0],f[2],f[3]]))) if f.shape[0]==4 else np.vstack((newFaces,f))
    newFaces=np.delete(newFaces,0,axis=0)
        
    return newFaces

def transform_dataframe(df:pd.DataFrame,transform:np.array,pointFields:List[str]=['x', 'y', 'z','Nx', 'Ny', 'Nz'])->pd.DataFrame:
    """apply rigid body transformation to the 3D point coordinates[x,y,z] in a pandas dataFrame.\n

    Args:
        1. df (pd.DataFrame)\n
        2. transform (np.array[4x4]): Rigid body transformation.\n
        3. pointFields (List[str], optional): names of the dataFrame columns. Defaults to ['x', 'y', 'z','Nx', 'Ny', 'Nz'].\n

    Raises:
        ValueError: 'No valid xyz data. Make sure column headers are names X,Y,Z'

    Returns:
        pd.DataFrame
    """
    if transform is not None:
        assert transform.shape[0]==4
        assert transform.shape[1]==4

    #validate pointfields    
    if pointFields == None:
        pointFields=['x', 'y', 'z','Nx', 'Ny', 'Nz']
    fields=[s.casefold() for s in pointFields]

    #transform XYZ
    if (all(elem.casefold() in fields for elem in ['X', 'Y', 'Z'])):
        xyz=df.get([pointFields[0], pointFields[1], pointFields[2]])
        newxyz=transform_points( xyz.to_numpy(),transform) if transform is not None else xyz.to_numpy()
        #replace column
        for i,n in enumerate([pointFields[0], pointFields[1], pointFields[2]]):
            df[n] = newxyz[:,i].tolist()

            # df.drop(n, axis = 1, inplace = True)
            # df[n] = newxyz[:,i].tolist()
    else:
        raise ValueError('No valid xyz data. Make sure column headers are names X,Y,Z')

    #transform ['Nx', 'Ny', 'Nz']
    if (all(elem.casefold() in pointFields for elem in ['Nx', 'Ny', 'Nz'])): 
        nxyz=df.get(['Nx', 'Ny', 'Nz'])
        newnxyz=transform_points( nxyz.to_numpy(),transform) if transform is not None else nxyz.to_numpy()
        #replace column
        for i,n in enumerate(['Nx', 'Ny', 'Nz']):
            df.drop(n, axis = 1, inplace = True)
            df[n] = newnxyz[:,i].tolist()
    return df

def transform_points(points:np.ndarray,transform:np.ndarray)->np.ndarray:
    """Transform points with transformation matrix.\n

    Args:
        1.points (np.array(:,3)): points to transform\n
        2.transform (np.array(4,4)): transformation Matrix\n

    Returns:
        np.array(:,3)
    """
    assert(points.shape[1] == 3)
    assert(transform.shape[0] == 4)
    assert(transform.shape[1]== 4)

    hpoints=np.hstack((points,np.ones((points.shape[0],1))))
    hpoints=transform @ hpoints.transpose()
    return hpoints[0:3].transpose()

def voxelgrid_to_mesh(voxel_grid:o3d.geometry.VoxelGrid,voxel_size:float=0.4,colorArray:np.array=None)-> o3d.geometry.TriangleMesh:
    """Create a TriangleMesh from a voxelGrid.
    
    .. image:: ../../../docs/pics/pcd2.PNG
    .. image:: ../../../docs/pics/voxelgrid_1.PNG
    
    Args:
        voxel_grid (o3d.geometry.VoxelGrid):
        voxel_size (float, optional): size of each voxel. Defaults to 0.4.
        colorArray (np.array,optional): optional colorArray np.Array(len(voxels),3) from [0-1]

    Returns:
        o3d.geometry.TriangleMesh
    """
    # get all voxels in the voxel grid
    voxels_all= voxel_grid.get_voxels()
    # geth the calculated size of a voxel
    voxel_size = voxel_grid.voxel_size
    # loop through all the voxels
    cubes=[]
    for i,voxel in enumerate(voxels_all):
        # create a cube mesh with a size 1x1x1
        cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        # paint it with the color of the current voxel
        cube.paint_uniform_color(voxel.color) if colorArray is None else cube.paint_uniform_color(colorArray[i])
        # scale the box using the size of the voxel
        cube.scale(voxel_size, center=cube.get_center())
        # get the center of the current voxel
        voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        # translate the box to the center of the voxel
        cube.translate(voxel_center, relative=False)
        # add the box to the TriangleMesh object
        cubes.append(cube)
    return join_geometries(cubes)

