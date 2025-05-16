"""Tools to combine meshes and pointclouds."""
import copy
import itertools
import geomapi
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import open3d as o3d
import numpy as np
from typing import Tuple, List
from scipy.spatial import Delaunay


# The main function to combine 2 aligned geometries
def combine_geometry(ogGeometry: o3d.geometry.Geometry, newGeometry :o3d.geometry.Geometry, distanceTreshold : float = 0.05, logProcess = False) -> o3d.geometry.Geometry:
    """Combines 2 aligned geometries assuming the ogGeometry is the reference and the newGeometry will suplement it.

    This is performed in a few steps:
        1) Create a convex hull of the newGeometry
        2) Filter the relevant points of the ogGeometry
        3) Perform a 2 step relevance check on the ogPoints if they fail either, they will be removed
            - Coverage check: perform a distance query, points that to far away from the mesh could be either out of date or not scanned
            - Visibility check: points that are inside the mesh are considered invisible and are kept, visible points are deemed out of date and are removed.
        4) Filter the newPoints to only add the points that are changed
        5) combine the changed-newPoints, the invisible-not-covered-ogPoints, the covered-ogPoints and the irrelevant points

    Args:
        ogGeometry (o3d.geometry): The reference geometry to be completed, will be sampled to a pointcloud if this is not already the case.
        newGeometry (o3d.geometry): The new geometry to be added
        distanceTreshold (float, optional): The treshold the distance-filtering is set to. Defaults to 0.05.

    Returns:
        o3d.geometry: the combined geometry as a pointcloud
    """

    # Step 0: Prepare the input data
    if(type(ogGeometry) == o3d.geometry.TriangleMesh):
        ogGeometry = mesh_to_pcd(ogGeometry, voxelSize=distanceTreshold/2)

    # Step 1: Create a convex hull of the newGeometry
    newGeoHull = gmu.get_convex_hull(newGeometry)
    if(logProcess): print("Covex hull created")
    # Step 2: Filter out the irrelevant points in the ogGeometry
    relevantOg, irrelevantOg = get_points_in_hull(ogGeometry, newGeoHull)
    if(logProcess): print("Irrelevant points filtered")
    # Step 3: Isolate the not covered points of the ogGeometry compared to the newGeometry
    newGeometryPoints = mesh_to_pcd(newGeometry,distanceTreshold/2)
    coveredPoints, unCoveredPoints = filter_pcd_by_distance(relevantOg, newGeometryPoints, distanceTreshold)
    if(logProcess): print("Covered poinys calculated")
    # Step 4: Perform the visibility check of the not covered points
    invisibleUncoveredPoints = get_invisible_points(unCoveredPoints, newGeometry)
    if(logProcess): print("invisible points detected")
    # Step 5: Filter the newGeometryPoints to only keep the changed geometry
    existingNewGeo, newNewGeo = filter_pcd_by_distance(newGeometryPoints, relevantOg, distanceTreshold)
    if(logProcess): print("new points filtered")
    # Step 6: Combine the irrelevant, unchanged and changed geometry
    newCombinedGeometry = irrelevantOg + coveredPoints + invisibleUncoveredPoints + newNewGeo
    if(logProcess): print("geometries combined")
    return newCombinedGeometry

# checks if the points is inside the (closed) mesh
def check_point_inside_mesh(points: List, mesh :o3d.geometry.Geometry) -> Tuple[List, List]:
    """Performs a visibility check to check if the points are inside the mesh or not

    Args:
        points (List): The points to check
        mesh (open3d.geometry): The mesh to check against

    Returns:
        Tuple[List, List]: The points withing the mesh, the points outside the mesh
    """

    # step 0: Set up a raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    insideList = []
    outsideList = []

    for point in points:
        # step 1: get the closest point to the mesh
        queryPoint = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
        queryResult = scene.compute_closest_points(queryPoint)
        closestPoint =  queryResult['points'].numpy()
        # step 2 Get the normal of the triangle
        closestTriangle = queryResult['primitive_ids'][0].item()
        triangleNormal = np.asarray(mesh.triangle_normals[closestTriangle])
        # step 3: compare the normal with the ray direction
        closestPointDirection = closestPoint - point
        dotProduct = (closestPointDirection @ triangleNormal)[0]
        if (dotProduct > 0):
            insideList.append(point)
        else:
            outsideList.append(point)

    return insideList, outsideList

# geturns all the points that are inside the (closed) mesh
def get_invisible_points(points : o3d.geometry.Geometry, mesh: o3d.geometry.Geometry) -> o3d.geometry.Geometry:
    """Returns all the points that are not visible because the are within a mesh

    Args:
        points (open3d.geometry): The points to chack
        mesh (open3d.geometry): The mesh to check against

    Returns:
        open3d.geometry: The filtered poincloud
    """
    insideList, outsideList = check_point_inside_mesh(points.points, mesh)
    visiblePoints = o3d.geometry.PointCloud()
    visiblePoints.points = o3d.utility.Vector3dVector(outsideList)
    invisiblePoints = o3d.geometry.PointCloud()
    invisiblePoints.points = o3d.utility.Vector3dVector(insideList)
    return invisiblePoints

def mesh_to_pcd(mesh:o3d.geometry.TriangleMesh,voxelSize : float = 0.1) -> o3d.geometry.PointCloud:
    """Sample a point cloud on a triangle mesh (Open3D).\n

    Args:
        1. mesh (o3d.geometry.TriangleMesh) : source geometry\n
        2. voxel_size (float) : spatial resolution of the point cloud e.g. 0.1m\n

    Returns:
        o3d.geometry.PointCloud
    """
    k = round(mesh.get_surface_area() * 1000)
    pcd = mesh.sample_points_uniformly(number_of_points = k, use_triangle_normal=True)
    pcd = pcd.voxel_down_sample(voxelSize)    
    return pcd

def get_points_in_hull(geometry : o3d.geometry.PointCloud, hull: o3d.geometry.TriangleMesh) -> Tuple[o3d.geometry.PointCloud,o3d.geometry.PointCloud]:
    """Separates a geometry in points inside and outside a convex hull.\n
    
    Args:
        1. geometry (open3d.geometry): The geometry to be filtered.\n
        2. hull (open3d.geometry): The hull to filter. \n

    Returns:
       Tuple[open3d.geometry, open3d.geometry]: Points inside the hull, Points outside the hull
    """
    hullVerts = np.asarray(hull.vertices)
    points = np.asarray(geometry.points)
    idxs = get_indices_in_hull(points, hullVerts)
    pcdInHull = geometry.select_by_index(idxs)
    pcdOutHull = geometry.select_by_index(idxs, True)
    return pcdInHull, pcdOutHull

def get_indices_in_hull(points : np.ndarray, hull :np.ndarray) -> List[int]:
    """Get the indices of all the points that are inside the hull.\n

    Args:
        1. points (numpy.array): should be a NxK coordinates of N points in K dimensions.\n
        2. hull (np.array): is either a scipy.spatial.Delaunay object or the MxK array of the coordinates of M points in K dimensions for which Delaunay triangulation will be computed.

    Returns:
        List[int]: The indices of the points that are in the hull.
    """
    hull = Delaunay(hull) if not isinstance(hull,Delaunay) else hull
    ind = hull.find_simplex(points)>=0

    intList = []
    for i,x in enumerate(ind):
        if ind[i]:
            intList.append(i)
    return intList

def filter_pcd_by_distance(sourcePcd : o3d.geometry.PointCloud, testPcd: o3d.geometry.PointCloud, maxDistance : float) -> Tuple[o3d.geometry.PointCloud,o3d.geometry.PointCloud]:
    """Splits the sourcePcd in close and too far points compared to the testPcd based on the Euclidean distance.\n

    Args:
        1. sourcePcd (open3d.geometry.PointCloud): The pcd to be split.\n
        2. testPcd (open3d.geometry.PointCloud): The pcd to test against.\n

    Returns:
        Tuple(o3d.geometry.PointCloud,o3d.geometry.PointCloud): The points close enough and the points too far away.
    """
    dists = sourcePcd.compute_point_cloud_distance(testPcd)
    ind = np.where(np.asarray(dists) < maxDistance)[0]
    return (sourcePcd.select_by_index(ind), sourcePcd.select_by_index(ind, True))

def filter_geometry_by_distance(geometries: List[o3d.geometry.Geometry], querry_point:np.ndarray, distance_threshold : float =500) -> List[o3d.geometry.Geometry]:
    """Filter out the parts of geometries that lie to far from a querry point.

    Args:
        geometries (List[o3d.geometry.Geometry]): list of geometries (point clouds, trianglemesh or lineset)
        querry_point (np.ndarray[1x3]): center point of the search
        distance_threshold (float, optional): search distance. Defaults to 500m.

    Returns:
        List[o3d.geometry.Geometry]
    """
    geometry_groups=copy.deepcopy(geometries)
    geometryArray=[]

    for i,g in enumerate(list(itertools.chain(*geometry_groups))):
    
        if type(g)==o3d.geometry.LineSet:
            #get indices outside range
            points=np.asarray(g.points)            
            dist = np.linalg.norm(points-querry_point,axis=1)         
            indices=(dist>distance_threshold).nonzero()[0]
            
            if indices.size >0 and indices.size < points.shape[0]:            
                # Remove points
                g.points = o3d.utility.Vector3dVector(np.delete(points, indices, axis=0))  
                # get linesindices to remove          
                lineindices = np.where(np.any(np.isin(np.asarray(g.lines), indices), axis=1))[0]
                if  lineindices.size==0:
                    #add in its totality
                    geometryArray.append(g)   
                else:
                    # Remove lines
                    g.lines = o3d.utility.Vector2iVector(np.delete(np.asarray(g.lines), lineindices, axis=0))
                    geometryArray.append(g)
        
        elif type(g)==o3d.geometry.PointCloud:
            #get indices outside range
            kd= o3d.geometry.KDTreeFlann(g)
            _, idx, _=kd.search_radius_vector_3d(querry_point,distance_threshold)
            indices=np.asarray(idx)
            
            if  indices.size == 0:
                #ignore this geometry
                continue
            elif  indices.size == np.asarray(g.points).shape[0]:
                #add in its totality
                geometryArray.append(g)
            else:
                #select some points
                g=g.select_by_index(indices)       
                geometryArray.append(g)
            
        elif type(g)==o3d.geometry.TriangleMesh:
            #get indices outside range
            points=np.asarray(g.vertices)            
            dist = np.linalg.norm(points-querry_point,axis=1) 
            indices=(dist>distance_threshold).nonzero()[0]
            
            if indices.size >0 and indices.size < points.shape[0]:
                # Remove points
                # g.vertices = o3d.utility.Vector3dVector(np.delete(points, indices, axis=0))
                # Remove triangles
                triangleindices = np.where(np.any(np.isin(np.asarray(g.triangles), indices), axis=1))[0]
                if  triangleindices.size==0:
                    #add in its totality
                    geometryArray.append(g)
                else:
                    #remove some triangles
                    g.remove_triangles_by_index(triangleindices) 
                    g.remove_unreferenced_vertices()    
                    geometryArray.append(g)           

    return geometryArray

