"""Tools to combine meshes and pointclouds."""
import geomapi
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import open3d as o3d
import numpy as np
from typing import Tuple, List

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
        ogGeometry = gmu.mesh_to_pcd(ogGeometry, voxelSize=distanceTreshold/2)

    # Step 1: Create a convex hull of the newGeometry
    newGeoHull = gmu.get_convex_hull(newGeometry)
    if(logProcess): print("Covex hull created")
    # Step 2: Filter out the irrelevant points in the ogGeometry
    relevantOg, irrelevantOg = gmu.get_points_in_hull(ogGeometry, newGeoHull)
    if(logProcess): print("Irrelevant points filtered")
    # Step 3: Isolate the not covered points of the ogGeometry compared to the newGeometry
    newGeometryPoints = gmu.mesh_to_pcd(newGeometry,distanceTreshold/2)
    coveredPoints, unCoveredPoints = gmu.filter_pcd_by_distance(relevantOg, newGeometryPoints, distanceTreshold)
    if(logProcess): print("Covered poinys calculated")
    # Step 4: Perform the visibility check of the not covered points
    invisibleUncoveredPoints = get_invisible_points(unCoveredPoints, newGeometry)
    if(logProcess): print("invisible points detected")
    # Step 5: Filter the newGeometryPoints to only keep the changed geometry
    existingNewGeo, newNewGeo = gmu.filter_pcd_by_distance(newGeometryPoints, relevantOg, distanceTreshold)
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