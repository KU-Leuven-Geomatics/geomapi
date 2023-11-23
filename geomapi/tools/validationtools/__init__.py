"""
validationtools - a Python library for validating objects.
"""
import csv
import os
import os.path
from typing import List, Tuple

from sklearn.neighbors import NearestNeighbors 

import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import xlsxwriter
from colour import Color
from geomapi.nodes import *
import ezdxf


def get_boundingbox_of_list_of_geometries(geometries:List[o3d.geometry.PointCloud]) -> np.array:
    """Determines the global boundingbox of a group of Node containing geometries.

    Args:
        geometries (List[Nodes]):  list of Nodes containing a resource of which the boundingbox must be determined"
            
    Returns:
        np.array[3x1]
    """
    pcd = o3d.geometry.PointCloud()
    for n in geometries:
        n.get_resource()
        if n.resource is not None:
            pcd.__iadd__(o3d.geometry.PointCloud(gmu.get_oriented_bounds(gmu.get_cartesian_bounds(n.resource))))
    cartesianBounds = gmu.get_cartesian_bounds(pcd.get_oriented_bounding_box())
    return cartesianBounds

def match_BIM_points(sources: List[o3d.geometry.PointCloud], references: List[o3d.geometry.PointCloud], referenceIds = None, resolution : float = 0.02, t00: float=0.15, wd: float = 0.62, wn: float = 0.38, threshold: float = 0.7) -> np.array:
    """Determines the accuracy of a BIM model or object using a pointcloud\n

    Args:
        1. sources(list(o3d.geometry.PointCloud)): The captured point clouds of the site.\n
        2. references(list(o3d.geometry.PointCloud)): A pointcloud per reference object\n
        3. referenceIds(list(str), optional): ID to be assigned to the points of the source point cloud\n
        4. resolution (float, optional): voxel size for the a voxel downsampling before the analysis default 0.02m\n
        5. t00(float): Threshold to which matching point will be searched default 0.15m\n
        6. wd (float): weight of the distances in the decision function default 0.62\n
        7. wn (float): weight of the normals in the decision function default 0.38\n
        8. threshold (float): minimum value to be seen as a match, default 0.7\n

    Returns:
        o3d.geometry.PointCloud: pointcloud used in the analysis linked to the identity and the distances of the points.\n
        identityArray: array containing the matching label to the BIM for each point from the point cloud.\n
        distances: array containing all the distances to the matching BIM for each point from the point cloud.\n
    """    
    #if no list, list
    sources=ut.item_to_list(sources)
    sourcePCDs=[]
    references=ut.item_to_list(references)
    referencePCDs=[]
    referenceIdentityArray=None
    
    #itterate over the reference clouds and create a Identity array
    for i,reference in enumerate(references):
        if 'PointCloud' in str(type(reference)) :
            referencePCD=reference.voxel_down_sample(resolution)
            if referenceIds:
                referenceIdentityArray=np.vstack((referenceIdentityArray,np.full((len(referencePCD.points), 1), referenceIds[i])))
            else:
                referenceIdentityArray=np.vstack((referenceIdentityArray,np.full((len(referencePCD.points), 1), i+1)))
        referencePCDs.append(referencePCD)
    #Join the reference clouds together to one reference cloud
    referencePCD=gmu.join_geometries(referencePCDs)
    
    #check if the reference cloud has normals, if not compute them, this will be less accurate than when using the normals of the meshes
    if not referencePCD.has_normals():
        referencePCD.estimate_normals()
    referencePCD.normalize_normals()

    #Flatten the identity array and remove the first (None) value
    referenceIdentityArray=referenceIdentityArray.flatten()
    referenceIdentityArray=np.delete(referenceIdentityArray,0)

    #Check if the boundingboxes of the source clouds lie within the boundingbox of the reference
    #when the reference covers more then the captured data
    inliers = gmu.get_box_inliers(referencePCD.get_oriented_bounding_box(), [geometry.get_oriented_bounding_box() for geometry in sources], t_d = 500) 
    #when the reference covers less than the captured data
    inliers += gmu.get_box_intersections(referencePCD.get_oriented_bounding_box(), [geometry.get_oriented_bounding_box() for geometry in sources])
    
    if not inliers:
        #If reference and source share no overlap the sources containing no overlap will be ignored
        return None
    
    for i,source in enumerate(sources): 
        if i in inliers:      
            if 'PointCloud' in str(type(source)) :
                sourcePCD=source.voxel_down_sample(resolution)
            sourcePCDs.append(sourcePCD)
    

    #Join them in one source cloud for computations can be limiting for very large clouds
    joinedPCD=gmu.join_geometries(sourcePCDs)
    
    identityArray= [0] * len(joinedPCD.points)
    distances = [0.0] * len(joinedPCD.points)


    #Create a KDtree of the reference
    referenceTree = o3d.geometry.KDTreeFlann(referencePCD)
    
    #compute the c2c distances between both clouds, without matching. This speeds up the computations significantly
    c2cdistances = joinedPCD.compute_point_cloud_distance(referencePCD)
    #Itterate over every captured point within the tresholddistance
    for i, point in enumerate(joinedPCD.points):
        
        if c2cdistances[i] < t00:
            #Search for the k=20 nearest neighbours with a maximum distance of t00
            [k,inliers ,ignore] = referenceTree.search_hybrid_vector_3d(point, t00, 20)
            #When neighbours are found within a distance t00 ierate over them
            if k > 0:
                localdistances = [0.0]*len(np.asarray(inliers))
                distanceparameters = [0.0]*len(np.asarray(inliers))
                normalparameters = [0.0]*len(np.asarray(inliers))
                similarities = [0.0]*len(np.asarray(inliers))

                #For every found neighbour compute the distance and the normal parameters
                for id, inlier in enumerate(inliers):
                    #Compute the actual 3d distance (is not equal to the distance returned by o3d.search_hybrid_vector!)
                    localdistances[id] = np.sqrt(np.sum((point-referencePCD.points[inlier])**2, axis=0))
                    #compute the distance parameter
                    distanceparameters[id] = (t00 - localdistances[id])/t00
                    #compute the normal parameter
                    normalparameters[id] = np.abs(np.dot(np.asarray(joinedPCD.normals[i]), np.asarray(referencePCD.normals[inlier])))
                    #compute the similarity between point p and the specifik neighbour k
                    similarities[id] = wn * np.power(normalparameters[id],3) + distanceparameters[id] * wd

                #Search for the maximal similarity which should be above the defined threshold and assign the correct label and distance to the according arrays
                if similarities[np.argmax(similarities)] > 0.7:
                    identityArray[i] = referenceIdentityArray[inliers[np.argmax(similarities)]]
                    distances[i] = localdistances[np.argmax(similarities)]
    #Filter out points not matched to any element
    resultPCD = joinedPCD.select_by_index(np.where(np.asarray(identityArray) != '0')[0])
    
    i = 0
    identityArray2 = []
    distances2 = []
    while i < len(identityArray):
        if not identityArray[i] == 0:
            identityArray2.append(identityArray[i])
            distances2.append(distances[i])
        i += 1

    return resultPCD, identityArray2, distances2

def compute_LOA(identities, distances, t00: float = 0.15, t10: float = 0.10, t20: float = 0.05, t30: float = 0.015, byElement: bool = False, limit: float = 0.95):
    """Function which uses distances and a linked identity array to determine the LOA percentages 

    Args:
        identities (nx1-array): Array containing the identity of the distance between two matched points
        distances (nx1 array): Array containing the distances between two matched points
        t00 (float, optional): Maximum distance to be used in the analysis. Defaults to 0.15.
        t10 (float, optional): Upper bound of the LOA10 bracket. Defaults to 0.10.
        t20 (float, optional): Upper bound of the LOA20 bracket. Defaults to 0.05.
        t30 (float, optional): Upper bound of the LOA10 bracket. Defaults to 0.015.
        byElement (bool, optional): If the LOA must be computed per element of for the enitre cloud. Defaults to False.
        limit (float, optional): Percentage of inliers between two brackets needed to assign the LOA label. Defaults to 0.95.

    Returns:
        LOA: List of LOAs per element (id, [LOA10, LOA20, LOA30], label)
    """
    LOA = []
    if byElement: 
        for identity in np.unique(identities):
            if not identity  == 0:
                places = np.where(np.asarray(identities) == identity)[0]

                LOA00Inliers = 0
                LOA10Inliers = 0
                LOA20Inliers = 0
                LOA30Inliers = 0

                for place in places :
                    if distances[place] <= t00:
                        LOA00Inliers += 1
                    if distances[place] <= t10:
                        LOA10Inliers += 1
                    if distances[place] <=t20:
                        LOA20Inliers += 1
                    if distances[place] <= t30:
                        LOA30Inliers += 1

                if LOA00Inliers > 0:
                    LOA10 = LOA10Inliers/LOA00Inliers
                    LOA20 = LOA20Inliers/LOA00Inliers
                    LOA30 = LOA30Inliers/LOA00Inliers

                    if LOA30 > limit:
                        label = 'LOA30'
                    elif LOA20 > limit:
                        label = 'LOA20'
                    elif LOA10 > limit:
                        label = 'LOA10'
                    else: 
                        label = None
                    LOA.append((identity,[LOA10, LOA20, LOA30], label))
    else:
        LOA00Inliers = 0
        LOA10Inliers = 0
        LOA20Inliers = 0
        LOA30Inliers = 0

        for d in distances :
            if d <= t00:
                LOA00Inliers += 1
            if d <= t10:
                LOA10Inliers += 1
            if d <=t20:
                LOA20Inliers += 1
            if d <= t30:
                LOA30Inliers += 1

        if LOA00Inliers > 0:
            LOA10 = LOA10Inliers/LOA00Inliers
            LOA20 = LOA20Inliers/LOA00Inliers
            LOA30 = LOA30Inliers/LOA00Inliers
            if LOA30 > limit:
                label = 'LOA30'
            elif LOA20 > limit:
                label = 'LOA20'
            elif LOA10 > limit:
                label = 'LOA10'
            else:
                label = None
                    
            LOA.append((None,[LOA10, LOA20, LOA30], label))
    return LOA

def plot_histogram(identities, distances, buckets: int = None, interval: float = None, dmax:float = 0.1, byElement = False, bins = None, directory = None, show = True):
    """Function to plot distances between the captured cloud and the reference cloud

    Args:
        identities (nx1-array): Array containing the identity of the distance between two matched points
        distances (nx1 array): Array containing the distances between two matched points
        buckets (int, optional): Number of intervals the data will be seperated. Defaults to None.
        interval (float, optional): distance between the upper and lower bound of an interval. Defaults to None.
        dmax (float, optional): Distances higher then this distance will be ignored. Defaults to 0.1.
        byElement (bool, optional): If the LOA must be computed per element of for the enitre cloud. Defaults to False.
        bins (1xn array): Can be used to describe custom bin boundries (intervals must be equal). Defaults to None.
        directory (path, optional): When provided the histograms will be saved in the form of a PNG to this directory. Defaults to None.
        show (bool, optional): When set on true the histograms will be visualized. Defaults to True.
    """
    if directory:
        if not os.path.exists(directory):
            os.mkdir(directory)

    if buckets:
        max = np.max(np.asarray(distances))
        if max > dmax:
            max = dmax
        
        min = np.min(np.asarray(distances))
        if min < 0:
            min = 0  
        range = max - min 
        interval = range / buckets
        lb = 0
        ub = lb+interval

        bins = [lb]
        while ub <= dmax:
            bins.append(ub)
            ub +=interval
    if interval:
        lb = 0
        ub = lb+interval

        bins = [lb]
        while ub <= dmax:
            bins.append(ub)
            ub +=interval 
    if not bins:
        max = np.max(np.asarray(distances))
        if max > dmax:
            max = dmax
        interval = 0.005
        lb = 0
        ub = lb+interval
        bins = [lb]
        while ub <= dmax:
            bins.append(ub)
            ub +=interval

    if byElement:
        for identity in np.unique(identities):
            elementDistances = []
            if not identity  == "0":
                places = np.where(np.asarray(identities) == identity)[0]
                for place in places :
                    elementDistances.append(distances[place])
                plt.hist(elementDistances, bins = bins)
                plt.title(identity.split("file:///")[1])
                if directory:
                    if not directory.endswith("histograms"):
                        directory = os.path.join(directory, "histograms")
                        if not os.path.exists(directory):
                            os.mkdir(directory)
                    filename = os.path.join(directory, identity.split("file:///")[1] + ".PNG")
                    plt.title(identity.split("file:///")[1])
                    plt.savefig(filename)
                    plt.clf()
                if show:
                    plt.show()

    else:
        places = np.where(np.asarray(identities) != '0')[0]
        elementDistances = [distances[place] for place in places]
        plt.hist(elementDistances, bins = bins)
        if directory:
            filename = os.path.join(directory, 'histogram' + ".PNG")
            plt.savefig(filename)
        if show:
            plt.show()

def color_point_cloud_by_LOA(pointcloud: o3d.geometry.PointCloud, identities, distances, t00: float = 0.15, t10: float = 0.10, t20: float = 0.05, t30: float = 0.015, byElement: bool = False): 
    """Colors each point by its computed LOA based on the distance between the matched points of the reference and the source cloud

    Args:
        pointcloud (o3d.geometry.PointCloud): Point cloud from the LOA determination or pointcloud matching its the returned indeces
        identities (nx1-array): Array containing the identity of the distance between two matched points
        distances (nx1 array): Array containing the distances between two matched points
        t00 (float, optional): Maximum distance to be used in the analysis. Defaults to 0.15.
        t10 (float, optional): Upper bound of the LOA10 bracket. Defaults to 0.10.
        t20 (float, optional): Upper bound of the LOA20 bracket. Defaults to 0.05.
        t30 (float, optional): Upper bound of the LOA10 bracket. Defaults to 0.015.
        byElement (bool, optional): If the LOA must be computed per element of for the enitre cloud. Defaults to False.
        
    Returns:
        o3d.geometry.PointCloud()
    """
    pointcloud.paint_uniform_color([0.5,0.5,0.5])
    if byElement:
        elementClouds = []
        for identity in np.unique(identities):
            elementDistances = []
            if not identity  == 0:
                places = np.where(np.asarray(identities) == identity)[0]
                elementCloud = pointcloud.select_by_index(places)
                for place in places :
                    elementDistances.append(distances[place])
                for i, d in enumerate(elementDistances):
                    if d <= t00:
                        np.asarray(elementCloud.colors)[i] = [1,0,0]
                    if d <= t10:
                        np.asarray(elementCloud.colors)[i] = [1,0.76,0]
                    if d <=t20:
                        np.asarray(elementCloud.colors)[i] = [1,1,0]
                    if d <= t30:
                        np.asarray(elementCloud.colors)[i] = [0,1,0]
                elementClouds.append(elementCloud)
        return elementClouds
    else:
        for i, d in enumerate(distances):
            if not identities[i] == 0:
                if d <= t00:
                    np.asarray(pointcloud.colors)[i] = [1,0,0]
                if d <= t10:
                    np.asarray(pointcloud.colors)[i] = [1,0.76,0]
                if d <=t20:
                    np.asarray(pointcloud.colors)[i] = [1,1,0]
                if d <= t30:
                    np.asarray(pointcloud.colors)[i] = [0,1,0]
        return pointcloud

def color_point_cloud_by_distance(pointcloud: o3d.geometry.PointCloud, identities:np.array, distances:np.array, buckets: int = 5, dmax:float = 0.1, byElement: bool = False)->o3d.geometry.PointCloud:
    """Colorizes the resulting point cloud of the LOA analysis in a gradient by distance between the matched points from the reference and the source (very slow).\n

    **NOTE**: use sklearn to make this faster.

    Args:
        1. pointcloud (o3d.geometry.PointCloud): Point cloud from the LOA determination or pointcloud matching its the returned indeces.\n
        2. identities (nx1-array): Array containing the identity of the distance between two matched points.\n
        3. distances (nx1 array): Array containing the distances between two matched points.\n
        4. buckets (int, optional): Number of intervals to be colored in. Defaults to 5.\n
        5. dmax (float, optional): Distances higher then this distance will be ignored. Defaults to 0.1m.\n
        6. byElement (bool, optional): If the LOA must be computed per element of for the enitre cloud. Defaults to False.\n

    Returns:
        o3d.geometry.PointCloud
    """
    max = np.max(np.asarray(distances))
    if max > dmax:
        max = dmax
    
    min = np.min(np.asarray(distances))
    range = max - min 
    interval = range / buckets
    lb = 0
    ub = lb+interval

    pointcloud.paint_uniform_color([0.5,0.5,0.5])

    green = Color("green")
    colors = list(green.range_to(Color("red"),buckets))
    colors = [c.rgb for c in colors]
    
    
    if byElement:
        elementClouds = []
        for identity in np.unique(identities):
            lb = 0
            ub = lb+interval
            elementDistances = []
            if not identity  == 0:
                bucket = 0
                places = np.where(np.asarray(identities) == identity)[0]
                elementCloud = pointcloud.select_by_index(places)
                for place in places :
                    elementDistances.append(distances[place])

                while ub <= max:
                    places2 = np.where(np.asarray(elementDistances) <= ub)[0]
                    places3 = np.where(np.asarray(elementDistances) > lb)[0]
                    for place2 in places2:
                        if place2 in places3:
                            np.asarray(elementCloud.colors)[place2] = colors[bucket]
                    lb = ub
                    ub += interval
                    bucket +=1
                elementClouds.append(elementCloud)
        return elementClouds
    else:
        bucket = 0
        while ub <= max:
            places2 = np.where(np.asarray(distances) <= ub)[0]
            places3 = np.where(np.asarray(distances) > lb)[0]
            for place2 in places2:
                if place2 in places3:
                    np.asarray(pointcloud.colors)[place2] = colors[bucket]
            lb = ub
            ub += interval
            bucket +=1
        return pointcloud

def csv_by_LOA(directory:str, LOAs, visibility=None):
    """Function to report the LOA analysis in a csv file.\n

    Args:
        1. directory (path): directory where the report must be saved.\n
        2. LOAs (_type_): results of the LOA computation.\n
        3. visibility (_type_, optional): array containing the per element visibility.\n

    Returns:
        returns true when succeded.
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
    csvFilename = "LOA_Report1.csv"
    csvPath = os.path.join(directory, csvFilename)
    if visibility:
        header = ['Name','LOA', 'LOA10', 'LOA20', 'LOA30', 'Theoretical visibility']
    else:
        header = ['Name','LOA', 'LOA10', 'LOA20', 'LOA30']
    csvFile = open(csvPath, 'w')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(header)

    for LOA in LOAs:
        if not LOA[0]  == '0':
            if visibility:
                vis = [v[1] for v in visibility if str(v[0]) == LOA[0]]
                data = [LOA[0], LOA[2], LOA[1][0], LOA[1][1], LOA[1][2], vis[0]]
            else: 
                data = [LOA[0], LOA[2], LOA[1][0], LOA[1][1], LOA[1][2]]
            csvWriter.writerow(data)
    csvFile.close()
    return True

def excel_by_LOA(directory, LOAs, visibility= None):
    """Function to report the LOA analysis in an excel file

    Args:
        directory (path): directory where the report must be saved
        LOAs (_type_): results of the LOA computation
        visibility (_type_, optional): array containing the per element visibility

    Returns:
        returns true when succeded
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
    xlsxFilename = "LOA_Report.xlsx"
    xlsxPath = os.path.join(directory, xlsxFilename)
    workbook = xlsxwriter.Workbook(xlsxPath)
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,'Name')
    worksheet.write(0,1,'LOA')
    worksheet.write(0,2, 'LOA10')
    worksheet.write(0,3, 'LOA20')
    worksheet.write(0,4, 'LOA30')
    if visibility:
        worksheet.write(0,5, 'Theoretical visibility')
    xlsxRow = 1

    for LOA in LOAs:
        if not LOA[0]  == '0':
            worksheet.write(xlsxRow, 0, LOA[0])
            worksheet.write(xlsxRow, 1, LOA[2])
            worksheet.write(xlsxRow, 2, LOA[1][0])
            worksheet.write(xlsxRow, 3, LOA[1][1])
            worksheet.write(xlsxRow, 4, LOA[1][2])
            if visibility:
                vis = [v[1] for v in visibility if str(v[0]) == LOA[0]]
                worksheet.write(xlsxRow, 5, vis[0])
        xlsxRow += 1 
    workbook.close()

    return True

def color_BIMNode(LOAs, BIMNodes: List[BIMNode]):
    """Colors the BIM mesh geometries in the computed LOA color

    Args:
        LOAs (_type_): results of the LOA analysis
        BIMNodes (List[BIMNode]): List of the BIMNodes in the project
    """
    for BIMNode in BIMNodes:
        if BIMNode.resource:
                BIMNode.resource.paint_uniform_color([0.5,0.5,0.5])
    for BIMNode in BIMNodes:
        for LOA in LOAs:
            if LOA[0] == BIMNode.subject:
                if not BIMNode.resource:
                    BIMNode.get_resource()
                if LOA[2] == 'LOA10':
                    BIMNode.resource.paint_uniform_color([1,0.76,0])
                if LOA[2] == 'LOA20':
                    BIMNode.resource.paint_uniform_color([1,1,0])
                if LOA[2] == 'LOA30':
                    BIMNode.resource.paint_uniform_color([0,1,0])


def cad_show_lines(dxf_path:str):

    doc = ezdxf.readfile(dxf_path)

    # Extract all line entities from the DXF file
    msp = doc.modelspace()
    lines = msp.query("LINE")

    # Plot all the lines using Matplotlib
    for line in lines:
        x1, y1, _ = line.dxf.start
        x2, y2, _ = line.dxf.end
        plt.plot([x1, x2], [y1, y2])

    plt.show()

# def create_lineset(line, points:np.ndarray):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.array(points))
#     colors = [line[2]] * len(points) #! this is sketchy
#     pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)
    
#     lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd, pcd, [(i, i+1) for i in range(len(points)-1)])
#     lineset.paint_uniform_color(np.array(line[2]) / 255.0)
#     lineset.line_width = line[4]
    
#     return lineset

def sample_pcd_from_linesets(linesets:List[o3d.geometry.LineSet],step_size:float=0.1)-> Tuple[o3d.geometry.PointCloud,np.ndarray]:
    """Sample a point cloud from a set of o3d.geometry.LineSet elements (color is inherited)

    Args:
        linesets (List[o3d.geometry.LineSet]): linesets to sample. 
        step_size(float,optional):spacing between points. Defaults to 0.1m.

    Returns:
        Tuple[List[o3d.geometry.PointCloud],np.ndarray]: point_clouds, identityarray with integers of the origin of the points
    """
    point_clouds=o3d.geometry.PointCloud()
    ilist=[]
    jlist=[]
    
    for i,lineset in enumerate(linesets):

        # Get line segments from the LineSet
        pointArray=np.asarray(lineset.points)
        points = []

        for j,line in enumerate(np.asarray(lineset.lines)):
            #get start and end
            start_point = pointArray[line[0]]
            end_point = pointArray[line[1]]
            #get direction and length
            direction = end_point - start_point
            length = np.linalg.norm(direction)
            #compute number of points
            num_points = int(length / step_size)
            if num_points > 0:
                step = direction / num_points
                p=[start_point + r * step for r in range(num_points + 1)]
                points.extend(p)
                
                #keep track of identity of the points
                ilist.extend(np.full((len(p), 1), i))
                jlist.extend(np.full((len(p), 1), j))
                
        # Convert the sampled points to an o3d PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        color=lineset.colors[0]
        point_cloud.paint_uniform_color(color)
        point_clouds+=point_cloud
        
    #compile identidyarray & point cloud
    indentityArray=np.column_stack((np.array(ilist),np.array(jlist)))

    return point_clouds,indentityArray

# def select_lineset_inliers(linesets,points)->o3d.geometry.LineSet:
#     linesetselection=np.empty((points.shape[0],2))

#     #iterate through linesets
#     for i,p in enumerate(points):
        
#         for linesetidx,lineset in enumerate(sublinesets):
            
#             #iterate through line
#             for lineidx,line in enumerate(lineset.lines):
#                 # # get p0 and p1 in the size of the input points
#                 # p0=np.tile(lineset.points[line[0]], (points.shape[0], 1)) 
#                 # p1=np.tile(lineset.points[line[1]], (points.shape[0], 1))
#                 p0=lineset.points[line[0]]
#                 p1=lineset.points[line[1]]
                
#                 #test if any point is on the line  -> # 0.0 <= dot(p1-p0,p-p0)/|p-p0| <= 1.0
#                 # print(np.sum((p-p0)**2))
#                 dot=np.dot(p1-p0,p-p0)/np.sum((p-p0)**2)
                
#                 if (dot>=0) & (dot <=1 ):
#                     linesetselection[i,0]=linesetidx
#                     linesetselection[i,1]=lineidx
#     return(linesetselection)               
#             # print('line')
#             # dot=np.sum(np.dot(p1-p0,(points-p0).T),axis=1) /np.sum((points-p0)**2,axis=1)
        
# #         # create tuple 
# #         np.where((dot>=0) & (dot <=1 ),
# #                  (linesetidx,lineidx,dot),
# #                  (np.nan,np.nan,dot))
            
        
# # return linesetselection.append((linesetidx,lineidx,distance))
        
# # return True if (dot>=0 or dot <=1 ) else False


# def create_unique_mapping(array:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
#     """Create a unique mapping of an array

#     Args:
#         array (np.ndarray): first column of the array will be used for the sorting.

#     Returns:
#         Tuple[np.ndarray,np.ndarray]: unique_values, mapping (shape of input array)
#     """
#     #get first array #! this is a bit flawed and supposes that every x-value is unique
#     a=array[:,0]
#     unique_values=np.unique(array,axis=0)
    
#     # build dictionary
#     fwd = np.argsort(a)
#     asorted = a[fwd]
#     keys = np.unique(asorted) 
#     lower = np.searchsorted(asorted, keys)
#     higher = np.append(lower[1:], len(asorted))

#     inv = {key: fwd[lower_i:higher_i]
#             for key, lower_i, higher_i
#             in zip(keys, lower, higher)}
    
#     # remap values to 0,1,2,....
#     mapping=np.zeros(array.shape[0])
#     i=0
#     for _,value in inv.items():
#             for v in value:
#                     mapping[v]=i
#             i+=1
    
#     return unique_values,mapping

def get_linesets_inliers_in_box(linesets:List[o3d.geometry.LineSet],box:o3d.geometry.OrientedBoundingBox,point_cloud:o3d.geometry.PointCloud,identityArray:np.ndarray) -> List[o3d.geometry.LineSet]:
    """Returns the segments of the linesets that have sampled pointcloud points falling within a certain bounding box.
    This function should be used together with:\\
        1. vt.sample_pcd_from_linesets(linesets,step_size=0.1)\\
        2.vt.create_selection_box_from_image_boundary_points(n,roi,meshNode.resource,z=5) \\

    Args:
        linesets (List[o3d.geometry.LineSet]): linesets from which the segments will be selected
        box (o3d.geometry.OrientedBoundingBox): bounding box that is used to filter the point cloud points
        point_cloud (o3d.geometry.PointCloud): sampled points on the linesets
        identityArray (np.ndarray): array with integers that reflect which point cloud point belongs to which lineset

    Returns:
        List[o3d.geometry.LineSet]: _description_
    """
    assert len(np.asarray(point_cloud.points))==identityArray.shape[0], f'length of point cloud and identityarray are not equal'
        
    #compute point_cloud inliers in box
    idxList=box.get_point_indices_within_bounding_box(point_cloud.points)
    if len(idxList)==0:
        return []    
        
    #retrieve which linesets are visible in the box
    idx=identityArray[idxList]
    unique_rows, _ = np.unique(idx, axis=0, return_inverse=True)
    
    #split lists per lineset -> create split_rows dictionary of mapping
    split_rows = {}
    for row in unique_rows:
        key = row[0]
        if key not in split_rows:
            split_rows[key] = []
        split_rows[key].append(row)
    
    #get linesegments and build new linesets
    sublinesets=[]
    for key,value in split_rows.items():
        #get lineset
        lineset=linesets[key]
        #get linesegments
        linesegments=[linesets[key].lines[v[1]] for v in value]
        #create new linesets -> currently still has all redundant points
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector(lineset.points)  
        line_set.lines = o3d.utility.Vector2iVector(linesegments)
        sublinesets.append(line_set)
    return sublinesets

def create_selection_box_from_image_boundary_points(n:ImageNode,roi:Tuple[int,int,int,int],mesh:o3d.geometry.TriangleMesh,z:float=5)->o3d.geometry.OrientedBoundingBox:
    """Create a selection box from an ImageNode, a region of interest (roi) and a mesh to raycast.
    A o3d.geometry.OrientedBoundingBox will be created on the location of the intersection of the rays with the mesh.
    The height of the box is determined by the offset of z in both positive and negative Z-direction

    Args:
        n (ImageNode): Imagenode used for the raycasting (internal and external camera paramters)
        roi (Tuple[int,int,int,int]): region of interest (rowMin,rowMax,columnMin,columnMax)
        mesh (o3d.geometry.TriangleMesh): mesh used for the raycasting
        z (float, optional): offset in height of the bounding box. Defaults to [-5m:5m].

    Returns:
        o3d.geometry.OrientedBoundingBox or None (if not all rays hit the mesh)
    """
    box=None
    
    #create rays for boundaries
    uvCoordinates=np.array([[roi[0],roi[2]], # top left
                            [roi[0],roi[3]], # top right
                            [roi[1],roi[2]], # bottom left
                            [roi[1],roi[3]] # bottom right
                            ])
    # transform uvcoordinates  to world coordinates to rays   
    rays=n.create_rays(uvCoordinates)
    
    # cast rays to 3D mesh 
    distances,_=gmu.compute_raycasting_collisions(mesh,rays)
    
    if all(np.isnan(distances)==False): #if all rays hit
        #compute endpoints 
        _,endpoints=gmu.rays_to_points(rays,distances)
        
        #create box of projected points
        points=np.vstack((gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z],[0,0,0,1]])),
                        gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-z],[0,0,0,1]]))))
        box=o3d.geometry.OrientedBoundingBox.create_from_points(o3d.cpu.pybind.utility.Vector3dVector(points))
        box.color=[1,0,0]     
    return box 
