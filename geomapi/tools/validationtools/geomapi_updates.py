import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
import copy
from typing import List

import csv
import json
from geomapi.nodes import *
import geomapi.utils as ut
import geomapi.utils.imageutils as iu

def decode_depthmap(source, resize = True, size = (8192,4096), show = False):
    """
    Function to decode the depthmaps generated by the navvis processing
    source: Location of the PNG files containing the depthmap
    resize(bool): If the resulting dethmap needs to be resized to match the size of the corresponding pano, by default True
    size: size of the corresponding pano, by default 8192x4096
    """
    depthmap = np.asarray(Image.open(source)).astype(float)
    converted_depthmap = np.empty([np.shape(depthmap)[0], np.shape(depthmap)[1]])
    r = 0
    while r < np.shape(depthmap)[0]:
        c = 0
        while c < np.shape(depthmap)[1]:
            value = depthmap[r,c]
            depth_value = value[0] / 256 * 256 + value[1] / 256 * 256 * 256 + value[2] / 256 * 256 * 256 * 256 + value[3] / 256 * 256 * 256 * 256 * 256
            converted_depthmap[r,c] = depth_value
            c = c + 1
        r = r + 1
    if resize:
        resized_depthmap = cv2.resize(converted_depthmap,size)
        if show:
            plt.imshow(resized_depthmap, cmap="plasma")
            plt.show()
        return resized_depthmap
    else:
        if show:
            plt.imshow(converted_depthmap, cmap="plasma")
            plt.show()
        return converted_depthmap
    
    
def plot_pano_positions(panos, colors=None, headings=False, z = False):
    """
    ppcs: list of PanoPoseCollection
    headings: boolean (default: False) - plots headings as vectors
    with size 1.
    """
    
    if z:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        _, ax = plt.subplots()

    n = len(panos)
    pos_xs, pos_ys = np.zeros(n), np.zeros(n)
    ori_xs, ori_ys = np.zeros(n), np.zeros(n)
    if z: 
        pos_zs = np.zeros(n)
        ori_zs = np.zeros(n)

    for i, ppc in enumerate(panos):
        kwargs = {}
        pos_xs[i] = ppc.pos_x
        pos_ys[i] = ppc.pos_y
        if z:
            pos_zs[i] = ppc.pos_z
        if colors is not None:
            kwargs['c'] = colors[i]
  
        if headings is True:
            pc_headings = get_heading(ppc.orientation)
            
            ori_xs[i] = np.cos(np.radians(pc_headings))
            ori_ys[i] = np.sin(np.radians(pc_headings))

            if z:
                pc_zenits = get_zenit(ppc.orientation)
                ori_zs[i] = np.sin(np.radians(pc_zenits))

    if z:
        ax.scatter(pos_xs, pos_ys, pos_zs)
        ax.quiver(pos_xs, pos_ys,pos_zs, ori_xs, ori_ys, ori_zs, length=0.5, color='k')
        ax.axis('auto')
        plt.title('XYZ')
    else:
        ax.scatter(pos_xs, pos_ys)
        ax.quiver(pos_xs, pos_ys, ori_xs, ori_ys, color='k')
        ax.axis('equal')
        plt.title('XY')
    plt.ion()
    plt.show()

    return plt


def get_heading(orientation):
    """
        Heading measured as angle from x to y axis. In equirectangular format
        this is the center of the pano. Headings are always positive to
        simplify subsequent calculations.

        See 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html'
    """
    heading = Rotation.from_quat(orientation).as_euler('xyz', degrees=True)[-1]
    if heading < 0:
        heading = 360 + heading
    
    return heading


def get_zenit(orientation):
    """
    Angle with the vertical
    """
    zenit = Rotation.from_quat(orientation).as_euler('xyz', degrees=True)[1]
    
    return zenit


def navvis_csv_to_nodes(csvPath :str,
                        panoPath : str = None, 
                        includeDepth : bool = True, 
                        depthPath : str = None, 
                        skip:int=None, 
                        filterByFolder = True, 
                        **kwargs) -> List[panonode.PanoNode]:
    """Parse Navvis csv file and output a set of PanoNodes

    Args:
        csvPath (str): csv file path e.g. "D:/Data/pano/pano-poses.csv"
        panoPath (str, optional): _description_. Defaults to None.
        includeDepth (bool, optional): _description_. Defaults to True.
        depthPath (str, optional): _description_. Defaults to None.
        skip (int, optional): select every nth image from the xml. Defaults to None.
        filterByFolder (bool, optional): _description_. Defaults to True.

    Returns:
        List[panonode.PanoNode]: 
    """
    
    assert skip == None or skip >0, f'skip == None or skip '    
    
    #open csv
    pano_csv_file = open(csvPath, mode = 'r')
    pano_csv_data = list(csv.reader(pano_csv_file))

    #Create Nodes per record
    nodes=[]
    for sublist in pano_csv_data[1::skip]:
        panoData=sublist.split('; ')
        
        fileName=panoData[1]
 
        r = Rotation.from_quat((float(sublist[7]),float(sublist[8]), float(sublist[9]), float(sublist[6]))).as_matrix()
        cartesianTransform = np.pad(r, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        cartesianTransform[0,3] = float(sublist[3])
        cartesianTransform[1,3] = float(sublist[4])
        cartesianTransform[2,3] = float(sublist[5])
        cartesianTransform[3,3] = float(1)

        panoPath = ut.get_folder(csvPath) if not panoPath else None
        depthFilename  = fileName.replace(".jpg","_depth.png")
        
        nodes.append(PanoNode(name= fileName.split(".")[0],
                              cartesianTransform=cartesianTransform,
                              timeStamp=panoData[2],
                              path=os.path.join(csvPath, fileName),
                              depthPath = os.path.join(depthPath,depthFilename),
                              use = True,
                              **kwargs))
    return nodes