"""Tools to inspect construction assets."""

import numpy as np
import open3d as o3d
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import geomapi.utils.imageutils as iu
from typing import List,Tuple
from colour import Color
from geomapi.nodes import ImageNode
import copy 
import os


def calculate_distance_between_images(image1, image2):
    """
    Calculate the Euclidean distance between the Cartesian transformation origins of two images.

    Args:
        image1: An object representing the first image with a 'cartesianTransform' attribute.
        image2: An object representing the second image with a 'cartesianTransform' attribute.

    Returns:
        float: The Euclidean distance between the origins of the Cartesian transformations of the two images.

    Example:
        image1 = ImageNode(cartesianTransform=np.array([[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 0]]))
        image2 = ImageNode(cartesianTransform=np.array([[1, 0, 0, 30], [0, 1, 0, 40], [0, 0, 1, 0]]))
        distance = calculate_distance(image1, image2)
    """
     
    x1 = image1.cartesianTransform[0,3]
    y1 = image1.cartesianTransform[1,3]
    x2 = image2.cartesianTransform[0,3]
    y2 = image2.cartesianTransform[1,3]
    
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def filter_images_by_overlap(bimNodes,imgNodes,overlap):

    """
    Filter a list of images based on specified overlap criteria.

    Args:
        bimNodes (list or BIMNode): A list of BIM nodes or a single BIM node.
        imgNodes (list): A list of image nodes to be filtered.
        overlap (float): The desired overlap percentage (in percentage, e.g., 20 for 20%).

    Returns:
        list: A filtered list of image nodes based on the overlap criteria.

    Example:
        bim_nodes = [BIMNode(cartesianBounds=[...]), ...]
        img_nodes = [ImageNode(cartesianTransform=[...]), ...]
        filtered_images = filteronoverlap(bim_nodes, img_nodes, overlap=20)
    """

    overlap=overlap/100
    if isinstance(bimNodes, list):
        total_height = sum(bim_node.cartesianBounds[5] for bim_node in bimNodes)
        mean_height = total_height / len(bimNodes)
    else:
        mean_height=bimNodes.cartesianBounds[5]

    fov_degrees=94
    fov=math.radians(fov_degrees)

    for im in imgNodes:
        height=im.cartesianTransform[2,3]-mean_height
        im.coverage = 2 * height * math.tan(fov / 2)
        im.o=overlap*im.coverage
        
    otot=sum(im.o for im in imgNodes)
    omean=otot/len(imgNodes)
    # print("the mean o: " + str(omean))

    images_copy = imgNodes.copy()
    # print(len(images_copy))
    filtered_image = [images_copy[0]]

    i = 0
    while i < len(images_copy):
        image1 = images_copy[i]
        j = i + 1
        exceeded_distance = False  # Flag to indicate if a distance exceeds the threshold

        while j < len(images_copy):
            image2 = images_copy[j]
            distance = calculate_distance(image1, image2)
            if distance > ((image1.coverage - omean) / 2):
                # If the distance exceeds the threshold, add the previous image to the filtered list
                if j > i + 1:
                    filtered_image.append(image2)
                i = j - 1  # Start from the exceeded image
                exceeded_distance = True
                break  # Exit the inner loop

            j += 1

        if not exceeded_distance:
            # If the distance threshold was not exceeded for this image, add it to the filtered list
            filtered_image.append(image1)

        i += 1
    

    imgNodes=filtered_image

    return(imgNodes)
