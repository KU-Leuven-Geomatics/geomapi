"""Tools to inspect construction assets."""

import cv2
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
import math


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
            distance = calculate_distance_between_images(image1, image2)
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

def project_lineset_on_image(self,linesets:List[o3d.geometry.LineSet],thickness:int=2,overwrite=True) ->np.ndarray:
    """Project Opend3D linesets onto the resource of the node.

    **NOTE**: this affects the original image if overwrite is True.
    
    .. image:: ../../../docs/pics/image_projection_1.png

    Args:
        - linesets (List[o3d.LineSet]): List of linesets. Note that the color of the lines is stored in the lineset.
        - thickness (int) : Thickness of the projected lines
        - overwrite (bool) : If True, the original image is overwritten. If False, a new image is created.

    Returns:
        resource : The resource of the ImageNode with the projected lines.
    """
    if self.resource is None:
        return None
    
    #copy if overwrite is False
    image=self.resource if overwrite else copy.deepcopy(self.resource)
    
    # Loop through each LineSet
    for lineset in ut.item_to_list(linesets):
        points = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)

        # Project points to image plane
        projected_points = self.world_to_pixel_coordinates(points)
        
        #reverse column 0 and 1 to get uv format (column,row)
        projected_points_switched = projected_points[:, [1, 0]]

        #get colors
        colors = np.asarray(np.asarray(lineset.colors)* 255).astype(int) if lineset.has_colors() else np.full((len(lines), 3), 255)
        # Draw lines on the image
        for i,line in enumerate(lines):
            pt1 = tuple(projected_points_switched[line[0]].astype(int))
            pt2 = tuple(projected_points_switched[line[1]].astype(int))
            color = tuple(colors[i])
            color = (int(color[0]), int(color[1]), int(color[2]))  # Ensure color values are integers
            if 0 <= pt1[0] < self.imageWidth and 0 <= pt1[1] < self.imageHeight and \
            0 <= pt2[0] < self.imageWidth and 0 <= pt2[1] < self.imageHeight:
                cv2.line(image, pt1, pt2,color, thickness=thickness)
    return image


def crop_image_within_lineset(self, lineset:o3d.geometry.LineSet, bufferDistance:int=0,overwrite=False) ->np.ndarray:
    """
    Crop an image within a 3D polygon (open3d LineSet). If the lineset is not an enclosed space,
    use a buffer distance to cut out the image up to this distance from the lineset.

    Args:
        - image (np.ndarray) : The input image to be cropped.
        - lineset (o3d.geometry.LineSet) : The open3d LineSet representing the 3D polygon.
        - buffer_distance (float) : The buffer distance to cut out the image if the lineset is not enclosed.
        - overwrite (bool) : If True, the original image is overwritten. If False, a new image is created.

    Returns:
        image : The cropped image
    """
    #copy if overwrite is False
    image=self.resource if overwrite else copy.deepcopy(self.resource)
    
    # Project the lineset onto the image plane
    points = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)

    # Project points to image plane
    projected_points = self.world_to_pixel_coordinates(points)

    # Reverse columns 0 and 1 to get uv format (column, row)
    projected_points_switched = projected_points[:, [1, 0]]

    # Create a mask to crop the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw lines on the mask
    for line in lines:
        pt1 = tuple(projected_points_switched[line[0]].astype(int))
        pt2 = tuple(projected_points_switched[line[1]].astype(int))
        cv2.line(mask, pt1, pt2, 255, thickness=bufferDistance * 2 if bufferDistance > 0 else 1)

    # Dilate the mask to account for the buffer distance
    kernel_size = max(1, bufferDistance)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel)

    # Find contours to determine the region to crop
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contours found. Unable to create a crop region.")
    # Create a new mask with the filled polygon
    fill_mask = np.zeros_like(mask)
    cv2.drawContours(fill_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the image to get the cropped result
    cropped_image = cv2.bitwise_and(image, image, mask=fill_mask)

    # Find the bounding box of the filled mask to crop the image
    x, y, w, h = cv2.boundingRect(fill_mask)
    cropped_image = cropped_image[y:y+h, x:x+w]

    return cropped_image