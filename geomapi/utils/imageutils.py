"""
Image Utils - functions for handling images.

Orientation is given as a quaternion in scalar-last format [x, y, z, w]. For an
equirectangular pano format this is the orientation of the center of the pano.
The coordinate system is right-handed with z as the vertical direction.
"""


from datetime import datetime
from typing import Tuple
import xml.etree.ElementTree as ET
from typing import List, Tuple
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import geomapi.utils as ut
import os

from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import open3d as o3d

def fill_black_pixels(image:np.array,region:int=5)->np.array:
    """Fill in the black pixels in an RGB image given a search distance.\n

    Args:
        image (np.array)\n
        region (int, optional): search distance. Defaults to 5.\n

    Returns:
        np.array: image
    """
    kernel = np.ones((region,region),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def subdivide_image(image:np.array,m:int=None,n:int=None,width:int=None,height:int=None,includeLast:bool=True)->Tuple[List[np.array],List[Tuple[int,int,int,int]]]:
    """Subdivide image into [mxn] parts.\n

    Args:\n
        1.image (np.array): OpenCV image\n
        2.m (int,optional): number of rows (evenly spaced)\n
        3.n (int,optional): number of columns (evenly spaced)\n   
        4.width (int,optional): length of each column (last member may deviate if included)\n        
        5.height (int,optional): length of each row (last member may deviate if included)\n  
        6.includeLast (bool): True= include last column and row. This is a potential risk as these images typically have a different shape.

    Returns:
        roiList(List[np.array]) : List per row of subdivided images.\n 
        roiBounds (List[tuple(int,int,int,int)])): Boundaries of each roi (nMin,nMax,mMin,mMax)\n
    """
    #includeLast or not
    t=None if includeLast else -1 
    
    #create intervals
    if m and n:
        rows=ut.split_list(range(image.shape[0]),n=m)
        columns=ut.split_list(range(image.shape[1]),n=n)
    elif width and height:
        rows=ut.split_list(range(image.shape[0]),l=height)
        columns=ut.split_list(range(image.shape[1]),l=width)
    else:
        raise ValueError('Invalid input. Enter mxn or width and height')
    
    #extract regions -> roiList, and region boundaries -> roiBounds
    roiList=[]
    roiBounds=[]
    for row in rows[0:t]: 
        for col in columns[0:t]:       
            roiList.append(image[row[0]:row[-1]+1,col[0]:col[-1]+1])   
            roiBounds.append((row[0],row[-1],col[0],col[-1])) 
    return roiList,roiBounds   


    
def get_features(img : np.array, featureType : str = "orb", max : int = 1000):
    """Compute the image features and descriptors

    Args:
        - img (np.array): The source image
        - method (str, optional): The type of features to detect, choose between: orb, sift or fast. Defaults to "Orb".
        - max (int, optional): The destection treshold. Defaults to 1000.

    Raises:
        - ValueError: If the provided method is incorrect

    Returns:
        - Keypoints, descriptors: The detected keypoints and descritors of the source image
    """

    im1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if(featureType.lower() == "orb"):
        detector = cv2.ORB_create(max)
    elif(featureType.lower() == "sift"):
        detector = cv2.SIFT_create(max)
    elif(featureType.lower() == "fast"):
        detector = cv2.FastFeatureDetector_create(max)
    else:
        raise ValueError("Invalid method name, please use one of the following: orb, sift, fast")
    
    return detector.detectAndCompute(im1Gray, None)

def draw_keypoints_on_image(image: np.array, featureType: str = "orb", max_features: int = 1000,keypoint_size: int = 200, overwrite:bool=True):
    """
    Detect and show keypoints on the image.

    Args:
        img (np.array): The input image.
        featureType (str): The type of features to detect ('orb', 'sift', 'fast').
        max_features (int): The maximum number of features to detect.

    Returns:
        np.array: The image with keypoints drawn.
    """
    # Detect features
    keypoints, _ = get_features(image, featureType, max_features)
    
    # Increase the size of keypoints
    for kp in keypoints:
        kp.size = keypoint_size
    # Draw keypoints on the image
    image=image if overwrite else copy.deepcopy(image)
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_with_keypoints

def match_features(des1: np.array, des2: np.array, matchMethod: str = "bfm_orb", 
                    k: int = 2, goodPercent: float = 0.75, checks: int = 100, 
                    table_number: int = 12, key_size: int = 20, multi_probe_level:  int = 2):
    """Matches 2 sets of features using different methods

    Args:
        des1 (np.array): The descriptors of the first image
        des2 (np.array): The descriptors of the first image
        method (str, optional): Use one of the following: bfm_orb, bfm_sift, flann_orb, flann_sift. Defaults to "bfm".
        k (int, optional): The nr of best matches per descriptor. Defaults to 2.
        goodPercent (float, optional): The filter value for the ratio test. Defaults to 0.75.
        checks (int, optional): the number of times the trees in the index should be recursively traversed. Higher values gives better precision, but also takes more time. Defaults to 100.
        table_number (int, optional): Flann orb parameter. Defaults to 12.
        key_size (int, optional): Flann orb parameter. Defaults to 20.
        multi_probe_level (int, optional): Flann orb parameter. Defaults to 2.

    Raises:
        ValueError: If the provided method is incorrect

    Returns:
        np.array: The matches
    """
    
    if(matchMethod.lower() == "bfm_orb"):
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
    
    elif(matchMethod.lower() == "bfm_sift"):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=k)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < goodPercent*n.distance:
                good.append([m])
        matches = good
    
    elif(matchMethod.lower() == "flann_sift"):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=checks)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=k)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < goodPercent*n.distance:
                good.append([m])
        matches = good
    
    elif(matchMethod.lower() == "flann_orb"):
        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = table_number, # 12
                        key_size = key_size,     # 20
                        multi_probe_level = multi_probe_level) #2
        search_params = dict(checks=checks)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=k)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < goodPercent*n.distance:
                good.append([m])
        matches = good
    
    else:
        raise ValueError("Invalid method name, please use one of the following: bfm_orb, bfm_sift, flann_orb, flann_sift")

    return matches

def match_images(img1: np.array, img2: np.array, featureType: str = "orb", matchMethod: str = "bfm"):
    """Matches 2 images using different methods

    Args:
        img1 (np.array): The first image
        img2 (np.array): The second image
        featureType (str, optional): The type of feature to detect, use: orb, sift. Defaults to "orb".
        matchMethod (str, optional): The matching method to use, use: bfm, flann. Defaults to "bfm".

    Raises:
        ValueError: if an incorrect feature or method name is provided

    Returns:
        np.array: The matches of the 2 images
    """
    
    if(featureType.lower() == "orb"):
        kp1, des1 = get_features(img1, "orb")
        kp2, des2 = get_features(img2, "orb")
        if(matchMethod.lower() == "bfm"):
            return match_features(des1, des2, "bfm_orb")
        if(matchMethod.lower() == "flann"):
            return match_features(des1, des2, "flann_orb")
        raise ValueError("Invalid method name, please use one of the following: bfm, flann")
    elif(featureType.lower() == "sift"):
        kp1, des1 = get_features(img1, "sift")
        kp2, des2 = get_features(img2, "sift")
        if(matchMethod.lower() == "bfm"):
            return match_features(des1, des2, "bfm_sift")
        if(matchMethod.lower() == "flann"):
            return match_features(des1, des2, "flann_sift")
        raise ValueError("Invalid method name, please use one of the following: bfm, flann")
    
    raise ValueError("Invalid feature name, please use one of the following: orb, sift")


def load_navvis_depth_map(self):        
    """Returns the full path of the depthMap from this Node. If no path is present, it is gathered from the following inputs.

    Args:
        - self._depthPath
        
    Returns:
        - np.array: depthMap
    """
    # Load depthmap image
    if(self.depthPath):
        depthmap = np.asarray(Image.open(self.depthPath)).astype(float)
        
        # Vectorized calculation for the depth values
        depth_value = (depthmap[:, :, 0] / 256) * 256 + \
                    (depthmap[:, :, 1] / 256) * 256 ** 2 + \
                    (depthmap[:, :, 2] / 256) * 256 ** 3 + \
                    (depthmap[:, :, 3] / 256) * 256 ** 4

        # Assign the computed depth values to the class attribute _depthMap
        self.depthMap = depth_value
        return self._depthMap 
    
def get_pcd_from_depth_map(self)->o3d.geometry.PointCloud:
    """
    Convert a depth map and resource colors to a 3D point cloud.
    
    Args:
        - self._depthMap: 2D numpy array containing depth values 
        - self._resource: 2D numpy array containing color values 
        - self._intrinsic_matrix: 3x3 numpy array containing camera intrinsics
    
    Returns:
        - An Open3D point cloud object
    """
    # Create Open3D image from depth map
    depth_o3d = o3d.geometry.Image(self._depthMap)

    # Check if the color image and depth map have the same dimensions
    resource=cv2.resize(self._resource,self._depthMap.shape) if self._resource and not self._resource.shape == self._depthMap.shape else self._resource

    # Create point cloud from the depth image using the camera intrinsics
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                                            depth_o3d, 
                                            self._intrinsic_matrix, 
                                            project_valid_depth_only=True
                                            )
    
    # Flatten the color image to correspond to the points
    colors = resource.reshape(-1, 3) if resource is not None else None
    pcd.colors = o3d.utility.Vector3dVector(colors) if colors is not None else None

    #transform to the cartesianTransform
    pcd.transform(self._cartesianTransform)
    
    return pcd
