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

import matplotlib.pyplot as plt
import numpy as np
import geomapi.utils as ut

from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
# NOTE this looks too basic to create a new function
def image_resize(img:np.array,width:int=None,height:int=None,scale:float=None)->np.array:
    """Resize an cv2 image (np.array).

    Args:
        1.img (np.array) \n
        2.width (int, optional): width in pixels. Defaults to None.\n
        3.height (int, optional): height in pixels. Defaults to None.\n
        4.scale (float, optional): percentual scale. Defaults to 0.5.\n

    Returns:
        np.array: resized image.
    """
    if scale:
        width = math.ceil(img.shape[1] * scale )
        height = math.ceil(img.shape[0] * scale )
    elif width and height:
        width=width
        height=height
    dim = (width, height)
    
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#NOTE these functions already exist in opencv
def grb01_to_rgb255(image:np.array)->np.array:
    """Return image RGB [0-1] interval as RGB image [0-255] interval.\n
    """
    norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    return norm_image.astype(np.uint8)

def rgb2gray(rgb:np.array)->np.array:
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gray2rgb(gray:np.array)->np.array:
    rgb = np.zeros((*gray.shape, 3))
    rgb[..., :] = gray[..., np.newaxis]
    return rgb
#NOTE endNote

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

def calibrate_camera(images, CheckerBoardPattern = (7,7), squareSize: float = 1, drawImages:bool = False):
    """Calibrates a camera and determines the intrinsic matrix and distortion coëfficients using a list of images of a chessboard pattern.

    Args:\n
        images (list(np.array)): a list of cv2 images\n
        CheckerBoardPattern (tuple, optional): The pattern of the checkerboard. Count the inner crosses. Defaults to (7,7).\n
        squareSize (float, optional): The size of 1 scuare of the checkerboard. Defaults to 1.\n
        drawImages (bool, optional): Display checker image after each itteration. Defaults to False.\n

    Returns:
        bool: The success of the calculation
        np.array(): The intrinsic matrix
        np.array(): The distortion coëfficients
        np.array(): The rotation vectors
        np.array(): The translation vectors
    """

    # Defining the Matching criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CheckerBoardPattern[0] * CheckerBoardPattern[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CheckerBoardPattern[0], 0:CheckerBoardPattern[1]].T.reshape(-1, 2) * squareSize
    prev_img_shape = None
    
    for i, image in enumerate(images):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CheckerBoardPattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)

            if(drawImages):
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CheckerBoardPattern, corners2, ret)
                cv2.imshow(str("image " + str(i)),img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    # Perform the camera calibration    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs
    
def get_features(img : np.array, featureType : str = "Orb", max : int = 1000):
    """Compute the image features and descriptors

    Args:
        img (np.array): The source image
        method (str, optional): The type of features to detect, choose between: orb, sift or fast. Defaults to "Orb".
        max (int, optional): The destection treshold. Defaults to 1000.

    Raises:
        ValueError: If the provided method is incorrect

    Returns:
        Keypoints, descriptors: The detected keypoints and descritors of the source image
    """

    im1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if(featureType.lower() == "orb"):
        detector = cv2.ORB_create(max)
    if(featureType.lower() == "sift"):
        detector = cv2.SIFT_create(max)
    if(featureType.lower() == "fast"):
        detector = cv2.FastFeatureDetector_create(max)
    else:
        raise ValueError("Invalid method name, please use one of the following: orb, sift, fast")
    
    return detector.detectAndCompute(im1Gray, None)

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
    
    if(matchMethod.lower() == "bfm_sift"):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=k)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < goodPercent*n.distance:
                good.append([m])
        matches = good
    
    if(matchMethod.lower() == "flann_sift"):
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
    
    if(matchMethod.lower() == "flann_orb"):
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
    
    if(featureType.lower() == "sift"):
        kp1, des1 = get_features(img1, "sift")
        kp2, des2 = get_features(img2, "sift")
        if(matchMethod.lower() == "bfm"):
            return match_features(des1, des2, "bfm_sift")
        if(matchMethod.lower() == "flann"):
            return match_features(des1, des2, "flann_sift")
        raise ValueError("Invalid method name, please use one of the following: bfm, flann")
    
    raise ValueError("Invalid feature name, please use one of the following: orb, sift")

# NOTE Utils function cannot be found -> should be named better
def create_transformation_matrix(R: np.array, t: np.array) -> np.array:
    """Combines a rotation matrix and translation vector into a 4x4 transformation matrix
    
    **NOTE**: this function should be in utils and is already present

    Args:
        R (np.array): 3x3 rotation matrix
        t (np.array): 3x1 translation vector

    Returns:
        np.array: 4x4 transformation matrix
    """

    return np.vstack((np.hstack((R,np.reshape(t,[3,1]))), np.array([0,0,0,1])))

# NOTE the name of the function should imply the returned values if named GET_
def split_transformation_matrix(T:np.array):
    """Splits a transformation matrix into a rotation and translationmatrix

    **NOTE**: this function should be in utils and is already present in get_translation
    
    Args:
        T (np.array): The 4x4 transormationmatrix

    Returns:
        np.array, np.array: 3x3 rotationmatrix R, 3x1 translationmatrix t
    """
    return T[:3,:3], T[:3,3]
# NOTE remove obsolete classes -> ImageNodes
class AlignmentPose():
    """
    An alignment pose is used to transform a collection of pano poses.
    """

    def __init__(self, x :float, y: float, z: float, orientation: tuple, name: str = None, validate : bool = True):
        """Creation of the alignmentpose

        Args:\n
            x (float): x-coordinate\n
            y (float): y-coordinate\n
            z (float): z-coordinate\n
            orientation (tuple): rotation quaternion\n
            name (str, optional): the name of the pose. Defaults to None.\n
            validate (bool, optional): check if the rotation is a valid quaternion. Defaults to True.\n
        """
        if validate is True:
            assert(isinstance(orientation, tuple) and len(orientation) == 4)

        self.x = x
        self.y = y
        self.z = z
        self.orientation = orientation
        self.name = name


class PanoPose(AlignmentPose):
    """
    A pano pose gives the position, orientation and optionally time and name of
    a pano.
    """

    def __init__(self, x, y, z, orientation, time=None, name=None, validate=True):
        """
        """
        if validate is True:
            if time is not None:
                assert(isinstance(time, datetime))

        super().__init__(x, y, z, orientation, name)

        self.time = time

    @property
    def heading(self):
        """
        Heading measured as angle from x to y axis. In equirectangular format
        this is the center of the pano. Headings are always positive to
        simplify subsequent calculations.

        See 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html'
        """
        heading = Rotation.from_quat(
                self.orientation).as_euler('xyz', degrees=True)[-1]

        if heading < 0:
            heading = 360 + heading

        return heading

    @property
    def zenit(self):
        """
        Angle with the vertical
        """
        zenit = Rotation.from_quat(
                self.orientation).as_euler('xyz', degrees=True)[0]

        return zenit


    def get_direction_to_other_pose(self, other_pose, validate=True):
        """
        Angle from x to y with 'self' in the origin. Angles are always positive
        to simplify subsequent calculations.

        other_pose: instance of 'self'
        """
        if validate is True:
            assert(isinstance(self, self.__class__))

        x = other_pose.x - self.x
        y = other_pose.y - self.y
        angle = np.degrees(np.arctan2(y, x))

        if angle < 0:
            angle = 360 + angle

        return angle


class PanoPoseCollection():
    """
    A collection of pano poses with position, orientation and time.
    """

    def __init__(self, pos_xs, pos_ys, pos_zs, ori_xs, ori_ys, ori_zs,
        ori_ws, times=None, validate=True):
        """
        """
        if validate is True:
            assert(len(pos_xs) == len(pos_ys) == len(pos_zs) == len(ori_xs) ==
                len(ori_ys) == len(ori_zs) == len(ori_ws))
            if times is not None:
                assert(len(times) == len(ori_ws))

        self.pos_xs = pos_xs
        self.pos_ys = pos_ys
        self.pos_zs = pos_zs
        self.ori_xs = ori_xs
        self.ori_ys = ori_ys
        self.ori_zs = ori_zs
        self.ori_ws = ori_ws
        self.times = times

    def __len__(self):
        """
        """
        return len(self.ori_ws)

    def __getitem__(self, i):
        """
        """
        pano_pose = PanoPose(self.pos_xs[i], self.pos_ys[i], self.pos_zs[i],
            (self.ori_xs[i], self.ori_ys[i], self.ori_zs[i], self.ori_ws[i]),
            self.times[i], name=str(i), validate=False)

        return pano_pose

    def __iter__(self):
        """
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def headings(self):
        """
        """
        return [p.heading for p in self]

    @property
    def zenits(self):
        """
        """
        return [p.zenit for p in self]

    @property
    def box(self):
        """
        Give bounding box of pano collection.
        """
        x_min, x_max = self.pos_xs.min(), self.pos_xs.max()
        y_min, y_max = self.pos_ys.min(), self.pos_ys.max()

        return x_min, x_max, y_min, y_max

    def transform(self, aligment_pose, validate=True):
        """
        """
        if validate is True:
            assert(isinstance(aligment_pose, AlignmentPose))

        r = Rotation.from_quat(aligment_pose.orientation)
        coordinates = np.matmul(
            r.as_matrix(), np.array([self.pos_xs, self.pos_ys, self.pos_zs]))
        pos_xs = coordinates[0] + aligment_pose.x
        pos_ys = coordinates[1] + aligment_pose.y
        pos_zs = coordinates[2] + aligment_pose.z

        # TODO: Optimise this part so operation is performed at once
        ori_xs, ori_ys, ori_zs, ori_ws = [], [], [], []
        for p in self:
            pr = r * Rotation.from_quat(p.orientation)
        
            ori_x, ori_y, ori_z, ori_w = pr.as_quat()
            ori_xs.append(ori_x)
            ori_ys.append(ori_y)
            ori_zs.append(ori_z)
            ori_ws.append(ori_w)

        ppc = PanoPoseCollection(
            pos_xs, pos_ys, pos_zs, ori_xs, ori_ys, ori_zs, ori_ws, self.times)

        return ppc

    def plot(self, headings=False, size=None):
        """
        headings: boolean (default: False) - plots headings as vectors
        with size 1.
        """
        plot_pose_collections_3D(
            [self], colors=['k'], headings=headings, size=size)


def read_leica_pano_poses_xml(filespec):
    """
    Read xml file from .e57 exported with Leica Cyclone.
    """
    root = ET.parse(filespec).getroot()

    ns = '{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}'

    datetimes = []
    pos_xs, pos_ys, pos_zs = [], [], []
    ori_xs, ori_ys, ori_zs = [], [], []
    ori_ws = []

    for i, pano in enumerate(root.findall('.//{}vectorChild'.format(ns))):
        datetimes.append(datetime.fromtimestamp(float(
            pano.find('{}acquisitionDateTime'.format(ns)).find(
                '{}dateTimeValue'.format(ns)).text)))

        pose = pano.find('{}pose'.format(ns))
        if pose is None and i == 0: ## no translation and rotation given for first point
            ori_x, ori_y, ori_z = 0, 0, 0
            ori_w = 1
            pos_x, pos_y, pos_z = 0, 0, 0
        else:
            rotation = pose.find('{}rotation'.format(ns))
            ori_x = rotation.find('{}x'.format(ns)).text
            ori_y = rotation.find('{}y'.format(ns)).text
            ori_z = rotation.find('{}z'.format(ns)).text
            ori_w = rotation.find('{}w'.format(ns)).text
            if ori_x is None:
                ori_x = 0
            if ori_y is None:
                ori_y = 0
            if ori_z is None:
                ori_z = 0
            if ori_w is None:
                ori_w = 1
            
            ori_x = float(ori_x)
            ori_y = float(ori_y)
            ori_z = float(ori_z)

            translation = pose.find('{}translation'.format(ns))
            pos_x = float(translation.find('{}x'.format(ns)).text)
            pos_y = float(translation.find('{}y'.format(ns)).text)
            pos_z = float(translation.find('{}z'.format(ns)).text)

        ## Correct for Leica pano's (at least export from Cyclone) pointing
        ## towards positive y axis with center. This should be the positive
        ## x-axis so have to rotate 90 degrees.
        r = (Rotation.from_quat((ori_x, ori_y, ori_z, ori_w)) *
            Rotation.from_euler('z', 90, degrees=True))
        ori_x, ori_y, ori_z, ori_w = r.as_quat()

        ori_xs.append(ori_x)
        ori_ys.append(ori_y)
        ori_zs.append(ori_z)
        ori_ws.append(ori_w)
        pos_xs.append(pos_x)
        pos_ys.append(pos_y)
        pos_zs.append(pos_z)

    ppc = PanoPoseCollection(
        np.array(pos_xs), np.array(pos_ys), np.array(pos_zs),
        np.array(ori_xs), np.array(ori_ys), np.array(ori_zs),
        np.array(ori_ws),
        np.array(datetimes), validate=False)

    return ppc


def read_navvis_pano_poses_csv(filespec):
    """
    NavVis provides a pano-poses.csv file in each post-processed dataset. It
    holds the timestamp, position and orientation of each pano in the dataset.
    """
    with open(filespec) as f:
        data = f.read().split('\n')

    n = len(data) - 2

    datetimes = []
    pos_xs, pos_ys, pos_zs = np.zeros(n), np.zeros(n), np.zeros(n)
    ori_xs, ori_ys, ori_zs = np.zeros(n), np.zeros(n), np.zeros(n)
    ori_ws = np.zeros(n)
    for i, l in enumerate(data[1:-1]):
        r = l.split('; ')
        datetimes.append(datetime.fromtimestamp(float(r[2])))
        pos_xs[i] = r[3]
        pos_ys[i] = r[4]
        pos_zs[i] = r[5]
        ori_xs[i] = r[7]
        ori_ys[i] = r[8]
        ori_zs[i] = r[9]
        ori_ws[i] = r[6]

    ppc = PanoPoseCollection(pos_xs, pos_ys, pos_zs, ori_xs, ori_ys, ori_zs,
        ori_ws, np.array(datetimes), validate=False)

    return ppc


def read_navvis_alignment_xml(filespec):
    """
    Read xml file generated by NavVis aligment tool.
    """
    root = ET.parse(filespec).getroot()

    alignment_poses = []
    for dataset in root.findall('.//dataset'):
        name = dataset.find('name').text
        position = dataset.find('.//position')
        pos_x = float(position.find('x').text)
        pos_y = float(position.find('y').text)
        pos_z = float(position.find('z').text)
        orientation = dataset.find('.//orientation')
        ori_x = float(orientation.find('x').text)
        ori_y = float(orientation.find('y').text)
        ori_z = float(orientation.find('z').text)
        ori_w = float(orientation.find('w').text)

        alignment_poses.append(AlignmentPose(
            pos_x, pos_y, pos_z, (ori_x, ori_y, ori_z, ori_w), name))

    return alignment_poses


def plot_pose_collections(ppcs, colors=None, headings=False, size=None):
    """
    ppcs: list of PanoPoseCollection
    headings: boolean (default: False) - plots headings as vectors
    with size 1.
    """
    _, ax = plt.subplots(figsize=size)

    for i, ppc in enumerate(ppcs):
        kwargs = {}
        if colors is not None:
            kwargs['c'] = colors[i]
        ax.scatter(ppc.pos_xs, ppc.pos_ys, **kwargs)

        if headings is True:
            pc_headings = ppc.headings
            xs = np.cos(np.radians(pc_headings))
            ys = np.sin(np.radians(pc_headings))
            ax.quiver(ppc.pos_xs, ppc.pos_ys, xs, ys, color='k')

    ax.axis('equal')

    plt.show()

def plot_pose_collections_3D(ppcs, colors=None, headings=False, size=None):
    """
    ppcs: list of PanoPoseCollection
    headings: boolean (default: False) - plots headings as vectors
    with size 1.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, ppc in enumerate(ppcs):
        kwargs = {}
        if colors is not None:
            kwargs['c'] = colors[i]
        ax.scatter(ppc.pos_xs, ppc.pos_ys, ppc.pos_zs, **kwargs)

        if headings is True:
            pc_headings = ppc.headings
            pc_zenits = ppc.zenits
            xs = np.cos(np.radians(pc_headings))
            ys = np.sin(np.radians(pc_headings))
            zs = np.sin(np.radians(pc_zenits))

            ax.quiver(ppc.pos_xs, ppc.pos_ys,ppc.pos_zs, xs, ys, zs,color='k')

    ax.axis('auto')

    plt.show()
def decode_depthmap(source, resize = True, size = (8192,4096), show = False):
    """
    Function to decode the depthmaps generated by the navvis processing
    source: Location of the PNG files containing the depthmap
    resize(bool): If the resulting dethmap needs to be resized to match the size of the corresponding pano, by default True
    size: size of the corresponding pano, by default 8192x4096
    show: if true the result wil be shown, by default False
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

