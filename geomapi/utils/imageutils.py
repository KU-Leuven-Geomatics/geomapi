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


def initiate_hed_line_detection():
        
    """Saves images with annotated additional lines (abLines).

    This function takes a list of images (`imgNodes`) and detects additional lines using the LSD algorithm. It annotates the detected additional lines on each image and saves the annotated images to an output directory.

    Args:
        imgNodes (list): A list of image nodes containing information about images.
        output_dir (str): The path to the directory containing masked images.
        output (str): The path to the output directory where images with detected lines will be saved.
        score_thr (float): The score threshold for line detection.
        dist_thr (float): The distance threshold for line detection.
        interpreter: The TensorFlow Lite interpreter for line detection.
        input_details: Input details for the interpreter.
        output_details: Output details for the interpreter.

    Returns:
        list: A list of image nodes with additional line information.

    Example:
        imgNodes = [ImageNode1, ImageNode2, ...]  # List of image nodes
        output_dir = "path/to/masked_images_directory"
        output = "path/to/output_directory"
        score_thr = 0.5
        dist_thr = 2.0
        interpreter = tf.lite.Interpreter(model_path="line_detection_model.tflite")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        updated_imgNodes = saveImageAbLine(imgNodes, output_dir, output, score_thr, dist_thr, interpreter, input_details, output_details)

    Note:
        - The function detects additional lines in each image using the LSD algorithm.
        - It annotates the detected lines on each image and saves the annotated images to the output directory.
    """

    protoPath = r"O:\Code\hed\examples\hed\deploy.prototxt"
    modelPath = r"O:\Code\hed\examples\hed\hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    class CropLayer(object):
        def __init__(self, params, blobs):
            self.startX = 0
            self.startY = 0
            self.endX = 0
            self.endY = 0

        def getMemoryShapes(self, inputs):
            (inputShape, targetShape) = (inputs[0], inputs[1])
            (batchSize, numChannels) = (inputShape[0], inputShape[1])
            (H, W) = (targetShape[2], targetShape[3])

            self.startX = int((inputShape[3] - targetShape[3]) / 2)
            self.startY = int((inputShape[2] - targetShape[2]) / 2)
            self.endX = self.startX + W
            self.endY = self.startY + H

            return [[batchSize, numChannels, H, W]]

        def forward(self, inputs):
            return [inputs[0][:, :, self.startY:self.endY,
                    self.startX:self.endX]]

    cv2.dnn_registerLayer("Crop", CropLayer)

    return(protoPath,modelPath,net)


def calculateGSD(imageNode,bimNodes,f=24,Sw=4000):
    """
    Calculates the Ground Sampling Distance (GSD) in millimeters.

    Args:
        imageNode: An object representing the image node.
        bimNodes (list): A list of BIM nodes.
        f (float): The focal length of the camera in millimeters. Default is 24mm.
        Sw (float): The sensor width of the camera in millimeters. Default is 4000mm.

    Returns:
        float: The Ground Sampling Distance (GSD) in millimeters.

    Note:
        - The imageNode should have the attribute 'imageWidth' representing the width of the image in pixels.
        - The imageNode and each bimNode in bimNodes should have the attribute 'cartesianTransform' representing the Cartesian transformation matrix.
        - Each bimNode in bimNodes should have the attribute 'cartesianBounds' representing the Cartesian bounds of the BIM.

    """
    if isinstance(bimNodes, list):
        total_height = sum(bim_node.cartesianBounds[5] for bim_node in bimNodes)
        mean_height = total_height / len(bimNodes)
    else:
        mean_height=bimNodes.cartesianBounds[5]
    
    f = f  # in mm
    imW = imageNode.imageWidth  # in pix
    Sw = Sw  # in mm
    H = imageNode.cartesianTransform[2, 3] - mean_height  # in m, the height between the ground and the BIM

    GSD = (Sw * H * 10) / (f * imW)  # GSD in mm
    GSD = GSD / 1000

    return GSD

def detect_hed_lines(imgNodes,output,net,hom_begin_points_ad,hom_end_points_ad):
    """Create Holistically-Nested Edge Detection (HED) images and masked HED images.

    This function takes a list of images (`imgNodes`), an HED network (`net`), and information about additional lines (beginning and ending points) in homogeneous coordinates. It processes the images with the HED model to create edge maps and saves both the HED images and masked HED images to the output directory.

    Args:
        imgNodes (list): A list of image nodes containing information about images.
        output (str): The path to the output directory where HED images will be saved.
        net: The initialized HED network.
        hom_begin_points_ad (list): A list of lists containing the homogeneous coordinates of beginning points of additional lines.
        hom_end_points_ad (list): A list of lists containing the homogeneous coordinates of ending points of additional lines.

    Returns:
        None

    Example:
        imgNodes = [ImageNode1, ImageNode2, ...]  # List of image nodes
        output = "path/to/output_directory"
        net = initialized_HED_network
        hom_begin_points_ad = [[point1, point2, ...], [point1, point2, ...], ...]  # List of homogeneous coordinates of beginning points
        hom_end_points_ad = [[point1, point2, ...], [point1, point2, ...], ...]  # List of homogeneous coordinates of ending points
        createHEDs(imgNodes, output, net, hom_begin_points_ad, hom_end_points_ad)

    Note:
        - The function processes each image with the HED model to create edge maps.
        - It saves both the HED images and masked HED images to the output directory.
    """
        
    blobs=[]
    for img in imgNodes:
        start_list=[]
        end_list=[]
        output_dir = os.path.join(output,"Images", "Masked_Image")
        imgcv=os.path.join(output_dir,'masked_image_'+img.name+'.jpg')
        if os.path.exists(imgcv):
            im=cv2.imread(imgcv)
            H,W=im.shape[0:2]
            blob = cv2.dnn.blobFromImage(im, scalefactor=1, size=(W, H),
                                        mean=(100, 180, 100),
                                        swapRB= False, crop=False)
            blobs.append(blob)
    heds=[]
    for blob in blobs:
        net.setInput(blob)
        hed = net.forward()
        hed = hed[0,0,:,:]  
        threshold = 0.85  
        hed = (hed > threshold) * 255  
        hed = hed.astype("uint8")
        heds.append(hed)

    output_dir1 = os.path.join(output,"Images","HED", "Image_HED")
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)

    for count,hed in enumerate(heds):
        output_image_name = os.path.join(output_dir1, f"image_HED_{imgNodes[count].name}.jpg")
        cv2.imwrite(output_image_name, hed)
    
    for tel,hed in enumerate(heds):
        mask = np.zeros(hed.shape[:2], dtype="uint8")
        for count, start_list in enumerate(hom_begin_points_ad):
            for i,start in enumerate (start_list):
                stop_list = hom_end_points_ad[count]
                stop = stop_list[i]
                imgNodes[tel].uvCoordinates_start = world_to_pixel(imgNodes[tel],start)
                imgNodes[tel].uvCoordinates_stop = world_to_pixel(imgNodes[tel],stop)
                u_start, v_start = int(imgNodes[tel].uvCoordinates_start[0]), int(imgNodes[tel].uvCoordinates_start[1])
                u_stop, v_stop = int(imgNodes[tel].uvCoordinates_stop[0]), int(imgNodes[tel].uvCoordinates_stop[1])

                cv2.line(mask, (u_start,v_start),(u_stop,v_stop), (255, 255, 255, 255), thickness=60)
        output_dir2 = os.path.join(output,"Images","HED", "Masked_Image_HED")
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)

        masked_img = cv2.bitwise_and(hed, hed, mask=mask)
        output_image_name = os.path.join(output_dir2, f"masked_image_HED_{imgNodes[tel].name}.jpg")
        cv2.imwrite(output_image_name, masked_img)


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
    """Calibrates a camera and determines the intrinsic matrix and distortion coefficients using a list of images of a chessboard pattern.

    Args:\n
        images (list(np.array)): a list of cv2 images\n
        CheckerBoardPattern (tuple, optional): The pattern of the checkerboard. Count the inner crosses. Defaults to (7,7).\n
        squareSize (float, optional): The size of 1 square of the checkerboard. Defaults to 1.\n
        drawImages (bool, optional): Display checker image after each iteration. Defaults to False.\n

    Returns:
        bool: The success of the calculation
        np.array(): The intrinsic matrix
        np.array(): The distortion coefficients
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



