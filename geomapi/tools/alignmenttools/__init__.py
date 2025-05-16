"Tools to align different datasets using both images and 3d meshes"

import datetime

import numpy as np
import quaternion
import cv2
import open3d as o3d
from typing import List

from geomapi.nodes import SetNode
import geomapi.utils.geometryutils as gu
import geomapi.tools.alignmenttools.params as params
import geomapi.tools.alignmenttools.match as match
from geomapi.tools.alignmenttools.match import Match, Match2d, Match3d, PoseEstimation


# returns all sessions that are within the bounding volume
def find_close_sessions(boundingVolume: np.array, referenceSessions: List[SetNode]) ->  List[SetNode]:
    """Determines the close ennoug sessions based on bounding volumes

    Args:
        boundingVolume (np.array 6x1): The test bounding volume
        referenceSessions (list[SetNode]): All the SetNodes to check

    Returns:
        list[SetNode]: All the close enough session nodes
    """

    boundingboxes = []
    for sess in referenceSessions:
        boundingboxes.append(sess.get_cartesian_bounds())
    inds = gu.get_box_intersections(boundingVolume, boundingboxes) #Returns inds of all the intersecting bounding boxes

    inlierSessions = []
    for idx in inds:
        inlierSessions.append(referenceSessions[idx])

    return inlierSessions

def get_weighted_pose(poses: PoseEstimation) -> np.array:
    """Calculates the weighted average of a list of PoseEstimations

    Args:
        poses (PoseEstimation): The list of Pose Estimations

    Returns:
        np.array 4x4: The Average transformation matrix
    """

    rotations = []
    positions = []
    weights = []
    for estimation in poses:
        rotations.append(quaternion.from_rotation_matrix(estimation.rotation))
        positions.append(estimation.position)
        weights.append(estimation.get_confidence())
    Q = np.array(rotations)
    T = np.array(positions)
    w = np.array(weights)/sum(weights)

    averageRotation = weighted_average_quaternions(Q,w)
    averagePosition = np.average(T,axis = 0,weights = w)

    transformation = np.vstack((cv2.hconcat((quaternion.as_rotation_matrix(averageRotation), averagePosition)), np.array([0,0,0,1])))

    return transformation

def estimate_session_position(testSession : SetNode, refSessions: List[SetNode]) -> np.array:
    """Estimates the global position of a session in relation to a list of reference sessions

    Args:
        testSession (SetNode): The Test Session where the positions needs to be determined from
        refSessions (list[SetNode]): The Reference sessions to help calculate the new position

    Returns:
        np.array: The estimated transformation
    """

    estimations = []
    #find the image with the best match rate
    for referenceSession in refSessions:
        estimations.append(match.match_session(testSession, referenceSession))

    # Perform a weighted average of all the estimations
    transform = get_weighted_pose(estimations)

    return transform


def execute_fast_global_registration(source_pcd : o3d.geometry.Geometry, target_pcd : o3d.geometry.Geometry, source_fpfh : np.array,target_fpfh : np.array, radius: float):
    """Performs a RANSAC based Fpfh feature matching of 2 pointclouds using a faster algorithm

    Args:
        source_down (o3d.geometry.Geometry): the source PCD
        target_down (o3d.geometry.Geometry): the Target PCD
        source_fpfh (np.array): The sourse features
        target_fpfh (np.array): The target features
        radius (float): The correspondance radius

    Returns:
         RegistrationResult: The resulting registration
    """

    print(":: Apply fast global registration with distance threshold %.3f" \
            % radius)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=radius))

    return result
def execute_global_registration(source_down : o3d.geometry.Geometry, target_down : o3d.geometry.Geometry, source_fpfh : np.array,target_fpfh : np.array, voxel_size : float):
    """Performs a RANSAC based Fpfh feature matching of 2 pointclouds

    Args:
        source_down (o3d.geometry.Geometry): the source PCD
        target_down (o3d.geometry.Geometry): the Target PCD
        source_fpfh (np.array): The sourse features
        target_fpfh (np.array): The target features
        voxel_size (np.array): The size to downsample to for the feature matching

    Returns:
        RegistrationResult: The resulting registration
    """

    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    return result

def weighted_average_quaternions(Q : np.array, w: np.array):
    """Calculates a weighted average of a list of quaternions 
    The weight vector w must be of the same length as the number of rows in the quaternion maxtrix Q

    Args:
        Q (List[np.array]): a list of quaternions
        w (List[float]): a lost of weights

    Returns:
        np.array: the weighted 4x1 quaternion
    """
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))
    weightSum = 0
    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]
    # scale
    A = (1.0/weightSum) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)

def calibrate_camera(images, CheckerBoardPattern = (7,7), squareSize: float = 1, drawImages:bool = False):
    """
    Calibrate a camera using a list of chessboard images.

    Args:
        - images (list[np.array]): A list of cv2 images.
        - CheckerBoardPattern (tuple, optional): Number of internal corners (rows, columns). Defaults to (7,7).
        - squareSize (float, optional): Size of one square in real-world units. Defaults to 1.
        - drawImages (bool, optional): If True, shows images with detected corners. Defaults to False.

    Returns:
        - bool: Success of the calibration.
        - np.array: Intrinsic matrix (camera matrix).
        - np.array: Distortion coefficients.
        - list[np.array]: Rotation vectors.
        - list[np.array]: Translation vectors.
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
        if image.ndim == 3: #color image
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        else: 
            gray = image
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CheckerBoardPattern)
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