"Tools to align different datasets using both images and 3d meshes"

import datetime

import numpy as np
import quaternion
import cv2
from typing import List

from geomapi.nodes import SessionNode
import geomapi.utils.geometryutils as gu
import geomapi.tools.alignmenttools.params as params
import geomapi.tools.alignmenttools.match as match
from geomapi.tools.alignmenttools.match import Match, Match2d, Match3d, PoseEstimation


# returns all sessions that are within the bounding volume
def find_close_sessions(boundingVolume: np.array, referenceSessions: List[SessionNode]) ->  List[SessionNode]:
    """Determines the close ennoug sessions based on bounding volumes

    Args:
        boundingVolume (np.array 6x1): The test bounding volume
        referenceSessions (list[SessionNode]): All the sessionNodes to check

    Returns:
        list[SessionNode]: All the close enough session nodes
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

    averageRotation = gu.weighted_average_quaternions(Q,w)
    averagePosition = np.average(T,axis = 0,weights = w)

    transformation = np.vstack((cv2.hconcat((quaternion.as_rotation_matrix(averageRotation), averagePosition)), np.array([0,0,0,1])))

    return transformation

def estimate_session_position(testSession : SessionNode, refSessions: List[SessionNode]) -> np.array:
    """Estimates the global position of a session in relation to a list of reference sessions

    Args:
        testSession (SessionNode): The Test Session where the positions needs to be determined from
        refSessions (list[SessionNode]): The Reference sessions to help calculate the new position

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


