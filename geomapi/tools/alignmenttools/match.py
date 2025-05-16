import datetime
import math
from typing import List

import cv2
import numpy as np
from scipy import optimize

import geomapi.tools.alignmenttools.params as params
from geomapi.nodes import MeshNode, ImageNode, Node, SetNode
import geomapi.utils.imageutils as iu
import geomapi.utils.geometryutils as gmu

class Match:
    """The generic match object to determine the relation between 2 geomapi.Node objects"""

    _matchType = "" # Defines the type of match

    def __init__(self, node1: Node, node2: Node) -> None:
        """The default constructor

        Args:
            node1 (Node): The first node, defines the transfrom from
            node2 (Node): The second node, defines the transformation to
        """
        self.node1: Node = node1                    #: The first node, defines the transfrom from
        self.node2: Node = node2                    #: The second node, defines the transformation to

        self.matches = None
        self.matchError: float = math.inf           #: A single value indicating the quality of the match, lower is better
        self.matchAmount: float = 0                 #: The number of good matches
        self.transformation: np.array = np.ones(4)  #: The 4x4 transformationmatrix defined from 1 to 2
        pass
    
    # The initial matching to evaliaute the quality of the match
    def get_matches(self):
        """Finds the matches between 2 objects"""
        print("WARNING: Calling 'get_matches()' on a generic Match object, please use a 2D or 3D match instead")

    # The full transormation calculation
    def get_transformation(self):
        """Returns the estimated transformation between the 2 objects"""
        print("WARNING: Calling 'get_transformation()' on a generic Match object, please use a 2D or 3D match instead")

class Match2d (Match):

    _matchType = "2d"

    def __init__(self, imageNode1: ImageNode, imageNode2: ImageNode) -> None:
        """The default constructor

        Args:
            imageNode1 (ImageNode): The first node, defines the transfrom from
            imageNode2 (ImageNode): The second node, defines the transformation to
        """
        super().__init__(imageNode1, imageNode2)

        self.image1 = None
        self.image2 = None
    
    def get_matches(self):
        """Finds matches between the 2 images"""

        # Inherit the super functionality
        super().get_matches()

        if(self.matches is None):
            # get cv2 ORb features
            self.image1 = self.node1.resource
            self.image2 = self.node2.resource

            if(not (self.image1 & self.image2)):
                print("Node 1 or 2 do not have an image resource to get matches from")
                return None

            # Get the features
            self.image1.get_image_features(max =params.MAX_2D_FEATURES)
            self.image2.get_image_features(max =params.MAX_2D_FEATURES)
            # Match the features
            matches = iu.match_features(self.image1.descriptors, self.image2.descriptors)
            # only use the best features
            if(len(matches) < params.MAX_2D_MATCHES):
                print("only found", len(matches), "good matches")
                matchError = math.inf
            else:
                matches = matches[:params.MAX_2D_MATCHES]
                # calculate the match score
                # right now, it's just the average distances of the best points
                matchError = 0
                for match in matches:
                    matchError += match.distance
                matchError /= len(matches)
            self.matches = matches
            self.matchError = matchError
            self.matchAmount = len(matches)
        return self.matches

    # The full transormation calculation
    def get_transformation(self):
        """Returns the estimated transformation between the 2 objects"""
        self.intrinsic1 = self.image1.get_intrinsic_camera_parameters(1).intrinsic_matrix
        self.intrinsic2 = self.image2.get_intrinsic_camera_parameters(1).intrinsic_matrix

        # Extract location of good matches
        points1 = np.zeros((len(self.matches), 2), dtype=np.float32)
        points2 = np.zeros((len(self.matches), 2), dtype=np.float32)
        for i, match in enumerate(self.matches):
            points1[i, :] = self.image1.keypoints[match.queryIdx].pt
            points2[i, :] = self.image2.keypoints[match.trainIdx].pt

        _, self.E, self.R, self.t, self.mask = cv2.recoverPose(
            points1= points1,
            points2= points2,
            cameraMatrix1= self.intrinsic1,
            distCoeffs1= None,
            cameraMatrix2= self.intrinsic2,
            distCoeffs2= None)

        self.transformation = gmu.get_cartesian_transform(self.t, self.R) #iu.create_transformation_matrix(self.R,self.t)
        return self.transformation    

    def get_pnp_pose(self, OtherMatch):
        """Calculates the pose of a third camera with matches """
        self.fidelity = params.ERROR_2D
        
        # match old descriptors against the descriptors in the new view
        self.get_matches()
        self.get_transformation()
        points_3D = []
        points_2D = []

        # loop over all the 3d points in the other match to find the same points in this match
        self.iterativeMatch = []
        for point in OtherMatch.pointMap:
            currentOtherPixel = np.around(point[1])
            for match in self.matches:
                currentSelfPixel = np.around(self.image1.keypoints[match.queryIdx].pt)
                currentSelfQueryPixel = np.around(self.image2.keypoints[match.trainIdx].pt)
                #print(currentSelfPixel)
                if(np.array_equal(currentOtherPixel,currentSelfPixel)):
                    #print("found match: ", currentSelfPixel)
                    points_3D.append(point[2])
                    points_2D.append(np.array(currentSelfQueryPixel).T.reshape((1, 2)))
                    self.iterativeMatch.append([np.around(point[0]), currentSelfPixel, currentSelfQueryPixel, point[2]])
                    break

        if(len(points_3D) < 10):
            return self.image1.transform
        
        # compute new inverse pose using solvePnPRansac
        _, R, t, _ = cv2.solvePnPRansac(np.array(points_3D), np.array(points_2D), self.image2.intrinsic_matrix, None,
                                        confidence=0.99, reprojectionError=8.0, flags=cv2.SOLVEPNP_DLS, useExtrinsicGuess=True)
        R, _ = cv2.Rodrigues(R)
        self.points3d = points_3D
        return gmu.get_cartesian_transform(-R.T @ t, R.T)# iu.create_transformation_matrix(R.T, -R.T @ t)
    
    def get_reference_scaling_factor(self):
        """Uses the real world distance to scale the translationvector"""
        t1 = gmu.get_translation(self.image1.get_cartesian_transform()) #iu.split_transformation_matrix(self.image1.get_cartesian_transform())
        t2 = gmu.get_translation(self.image2.get_cartesian_transform()) #iu.split_transformation_matrix(self.image2.get_cartesian_transform())
        scalingFactor = np.linalg.norm(t2 - t1)
        self.t = self.t * scalingFactor / np.linalg.norm(self.t)
        return scalingFactor

    def set_scaling_factor(self, scalingFactor):
        """Uses the real world distance to scale the translationvector"""

        self.t = self.t * scalingFactor / np.linalg.norm(self.t)
        return self.t

    def get_image2_pos(self, local = False):
        """Return the global/local position of image2 with t and R """
        if(local):
            return self.R, self.t
        else:
            R_local = gmu.get_rotation_matrix(self.image1.get_cartesian_transform()) 
            t_local = gmu.get_translation(self.image1.get_cartesian_transform())   #iu.split_transformation_matrix(self.image1.get_cartesian_transform())
            R = R_local @ self.R.T
            t = t_local - np.reshape(R @ self.t,(3,1))
            return R,t

class Match3d (Match):
    
    _matchType = "3d"

    def __init__(self, geometryNode1: MeshNode, geometryNode2: MeshNode) -> None:
        """The default constructor

        Args:
            imageNode1 (ImageNode): The first node, defines the transfrom from
            imageNode2 (ImageNode): The second node, defines the transformation to
        """
        super().__init__(geometryNode1, geometryNode2)

        self.geometry1 = None
        self.geometry2 = None
    
    def get_matches(self):
        """Finds matches between the 2 images"""

        # Inherit the super functionality
        super().get_matches()

        if(self.matches is None):
            # get cv2 ORb features
            self.image1 = self.node1.get_resource()
            self.image2 = self.node2.get_resource()

            if(not (self.image1 & self.image2)):
                print("Node 1 or 2 do not have an image resource to get matches from")
                return None

            #image1.get_cv2_features(params.MAX_2D_FEATURES)

            # Match features.
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(self.image1.descriptors, self.image2.descriptors, None)
            # Sort matches by score
            matches = sorted(matches, key = lambda x:x.distance)
            # only use the best features
            if(len(matches) < params.MAX_2D_MATCHES):
                print("only found", len(matches), "good matches")
                matchError = math.inf
            else:
                matches = matches[:params.MAX_2D_MATCHES]
                # calculate the match score
                # right now, it's just the average distances of the best points
                matchError = 0
                for match in matches:
                    matchError += match.distance
                matchError /= len(matches)
            self.matches = matches
            self.matchError = matchError
            self.matchAmount = len(matches)
        return self.matches


class PoseEstimation():
    """Contains an estimated pose and all it's parameters to calculate it's validity"""

    transformation = np.array([])
    matches: Match = []
    method = ""
    referenceSessionId = ""


    def __init__(self, transformation, matches, method) -> None:
        self.transformation = transformation
        self.matches = matches
        self.method = method

    def get_confidence(self, session) -> float:
        """Returns the confidence of an estimation based on the matching parameter value from 0 to 1"""
        
        # the starting confidence is 1
        factors = []

        # The match specific parameters
        matchErrorFactor = 1        # the error radius of the match
        matchAmountFactor = 1       # the amount of good matches/inliers

        for match in self.matches:
            if(isinstance(match, Match2d)):
                matchErrorFactor = 1 - (min(params.MAX_ERROR_2D, match.matchError)/params.MAX_ERROR_2D) #remap from 0-MaxError to 1-0
                matchAmountFactor = match.matchAmount / params.MAX_2D_MATCHES
                factors.append([matchErrorFactor, params.ERROR_2D])
                factors.append([matchAmountFactor, params.MATCHES_2D])
            elif(isinstance(match, Match3d)):
                matchErrorFactor = 1 - (min(params.MAX_ERROR_3D, match.matchError)/params.MAX_ERROR_3D) #remap from 0-MaxError to 1-0
                matchAmountFactor = match.matchAmount / params.MAX_3D_MATCHES
                factors.append([matchErrorFactor, params.ERROR_3D])
                factors.append([matchAmountFactor, params.MATCHES_3D])

        # The method Parameters
        methodFactor = 1
        if (self.method == "leastDistance"):
            methodFactor *= params.LEAST_DISTANCE
        elif (self.method == "incremental"):
            methodFactor *= params.INCREMENTAL
        elif (self.method == "raycasting"):
            methodFactor *= params.RAYCASTING
        factors.append([methodFactor, params.METHOD])

        # The Other session Parameters
        dateFactor = (datetime.datetime.now() - session.recordingDate).total_seconds()
        factors.append([dateFactor, params.SESSION_DATE])
        sensorFactor = session.fidelity
        factors.append([sensorFactor, params.SENSOR_TYPE])

        confidence = np.average(np.array(factors)[:,0], weights = np.array(factors)[:,1])

        return confidence


# Match 2 sessions using the different matching methods
def match_session(testSession : SetNode, refSession : SetNode):
    estimations = []
    
    # loop over every test node in the session, and find compatible estimation methods
    for testNode  in testSession.linkedNodes:

        # Perform the 2D check, only if it is an ImageNode
        if(type(testNode) is ImageNode):
            if(sum(isinstance(e,ImageNode) for e in refSession.linkedNodes) > 1): # we need 2 ref images to match
                estimations.append(match_crossref(testNode, refSession))
                estimations.append(match_incremental(testNode, refSession))
            if(sum(isinstance(e,MeshNode) for e in refSession.linkedNodes) > 0
            and sum(isinstance(e,ImageNode) for e in refSession.linkedNodes) > 0): # we need a mesh to raycast against
                estimations.append(match_raycast(testNode, refSession))

        # Perform the 3D check, only if it is a GeometryNode
        if(type(testNode) is MeshNode):
            if(sum(isinstance(e,MeshNode) for e in refSession.linkedNodes) > 0): # we at least one other geometry to match against
                estimations.append(match_fgr(testNode, refSession))
                estimations.append(match_super4pcs(testNode, refSession))

    return estimations

# Different Matching methods for 2D objects (images)

# Searches for the best match in the refImages, then searches for its best match in the other refimages
# Creates a match between the 2 linked images and iterrates the third one. using pnp pose
def match_incremental(testImage: Node , refImages: List[Node]) -> PoseEstimation:
    # Find the best test-ref image match
    bestMatch = get_best_matches(testImage, refImages, nr=1)
    # Find the best ref-ref image match with the chosen ref image
    bestRefMatch = get_best_session_match(bestMatch.image1, refImages)
    # Calculate the transformation of the ref-ref-match
    # Get the PnP pose of the test image
    transformation = bestMatch.get_pnp_pose(bestRefMatch)
    # return a transformation matrix and the matches
    return PoseEstimation(transformation, [bestMatch, bestRefMatch], "incremental")

def match_crossref(testImage: Node , refImages: List[Node]):
    # Find the 2 best test-ref image match
    bestMatches = get_best_matches(testImage, refImages, nr=2)
    match1 = bestMatches[0]
    match2 = bestMatches[1]
    # Calculate the transformation of each of the matches
    # Find the closest point between the 2 direction
    def get_position(scaleFactor, match : Match2d):
        """Returns the translation in function of a scale factor"""
        match.set_scaling_factor(scaleFactor)
        _,t = match.get_image2_pos()
        #newPosition = imageTransform.pos + scaleFactor * (imageTransform.get_rotation_matrix() @ translation).T
        return t

    def get_distance_array(x):
        pos1 = get_position(x[0], match1)
        pos2 = get_position(x[1], match2)
        return np.linalg.norm(pos2-pos1)

    minimum = optimize.fmin(get_distance_array, [1,1])

    pos1 = get_position(minimum[0], match1)
    pos2 = get_position(minimum[1], match2)
    t =(pos1 + pos2)/2 #return the average of the 2 positions
    R,_ = match1.get_image2_pos()
    # return a transformation matrix and the matches
    return PoseEstimation(gmu.get_cartesian_transform(t,R), bestMatches, "crossref")# iu.create_transformation_matrix(R,t), bestMatches, "crossref")

def match_raycast(testImage: Node , refImages: List[Node], geometry: Node) -> PoseEstimation:
    pass


# Different 3D matching methods

def match_fgr(testGeometry: MeshNode, refSession : SetNode) -> PoseEstimation:

    estimations = []

    for node in refSession.linkedNodes:
        if(type(node) is MeshNode):
            #The node is a geometrynode
            newMatch = Match3d(testGeometry, node)

    pass

def match_super4pcs(testGeometry: MeshNode, refSession : SetNode) -> PoseEstimation:
    return None


def get_best_session_match(image: Node , imageList: List[Node]):
    """Finds the best match in the same session"""

    if(image not in imageList): 
        print("ERROR: Image not in list")
        return None
    newList = imageList.copy()
    newList.remove(image)
    bestRefMatch = get_best_matches(image, newList)
    #Calculate the 3D points in the scene with the know real world locations of the 2 reference images
    
    bestRefMatch.get_essential_matrix() #calculate the essential matrix and inliers
    bestRefMatch.get_reference_scaling_factor() # get the scene scale by using the real world distances
    bestRefMatch.triangulate(True) #calulate the 3d points

    return bestRefMatch

# Check a test image against a list of reference images. Returns a list of the "nr" best matches
def get_best_matches(testImage, refImages, nr: int = 1) -> Match:
    results = [] # a list of all the results
    bestResults = [] # a list of the best results
    nrCheck = 0
    totalCheck = len(refImages)

    for refImage in refImages:
            newMatch = Match2d(refImage, testImage) #create a new match between 2 images
            newMatch.find_matches() # find the best matches 
            results.append(newMatch)

            # check if the newResult is in the top of results
            bestResults.append(newMatch)
            bestResults = sorted(bestResults, key= lambda x: x.matchError) #sort them from low to High
            if(len(bestResults) > nr): #remove the worst match
                bestResults = bestResults[:nr]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete with matchError:" + str(newMatch.matchError))

    for result in bestResults:
        result.get_transormation() # determin the transformation and inliers

    if(nr == 1): return bestResults[0]
    return bestResults