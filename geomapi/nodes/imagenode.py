"""
**ImageNode** is a Python Class to govern the data and metadata of pinhole camera data i.e. normal cameras. 

This node builds upon the [OpenCV](https://opencv.org/), [Open3D](https://www.open3d.org/) and [PIL](https://pillow.readthedocs.io/en/stable/) API for the image definitions.
Be sure to check the properties defined in those abstract classes to initialise the Node.

.. image:: ../../../docs/pics/graph_img_1.png

**IMPORTANT**: This Node class is designed to manage geolocated imagery. It works best when the camera interior and exterior parameters are known. The class can be used to generate virtual images, raycasting, and other geometric operations.

"""
#IMPORT PACKAGES
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import PIL
from matplotlib import pyplot as plt
from rdflib import XSD, Graph, URIRef
import numpy as np
import os
import open3d as o3d

from scipy.spatial.transform import Rotation as R
import copy
from typing import List, Optional,Tuple,Union
from PIL import Image

#IMPORT MODULES
from geomapi.nodes import Node
import geomapi.utils as ut
from geomapi.utils import rdf_property, GEOMAPI_PREFIXES
import geomapi.utils.geometryutils as gmu
import geomapi.utils.imageutils as iu
import geomapi.utils.geospatialutils as gsu


class ImageNode(Node):
    def __init__(self, 
                subject: Optional[URIRef] = None,
                graph: Optional[Graph] = None,
                graphPath: Optional[Path] = None,
                name: Optional[str] = None,
                path: Optional[Path] = None,
                timestamp: Optional[str] = None,
                resource = None,
                cartesianTransform: Optional[np.ndarray] = None,
                orientedBoundingBox: Optional[o3d.geometry.OrientedBoundingBox] = None,
                convexHull: Optional[o3d.geometry.TriangleMesh] =None,
                loadResource: bool = False,
                xmpPath: Path = None,
                xmlPath: Path = None,
                imageWidth:int = None, #640
                imageHeight:int = None, #480
                principalPointU:float =None,  
                principalPointV:float= None, 
                focalLength35mm:float = None, #2600
                intrinsicMatrix:np.ndarray=None,
                keypoints:np.ndarray=None,
                descriptors:np.ndarray=None,
                depth: float= None,
                **kwargs):
        """Creates an ImageNode. Overloaded function.
                
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to

        Args:
            - subject (RDFlib URIRef) : subject to be used as the main identifier in the RDF Graph
            
            - graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - graphPath (Path) :  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - path (Path) : Path to an image .jpg, jpeg, .png file (data is not automatically loaded)
            
            - resource (ndarray, PIL, Open3D) : Image data from [Open3D](https://www.open3d.org/), [OpenCV](https://opencv.org/) or [PIL](https://pillow.readthedocs.io/en/stable/). 

            - xmlPath (Path) : Xml file path from Agisoft Metashape
            
            - xmpPath (Path) :  xmp file path from RealityCapture
                        
            - getResource (bool, optional= False) : If True, the node will search for its physical resource on drive 
            
            - imageWidth (int, optional) : width of the image in pixels (u). Defaults to 640p
            
            - imageHeight (int, optional) : height of the image in pixels (v). Defaults to 480p
            
            - intrinsicMatrix (np.array, optional) : intrinsic camera matrix (3x3) k=[[fx 0 cx] [0 fy cy][0 0  1]]
            
            - focalLength35mm (float, optional) : focal length with a standardised Field-of-View in pixels. Defaults to circa 2600pix for a 25.4mm lens
            
            - keypoints (np.array, optional) : a set of image keypoints, generated through sift or orb. 
            
            - descriptors (np.array, optional) : a set of image descriptors, generated through sift or orb.
            
        Returns:
            ImageNode : An ImageNode with metadata 
        """ 

        #instance variables
        self.xmpPath=xmpPath
        self.xmlPath=xmlPath
        self.imageWidth=imageWidth
        self.imageHeight=imageHeight
        self.principalPointU=principalPointU
        self.principalPointV=principalPointV
        self.focalLength35mm=focalLength35mm
        self.keypoints=keypoints
        self.descriptors = descriptors
        self.intrinsicMatrix = intrinsicMatrix
        self.depth = depth

        super().__init__(   subject = subject,
                            graph = graph,
                            graphPath = graphPath,
                            name = name,
                            path = path,
                            timestamp = timestamp,
                            resource = resource,
                            cartesianTransform = cartesianTransform,
                            orientedBoundingBox = orientedBoundingBox,
                            convexHull = convexHull,
                            loadResource = loadResource,
                            **kwargs)   
        
        #Set image based parameters from metadata
        self.get_metadata_from_xmp_path() if xmpPath and os.path.exists(xmpPath) else None
        self.get_metadata_from_exif_data(path) if path and os.path.exists(path) else None


#---------------------PROPERTIES----------------------------

  #---------------------xmpPath----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def xmpPath(self): 
        """Get the xmpPath (str) of the node. This is the RealityCapture xmp file path."""
        return self._xmpPath#ut.parse_path(self._xmpPath)

    @xmpPath.setter
    def xmpPath(self,value:Path):
        if value is None:
            self._xmpPath = None
        elif Path(value).suffix =='.xmp':
            self._xmpPath=Path(value)
        else:
            raise ValueError('self.xmpPath has invalid type, path or extension.')
        
#---------------------xmlPath----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def xmlPath(self): 
        """Get the xmlPath (str) of the node. This is the RealityCapture xml file path."""
        return self._xmlPath#ut.parse_path(self._xmpPath)

    @xmlPath.setter
    def xmlPath(self,value:Path):
        if value is None:
            self._xmlPath = None
        elif Path(value).suffix =='.xml':
            self._xmlPath=Path(value)
        else:
            raise ValueError('self.xmlPath has invalid type, path or extension.')   

    #---------------------imageWidth----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES['exif'].imageWidth, datatype=XSD.int)
    def imageWidth(self):
        """Get the imageWidth (int) or number of columns of the resource of the node."""
        if self._imageWidth is None:
            self._imageWidth = 640
            if self.resource is not None:
                self._imageWidth=self._resource.shape[1]
        return self._imageWidth
    
    @imageWidth.setter
    def imageWidth(self,value:int):
        if value is None:
            self._imageWidth = None
        elif type(int(value)) is int:
            self._imageWidth=int(value)
        else:
            raise ValueError('self.imageWidth must be an integer')
        
    #---------------------imageHeight----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES['exif'].imageLength, datatype=XSD.int)
    def imageHeight(self):
        """Get the imageHeight (int) or number of rows of the resource of the node."""
        if self._imageHeight is None:
            self._imageHeight = 480
            if self.resource is not None:
                self._imageHeight=self.resource.shape[0]
        return self._imageHeight
    
    @imageHeight.setter
    def imageHeight(self,value:int):
        if value is None:
            self._imageHeight = None
        elif type(int(value)) is int:
            self._imageHeight=int(value)
        else:
            raise ValueError('self.imageHeight must be an integer')
    
    #---------------------principalPointU----------------------------
    @property
    @rdf_property(datatype=XSD.float)
    def principalPointU(self):
        """Get the principalPointU (float) of the node. Note that this is the deviation, not the actual value."""
        if self._principalPointU is None:
            self._principalPointU = 0
        return self._principalPointU
    
    @principalPointU.setter
    def principalPointU(self,value:float):
        if value is None:
            self._principalPointU = None
        elif type(float(value)) is float:
            self._principalPointU=float(value)
        else:
            raise ValueError('self.principalPointU must be a float')
        
    #---------------------principalPointV----------------------------
    @property
    @rdf_property(datatype=XSD.float)
    def principalPointV(self):
        """Get the principalPointV (float) of the node. Note that this is the deviation, not the actual value."""
        if self._principalPointV is None:
            self._principalPointV = 0
        return self._principalPointV
    
    @principalPointV.setter
    def principalPointV(self,value:float):
        if value is None:
            self._principalPointV = None
        elif type(float(value)) is float:
            self._principalPointV=float(value)
        else:
            raise ValueError('self.principalPointV must be a float')
        
    #---------------------focalLength35mm----------------------------
    @property
    @rdf_property(datatype=XSD.float)
    def focalLength35mm(self):
        """Get the focalLength35mm (float) of the node."""
        if self._focalLength35mm is None:
            self._focalLength35mm = 2600
        return self._focalLength35mm
    
    @focalLength35mm.setter
    def focalLength35mm(self,value:float):
        if value is None:
            self._focalLength35mm = None
        elif type(float(value)) is float:
            self._focalLength35mm=float(value)
        else:
            raise ValueError('self.focalLength35mm must be a float')
        
    #---------------------keypoints----------------------------
    @property
    def keypoints(self):
        """Get the keypoints (np.array) of the node. These are the distinct pixels in the image."""
        if(self._keypoints is None):
            print("Keypoints are None, use get_image_features() to calculate keypoints and descriptors")
        return self._keypoints
    
    @keypoints.setter
    def keypoints(self,value:np.ndarray):
        if value is None:
            self._keypoints = None
        elif type(np.array(value)) is np.ndarray:
            self._keypoints=np.array(value)
        else:
            raise ValueError('self.keypoints must be a numpy array')
        
    #---------------------descriptors----------------------------
    @property
    def descriptors(self):
        """Get the descriptors (np.array) of the node. These are the unique features of the image."""
        if(self._descriptors is None):
            print("Descriptors are None, use get_image_features() to calculate keypoints and descriptors")
        return self._descriptors

    @descriptors.setter
    def descriptors(self,value:np.ndarray):
        if value is None:
            self._descriptors = None
        elif type(np.array(value)) is np.ndarray:
            self._descriptors=np.array(value)
        else:
            raise ValueError('self.descriptors must be a numpy array')
        
    #---------------------intrinsicMatrix----------------------------
    @property
    @rdf_property()
    def intrinsicMatrix(self):
        """Get the intrinsic camera matrix (np.array) of the node.
        k=
        [fx 0 cx]
        [0 fy cy]
        [0 0  1]
        
        """
        if self._intrinsicMatrix is not None:
            return self._intrinsicMatrix
        else:
            pinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(self.imageWidth,
                                                                       self.imageHeight,
                                                                       self.focalLength35mm,
                                                                       self.focalLength35mm,
                                                                       self.imageWidth/2+self.principalPointU,
                                                                       self.imageHeight/2+self.principalPointV)
            self._intrinsicMatrix = pinholeCameraIntrinsic.intrinsic_matrix
            return self._intrinsicMatrix

    @intrinsicMatrix.setter
    def intrinsicMatrix(self,value:np.ndarray):
        if value is None:
            self._intrinsicMatrix = None
        elif type(np.array(value)) is np.ndarray and value.size==9:
            value=value.reshape(3,3)
            self._intrinsicMatrix=np.array(value)
        else:
            raise ValueError('self.descriptors must be a numpy array')

    #---------------------Depth----------------------------
    @property
    @rdf_property(datatype=XSD.float)
    def depth(self):
        """Get the maximum depth of the image, defaults to one"""
        if self._depth is None:
            self._depth = 1
        return self._depth
    
    @depth.setter
    def depth(self,value:float):
        if value is None:
            self._depth = None
        elif type(float(value)) is float:
            self._depth=float(value)
        else:
            raise ValueError('depth must be a float')

#---------------------PROPERTY OVERRIDES----------------------------
   
    @Node.resource.setter
    def resource(self,value):
        """Set the resource of the Node from various inputs.

        Args:
            - np.ndarray (OpenCV)
            - PIL Image
            - Open3D Image

        Raises:
            ValueError: Resource must be np.ndarray (OpenCV), PIL Image or Open3D Image.
        """
        if(value is None):
            self._resource = None
        elif isinstance(np.asarray(value),np.ndarray) : #OpenCV
            self._resource = np.asarray(value)
        elif isinstance(value,PIL.MpoImagePlugin.MpoImageFile): 
            self._resource=  np.array(value)#cv2.cvtColor(np.array(value), cv2.COLOR_RGB2BGR) #not sure if this is needed
        elif isinstance(value,PIL.Image.Image): 
            self._resource=  np.array(value)#cv2.cvtColor(np.array(value), cv2.COLOR_RGB2BGR)
        elif isinstance(value,o3d.geometry.Image):
            self._resource = np.array(value)
        else:
            raise ValueError('Resource must be np.ndarray (OpenCV) or PIL Image')
 

#---------------------METHODS----------------------------

    def get_metadata_from_exif_data(self,path) -> bool:
        """Returns the metadata from a resource. 

        Args:
            - GPSInfo (geospatialTransform (np.array(3,1))
            - coordinateSystem (str) 
            - DateTime ('%Y-%m-%dT%H:%M:%S')
            - XResolution (int)
            - YResolution (int)
            - ResolutionUnit (int)
            - ExifImageWidth (int)
            - ExifImageHeight (int)

        Returns:
            - bool: True if meta data is successfully parsed
        """
        
        if hasattr(self,'graph'):
            return True
        
        if getattr(self,'timestamp',None) is None :
            self.timestamp=ut.get_timestamp(path)
        
        if getattr(self,'name',None) is None:
            self.name=Path(path).stem

        if (getattr(self,'imageWidth',None) is not None and
            getattr(self,'imageHeight',None) is not None and
            getattr(self,'geospatialTransform',None) is not None):
            return True

        # pix = PIL.Image.open(self.path) 
        with Image.open(path) as pix:
            exifData=gsu.get_exif_data(pix)

        if exifData is not None:
            self.timestamp=exifData.get("DateTime")
            # self.resolutionUnit=exifData.get("ResolutionUnit")
            self.imageWidth=exifData.get("ExifImageWidth")
            self.imageHeight=exifData.get("ExifImageHeight")
            
            if 'GPSInfo' in exifData:
                gps_info = exifData["GPSInfo"]
                if gps_info is not None:
                    # self.GlobalPose=GlobalPose # (structure) SphericalTranslation(lat,long,alt), Quaternion(qw,qx,qy,qz)
                    latitude=gps_info.get("GPSLatitude")
                    latReference=gps_info.get("GPSLatitudeRef")
                    newLatitude=gsu.parse_exif_gps_data(latitude,latReference)
                    longitude=gps_info.get( "GPSLongitude")
                    longReference=gps_info.get("GPSLongitudeRef")
                    newLongitude=gsu.parse_exif_gps_data(longitude,longReference)
                    if newLatitude is not None and newLongitude is not None:
                        self.geospatialTransform=[  newLatitude, 
                                                    newLongitude,
                                                    gps_info.get("GPSAltitude")]
                        self.coordinateSystem='geospatial-wgs84'
            
            return True
        else:
            return False
    
    def get_metadata_from_xmp_path(self)->bool:
        """Read Metadata from .xmp file generated by https://www.capturingreality.com/.

        Smple Data:
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description xcr:Version="3" xcr:PosePrior="locked" xcr:Coordinates="absolute"
                xcr:DistortionModel="brown3" xcr:FocalLength35mm="24.3765359225552"
                xcr:Skew="0" xcr:AspectRatio="1" xcr:PrincipalPointU="-0.000464752782510192"
                xcr:PrincipalPointV="-0.000823593392050301" xcr:CalibrationPrior="exact"
                xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">
                <xcr:Rotation>0.412555151152903 0.910841439690343 0.0128887159933671 0.0412058430083455 -0.00452553581407911 -0.999140429583083 -0.910000178988533 0.412731621357468 -0.0393990224812024</xcr:Rotation>
                <xcr:Position>66.8850552499877 45.2551194778559 5.45377092514118</xcr:Position>
                <xcr:DistortionCoeficients>-0.124217384759894 0.107339706650415 -0.0104748224573926 0 0 0</xcr:DistortionCoeficients>
                </rdf:Description>
            </rdf:RDF>
            </x:xmpmeta>

        Returns:
            - bool: True if metadata is sucesfully parsed
        """
        # read the xmpPath
        mytree = ET.parse(self.xmpPath)
        root = mytree.find('.//rdf:Description', GEOMAPI_PREFIXES)
        
        self.timestamp=ut.get_timestamp(self.xmpPath)
        self.name=self.xmpPath.stem
        self.subject=self.name
        for child in root.iter():
            #Attributes
            for attribute in child.attrib:
                print(attribute)
                if ('latitude' in attribute and
                    'longitude'in attribute and
                    'altitude' in attribute):
                    lat=ut.xcr_to_lat(child.attrib[GEOMAPI_PREFIXES['xcr'].latitude])
                    long=ut.xcr_to_long(child.attrib[GEOMAPI_PREFIXES['xcr'].longitude])
                    alt=ut.xcr_to_alt(child.attrib[GEOMAPI_PREFIXES['xcr'].altitude])
                    self.geospatialTransform=np.array([lat, long, alt]) #?
                if 'Coordinates' in attribute:
                    self.coordinateSystem=child.attrib[attribute]
                if 'FocalLength35mm' in attribute:
                    self.focalLength35mm=ut.xml_to_float(child.attrib[attribute])
                if 'PrincipalPointU' in attribute:
                    self.principalPointU=ut.xml_to_float(child.attrib[attribute])
                if 'PrincipalPointV' in attribute:
                    self.principalPointV=ut.xml_to_float(child.attrib[attribute])

        #Nodes
        rotationnode=root.find('xcr:Rotation', GEOMAPI_PREFIXES)# '{http://www.capturingreality.com/ns/xcr/1.1#}Rotation')
        rotation=None
        if rotationnode is not None:
            rotation=np.reshape(ut.literal_to_matrix(rotationnode.text), (3,3)).T #! RC uses column-based rotation matrix

        positionnode=root.find('xcr:Position', GEOMAPI_PREFIXES)
        translation=None
        if positionnode is not None:
            translation=np.asarray(ut.literal_to_matrix(positionnode.text))
            
        self.cartesianTransform=gmu.get_cartesian_transform(translation=translation,rotation=rotation)
        
        coeficientnode=root.find('xcr:DistortionCoeficients', GEOMAPI_PREFIXES)
        if coeficientnode is not None:
            self.distortionCoeficients=ut.literal_to_list(coeficientnode.text)  
        return True   


    
    def load_resource(self)->np.ndarray: 
        """Returns the data in the node. If none is present, it will search for the data on using the attributes below.

        Args:
            - self.path

        Returns:
            np.ndarray or None
        """
        # Perform path checks
        if(not super().load_resource()):
            return None

        if self.path:
            self.resource = np.array(Image.open(self.path)) # PIL is 5% faster than OpenCV cv2.imread(self.path)
        return self._resource 

    def save_resource(self, directory:Path |str=None,extension :str = '.jpg') ->bool:
        """Export the resource of the Node.

        Args:
            - directory (str, optional) : directory folder to store the data.
            - extension (str, optional) : file extension. Defaults to '.jpg'.

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.

        Returns:
            bool: return True if export was succesfull
        """     
                # perform the path check and create the directory
        if not super().save_resource(directory, extension):
            return False
        
        #write files
        try:
            img = Image.fromarray(self._resource) # if cv2.imwrite(self.path, self.resource) is 5 times slower
            img.save(self._path)        
            return True
        except:
            return False

    def get_image_features(self, featureType:str = "orb", max:int = 1000) -> Tuple[np.array, np.array]:
        """Get the keypoints and the descriptors of this Nodes Image resource

        Args:
            - keypoints (cv2.keyPoints) : The featuretype to detect, use: orb, sift. Defaults to "orb".

        Returns:
            img_with_keypoints : Tuple[np.array, np.array] with the keypoints and the descriptors
        """

        if(self.keypoints is None or self.descriptors is None):
            self.keypoints, self.descriptors = iu.get_features(self.resource, featureType, max = max)
        return self._keypoints, self._descriptors

    def draw_keypoints_on_image(self,keypoint_size: int = 200,overwrite:bool=False)->np.array:
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
        self.get_image_features() if self.keypoints is None else None
         # Increase the size of keypoints
        for kp in self.keypoints:
            kp.size = keypoint_size
        # Draw keypoints on the image
        image=self.resource if overwrite else copy.deepcopy(self.resource)
        img_with_keypoints = cv2.drawKeypoints(image, self._keypoints, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return img_with_keypoints

    def _set_geometric_properties(self, _cartesianTransform=None, _convexHull=None, _orientedBoundingBox=None):
    
        self.cartesianTransform = _cartesianTransform
        self.convexHull = _convexHull
        self.orientedBoundingBox = _orientedBoundingBox

        # --- Handle Transform ---
        if self.cartesianTransform is None:
            if self.convexHull is not None:
                self.cartesianTransform = gmu.create_transform_from_pyramid_points(np.asarray(self.convexHull.vertices))
            elif self.orientedBoundingBox is not None:
                self.cartesianTransform = gmu.get_cartesian_transform(translation=self.orientedBoundingBox.get_center(), rotation=self.orientedBoundingBox.R)
            else:
                self.cartesianTransform = np.eye(4)  # Default to identity transform

        # --- Handle Convex Hull ---
        if self.convexHull is None:
            if self.orientedBoundingBox is not None:
                self.convexHull = gmu.get_convex_hull(self.orientedBoundingBox)
            else:
                self.convexHull = gmu.create_camera_frustum_mesh(self.cartesianTransform, self.focalLength35mm, depth = self.depth)
    
        # --- Handle Oriented Bounding Box ---
        if self.orientedBoundingBox is None:
            # Align with the frustum shape
            self.orientedBoundingBox = gmu.get_oriented_bounding_box(self.convexHull)
    
    def create_rays(self, imagePoints: np.ndarray = None, depths: np.ndarray = None) -> np.ndarray:
        """
        Generate a grid of rays from the camera location through given imagePoints.
        Args:
            imagePoints (np.ndarray): (n,2) Pixel coordinates (row, column).
            depths (np.ndarray, optional): NOT used inside this function anymore.
        Returns:
            np.ndarray: (n,6) array where [:,0:3] are the camera origins, [:,3:6] are unit direction vectors.
        """
        if imagePoints is None:
            points = np.array([
                [0, 0],
                [0, self.imageWidth],
                [self.imageHeight, 0],
                [self.imageHeight, self.imageWidth]])
        else:
            points = ut.map_to_2d_array(imagePoints)

        points = np.reshape(points, (-1, 2))  # ensure shape (n,2)
        n = points.shape[0]
        
        # Camera parameters
        f = self.focalLength35mm
        k = self.intrinsicMatrix
        m = self.cartesianTransform
        t = gmu.get_translation(m)

        # Convert pixel coordinates to normalized image plane
        u = +points[:,1] - self.imageWidth / 2
        v = +points[:,0] - self.imageHeight / 2
        camera_coords = np.vstack((u, v, np.full(n, f)))

        # Homogeneous coordinates
        camera_coords = np.vstack((camera_coords, np.ones((1, n))))  # (4,n)
        
        # Transform to world coordinates
        world_coords = m @ camera_coords  # (4,n)

        # Compute ray directions
        directions = (world_coords[:3, :].T - t)  # (n,3)
        directions = gmu.normalize_vectors(directions)

        # Build ray (origin + direction)
        rays = np.hstack((np.tile(t, (n, 1)), directions))  # (n,6)
        
        return rays

    
    def world_to_pixel_coordinates(self,worldCoordinates: np.ndarray) -> np.ndarray:
        """Converts 3D world coordinates to pixel coordinates in an image.

        This function takes 3D world coordinates and converts them to pixel coordinates in an image. It uses camera parameters such as the transformation matrix, focal length, image width, and image height.

        **NOTE**: the pixel coordinates have a (row, column) format. This fitts well with array indexing, but not with Matplotlib's imshow function

        Args:
            - worldCoordinates (np.ndarray (n,3) or (n,4) homogenous coordinates) : A set of 3D points in world coordinates to be converted.

        Returns:
           pixels : A 2D array containing the pixel coordinates (row, column) in the image.

        Note:
            - The function performs a series of transformations, including world to camera, camera to image, and image centering.
            - It returns the imageCoordinates as a 2D array.
        """
        # Ensure homogeneous coordinates
        worldCoordinates = gmu.convert_to_homogeneous_3d_coordinates(worldCoordinates)  # (n,4)

        # Transform world coordinates to camera coordinates
        cameraCoordinates = np.linalg.inv(self.cartesianTransform) @ worldCoordinates.T  # (4,n)

        # Normalize to 3D (homogeneous division)
        xy = cameraCoordinates[:3, :] / cameraCoordinates[2, :]

        # Apply focal length scaling
        xy[:2, :] *= self.focalLength35mm

        # Convert to image coordinates (row, column)
        uv = np.zeros((2, xy.shape[1]))
        uv[0, :] = xy[1, :] + self.imageHeight / 2  # row (y)
        uv[1, :] = xy[0, :] + self.imageWidth / 2   # column (x)

        uv = uv.T  # (n,2)

        # Optionally flatten if it's a single point
        if uv.shape[0] == 1:
            uv = uv.flatten()

        return uv
    
    
    def pixel_to_world_coordinates(self, pixels: np.array, depths: np.array = None) -> np.ndarray:
        """Converts pixel coordinates in an image to 3D world coordinates.

        This function takes pixel coordinates and optional depths and converts them to 3D world coordinates. It uses camera parameters such as the transformation matrix, focal length, image width, and image height.

        Args:
            - pixels (np.array[n,2]) : Pixel coordinates in the image (row, column).
            - depths (np.array[n,1], optional) : Depths for the corresponding pixel coordinates. Defaults to 50m for each point.

        Returns:
            - worldCoordinates (np.ndarray): A 2D array (n, 3) containing the 3D world coordinates (X, Y, Z).
        """
        rays = self.create_rays(pixels)  # (n,6)
    
        if rays.ndim == 1:
            camera_center = rays[:3]
            direction = gmu.normalize_vectors(rays[3:])
        else:
            camera_center = rays[:, :3]
            direction = gmu.normalize_vectors(rays[:, 3:])

        if depths is None:
            depths = np.full((direction.shape[0], 1), self.depth)
        else:
            depths = np.asarray(depths)
            if depths.ndim == 1:
                depths = depths[:, np.newaxis]

        world_coordinates = camera_center + direction * depths

        return world_coordinates
    

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
        
    def show(self):
        super().show()
        # Converts from one colour space to the other. this is needed as RGB
        # is not the default colour space for OpenCV
        image = cv2.cvtColor(self.resource, cv2.COLOR_BGR2RGB)

        # Show the image
        plt.imshow(image)

        # remove the axis / ticks for a clean looking image
        plt.xticks([])
        plt.yticks([])

        # if a title is provided, show it
        plt.title(self.name)

        plt.show()