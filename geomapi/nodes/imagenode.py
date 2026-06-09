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
            
            - focalLength35mm (float, optional) : focal length with a standardized Field-of-View in mm.
            
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
            self._focalLength35mm = 35
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
        if self._intrinsicMatrix is None:
            # create a default matrix
            fx = self.focalLength35mm * self.imageWidth / 36.0
            fy = self.focalLength35mm * self.imageHeight / 24.0
            cx = self.imageWidth / 2.0 + self.principalPointU
            cy = self.imageHeight / 2.0 + self.principalPointV
            self._intrinsicMatrix = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])
        return self._intrinsicMatrix

    @intrinsicMatrix.setter
    def intrinsicMatrix(self,value:np.ndarray):
        if value is None:
            self._intrinsicMatrix = None
        elif type(np.array(value)) is np.ndarray and value.size==9:
            value=value.reshape(3,3)
            self._intrinsicMatrix=np.array(value)
        else:
            raise ValueError('must be a 3x3 np array')

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

        with Image.open(path) as pix:
            exifData=gsu.get_exif_data(pix)
            print(exifData)

        if exifData is not None:
            # get the relevant exif properties
            timestamp=exifData.get("DateTime")
            imageWidth=exifData.get("ExifImageWidth")
            imageHeight=exifData.get("ExifImageHeight")
            focalLength = exifData.get("FocalLength")
            focalLength35mm = exifData.get("FocalLengthIn35mmFilm")

            # Get the cameraParameters
            cropFactor = focalLength35mm/focalLength
            sensor_width_mm = 36.0 / cropFactor
            sensor_height_mm = 24.0 / cropFactor
            fx = (focalLength / sensor_width_mm) * imageWidth
            fy = (focalLength / sensor_height_mm) * imageHeight
            cx = imageWidth / 2.0
            cy = imageHeight / 2.0
            K = np.array([
                [fx,  0,  cx],
                [ 0, fy,  cy],
                [ 0,  0,   1]
            ])

            if(self._timestamp is None): self.timestamp = timestamp
            if(self._imageWidth is None): self.imageWidth = imageWidth
            if(self._imageHeight is None): self.imageHeight = imageHeight
            if(self._focalLength35mm is None): self.focalLength35mm = focalLength35mm
            if(self._intrinsicMatrix is None): self.intrinsicMatrix = K
            
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
    
    def create_rays(self, imagePoints: np.ndarray, depths: np.ndarray = None) -> np.ndarray:
        """
        Generate rays from the camera center through given image points.

        Args:
            imagePoints (np.ndarray): (n, 2) Pixel coordinates (row, column).
            depths (np.ndarray, optional): If None, rays are cast at unit distance along camera direction.

        Returns:
            np.ndarray: (n, 6) where [:, 0:3] is origin, [:, 3:6] is unit direction.
        """

        points = ut.map_to_2d_array(imagePoints)

        points = np.reshape(points, (-1, 2))  # Ensure shape (n, 2)
        n = points.shape[0]

        # Default to unit distances if no depths provided
        if depths is None:
            depths = np.ones(n)

        # Compute 3D points in world space along each ray
        world_pts = self.pixel_to_world_coordinates(
            pixels=points,
            depths=depths,
            use_z_depth=False  # using Euclidean distance by default
        )

        # Camera origin
        origin = self.cartesianTransform[:3, 3]
        origins = np.tile(origin, (n, 1))

        # Ray directions
        directions = gmu.normalize_vectors(world_pts - origins)

        # Final ray array: origin + direction
        rays = np.hstack((origins, directions))  # (n, 6)
        return rays

    
    def world_to_pixel_coordinates(self,worldCoordinates: np.ndarray, use_z_depth=False) -> np.ndarray:
        """
        Projects 3D world points to 2D image coordinates.
        Optionally returns z-depths instead of Euclidean distances.

        Args:
            worldCoordinates: (N, 3) array of world coordinates [X, Y, Z]
            use_z_depth: if True, return z-axis depth instead of Euclidean distance
        Returns:
            image_points: (N, 2) pixel coordinates
            depths: (N,) z-depths or Euclidean distances
        """
        worldCoordinates = np.atleast_2d(worldCoordinates)
        N = worldCoordinates.shape[0]

        # Invert camera pose to get world-to-camera transform
        T_inv = np.linalg.inv(self.cartesianTransform)
        worldCoordinates_h = np.hstack([worldCoordinates, np.ones((N, 1))])
        cam_points = (T_inv @ worldCoordinates_h.T).T[:, :3]

        # Project to image
        proj = (self.intrinsicMatrix @ cam_points.T).T
        image_points = proj[:, :2] / proj[:, 2:3]

        if use_z_depth:
            depths = cam_points[:, 2]  # z-depth
        else:
            cam_origin = np.zeros((1, 3))
            distances = np.linalg.norm(cam_points - cam_origin, axis=1)
            depths = distances

        return image_points, depths
    
    
    def pixel_to_world_coordinates(self, pixels: np.array, depths: np.array = None, use_z_depth=False) -> np.ndarray:
        """
        Back-projects image points to 3D world coordinates using either z-depth or Euclidean distance.

        Args:
            pixels: (N, 2) array of pixel coordinates
            depths: (N,) array of z-depths or Euclidean distances
            use_z_depth: if True, interpret depths as z-axis depth
        Returns:
            world_points: (N, 3) array of world coordinates
        """
        pixels = np.atleast_2d(pixels)
        depths = np.atleast_1d(depths)
        N = pixels.shape[0]

        # Back-project to camera rays
        K_inv = np.linalg.inv(self.intrinsicMatrix)
        pixels_h = np.hstack([pixels, np.ones((N, 1))])
        cam_dirs = (K_inv @ pixels_h.T).T

        if not use_z_depth:
            # Normalize ray directions to unit vectors for Euclidean depth
            cam_dirs = cam_dirs / np.linalg.norm(cam_dirs, axis=1, keepdims=True)

        cam_points = cam_dirs * depths[:, np.newaxis]

        # Convert to homogeneous and transform to world space
        cam_points_h = np.hstack([cam_points, np.ones((N, 1))])
        world_points = (self.cartesianTransform @ cam_points_h.T).T[:, :3]

        return world_points
    
    def show(self, convertColorspace = False, show3d = False):
        super().show()
        if(show3d):
            frustum_lines, image_plane = gmu.create_camera_frustum_mesh_with_image(
                self.cartesianTransform,
                self.imageWidth, 
                self.imageHeight, 
                self.focalLength35mm, 
                self.depth,
                image_cv2=self.resource)
            o3d.visualization.draw_geometries([frustum_lines, image_plane])
        else:
            # Converts from one colour space to the other. this is needed as RGB
            # is not the default colour space for OpenCV
            if(convertColorspace):
                image = cv2.cvtColor(self.resource, cv2.COLOR_BGR2RGB)
            else:
                image = self.resource

            # Show the image
            plt.imshow(image)

            # remove the axis / ticks for a clean looking image
            plt.xticks([])
            plt.yticks([])

            # if a title is provided, show it
            plt.title(self.name)

            plt.show()