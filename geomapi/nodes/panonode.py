"""
**PanoNode** is a Python Class to govern the data and metadata of panoramic image data. 

This node builds upon the [OpenCV](https://opencv.org/), [Open3D](https://www.open3d.org/) and [PIL](https://pillow.readthedocs.io/en/stable/) API for the image definitions.
Be sure to check the properties defined in those abstract classes to initialise the Node.

.. image:: ../../../docs/pics/graph_img_1.png

**IMPORTANT**: This Node class is designed to manage geolocated imagery. It works best when the heading is known. The class can be used to generate virtual images, raycasting, and other geometric operations.

"""
#IMPORT PACKAGES
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import PIL
from rdflib import XSD, Graph, URIRef
import numpy as np
import os
import open3d as o3d
import json
import pandas as pd



import matplotlib.pyplot as plt
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



class PanoNode(Node):
    # class attributes
    
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
                jsonPath: Path = None,
                imageWidth:int = None,
                imageHeight:int = None, 
                depth: float= None,
                **kwargs): 
        """Creates an PanoNode. Overloaded function.
                
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to
        
        Args:
            - subject (RDFlib URIRef) : subject to be used as the main identifier in the RDF Graph
            
            - graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - graphPath (Path) :  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - path (Path) : Path to an image .jpg, jpeg, .png file (data is not automatically loaded)
            
            - resource (ndarray, PIL, Open3D) : Image data from [Open3D](https://www.open3d.org/), [OpenCV](https://opencv.org/) or [PIL](https://pillow.readthedocs.io/en/stable/). 
                                    
            - getResource (bool, optional= False) : If True, the node will search for its physical resource on drive. In this case, that includes the image and the depthmap. Defaults to False
            
            - imageWidth (int, optional) : Width of the image in pixels (u). 
            
            - imageHeight (int, optional) : Height of the image in pixels (v). 
                                    
            - depthMap (np.array, optional) : The depthmap of the image. Defaults to None
            
            - depthPath (Path, optional) : Path to the depthmap of the image. Defaults to None
            
        Returns:
            ImageNode : An ImageNode with metadata 
        """  

        #instance variables
        self.jsonPath=jsonPath
        self.imageWidth=imageWidth
        self.imageHeight=imageHeight
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
        
        #initialise functionality
        self.get_metadata_from_json_path() if self.jsonPath is not None else None
        self.get_metadata_from_exif_data(path) if self.path is not None else None
        

#---------------------PROPERTIES----------------------------

    #---------------------jsonPath----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def jsonPath(self): 
        """Get the jsonPath (Path) of the node."""
        return self._jsonPath

    @jsonPath.setter
    def jsonPath(self,value:Path):
        if value is None:
            self._jsonPath = None
        elif Path(value).suffix.upper() ==".JSON":
            self._jsonPath=Path(value)
        else:
            raise ValueError('self.jsonPath has invalid type, path or extension.')    
        
    #---------------------imageWidth----------------------------
    #---------------------imageWidth----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES['exif'].imageWidth, datatype=XSD.int)
    def imageWidth(self):
        """Get the imageWidth (int) or number of columns of the resource of the node."""
        if self._imageWidth is None:
            self._imageWidth = 2000
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
            self._imageHeight = 1000
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
# #---------------------METHODS----------------------------

    def load_resource(self)->np.ndarray: 
        """Loads the resource from the path

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
            self.name=path.stem

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
    
    def get_metadata_from_json_path(self)->bool:
        """Read Metadata from Navvis json.

        Args:
            - geospatialTransform (np.array(3x1))
            - coordinateSystem (str)
            - focalLength35mm (float)
            - principalPointU (float)
            - principalPointV (float)
            - cartesianTransform (np.array(4x4))

        Returns:
            - bool: True if metadata is sucesfully parsed
        """
        if self.jsonPath is None or not os.path.exists(self.jsonPath):
            return False
        
        if hasattr(self,'graph'):
            return True
        
        # Load JSON data
        with open(self.jsonPath, 'r') as file:
            data= json.load(file)

        # Extract the necessary data
        footprint_position = data.get('footprint', {}).get('position', [])
        footprint_quaternion = data.get('footprint', {}).get('quaternion', [])
        timestamp = data.get('timestamp', None)
        self.subject=self.jsonPath.stem
        self.name=self.jsonPath.stem
                            
        #convert to the right format
        self.timestamp=ut.get_timestamp(timestamp)
        self.cartesianTransform=gmu.get_cartesian_transform(translation=footprint_position,rotation=footprint_quaternion)
        #reset the bb and convex hull
        self._set_geometric_properties(self.cartesianTransform)

        return True   
    
   
    def _set_geometric_properties(self, _cartesianTransform=None, _convexHull=None, _orientedBoundingBox=None):
    
        self.cartesianTransform = _cartesianTransform
        self.convexHull = _convexHull
        self.orientedBoundingBox = _orientedBoundingBox

        # --- Handle Transform ---
        if self.cartesianTransform is None:
            if self.convexHull is not None:
                self.cartesianTransform = gmu.get_cartesian_transform(translation = self.convexHull.get_center())
            elif self.orientedBoundingBox is not None:
                self.cartesianTransform = gmu.get_cartesian_transform(translation=self.orientedBoundingBox.get_center(), rotation=self.orientedBoundingBox.R)
            else:
                self.cartesianTransform = np.eye(4)  # Default to identity transform

        # --- Handle Convex Hull ---
        if self.convexHull is None:
            if self.orientedBoundingBox is not None:
                self.convexHull = o3d.geometry.TriangleMesh.create_sphere(
                    center = self.orientedBoundingBox.get_center(),
                    radius = max(self.orientedBoundingBox.extend)/2)
            else:
                self.convexHull = o3d.geometry.TriangleMesh.create_sphere(radius=self.depth)
    
        # --- Handle Oriented Bounding Box ---
        if self.orientedBoundingBox is None:
            # Align with the frustum shape
            self.orientedBoundingBox = gmu.get_oriented_bounding_box(self.convexHull)

    def create_rays(self,imagePoints:np.array=None,depths:np.array=None)->o3d.core.Tensor:
        """Generate a grid a rays from the camera location to a given set of imagePoints.
                
        **NOTE**: This function targets a subselection of imagePoints, use o3d.t.geometry.RaycastingScene.create_rays_pinhole if you want a dense raytracing for the full image.
        
        .. image:: ../../../docs/pics/Raycasting_1.PNG
        
        Args:
            - imagePoints (np.array[n,2]) : imagePoints are conform (row, column) image coordinates system. so top left is (0,0). The camera intrinsic matrix is used to map it to the proper image coordinates. Defaults to np.array([[0, 0],[0, self._imageWidth],[self._imageHeight, 0],[self._imageHeight, self._imageWidth]]).
            - depths (np.array[n,1], optional) : depths of the rays. Defaults to 50m for each point.

        Returns:
            rays: o3d.core.Tensor (n,6) [:,0:3] with the camera center and [:,3:6] are the directions of the rays towards the imagePoints.
        """
        width=self.imageWidth
        height=self.imageHeight
        if imagePoints is None: # watch out near poles as cos and sin are highly sensitive in those areas.
            points=np.array([[height/2,width/2 ],   # front     [0,0,1]
                            [0,width/2],            # top       [0,1,0]
                            [ height/2,width/4],    # left      [-1,0,0]
                            [height/2,width*3/4 ],  # right     [1,0,0]
                            [height/2,0],           # back      [0,0,-1]
                            [height,width/2]])      # bottom    [0,-1,0]
        else:
            points=ut.map_to_2d_array(imagePoints)
            
        if depths is None:
            depths=np.full(points.shape[0],self.depth)
        else:
             depths = np.asarray(depths).flatten()  # Ensure depths is a 1D array
        
        #validate inputs
        points=np.reshape(points,(-1,2)) #if len(points.shape) >2 else points
        
        #transform pixels to image coordinates (rows are first)
        u=+points[:,1]
        v=+points[:,0]  
        
        # Field of view in radians
        fov_horizontal_rad = 2 * np.pi  # 360 degrees
        fov_vertical_rad = np.pi        # 180 degrees
        
        # Calculate azimuth (theta) and elevation (phi)
        theta = (u / (width - 1)) * fov_horizontal_rad - np.pi  # Maps [0, width] to [-pi, pi]
        phi = (v / (height - 1)) * fov_vertical_rad - (np.pi / 2)  # Maps [0, height] to [-pi/2, pi/2]


        # Spherical to Cartesian conversion in camera coordinate system (z-forward, y-up)
        x = np.cos(phi) * np.sin(theta)  # x-axis (left-right in pinhole model)
        y = -np.sin(phi)                   # y-axis (up-down in pinhole model)
        z = np.cos(phi) * np.cos(theta)   # z-axis (forward-backward in pinhole model)

        # Stack x, y, z to form 3D coordinates (on the unit sphere)
        unit_sphere_coords = np.stack((x, y, z), axis=-1)

        # Scale the unit sphere coordinates by the depth values
        world_coords = unit_sphere_coords * depths[:, np.newaxis] if depths is not None else unit_sphere_coords
        
        #note that Z goes through the center of the image, so we need to rotate the point cloud 90° clockwise around the x-axis
        # #rotation matrix for 90° around the x-axis
        # r = R.from_euler('x', 90, degrees=True).as_matrix()
        # #apply rotation
        # world_coords = (r @ world_coords.T).T        

        # Get the transformation matrix (4x4), and extract rotation and translation
        transformation = self._cartesianTransform
        r = transformation[:3, :3]  # Rotation matrix (3x3)
        t = transformation[:3, 3]   # Translation vector (3x1)

        # Transform the world coordinates by applying the transformation matrix
        directions = (r @ world_coords.T).T 

        # Camera origin is the translation vector (assume camera at t)
        camera_origin = np.tile(t, (directions.shape[0], 1))

        # Combine camera origin and directions into the final ray tensor (n, 6)
        rays = np.hstack((camera_origin, directions))
              
        return rays 
    
    def world_to_pixel_coordinates(self,worldCoordinates: np.ndarray) -> np.ndarray:
        """
        Project 3D world coordinates to panoramic (equirectangular) image coordinates.

        Args:
            - worldCoords (np.ndarray): The world coordinates (n, 3) as an Nx3 numpy array (X, Y, Z).

        Returns:
            - panoramicCoords (np.ndarray): The corresponding panoramic coordinates (n, 2) in (u, v) format (row,column).
        """
        # Extract X, Y, Z components of the world coordinates
        x = worldCoordinates[:, 0]
        y = worldCoordinates[:, 1]
        z = worldCoordinates[:, 2]

        # Convert Cartesian (x, y, z) to spherical coordinates
        # Compute azimuth (longitude) and elevation (latitude)
        azimuth = np.arctan2(y, x)  # Azimuth angle [-pi, pi]
        elevation = np.arcsin(z / np.linalg.norm(worldCoordinates, axis=1))  # Elevation angle [-pi/2, pi/2]

        # Map azimuth and elevation to 2D panoramic image coordinates -> not sure if this is correct
        u = (azimuth + np.pi) / (2 * np.pi)  # Normalize azimuth to [0, 1]
        v = (np.pi / 2 - elevation) / np.pi  # Normalize elevation to [0, 1]

        # Scale to image size
        u = u * self._imageWidth   # Map to pixel coordinates in the range [0, image_width]
        v = v * self._imageHeight  # Map to pixel coordinates in the range [0, image_height]

        # Combine u and v into final 2D coordinates
        panoramic_coords = np.stack((u, v), axis=-1)

        return panoramic_coords
    
    
    def pixel_to_world_coordinates(self, pixels: np.array, depths: np.array = None) -> np.ndarray:
        """Converts pixel coordinates in an panoramic image to 3D world coordinates.

        This function takes pixel coordinates and optional depths and converts them to 3D world coordinates. The function assumes that the camera is at the origin and the image is projected onto a unit sphere.

        Args:
            - pixels (np.ndarray) : Pixel coordinates (n, 2) in the image (row, column).
            - depths (np.ndarray, optional) : Depths (n, 1) for the corresponding pixel coordinates. Defaults to 50m for each point.

        Returns:
            - worldCoordinates (np.ndarray): A 2D array (n, 3) containing the 3D world coordinates (X, Y, Z).
        """
        # Generate rays from the pixel coordinates and depths
        rays = self.create_rays(pixels, depths)
        
        # Extract camera center and direction
        if rays.ndim == 1:
            camera_center = rays[:3]
            direction = rays[3:]
        else:
            camera_center = rays[:, :3]
            direction = rays[:, 3:]
        
        direction=gmu.normalize_vectors(direction)

        # Calculate the world coordinates
        if depths is None:
            depths = np.full((direction.shape[0], 1), self.depth)
        world_coordinates = camera_center + direction * depths

        return world_coordinates
    

    
    def project_lineset_on_image(self,linesets:List[o3d.geometry.LineSet],thickness:int=2,overwrite=True) ->np.ndarray:
        """Project Opend3D linesets onto the resource of the node.
        
        **NOTE**: WORK IN PROGRESS

        **NOTE**: this affects the original image if overwrite is True.
        
        .. image:: ../../../docs/pics/image_projection_1.png

        Args:
            - linesets (List[o3d.LineSet]): List of linesets. Note that the color of the lines is stored in the lineset.
            - thickness (int) : Thickness of the projected lines
            - overwrite (bool) : If True, the original image is overwritten. If False, a new image is created.

        Returns:
            resource : The resource of the ImageNode with the projected lines.
        """
        return
    

    
    def show(self, convertColorspace = False, subsample = None):
        super().show()
        # Converts from one colour space to the other. this is needed as RGB
        # is not the default colour space for OpenCV
        if(convertColorspace):
            image = cv2.cvtColor(self.resource, cv2.COLOR_BGR2RGB)
        else:
            image = self.resource
        if not subsample == None:
            image = cv2.resize(image,[int(self.imageWidth/subsample),int(self.imageHeight/subsample)])

        # Show the image
        plt.imshow(image)

        # remove the axis / ticks for a clean looking image
        plt.xticks([])
        plt.yticks([])

        # if a title is provided, show it
        plt.title(self.name)

        plt.show()
        
