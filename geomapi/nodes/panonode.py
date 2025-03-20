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
from rdflib import Graph, URIRef
import numpy as np
import os
import open3d as o3d
import json
import pandas as pd


from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import copy
from typing import List, Optional,Tuple,Union
from PIL import Image


#IMPORT MODULES
from geomapi.nodes import Node
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import geomapi.utils.imageutils as iu
import geomapi.utils.geospatialutils as gsu



class PanoNode(Node):
    # class attributes
    
    def __init__(self,  graph : Graph = None, 
                        graphPath:Path=None,
                        subject : URIRef = None,
                        name:str=None,
                        path : str=None, 
                        resource = None,
                        depthMap: np.ndarray | float= None,
                        depthPath: Path = None,         
                        jsonPath: Path = None,               
                        imageWidth:int = None,
                        imageHeight:int = None, 
                        cartesianTransform: np.ndarray = None,
                        getResource : bool = False,
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
        #private attributes 
        self._depthPath=None
        self._depthMap=None
        self._imageWidth = None 
        self._imageHeight = None          
        self._jsonPath=None
        self._resource=None 
        self._path=None
        self._subject=None
        self._cartesianTransform=None
        self._name=None
        self._graphPath=None
        self._timestamp=None

        #instance variables
        self.subject=subject
        self.path=path        
        self.name=name
        self.resource=resource                
        self.cartesianTransform=cartesianTransform
        self.graphPath=graphPath
        self.imageWidth=imageWidth
        self.imageHeight=imageHeight
        self.depthPath=depthPath
        self.depthMap=depthMap
        self.jsonPath=jsonPath
        
        self.get_resource() if getResource else None
        self.get_depth_map() if getResource else None
        

        #initialise functionality
        self.get_metadata_from_json_path() if self._jsonPath is not None else None
        self.get_metadata_from_exif_data(path) if self._path is not None else None
        
        # self.get_image_width()
        # self.get_image_height()
        
        super().__init__(   graph= graph,
                            graphPath= self._graphPath,
                            subject= self._subject,
                            path=path,
                            name=self._name,
                            resource = resource,
                            getResource=getResource,
                            cartesianTransform=self._cartesianTransform,
                            timestamp=self._timestamp,
                            **kwargs) 
        

#---------------------PROPERTIES----------------------------

    #---------------------depthPath----------------------------
    @property
    def depthPath(self): 
        """Get the depthPath (Path) of the node."""
        return self._depthPath#ut.parse_path(self._xmlPath)

    @depthPath.setter
    def depthPath(self,value:Path):
        if value is None:
            pass
        elif Path(value).suffix.upper() in ut.IMG_EXTENSIONS:
            self._depthPath=Path(value)
        else:
            raise ValueError('self.depthPath has invalid type, path or extension.')    
        
    #---------------------jsonPath----------------------------
    @property
    def jsonPath(self): 
        """Get the jsonPath (Path) of the node."""
        return self._jsonPath

    @jsonPath.setter
    def jsonPath(self,value:Path):
        if value is None:
            pass
        elif Path(value).suffix.upper() ==".JSON":
            self._jsonPath=Path(value)
        else:
            raise ValueError('self.jsonPath has invalid type, path or extension.')    
        
    #---------------------imageWidth----------------------------
    @property
    def imageWidth(self):
        """Get the imageWidth (int) or number of columns of the resource of the node."""
        if self._imageWidth is None and self._resource is not None:
            self._imageWidth=self._resource.shape[1]
        return self._imageWidth
    
    @imageWidth.setter
    def imageWidth(self,value:int):
        if value is None:
            pass
        elif type(int(value)) is int:
            self._imageWidth=int(value)
        else:
            raise ValueError('self.imageWidth must be an integer')
  
    #---------------------imageHeight----------------------------
    @property
    def imageHeight(self):
        """Get the imageHeight (int) or number of rows of the resource of the node."""
        if self._imageHeight is None and self._resource is not None:
            self._imageHeight=self._resource.shape[0]
        return self._imageHeight
    
    @imageHeight.setter
    def imageHeight(self,value:int):
        if value is None:
            pass
        elif type(int(value)) is int:
            self._imageHeight=int(value)
        else:
            raise ValueError('self.imageHeight must be an integer')
    
    #---------------------depthMap----------------------------
    @property
    def depthMap(self):
        """Get the depthMap (np.ndarray) of the Image. This is used for the convex hull and oriented bounding box."""
        return self._depthMap
    
    @depthMap.setter
    def depthMap(self,value:float|np.ndarray):
        if value is None:
            pass
        elif type(value) is np.ndarray:
            assert value.ndim==2, 'two-dimensional np.ndarray expected'
            if self.imageHeight is not None and self.imageWidth is not None:
                assert value.shape[0]==self.imageHeight and value.shape[1]==self.imageWidth, 'np.ndarray must have the same dimensions as the image'
            self._depthMap=value
        elif type(float(value)) is float and float(value)>0:
            #create a depthmap with the same dimensions as the image and fill it with the value
            assert self.imageHeight is not None and self.imageWidth is not None, 'self.imageHeight and self.imageWidth must be set before setting self.depthMap'
            self._depthMap=np.full((self.imageHeight,self.imageWidth),fill_value=float(value))
        else:
            raise ValueError('self.depthMap must be a float > 0 or a np.ndarray')   
        
# #---------------------METHODS----------------------------

    def set_resource(self,value):
        """Set the resource of the Node from various inputs.

        Args:
            - np.ndarray (OpenCV)
            - PIL Image
            - Open3D Image

        Raises:
            ValueError: Resource must be np.ndarray (OpenCV), PIL Image or Open3D Image.
        """

        if type(np.asarray(value)) is np.ndarray : 
            self._resource = np.asarray(value)
        else:
            raise ValueError('Resource must be np.ndarray (OpenCV) or PIL Image')
                
    def get_resource(self)->np.ndarray: 
        """Returns the data in the node. If none is present, it will search for the data on using the attributes below.

        Args:
            - self.path

        Returns:
            np.ndarray or None
        """
        if self._resource is not None :
            return self._resource
        elif self.get_path() and self._path.exists():
            self._resource   = np.array(Image.open(self._path)) # PIL is 5% faster than OpenCV cv2.imread(self.path)
        return self._resource  
        
    def set_path(self, value:Path):
        """sets the path for the Node type. 
        """
        if value is None:
            pass
        elif Path(value).suffix.upper() in ut.IMG_EXTENSIONS:
            self._path = Path(value) 
        else:
            raise ValueError('Invalid extension')

    def get_path(self) -> Path:
        """Returns the full path of the resource from this Node. If no path is present, it is gathered from the following inputs.

        Args:
            - self._path
            - self._graphPath
            - self._name
            - self._subject
            - self._xmpPath

        Returns:
            path 
        """      
        if self._path is not None: 
            return self._path
        
        elif self._graphPath and (self._name or self._subject):
            folder=self._graphPath.parent 
            nodeExtensions=ut.get_node_resource_extensions(str(type(self)))
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if path.suffix.upper() in nodeExtensions:
                    if self.get_name() == path.stem :
                        self.path = path    
                        return self._path
            if self._name:
                self.path=os.path.join(folder,self._name+nodeExtensions[0])
            else:
                self.path=os.path.join(folder,self._subject+nodeExtensions[0])
            return self._path
        
        else:
            return None
       
    def get_depth_map(self):
        """
        Function to decode the depthmaps generated by the navvis processing

        Args:
            - None
            
        Returns:
            - np.array: Depthmap
        """
        if isinstance(self._depthMap,np.ndarray):
            return self._depthMap
        elif self._depthPath is None:
            return None
        
        # Load depthmap image
        depthmap = np.asarray(Image.open(self._depthPath)).astype(float)
        
        # Vectorized calculation for the depth values
        depth_value = (depthmap[:, :, 0] / 256) * 256 + \
                    (depthmap[:, :, 1] / 256) * 256 ** 2 + \
                    (depthmap[:, :, 2] / 256) * 256 ** 3 + \
                    (depthmap[:, :, 3] / 256) * 256 ** 4

        # Assign the computed depth values to the class attribute _depthMap
        self._depthMap = depth_value/1000 # Convert to meters
        return self._depthMap 
    
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
        #check path
        if self.resource is None:
            return False
        
        #validate extension
        if extension.upper() not in ut.IMG_EXTENSIONS:
            raise ValueError('Invalid extension')
        
        filename=ut.validate_string(self.name)

        # check if already exists
        if directory and os.path.exists(os.path.join(directory,self.get_name() + extension)):
            self.path=os.path.join(directory,self.get_name() + extension)
            return True
        elif not directory and self.get_path() and os.path.exists(self.path) and extension.upper() in ut.IMG_EXTENSIONS:
            return True    
          
        #get path        
        if directory:
            self.path=os.path.join(directory,filename + extension)
        else:
            if self.get_path():
                directory =self._path.parent
                
            elif self._graphPath: 
                dir=self.graphPath.parent
                directory=os.path.join(dir,'PANO')   
                self.path=os.path.join(dir,filename + extension)
            else:
                directory=os.path.join(os.getcwd(),'PANO')
                self.path=os.path.join(dir,filename + extension)
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        #write files
        try:
            img = Image.fromarray(self.resource) # if cv2.imwrite(self.path, self.resource) is 5 times slower
            img.save(self.path)        
            return True
        except:
            return False
    


    def get_preview(self, subsample = None):
        if not np.any(self.preview):
            if not subsample == None:
                    img = copy.deepcopy(self.resource)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    self.preview = cv2.resize(img,[int(self.imageWidth/subsample),int(self.imageHeight/subsample)])
            else:
                self.preview = self.resource
        return self.preview
    
    
    
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
        if self._jsonPath is None or not os.path.exists(self._jsonPath):
            return False
        
        if hasattr(self,'graph'):
            return True
        
        # Load JSON data
        with open(self._jsonPath, 'r') as file:
            data= json.load(file)

        # Extract the necessary data
        footprint_position = data.get('footprint', {}).get('position', [])
        footprint_quaternion = data.get('footprint', {}).get('quaternion', [])
        timestamp = data.get('timestamp', None)
        self.subject=self._jsonPath.stem
        self.name=self._jsonPath.stem
                            
        #convert to the right format
        self.timestamp=ut.get_timestamp(timestamp)
        self.cartesianTransform=gmu.get_cartesian_transform(translation=footprint_position,rotation=footprint_quaternion)
     
        return True   
    


    def save_depth_map(self, directory:Path |str=None,extension :str = '.jpg') ->bool:
        """Export the depthMap of the Node.

        Args:
            - directory (str, optional) : directory folder to store the data.
            - extension (str, optional) : file extension. Defaults to '.jpg'.

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.

        Returns:
            bool: return True if export was succesfull
        """     
                #check path
        if not isinstance(self._depthMap,np.ndarray):
            return False
        
        #validate extension
        if extension.upper() not in ut.IMG_EXTENSIONS:
            raise ValueError('Invalid extension')
        
        filename=ut.validate_string(self._name)+'_depth'
    
        #get path        
        if directory:
            self.depthPath=os.path.join(directory,filename + extension)
        else:
            if self.get_path():
                directory =self._path.parent
                
            elif self._graphPath: 
                dir=self.graphPath.parent
                directory=os.path.join(dir,'PANO')   
                self.depthPath=os.path.join(dir,filename + extension)
            else:
                directory=os.path.join(os.getcwd(),'PANO')
                self.depthPath=os.path.join(dir,filename + extension)
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        #write files
        try:
            img = Image.fromarray(self._depthMap) # if cv2.imwrite(self.path, self.resource) is 5 times slower
            img.save(self._depthPath)        
            return True
        except:
            return False
   
    def get_cartesian_transform(self) -> np.ndarray:
        """Get the cartesianTransform of the node from various inputs. if no cartesianTransform is present, a default np.eye(4) is used.

        Returns:
            - cartesianTransform(np.ndarray(4x4))
        """
        if self._cartesianTransform is not None:
            return self._cartesianTransform
        
        if self._cartesianTransform is None and isinstance(self._depthMap,np.ndarray):
            #get points from the depthmap
            pcd=self.get_pcd_from_depth_map()
            self._convexHull=pcd.compute_convex_hull()[0]            
        if self._cartesianTransform is None and self._convexHull is not None:
            self._cartesianTransform = gmu.get_cartesian_transform(translation=self._convexHull.get_center()) 
        if self._cartesianTransform is None and self._orientedBoundingBox is not None:
            self._cartesianTransform = gmu.get_cartesian_transform(translation=self._orientedBoundingBox.get_center())    
        if self._cartesianTransform is None:
            #you could initialize a pano in an upright position instead of forward to match a terrestrial vantage point
            #rotation_matrix_90_x=   np.array( [[ 1, 0 , 0.        ],
            #                            [ 0,  0,  -1        ],
            #                            [ 0.   ,       1    ,      0        ]])  
            self._cartesianTransform = gmu.get_cartesian_transform()#rotation=rotation_matrix_90_x)    
        return self._cartesianTransform    
    
    
    def get_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        """Gets the Open3D OrientedBoundingBox of the node. If no orientedBoundingBox is present, it is gathered from the following inputs.
        
        Args:
            - self._resource
            - self._convexHull
            - self._cartesianTransform

        Returns:
            - o3d.geometry.OrientedBoundingBox
        """
        if self._orientedBoundingBox is not None:
            return self._orientedBoundingBox
    
        if self._orientedBoundingBox is None and self._convexHull is not None:
            self._orientedBoundingBox = gmu.get_oriented_bounding_box(self._convexHull)
        if self._orientedBoundingBox is None and self._cartesianTransform is not None: # this is different for imagery
            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            box.translate([-0.5, -0.5, -0.5])
            box.transform(self._cartesianTransform)
            self._orientedBoundingBox = box.get_oriented_bounding_box()
        return self._orientedBoundingBox
    
        
    def get_convex_hull(self) -> o3d.geometry.TriangleMesh:
        """Gets the Open3D Convex Hull of the node. If no convex hull is present, it is gathered from the following inputs.
        
        Args:
            - self._orientedBoundingBox
            - self._cartesianTransform

        Returns:
            - o3d.geometry.TriangleMesh
        """
        if self._convexHull is not None:
            return self._convexHull
        
        if self._convexHull is None and isinstance(self._depthMap,np.ndarray):
            #get points from the depthmap
            pcd=self.get_pcd_from_depth_map()
            self._convexHull=pcd.compute_convex_hull()[0]
        
        if self._convexHull is None and self._orientedBoundingBox is not None and self._cartesianTransform is not None:
            #get radius of the sphere
            points=self._orientedBoundingBox.get_box_points()
            radius=np.linalg.norm(points[0,:]-points[1,:])/2 #not the best way to get the radius
            ball=o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            ball.transform(self._cartesianTransform)            
            self._convexHull=ball

        if self._convexHull is None and self._cartesianTransform is not None:
            #create standard sphere
            ball=o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            ball.transform(self._cartesianTransform)            
            self._convexHull=ball
        
        return self._convexHull
    
    
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
        width=self._imageWidth
        height=self._imageHeight
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
            depths=np.full(points.shape[0],self._depthMap.max() if isinstance(self._depthMap,np.ndarray) else 10)
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
            depths = np.full((direction.shape[0], 1), self._depthMap.max() if isinstance(self._depthMap,np.ndarray) else 10)
        world_coordinates = camera_center + direction * depths

        return world_coordinates
    
    def transform(self, 
                  transformation: Optional[np.ndarray] = None, 
                  rotation: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None, 
                  translation: Optional[np.ndarray] = None, 
                  rotate_around_center: bool = True):
        """
        Apply a transformation to the Node's cartesianTransform, resource, and convexHull.
        
        **NOTE**: WORK IN PROGRESS
        
        Args:
            - transformation (Optional[np.ndarray]) : A 4x4 transformation matrix.
            - rotation (Optional[Union[np.ndarray, Tuple[float, float, float]]]) : A 3x3 rotation matrix or Euler angles $(R_z,R_y,R_x)$ for rotation.
            - translation (Optional[np.ndarray]) : A 3-element translation vector.
            - rotate_around_center (bool) : If True, rotate around the object's center.
        """
        return
    
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
    
    def get_pcd_from_depth_map(self)->o3d.geometry.PointCloud:
        """
        Convert a panoramic depth map and image colors (equirectangular) to a 3D point cloud.
        
        Args:
            - self._depthMap: 2D numpy array containing depth values (equirectangular depth map)
            - self._resource: 2D numpy array containing color values (equirectangular color image)
        
        Returns:
            - An Open3D point cloud object
        """
        if not isinstance(self._depthMap,np.ndarray):
            return None
        
        # Get the dimensions of the depth map
        height, width = self._depthMap.shape
        
        # Check if the color image and depth map have the same dimensions
        resource=cv2.resize(self._resource,(self._depthMap.shape[1],self._depthMap.shape[0])) if (isinstance(self._resource,np.ndarray) and not self._resource.shape[0] == self._depthMap.shape[0]) else self._resource

        # field of view in radians
        fov_horizontal_rad =  2*np.pi
        fov_vertical_rad = np.pi

        # Generate arrays for pixel coordinates
        # u: horizontal pixel coordinates (0 to width-1), v: vertical pixel coordinates (0 to height-1)
        u = np.linspace(0, width - 1, width)
        v = np.linspace(0, height - 1, height)#[::-1]  # Flip vertically (top to bottom)
        u, v = np.meshgrid(u, v)

        # Map pixels to spherical coordinates
        # Azimuth (longitude) theta is mapped from 0 to 2*pi across the width of the image
        theta = u / (width - 1) * fov_horizontal_rad - np.pi  # Map [0, width-1] to [-pi, pi]

        # Elevation (latitude) phi is mapped from 0 to pi across the height of the image
        phi = v / (height - 1) * fov_vertical_rad - (np.pi / 2)  # Map [0, height-1] to [-pi/2, pi/2]

        # Spherical to Cartesian conversion in camera coordinate system (z-forward, y-up)
        x = self._depthMap * np.cos(phi) * np.sin(theta)    # x-axis (left-right in pinhole model)
        y = self._depthMap * -np.sin(phi)                   # y-axis (up-down in pinhole model)
        z = self._depthMap * np.cos(phi) * np.cos(theta)    # z-axis (forward-backward in pinhole model)
      
        # Flatten the x, y, z arrays to create a list of points
        points = np.stack((x.flatten(order='C'), y.flatten(order='C'), z.flatten(order='C')), axis=-1)
        
        # Flatten the color image to correspond to the points
        colors=None
        if isinstance(resource,np.ndarray):
            colors = resource.reshape(-1, 3, order='C')  # Ensure row-major order (C-style)
            #divide by 255 if the colors are in the range 0-255
            colors = colors / 255 if np.max(colors) > 1 else colors
        
        # Create an Open3D point cloud from the points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if isinstance(colors,np.ndarray):
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # #note that Z goes through the center of the image, so we need to rotate the point cloud 90° clockwise around the x-axis
        # r = R.from_euler('x', -90, degrees=True).as_matrix()
        # pcd.rotate(r,center = [0,0,0])
        
        #transform to the cartesianTransform
        pcd.transform(self._cartesianTransform) if self._cartesianTransform is not None else None
        return pcd
    
    
    # def crop_pano(pano,project_node,crop_size, crop_output_folder):
    #     tile_count = 0
    #     if pano.resource is not None or pano.path is not None and pano.use and pano.depth is not None:
    #             pano.linkedNodes = []
    #             small_images = geomapi.utils.imageutils.subdivide_image(pano.resource, width = crop_size[0], height=crop_size[1], includeLast= True)
    #             for i, small_image in enumerate(small_images[0]):
    #                 tile_count += 1
    #                 name = "%s_%s_%s_%s_%s_%s" %(project_node.name, pano.name.split(".")[0], int(small_images[1][i][0]),int(small_images[1][i][1]),int(small_images[1][i][2]),int(small_images[1][i][3]))
    #                 row = int(small_images[1][i][0]/8)
    #                 depth = np.zeros(shape=(int(crop_size[0]/8),int(crop_size[1]/8)))
    #                 while row <= int(small_images[1][i][1]/8):
    #                     column = int(small_images[1][i][2]/8)
    #                     while column <= int(small_images[1][i][3]/8):
    #                         depth[row-int(small_images[1][i][0]/8),column-int(small_images[1][i][2]/8)] = pano.depth[row,column]/1000
    #                         column +=1
    #                     row +=1
    #                 depth = np.asarray(depth)
    #                 average_depth = np.average(depth)
    #                 gray_image = np.dot(small_image, [0.2989, 0.5870, 0.1140])
    #                 small_image_node = panonode.PanoNode(_name = name,
    #                                                     path = os.path.join(os.path.join(crop_output_folder), "%s.png" %(name)),
    #                                                     roi = small_images[1][i],
    #                                                     resource = small_image,
    #                                                     source = pano.subject,
    #                                                     imageWidth = crop_size[0],
    #                                                     imageHeight = crop_size[1],
    #                                                     use = "TRUE",
    #                                                     average_depth = average_depth,
    #                                                     contrast = int(np.max(gray_image)) - int(np.min(gray_image))
    #                                                     )
    #                 small_image_node.save_resource()
    #                 pano.linkedNodes.append(small_image_node)
    #     return tile_count

    # def pano_localization(pano):
    #     points = []
    #     for object in pano.objecten:
    #         points.append(object.poi)
    #     points = np.array(points)
    #     points.reshape(int(points.size/3),3)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    #     r = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    #     pcd_rotated3 = copy.deepcopy(pcd)#pcd_rotated2)
    #     pcd_rotated3.rotate(R=r,center = [0,0,0])
    #     T = pano.cartesianTransform
    #     pcd_trans = copy.deepcopy(pcd_rotated3)
    #     pcd_trans.transform(T)
    #     for i, object in enumerate(pano.objecten):
    #         object.location = np.asarray(pcd_trans.points)[i]
        
def navvis_csv_to_nodes(csvPath :Path, 
                        directory : Path = None, 
                        includeDepth : bool = True, 
                        depthPath : Path = None, 
                        skip:int=None, **kwargs) -> List[PanoNode]:
    """Parse Navvis csv file and return a list of PanoNodes with the csv metadata.
    
    Args:
        - csvPath (Path): csv file path e.g. "D:/Data/pano/pano-poses.csv"
        - skip (int, Optional): select every nth image from the xml. Defaults to None.
        - Path (Path, Optional): path to the pano directory. Defaults to None.
        - includeDepth (bool, Optional): include depth images. Defaults to True.
        - depthPath (Path, Optional): path to the depth images. Defaults to None.
        - kwargs: additional keyword arguments for the PanoNode instances
                
    Returns:
        - A list of PanoNodes with the csv metadata
        
    """
    assert skip == None or skip >0, f'skip == None or skip '
    assert os.path.exists(csvPath), f'File does not exist.'
    assert csvPath.endswith('.csv'), f'File does not end with csv.'
    
    #open csv
    pano_csv_file = open(csvPath, mode = 'r')
    pano_csv_data = list(pd.reader(pano_csv_file))
    
    #get pano information
    pano_data = []
    for sublist in pano_csv_data[1:]:
        pano_data.append(sublist[0].split('; '))
        
    pano_filenames = []
    for sublist in pano_data:
        pano_filenames.append(sublist[1])
        
    pano_timestamps = []
    for sublist in pano_data:
        pano_timestamps.append(ut.get_timestamp(sublist[2]))
        
    pano_cartesianTransforms = []
    for sublist in pano_data:
        r = R.from_quat((float(sublist[7]),float(sublist[8]), float(sublist[9]), float(sublist[6]))).as_matrix()
        T = np.pad(r, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        T[0,3] = float(sublist[3])
        T[1,3] = float(sublist[4])
        T[2,3] = float(sublist[5])
        T[3,3] = float(1)
        pano_cartesianTransforms.append(T)
        
    #get pano path
    directory = csvPath.parent if not directory else directory
        
    if includeDepth:
        depth_filenames = []
        for pano_filename in pano_filenames:
            depth_filename  = pano_filename.replace(".jpg","_depth.png") #navvis depth images are png
            depth_filenames.append(depth_filename)
        if not depthPath:
            depthPath = directory.replace("pano", "pano_depth")
    
    #create nodes     
    nodelist=[]
    for i,pano in enumerate(pano_filenames):
        if os.path.exists(os.path.join(directory, pano)):
            if includeDepth:
                node=PanoNode(name= pano.split(".")[0],
                            cartesianTransform=pano_cartesianTransforms[i],
                            timestamp = pano_timestamps[i],
                            path = directory / pano,
                            depthPath = depthPath / depth_filenames[i],
                            **kwargs)
            nodelist.append(node)
    return nodelist