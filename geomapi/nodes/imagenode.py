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
from rdflib import Graph, URIRef
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
import geomapi.utils.geometryutils as gmu
import geomapi.utils.imageutils as iu
import geomapi.utils.geospatialutils as gsu


class ImageNode(Node):
    # class attributes
    
    def __init__(self,  graph : Graph = None, 
                        graphPath:Path=None,
                        subject : URIRef = None,
                        name:str=None,
                        path : Path=None, 
                        resource = None,
                        xmpPath: Path = None,
                        xmlPath: Path = None,
                        # csvPath: Path = None,
                        imageWidth:int = None, #640
                        imageHeight:int = None, #480
                        principalPointU:float =None,  
                        principalPointV:float= None, 
                        focalLength35mm:float = None, #2600 
                        cartesianTransform: np.ndarray = None,
                        intrinsicMatrix:np.ndarray=None,
                        keypoints:np.ndarray=None,
                        descriptors:np.ndarray=None,
                        depthMap: np.ndarray | float= None,
                        depthPath: Path = None,       
                        getResource : bool = False,
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
        #private attributes 
        self._xmlPath=None
        self._xmpPath=None
        # self._csvPath=None
        self._imageWidth = None 
        self._imageHeight = None  
        self._principalPointU=None #this is the deviation, not the actual value
        self._principalPointV=None
        self._focalLength35mm = None 
        self._keypoints = None
        self._descriptors = None 
        self._intrinsicMatrix = None
        self._depthMap=None
        self._depthPath=None
        
        self._resource=None 
        self._path=None
        self._subject=None
        self._cartesianTransform=None
        self._name=None
        self._graphPath=None

        self.subject=subject
        self.path=path        
        self.name=name
        self.resource=resource                
        self.cartesianTransform=cartesianTransform
        self.graphPath=graphPath
        
        #instance variables
        self.xmlPath=xmlPath
        self.xmpPath=xmpPath
        # self.csvPath=csvPath
        self.imageWidth=imageWidth
        self.imageHeight=imageHeight
        self.principalPointU=principalPointU
        self.principalPointV=principalPointV
        self.focalLength35mm=focalLength35mm
        self.keypoints=keypoints
        self.descriptors=descriptors
        self.intrinsicMatrix = intrinsicMatrix
        self.depthMap=depthMap
        self.depthPath=depthPath
        self.get_resource() if getResource else None
        
        #initialise functionality
        self.get_metadata_from_xmp_path() if xmpPath and os.path.exists(xmpPath) else None
        self.get_metadata_from_xml_path() if xmlPath and os.path.exists(xmlPath) else None
        # self.get_metadata_from_csv_path() if csvPath and os.path.exists(csvPath) else None 
        self.get_metadata_from_exif_data(path) if path and os.path.exists(path) else None
        
        self.get_image_width()
        self.get_image_height()
        self.get_focal_length()
        
        super().__init__(   graph= graph,
                            graphPath= self._graphPath,
                            subject= self._subject,
                            path=path,
                            name=self._name,
                            resource = resource,
                            getResource=getResource,
                            cartesianTransform=self._cartesianTransform,
                            **kwargs) 
        
        self.get_principal_point_u()
        self.get_principal_point_v()
        self.get_intrinsic_matrix()

#---------------------PROPERTIES----------------------------

    #---------------------xmlPath----------------------------
    @property
    def xmlPath(self): 
        """Get the xmlPath (Path) of the node. This is the Agisoft Metashape xml file path."""
        return self._xmlPath#ut.parse_path(self._xmlPath)

    @xmlPath.setter
    def xmlPath(self,value:Path):
        if value is None:
            pass
        elif Path(value).suffix =='.xml':
            self._xmlPath=Path(value)
        else:
            raise ValueError('self.xmlPath has invalid type, path or extension.')    

  #---------------------xmpPath----------------------------
    @property
    def xmpPath(self): 
        """Get the xmpPath (str) of the node. This is the RealityCapture xmp file path."""
        return self._xmpPath#ut.parse_path(self._xmpPath)

    @xmpPath.setter
    def xmpPath(self,value:Path):
        if value is None:
            pass
        elif Path(value).suffix =='.xmp':
            self._xmpPath=Path(value)
        else:
            raise ValueError('self.xmpPath has invalid type, path or extension.')   
         
#   #---------------------csvPath----------------------------
#     @property
#     def csvPath(self): 
#         """Get the xmpPath (str) of the node."""
#         return self._csvPath#ut.parse_path(self._csvPath)

#     @csvPath.setter
#     def csvPath(self,value):
#         if value is None:
#             pass
#         elif Path(value).suffix =='.csv':
#             self._csvPath=Path(value)
#         else:
#             raise ValueError('self.csvPath has invalid type, path or extension.')    


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
    
    #---------------------principalPointU----------------------------
    @property
    def principalPointU(self):
        """Get the principalPointU (float) of the node. Note that this is the deviation, not the actual value."""
        if self._principalPointU is None :
            pass
        return self._principalPointU
    
    @principalPointU.setter
    def principalPointU(self,value:float):
        if value is None:
            pass
        elif type(float(value)) is float:
            self._principalPointU=float(value)
        else:
            raise ValueError('self.principalPointU must be a float')
        
    #---------------------principalPointV----------------------------
    @property
    def principalPointV(self):
        """Get the principalPointV (float) of the node. Note that this is the deviation, not the actual value."""
        if self._principalPointV is None :
            pass
        return self._principalPointV
    
    @principalPointV.setter
    def principalPointV(self,value:float):
        if value is None:
            pass
        elif type(float(value)) is float:
            self._principalPointV=float(value)
        else:
            raise ValueError('self.principalPointV must be a float')
        
    #---------------------focalLength35mm----------------------------
    @property
    def focalLength35mm(self):
        """Get the focalLength35mm (float) of the node."""
        return self._focalLength35mm
    
    @focalLength35mm.setter
    def focalLength35mm(self,value:float):
        if value is None:
            pass
        elif type(float(value)) is float:
            self._focalLength35mm=float(value)
        else:
            raise ValueError('self.focalLength35mm must be a float')
    
    #---------------------depthMap----------------------------
    @property
    def depthMap(self):
        """Get the depthMap (np.array) of the Image. This is used for the convex hull and oriented bounding box."""
        return self._depthMap
    
    @depthMap.setter
    def depthMap(self,value:float|np.ndarray):
        if value is None:
            pass
        elif isinstance(np.asarray(value),np.ndarray):
            self._depthMap=np.asarray(value)
        elif type(float(value)) is float and float(value)>0:
            self._depthMap=np.full((self._imageWidth,self._imageHeight), float(value))
        else:
            raise ValueError('self.depth must be  np.array, o3d.geometry.Image or float greater than 0')
        
    #---------------------keypoints----------------------------
    @property
    def keypoints(self):
        """Get the keypoints (np.array) of the node. These are the distinct pixels in the image."""
        return self._keypoints
    
    @keypoints.setter
    def keypoints(self,value:np.ndarray):
        if value is None:
            pass
        elif type(np.array(value)) is np.ndarray:
            self._keypoints=np.array(value)
        else:
            raise ValueError('self.keypoints must be a numpy array')
        
    #---------------------descriptors----------------------------
    @property
    def descriptors(self):
        """Get the descriptors (np.array) of the node. These are the unique features of the image."""
        return self._descriptors

    @descriptors.setter
    def descriptors(self,value:np.ndarray):
        if value is None:
            pass
        elif type(np.array(value)) is np.ndarray:
            self._descriptors=np.array(value)
        else:
            raise ValueError('self.descriptors must be a numpy array')
        
    #---------------------intrinsicMatrix----------------------------
    @property
    def intrinsicMatrix(self):
        """Get the intrinsic camera matrix (np.array) of the node.
        k=
        [fx 0 cx]
        [0 fy cy]
        [0 0  1]
        
        """
        return self._intrinsicMatrix

    @intrinsicMatrix.setter
    def intrinsicMatrix(self,value:np.ndarray):
        if value is None:
            pass
        elif type(np.array(value)) is np.ndarray and value.size==9:
            value=value.reshape(3,3)
            self._intrinsicMatrix=np.array(value)
        else:
            raise ValueError('self.descriptors must be a numpy array')
        
#---------------------METHODS----------------------------
   
    def set_resource(self,value):
        """Set the resource of the Node from various inputs.

        Args:
            - np.ndarray (OpenCV)
            - PIL Image
            - Open3D Image

        Raises:
            ValueError: Resource must be np.ndarray (OpenCV), PIL Image or Open3D Image.
        """

        if isinstance(np.asarray(value),np.ndarray) : #OpenCV
            self._resource = np.asarray(value)
        # elif isinstance(value,PIL.MpoImagePlugin.MpoImageFile): 
        #     self._resource=  np.array(value)#cv2.cvtColor(np.array(value), cv2.COLOR_RGB2BGR) #not sure if this is needed
        # elif isinstance(value,PIL.Image.Image): 
        #     self._resource=  np.array(value)#cv2.cvtColor(np.array(value), cv2.COLOR_RGB2BGR)
        # elif isinstance(value,o3d.geometry.Image):
        #     self._resource = np.array(value)
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
        if self._path is not None: #self._path.exists():
            return self._path
        
        elif self._graphPath and (self._name or self._subject):
            folder=self._graphPath.parent 
            nodeExtensions=ut.get_node_resource_extensions(str(type(self)))
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if path.suffix.upper() in nodeExtensions:
                    if self.get_name() in path.stem :
                        self.path = path    
                        return self._path
            if self._name:
                self.path=os.path.join(folder,self._name+nodeExtensions[0])
            else:
                self.path=os.path.join(folder,self._subject+nodeExtensions[0])
            return self._path
        
        elif self._xmpPath :
            folder=self._xmpPath.parent 
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if str(Path(path).suffix) in ut.IMG_EXTENSIONS:
                    if str(self._xmpPath.stem) in str(path) :
                        self.name=self._xmpPath.stem
                        self.subject=self._name
                        self.path = path    
                        return self._path
                    
        elif self._xmlPath and (self._name or self._subject):
            folder=self._xmlPath.parent
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if path.suffix.upper() in ut.IMG_EXTENSIONS:
                    if self.get_name() in path.stem :
                        self.path = path    
                        return self._path
        else:
            return None
        
    def get_depth_map(self):        
        """Returns the full path of the depthMap from this Node. If no path is present, it is gathered from the following inputs.

        Args:
            - self._depthPath
            
        Returns:
            - np.array: depthMap
        """
        # Load depthmap image
        depthmap = np.asarray(Image.open(self._depthPath)).astype(float)
        
        # Vectorized calculation for the depth values
        depth_value = (depthmap[:, :, 0] / 256) * 256 + \
                    (depthmap[:, :, 1] / 256) * 256 ** 2 + \
                    (depthmap[:, :, 2] / 256) * 256 ** 3 + \
                    (depthmap[:, :, 3] / 256) * 256 ** 4

        # Assign the computed depth values to the class attribute _depthMap
        self._depthMap = depth_value
        return self._depthMap 
    
    # def get_xmp_path(self)->Path: 
    #     """Returns the xmpPath in the node. If none is present, it will search for the data on drive from the following inputs.\n

    #     Args:
    #         1. self.graphPath 
    #         2. self.name 
    #         3. self.subject

    #     Returns:
    #         str or None
    #     """
    #     if self._xmpPath is not None:
    #         return self._xmpPath       
             
    #     elif self._graphPath and (self._name or self._subject):
    #         folder=ut.get_folder_path(self._graphPath)
    #         allSessionFilePaths=ut.get_list_of_files(folder) 
    #         for path in allSessionFilePaths:
    #             if ut.get_extension(path).endswith('xmp'):
    #                 if self.get_name() in path or self.get_subject() in path :
    #                     self._xmpPath = path    
    #                     return self._xmpPath
    #     else:
    #         return None

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
        if self._resource is None:
            return False
        
        #validate extension
        if extension.upper() not in ut.IMG_EXTENSIONS:
            raise ValueError('Invalid extension')
        
        filename=ut.validate_string(self._name)

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
                directory=os.path.join(dir,'IMG')   
                self.path=os.path.join(dir,filename + extension)
            else:
                directory=os.path.join(os.getcwd(),'IMG')
                self.path=os.path.join(dir,filename + extension)
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

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
        for kp in self._keypoints:
            kp.size = keypoint_size
        # Draw keypoints on the image
        image=self.resource if overwrite else copy.deepcopy(self.resource)
        img_with_keypoints = cv2.drawKeypoints(image, self._keypoints, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return img_with_keypoints

    # def get_mesh_geometry(self, depth:float=10, focalLength35mm:float=24)->o3d.geometry.TriangleMesh:
    #     """Generate a concical mesh representation using the Image's cartesianTransform and focalLength35mm.\n

    #     **DEPECATED**: This function is deprecated and will be removed in future versions. Use self.convexHull instead.
    
    #     .. image:: ../../../docs/pics/virtual_image2.PNG

    #     Args:
    #         1. depth (float, optional): Viewing depth of the image. Defaults to 10m.
    #         2. focalLength35mm (float,optional): standardised focal length on 35mm film (w=36mm, h = 24mm)

    #     Returns:
    #         o3d.geometry.TriangleMesh 
    #     """
    #     if self.cartesianTransform is not None:
    #         radius=35/(focalLength35mm*2)*depth        
    #         mesh= o3d.geometry.TriangleMesh.create_cone(radius=radius, height=depth, resolution=20, split=1)
    #         rotation=gmu.get_rotation_matrix(self.cartesianTransform)
    #         r=R.from_matrix(rotation)
    #         rz=R.from_euler('xyz' ,[0, 0, 0], degrees=True)
    #         t=gmu.get_translation(self.cartesianTransform)
    #         mesh=mesh.translate(t)
    #         r=rz*r
    #         # t2=r.as_matrix() * np.array([[1],[0],[0]]) *depth
    #         A = np.dot( r.as_matrix(),np.array([0,0,-1]) )*depth
    #         mesh=mesh.translate(A)
    #         rot=r.as_matrix()
    #         mesh=mesh.rotate(rot)
    #         return mesh
    #     else:
    #         return None

    # def get_virtual_image(self, geometries: o3d.geometry, downsampling:int=2)-> o3d.geometry.Image:
    #     """Generates a virtual image of a set of geometries given the ImageNode's pose and piholeModel.

    #     .. image:: ../../../docs/pics/rendering3.PNG

    #     Args:
    #         - geometries (o3d.geometry): geometries to include in the scene of the virtual image.
    #         - downsampling (int, optional): pixel downsampling of the image both in height and width (each step reduces the density by factor 4). Defaults to 2.

    #     Returns:
    #         o3d.geometry.Image
    #     """
    #     pinholeCamera=o3d.camera.PinholeCameraParameters()
    #     pinholeCamera.extrinsic=self.cartesianTransform          
    #     intrinsic=o3d.camera.PinholeCameraIntrinsic(width=int(self.imageWidth/downsampling), 
    #                                         height=int(self.imageHeight/downsampling), 
    #                                         fx=self.focalLength35mm/downsampling, 
    #                                         fy=self.focalLength35mm/downsampling, 
    #                                         cx=self.intrinsicMatrix[0,2]/downsampling, 
    #                                         cy=self.intrinsicMatrix[1,2]/downsampling)
          
    #     pinholeCamera.intrinsic=intrinsic
    #     return gmu.generate_virtual_image(geometries,pinholeCamera)


    # def get_pinhole_camera_parameters(self, downsampling:int=1) -> o3d.camera.PinholeCameraParameters:
    #     """Returns the intrinsic and extrinsix camera parameters based on the following attributes.

    #     .. image:: ../../../docs/pics/pinholemodel1.PNG

    #     Args:
    #         - self.imageWidth: width of the image in pixels (u) 
    #         - self.imageHeight: height of the image in pixels (v) 
    #         - self.focalLength35mm: focal length with a standardised Field-of-View.
    #         - self.cartesianTransform: External camera pose.
    #         - downsampling (int, optional): pixel downsampling of the image both in height and width (each step reduces the density by factor 4). Defaults to 1.

    #     Returns:
    #         o3d.camera.PinholeCameraParameters()
    #     """
    #     param=o3d.camera.PinholeCameraParameters()
    #     # param.extrinsic=np.linalg.inv(self.cartesianTransform) #! unsure why this was inverted
    #     param.extrinsic=self.cartesianTransform            
    #     param.intrinsic=self.get_intrinsic_matrix()/downsampling 
    #     return param


    def get_intrinsic_matrix(self) -> np.ndarray:
        """Returns the intrinsic camera matrix based on the following attributes.
        
        intrinsic camera matrix (3x3) k=
        [fx 0 cx]
        [0 fy cy]
        [0 0  1]

        Args:
            - self.imageWidth: width of the image in pixels (u) 
            - self.imageHeight: height of the image in pixels (v) 
            - self.focalLength35mm: focal length with a standardised Field-of-View.
            - self.PrincipalPointU: cx 
            - self.PrincipalPointV: cy 

        Returns:
            - np.array(3x3)
        """
        if self._intrinsicMatrix is not None:
            return self._intrinsicMatrix
        else:
            pinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(self._imageWidth,
                                                                       self._imageHeight,
                                                                       self._focalLength35mm,
                                                                       self._focalLength35mm,
                                                                       self._imageWidth/2+self._principalPointU,
                                                                       self._imageHeight/2+self._principalPointV)
            self._intrinsicMatrix = pinholeCameraIntrinsic.intrinsic_matrix
            return self._intrinsicMatrix
    
    def get_principal_point_u(self) -> float:
        if self._principalPointU is not None:
            pass
        else:
            self._principalPointU= 0
        return self._principalPointU
    
    def get_principal_point_v(self) -> float:
        if self._principalPointV is not None:
            pass
        else:
            self._principalPointV= 0
        return self._principalPointV
    
    def get_image_width(self) -> int:
        if self._imageWidth:
            pass
        elif self._resource is not None:
            self._imageWidth=self._resource.shape[1]  
        else:
            self._imageWidth=640
        return self._imageWidth
    
    def get_image_height(self) -> int:
        if self._imageHeight:
            pass
        elif self._resource is not None:
            self._imageHeight=self._resource.shape[0]  
        else:
            self._imageHeight=480
        return self._imageHeight
    
    def get_focal_length(self) -> float:
        if self._focalLength35mm:
            pass
        else:
            self._focalLength35mm=2600
        return self._focalLength35mm
    
    # def get_intrinsic_camera_parameters(self, downsampling:int=1) -> o3d.camera.PinholeCameraIntrinsic():
    #     """Returns the intrinsic camera parameters based on the following attributes.
        
    #     Args:
    #         - self.imageWidth: width of the image in pixels (u). Defaults to 640p 
    #         - self.imageHeight: height of the image in pixels (v). Defaults to 480p  
    #         - self.focalLength35mm: focal length with a standardised Field-of-View. Defaults to 2600pix 
    #         - self.PrincipalPointU: cx 
    #         - self.PrincipalPointV: cy 

    #     Returns:
    #         o3d.camera.PinholeCameraIntrinsic(width,height,fx,fy,cx,cy)
    #     """
    #     #validate inputs
    #     width=int(self._imageWidth/downsampling) 
    #     height=int(self._imageHeight/downsampling) 
    #     f=self._focalLength35mm 
        
    #     #! deprecated
    #     # pixX=width/36 #these are standard 35mm film properties
    #     # pixY=height/24 #these are standard 35mm film properties
    #     # fx=pixX*f
    #     # fy=pixY*f        

    #     if (getattr(self,'principalPointU',None) is not None and
    #         getattr(self,'principalPointV',None) is not None ):
    #         cx=width/2-0.5+self.principalPointU
    #         cy=height/2-0.5+self.principalPointV
    #     else:
    #         cx=width/2-0.5
    #         cy=height/2-0.5
    #     pinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width,height,f,f,cx,cy)
    #     self._intrinsic_matrix = pinholeCameraIntrinsic.intrinsic_matrix
    #     return pinholeCameraIntrinsic

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
    
    def get_metadata_from_xmp_path(self)->bool:
        """Read Metadata from .xmp file generated by https://www.capturingreality.com/.

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
        #check if imageWidth and Height are present, else gather it from the image
        if self._imageHeight is None or self._imageWidth is None:
            folder=self._xmpPath.parent 
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if Path(path).suffix in ut.IMG_EXTENSIONS:
                    if str(self._xmpPath.stem) in str(path) :
                        self.name=self._xmpPath.stem
                        self.subject=self._name
                        self.path = path    
                        self.get_resource() 
                        self.get_image_height()
                        self.get_image_width()
                        break
        if self._path is None:
            return False
        
        if hasattr(self,'graph'):
            return True

        if (getattr(self,'principalPointU',None) is not None and
            getattr(self,'principalPointV',None) is not None and
            getattr(self,'distortionCoeficients',None) is not None and
            getattr(self,'geospatialTransform',None) is not None ):         
            return True
        
        mytree = ET.parse(self.xmpPath)
        root = mytree.getroot()                       
        
        self.timestamp=ut.get_timestamp(self.xmpPath)
        self.name=self.xmpPath.stem
        self.subject=self.name
        for child in root.iter('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'):

            #Attributes
            for attribute in child.attrib:
                if ('latitude' in attribute and
                    'longitude'in attribute and
                    'altitude' in attribute):
                    lat=ut.xcr_to_lat(child.attrib['{http://www.capturingreality.com/ns/xcr/1.1#}latitude'])
                    long=ut.xcr_to_long(child.attrib['{http://www.capturingreality.com/ns/xcr/1.1#}longitude'])
                    alt=ut.xcr_to_alt(child.attrib['{http://www.capturingreality.com/ns/xcr/1.1#}altitude'])
                    self.geospatialTransform=np.array([lat, long, alt])
                if 'Coordinates' in attribute:
                    self.coordinateSystem=child.attrib[attribute]
                if 'FocalLength35mm' in attribute:
                    f=ut.xml_to_float(child.attrib[attribute])
                    self.focalLength35mm=f/36*self.imageWidth #! 35mm film width is 36mm doesn't seem super correct
                if 'PrincipalPointU' in attribute:
                    self.principalPointU=ut.xml_to_float(child.attrib[attribute])
                if 'PrincipalPointV' in attribute:
                    self.principalPointV=ut.xml_to_float(child.attrib[attribute])

            #Nodes
            rotationnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}Rotation')
            rotation=None
            if rotationnode is not None:
                rotation=ut.string_to_rotation_matrix(rotationnode.text).T #! RC uses column-based rotaton matrix

            positionnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}Position')
            translation=None
            if positionnode is not None:
                translation=np.asarray(ut.literal_to_list(positionnode.text))
              
            self.cartesianTransform=gmu.get_cartesian_transform(translation=translation,rotation=rotation)
            
            coeficientnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}DistortionCoeficients')
            if coeficientnode is not None:
                self.distortionCoeficients=ut.literal_to_list(coeficientnode.text)  
        return True   
    
    # def get_metadata_from_csv_path(self) ->bool:
    #     """Extract image metadata from XML Node generated by Agisoft Metashape (self.xmlData and self.subject should be present).

    #     Args:
    #         - cartesianTransform (np.array(4x4))
    #         - sxy: accuracy in XY (m)
    #         - sz: accuracy in Z (m) 

    #     Returns:
    #         bool: True if metadata is successfully parsed
    #     """

    #     if (getattr(self,'cartesianTransform',None) is not None and
    #         getattr(self,'sxy',None) is not None and
    #         getattr(self,'sz',None) is not None ):
    #         return True
        
    #     self.timestamp=ut.get_timestamp(self.csvPath)
    #     with open(self.csvPath, 'r') as file:
    #         csvData = file.readlines()
    #         for line in csvData:
    #             if 'Image' in line:
    #                 data = line.split(',')
    #                 if data[1] == self.name or data[1] == ut.get_subject_name(self.subject):
    #                     self.cartesianTransform = ut.string_to_list(data[2])
    #                     self.sxy = float(data[3])
    #                     self.sz = float(data[4])
    #                     return True
    #     return False

    def get_metadata_from_xml_path(self) ->bool:
        """Extract image metadata from XML Node generated by Agisoft Metashape (self.xmlData and self.subject should be present).
        
        **NOTE**: this is relative to the xml overall position. use tools.xml_to_nodes to get the absolute information

        Args:
            - cartesianTransform (np.array(4x4))
            - sxy: accuracy in XY (m)
            - sz: accuracy in Z (m) 

        Returns:
            - bool: True if metadata is successfully parsed
        """

        if self._cartesianTransform is not None:
            return True
        
        if hasattr(self,'graph'):
            return True
        
        #open xml

        self.timestamp=ut.get_timestamp(self.xmlPath)        
        mytree = ET.parse(self.xmlPath)
        root = mytree.getroot()   

        #get reference
        chunk=root.find('chunk')
        globalTransform=gmu.get_cartesian_transform(rotation=ut.literal_to_matrix(chunk.find('transform').find('rotation').text),
                                                translation= ut.literal_to_matrix(chunk.find('transform').find('translation').text))
    
        globalScale=float(chunk.find('transform').find('scale').text)


        #get components -> in some xml files, there are no components.
        components=[]
        for component in root.iter('component'):       
            try:
                transform=component.find('transform')
                region=component.find('region')
                scale=float(transform.find('scale').text)
                components.append({'componentid':  int(component.get('id')),        
                                'refTransform': gmu.get_cartesian_transform(rotation=ut.literal_to_matrix(transform.find('rotation').text),
                                                    translation= ut.literal_to_matrix(transform.find('translation').text)),
                                'scale': scale,
                                'center': gmu.get_cartesian_transform( translation=ut.literal_to_matrix(region.find('center').text)),
                                'size': ut.literal_to_matrix(region.find('size').text),
                                'R': ut.literal_to_matrix(region.find('R').text)})     
            except:
                components.append(None)
                continue

        #get sensors
        sensors=[]
        for sensor in root.iter('sensor'):       
            try:
                calibration=sensor.find('calibration')
                focalLength35mm= calibration.find('f').text if calibration.find('f') is not None else calibration.find('fx').text # sometimes the focal length is named diffirently
                sensors.append({'sensorid':  int(sensor.get('id'))   ,        
                                'imageWidth': int(calibration.find('resolution').get('width')),
                                'imageHeight': int(calibration.find('resolution').get('height')),
                                'focalLength35mm': float(focalLength35mm)})     
            except:
                sensors.append(None)
                continue
        #! this is different
        
        #grab the right camera
        if self.get_name():
            cam = next(cam for cam in root.iter('camera') if (Path(cam.get('label')).stem == self._name))
        else:
            #take first
            cam = next((cam for cam in root.iter('camera')) ,None)
            self.subject=Path(cam.get('label')).stem

        #get component
        componentid=cam.get('component_id')  
        if componentid:
            componentInformation= next(c for c in components if c['componentid']==int(componentid))  
            refTransform=componentInformation['refTransform']
            scale=componentInformation['scale']
        else:
            refTransform=globalTransform
            scale=globalScale

        #get transform
        transform=np.reshape(ut.literal_to_matrix(cam.find('transform').text),(4,4))
        transform=gmu.get_cartesian_transform(rotation=transform[0:3,0:3],
                                        translation=transform[0:3,3]*scale)
        
        transform=refTransform  @ transform

        #get sensor information
        sensorid=int(cam.get('sensor_id'))      
        sensorInformation= next(s for s in sensors if s is not None and s.get('sensorid')==sensorid)

        self.cartesianTransform=transform
        self.imageWidth=sensorInformation['imageWidth']
        self.imageHeight=sensorInformation['imageHeight']
        self.focalLength35mm=sensorInformation['focalLength35mm']


    def get_cartesian_transform(self) -> np.ndarray:
        """Get the cartesianTransform of the node from various inputs. if no cartesianTransform is present, a default np.eye(4) is used.

        Returns:
            - cartesianTransform(np.ndarray(4x4))
        """
        if self._cartesianTransform is not None:
            return self._cartesianTransform
        
        if self._cartesianTransform is None and self.get_convex_hull() is not None:
            #get pyramid points
            points = np.asarray(self._convexHull.vertices)
            #get top of pyramid
            top=points[0,:]
            #get baseCenter
            baseCenter=np.mean(points[1:,:],axis=0)
            #compute vector
            vector=baseCenter-top
            
            #get translation -> top of pyramid
            translation = top
            
            #get rotation
            #1.Normalize the vector (in this case, it's already normalized)
            vector = vector / np.linalg.norm(vector)
            #2.Determine the axis and angle
            #Rotate from z-axis to the target vector
            z_axis = np.array([0, 0, 1])
            cross_product = np.cross(z_axis, vector)
            dot_product = np.dot(z_axis, vector)
            norm_cross_product = np.linalg.norm(cross_product)

            if norm_cross_product != 0:
                axis = cross_product / norm_cross_product
                angle = np.arctan2(norm_cross_product, dot_product)
            else:
                # If the vector is already aligned with the z-axis, no rotation is needed.
                axis = np.array([1, 0, 0])  # Arbitrary axis
                angle = 0.0 if dot_product > 0 else np.pi  # 0 if aligned, pi if opposite

            #3.Compute the rotation matrix using the axis and angle
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            self._cartesianTransform = gmu.get_cartesian_transform(translation=translation,rotation=rotation_matrix) 
            
        if self._cartesianTransform is None:
            self._cartesianTransform = np.eye(4)
        
        return self._cartesianTransform
    
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
        
        if self._convexHull is None and self._orientedBoundingBox is not None:
            points=np.asarray(self._orientedBoundingBox.get_box_points())
            #get 4 base points: point 0,1,2,7
            basePoints=np.vstack((points[0,:],points[1,:],points[2,:],points[7,:]))
            #get 4 top points: point 4,5,6,3
            topPoints=np.vstack((points[4,:],points[5,:],points[6,:],points[3,:]))
            top_mean=np.mean(topPoints,axis=0)
            #repeat top points 4 times
            points=np.vstack((top_mean,top_mean,top_mean,top_mean,basePoints))
            self._convexHull=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).compute_convex_hull()[0]

        if self._convexHull is None and self._cartesianTransform is not None:
            startpoints, endpoints=gmu.rays_to_points(self.create_rays())
            points=np.vstack((startpoints,endpoints))
            self._convexHull=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).compute_convex_hull()[0]
        
        return self._convexHull
    
    
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
    
        if self._orientedBoundingBox is None and self.get_convex_hull() is not None and self._cartesianTransform is not None: 
            #gather width and height from convex hull
            points=np.asarray(self._convexHull.vertices)
            width=np.linalg.norm(points[1,:]-points[2,:])
            height=np.linalg.norm(points[3,:]-points[2,:])
            #create box at origin
            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            box.translate((-0.5,-0.5,-0.5))  
            #get oriented bounding box
            box = box.get_oriented_bounding_box() # u is x, v is y, w is z in this 
            depth=self._depthMap.max() if self._depthMap is not None else 10 #default depth
            box=gmu.expand_box(box,width-1,height-1,depth-1)
            box.translate([0,0,depth/2]) #center at the base so move upwards   
            #rotate and translate to correct position     
            box.rotate(self._cartesianTransform[0:3,0:3],center=(0,0,0)) #center (0,0,0) is at the top of the pyramid
            box.translate(gmu.get_translation(self._cartesianTransform))  
            self._orientedBoundingBox = box        
        return self._orientedBoundingBox
    
    def create_rays(self,imagePoints:np.array=None,depths:np.array=None)->o3d.core.Tensor:
        """Generate a grid a rays from the camera location to a given set of imagePoints.
                
        **NOTE**: This function targets a subselection of imagePoints, use o3d.t.geometry.RaycastingScene.create_rays_pinhole if you want a dense raytracing for the full image.
        
        .. image:: ../../../docs/pics/Raycasting_1.PNG
        
        Args:
            - imagePoints (np.array[n,2]) : imagePoints are conform (row,column) image coordinates system. so top left is (0,0). The camera intrinsic matrix is used to map it to the proper image coordinates. Defaults to np.array([[0, 0],[0, self._imageWidth],[self._imageHeight, 0],[self._imageHeight, self._imageWidth]]).
            - depths (np.array[n,1], optional) : depths of the rays. Defaults to 50m for each point.

        Returns:
            rays: o3d.core.Tensor (n,6) [:,0:3] with the camera center and [:,3:6] are the directions of the rays towards the imagePoints.
        """
        if imagePoints is None:
            points=np.array([[0, 0], # top left
                                  [0, self._imageWidth], # top right
                                  [self._imageHeight, 0],  # bottom left
                                  [self._imageHeight, self._imageWidth]]) # bottom right
        else:
            points=ut.map_to_2d_array(imagePoints)
            
        if depths is None:
            depths=np.full(points.shape[0],self._depthMap.max() if self._depthMap is not None else 10)
        else:
             depths = np.asarray(depths).flatten()  # Ensure depths is a 1D array

            
        #validate inputs
        points=np.reshape(points,(-1,2)) #if len(points.shape) >2 else points
            
        f=self._focalLength35mm 
        k=self._intrinsicMatrix #get_intrinsic_camera_parameters().intrinsic_matrix
        m=self._cartesianTransform 
        t=gmu.get_translation(m)  
        n=points.shape[0]
        
        #transform pixels to image coordinates (rows are first)
        u=+points[:,1]-self._imageWidth/2
        v=+points[:,0]-self._imageHeight/2    
        camera_coordinates=np.vstack((u,v,np.ones(n)))
        
        #transform to world coordinates
        camera_coordinates=np.vstack((camera_coordinates[0:2,:],np.full(n, f).T,np.ones((n,1)).T))
        world_coordinates=m @ camera_coordinates
        
        #normalize direction
        displacement=world_coordinates[0:3,:].T-t
        direction=gmu.normalize_vectors(displacement)
  
        if depths is not None:
            direction=direction * depths[:, np.newaxis]
          
        # Create rays [camera.center, direction]
        rays = np.hstack((np.tile(t, (n, 1)), direction))
        
        # #flatten if it is a single point
        # if rays.shape[0] == 1:
        #     rays = rays.flatten()
              
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
        worldCoordinates=ut.convert_to_homogeneous_3d_coordinates(worldCoordinates)
        
        imageCoordinates= np.linalg.inv(self.cartesianTransform) @ worldCoordinates.T

        xy=copy.deepcopy(imageCoordinates)
        xy[0]= imageCoordinates[0]/imageCoordinates[2]*self.focalLength35mm
        xy[1]= imageCoordinates[1]/imageCoordinates[2]*self.focalLength35mm
        xy[2]= imageCoordinates[2]/imageCoordinates[2]*self.focalLength35mm

        uv=copy.deepcopy(xy)
        uv[0]=xy[1]+self.imageHeight/2
        uv[1]=xy[0]+self.imageWidth/2
        uv=uv[0:2] #these are (row,column) coordinates
        
        #flatten it if it is a single point
        # if uv.shape[1] == 1:
        #     uv=uv.flatten()
                
        return uv.T
    
    
    def pixel_to_world_coordinates(self, pixels: np.array, depths: np.array = None) -> np.ndarray:
        """Converts pixel coordinates in an image to 3D world coordinates.

        This function takes pixel coordinates and optional depths and converts them to 3D world coordinates. It uses camera parameters such as the transformation matrix, focal length, image width, and image height.

        Args:
            - pixels (np.array[n,2]) : Pixel coordinates in the image (row, column).
            - depths (np.array[n,1], optional) : Depths for the corresponding pixel coordinates. Defaults to 50m for each point.

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
            depths = np.full((direction.shape[0], 1), self._depthMap.max() if self._depthMap is not None else 10)
        world_coordinates = camera_center + direction * depths

        return world_coordinates
    
    def transform(self, 
                  transformation: Optional[np.ndarray] = None, 
                  rotation: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None, 
                  translation: Optional[np.ndarray] = None, 
                  rotate_around_center: bool = True):
        """
        Apply a transformation to the Node's cartesianTransform, resource, and convexHull.
        
        Args:
            - transformation (Optional[np.ndarray]) : A 4x4 transformation matrix.
            - rotation (Optional[Union[np.ndarray, Tuple[float, float, float]]]) : A 3x3 rotation matrix or Euler angles $(R_z,R_y,R_x)$ for rotation.
            - translation (Optional[np.ndarray]) : A 3-element translation vector.
            - rotate_around_center (bool) : If True, rotate around the object's center.
        """
        if self.cartesianTransform is None:
            self.get_cartesian_transform()
            
        if transformation is not None:
            transformation=np.reshape(np.asarray(transformation),(4,4))            
        elif translation is not None or rotation is not None:
            transformation = gmu.get_cartesian_transform(rotation=rotation, translation=translation)
            
   
        if rotate_around_center:
            #cartesian transformation                
            transform_to_center = gmu.get_cartesian_transform (translation=-self.cartesianTransform[:3,3] ) 
            transform_back = gmu.get_cartesian_transform (translation=self.cartesianTransform[:3,3] )
            self._cartesianTransform = transform_back @ transformation @ transform_to_center @ self.cartesianTransform
                        
            #oriented bounding box
            if self._orientedBoundingBox is not None:
                # transform_to_center = gmu.get_cartesian_transform (translation=-self._orientedBoundingBox.get_center() )
                # transform_back = gmu.get_cartesian_transform (translation=self._orientedBoundingBox.get_center() )            
                points=self._orientedBoundingBox.get_box_points()
                pcd=o3d.geometry.PointCloud(points)      
                pcd.transform(transform_to_center)
                pcd.transform(transformation)      
                pcd.transform(transform_back )
                self._orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            
            #convex hull
            if self._convexHull is not None:
                # transform_to_center = gmu.get_cartesian_transform (translation=-self._convexHull.get_center() )
                # transform_back = gmu.get_cartesian_transform (translation=self._convexHull.get_center() )
                self._convexHull.transform(transform_to_center)
                self._convexHull.transform( transformation)
                self._convexHull.transform(transform_back)
                            
        else: #not sure about this one!
            #cartesian transformation                
            self._cartesianTransform = transformation @ self.cartesianTransform 
            
            #oriented bounding box
            if self._orientedBoundingBox is not None:
                points=self._orientedBoundingBox.get_box_points()
                pcd=o3d.geometry.PointCloud(points)            
                pcd.transform(transformation )
                self._orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            
            #convex hull
            self._convexHull.transform( transformation ) if self._convexHull is not None else None
    
    
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
        if self.get_resource() is None:
            return None
        
        #copy if overwrite is False
        image=self._resource if overwrite else copy.deepcopy(self._resource)
        
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
        image=self._resource if overwrite else copy.deepcopy(self._resource)
        
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