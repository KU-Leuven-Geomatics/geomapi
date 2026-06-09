"""
**OrthoNode** is a Python Class to govern the data and metadata of orthomosaic data. 

This node builds upon the [OpenCV](https://opencv.org/), [Open3D](https://www.open3d.org/) and [PIL](https://pillow.readthedocs.io/en/stable/) API for the image definitions.
Be sure to check the properties defined in those abstract classes to initialise the Node.

.. image:: ../../../docs/pics/graph_ortho_1.png

**IMPORTANT**: This Node class is designed to manage geolocated orthomosaics. It works best when the location parameters are known.

"""
#IMPORT PACKAGES
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import open3d as o3d
import ezdxf
import geomapi.utils.cadutils as cadu

from rdflib import XSD, Graph, URIRef
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import copy
from typing import List, Optional,Tuple,Union

#IMPORT MODULES
from geomapi.nodes import Node
import geomapi.utils as ut
from geomapi.utils import rdf_property, GEOMAPI_PREFIXES

import geomapi.utils.imageutils as iu
import geomapi.utils.geometryutils as gmu

class OrthoNode(Node):
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
                dxfPath : Path = None, 
                tfwPath : Path = None,
                imageWidth:int = None, #2000
                imageHeight:int = None, #1000
                gsd: float = None,
                depth:float=10,
                height:float=None,
                **kwargs): 
        """
        Creates an OrthoNode. Overloaded function.
        
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to

        Args:
            - subject (RDFlib URIRef) : subject to be used as the main identifier in the RDF Graph
            
            - graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - graphPath (Path) :  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - path (Path) : Path to an image .jpg, jpeg, .png file (data is not automatically loaded)
            
            - resource (ndarray, PIL, Open3D) : Image data from [Open3D](https://www.open3d.org/), [OpenCV](https://opencv.org/) or [PIL](https://pillow.readthedocs.io/en/stable/). 
  
            - imageWidth (int, optional) : width of the image in pixels (u). Defaults to 2000pix
            
            - imageHeight (int, optional) : height of the image in pixels (v). Defaults to 1000pix
            
            - gsd (float, optional) : Ground Sampling Distance in meters. Defaults to 0.01m
            
            - depth (float, optional) : Average depth of the image in meters. Defaults to 10m
            
            - height (float, optional) : Average height of the cameras that generated the image in meters. Defaults to None
            
            - dxfPath (Path, optional) : Path to the dxf file with the orthometadata from MetaShape. Defaults to None.
            
            - tfwPath (Path, optional) : Path to the tfw file with the orthometadata from MetaShape. Defaults to None.
            
            - getResource (bool, optional) : If True, the resource is loaded from the path. Defaults to False.
            
            - **kwargs : Additional keyword arguments to be used in the Node class.
            
        Returns:
            OrthoNode : A OrthoNode with metadata
        """  
   
        self.gsd=gsd
        self.imageWidth=imageWidth
        self.imageHeight=imageHeight
        self.depth=depth
        self.height=height
        self.dxfPath=dxfPath
        self.tfwPath=tfwPath

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
        self.get_metadata_from_tfw_path() if self.tfwPath else None 
 

#---------------------PROPERTIES----------------------------

    #---------------------dxfPath----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def dxfPath(self): 
        """The path (Path) of the dxf file with the orthometadata from MetaShape."""
        return self._dxfPath

    @dxfPath.setter
    def dxfPath(self,value:Path):
        if value is None:
           self._dxfPath = None
        elif Path(value).suffix.upper() == ".DXF":
            self._dxfPath=Path(value)
        else:
            raise ValueError('dxfPath invalid extension.')
        
    #---------------------tfwPath----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def tfwPath(self): 
        """The path (Path) of the tfw file with the orthometadata from MetaShape.
        The tfw world file is a text file used to georeference the GeoTIFF raster images, like the orthomosaic and the DSM.
        The tfw file is a 6-line file:

            Line 0: pixel size in the x-direction in map units (GSD).
            Line 1: rotation about y-axis.
            Line 2: rotation about x-axis.
            Line 3: pixel size in the y-direction in map in map units (GSD).
            Line 4: x-coordinate of the upper left corner of the image.
            Line 5: y-coordinate of the upper left corner of the image.
        """
        return self._tfwPath

    @tfwPath.setter
    def tfwPath(self,value:Path):
        if value is None:
           self._tfwPath = None
        elif Path(value).suffix.upper() in ut.CAD_EXTENSIONS:
            self._tfwPath=Path(value)
        else:
            raise ValueError('tfwPath invalid extension.')    
            
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
        
    #---------------------gsd----------------------------
    @property
    @rdf_property(datatype=XSD.float)
    def gsd(self):
        """Get the Ground Sampling Distance (float) of the node."""
        if(self._gsd is None):
            self._gsd = 0.01
        return self._gsd
    
        if self._imageWidth and self._orientedBoundingBox:
            #get most common value
            array1 = self._orientedBoundingBox.extent/self._imageHeight
            array2 = self._orientedBoundingBox.extent/self._imageWidth
            rounded_result = np.round(np.stack((array1, array2), axis=0),4)
            unique, counts = np.unique(rounded_result, return_counts=True)
            self._focalLength35mm = unique[np.argmax(counts)]
    
    @gsd.setter
    def gsd(self,value:float):
        if value is None:
            self._gsd = None
        elif type(float(value)) is float and float(value)>0:
            self._gsd=float(value)
        else:
            raise ValueError('self.gsd must be a float and greater than 0')
    
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
        
    #---------------------height----------------------------
    @property
    @rdf_property(datatype=XSD.float)
    def height(self):
        """Get the average height (float) of cameras that generated the image. This is used for the convex hull and oriented bounding box."""
        if self._height is None:
            self._height = 0
        return self._height
    
    @height.setter
    def height(self,value:float):
        if value is None:
            self._height=None
        elif type(float(value)) is float:
            self._height=float(value)
        else:
            raise ValueError('self.height must be a float')

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
    
    def _set_geometric_properties(self, _cartesianTransform=None, _convexHull=None, _orientedBoundingBox=None):
    
        self.cartesianTransform = _cartesianTransform
        self.convexHull = _convexHull
        self.orientedBoundingBox = _orientedBoundingBox

        # --- Handle Transform ---
        if self.cartesianTransform is None:
            if self.orientedBoundingBox is not None: # the boundingbox has more info, the orientation can return the backface as the origin
                self.cartesianTransform = gmu.get_backface_center_transform(self.orientedBoundingBox)
            elif self.convexHull is not None:
                self.cartesianTransform = gmu.get_cartesian_transform(translation=np.asarray(self.convexHull.get_center()))
            else:
                self.cartesianTransform = np.eye(4)  # Default to identity transform
        
        # --- Handle Oriented Bounding Box ---
        if self.orientedBoundingBox is None:
            if self.convexHull is not None:
                self.orientedBoundingBox = gmu.get_oriented_bounding_box(self.convexHull)
            else:
                # Create a boundingbox from the image parameters
                self.orientedBoundingBox = gmu.create_obb_from_orthophoto(self.cartesianTransform, self.imageWidth, self.imageHeight, self.gsd, self.depth)
        
        # --- Handle Convex Hull ---
        if self.convexHull is None:
            # get from bb
            self.convexHull = gmu.get_convex_hull(self.orientedBoundingBox)
    
    def get_metadata_from_tfw_path(self):
        """Get metadata from the tfw file. Uses the following information.
        
        **NOTE**: Only Geographic projection information is currently supported. 
        
        The tfw world file is a text file used to georeference the GeoTIFF raster images, like the orthomosaic and the DSM.
        The tfw file is a 6-line file:

            Line 0: pixel size in the x-direction in map units (GSD).
            Line 1: rotation about y-axis.
            Line 2: rotation about x-axis.
            Line 3: pixel size in the y-direction in map in map units (GSD).
            Line 4: x-coordinate of the upper left corner of the image.
            Line 5: y-coordinate of the upper left corner of the image.
        
         Args:
            - self._imageWidth
            - self._imageHeight
            - self._height
            - self._depth

        Returns:
            - self._gsd
            - self._cartesianTransform        
        """
        if self._graph:
            return None
        
        self.name=Path(self.tfwPath).stem
        self.subject=URIRef(self.name)

        with open(self.tfwPath, 'r') as f:
            A = float(f.readline())  # pixel size X
            D = float(f.readline())  # rotation
            B = float(f.readline())  # rotation
            E = float(f.readline())  # pixel size Y
            C = float(f.readline())  # top-left X position of the center of the pixel
            F = float(f.readline())  # top-left Y position of the center of the pixel
            
        # Calculate the ground sample distance (GSD)
        self.gsd = abs(A)  # usually GSD is the same in both X and Y directions

        # Translation (C, F) represents the position of the top-left pixel
        x=C+(self.imageWidth/2.0-0.5) * self._gsd
        y=F-(self.imageHeight/2.0-0.5) * self._gsd
        translation=np.array([x,y,self.height])
        #get rotation -> we apply downwards rotation similar to pinhole camera coordinate systems
        rotation_matrix_180_x=   np.array( [[1,0,0],
                                        [ 0,-1,0],
                                        [ 0,0,-1]])  
        rotation_x=  R.from_euler('x',B,degrees=False).as_matrix()
        rotation_y=  R.from_euler('y',D,degrees=False).as_matrix()
        #unsure how to combine the rotations -> looks about right
        rotation_matrix=rotation_matrix_180_x*rotation_x*rotation_y
        self.cartesianTransform=gmu.get_cartesian_transform(translation=translation,rotation=rotation_matrix)
        
        #reset bounding box and convexhull
        self._set_geometric_properties(self.cartesianTransform)
    
    def project_lineset_on_image(self,lineSet:o3d.geometry.LineSet) -> o3d.geometry.LineSet:
        """Project a LineSet on the image. The LineSet is projected on the image using the camera model of the orthomosaic.
        
        Args:
            - lineSet (o3d.geometry.LineSet) : LineSet to be projected on the image
        
        Returns:
            - o3d.geometry.LineSet projected on the image
        """
        #get the camera model
        cameraModel=self.get_camera_model()
        
        #project the lineset
        projectedLineSet=cameraModel.project_line_set(lineSet)
        return projectedLineSet
    
    def create_rays(self,imagePoints:np.array=None,depths:np.array=None)->o3d.core.Tensor:
        """Generate a grid a rays from the camera location to a given set of imagePoints.
                
        **NOTE**: This function targets a subselection of imagePoints, use o3d.t.geometry.RaycastingScene.create_rays_pinhole if you want a dense raytracing for the full image.
        
        .. image:: ../../../docs/pics/Raycasting_1.PNG
        
        Args:
            - imagePoints (np.array[n,2]) : imagePoints are conform uv (column, row) image coordinates system. so top left is (0,0). The camera intrinsic matrix is used to map it to the proper image coordinates. Defaults to np.array([[0, 0],[0, self._imageWidth],[self._imageHeight, 0],[self._imageHeight, self._imageWidth]]).
            - depths (np.array[n,1], optional) : Depths of the rays. Defaults to 50m for each point.

        Returns:
            - o3d.core.Tensor (n,6) where [:,0:3] is the camera center and [:,3:6] are the directions of the rays towards the imagePoints.
        """
        if imagePoints is None:
            points=np.array([[0, 0], # top left
                            [0, self._imageWidth], # top right
                            [self._imageHeight, 0],  # bottom left
                            [self._imageHeight, self._imageWidth]]) # bottom right
        else:
            points=ut.map_to_2d_array(imagePoints)
            
        if depths is None:
            depths=np.full(points.shape[0],self.depth)
        else:
             depths = np.asarray(depths).flatten()  # Ensure depths is a 1D array

            
        #validate inputs
        points=np.reshape(points,(-1,2)) #if len(points.shape) >2 else points
            
        n=points.shape[0]
        
        #transform pixels to image coordinates (rows are first)
        u=(+points[:,1]-self._imageWidth/2)*self._gsd
        v=(self._imageHeight/2-points[:,0])*self._gsd
        
        #transform to world coordinates
        camera_coordinates=np.vstack((u,v,np.full(n, 0).T,np.ones((n,1)).T))
        world_coordinates=self._cartesianTransform  @ camera_coordinates
        world_coordinates=world_coordinates[0:3,:].T
        
        #get the direction of the rays
        boxPoints=np.asarray(self._convexHull.vertices)
        meanBase=np.mean(boxPoints[:4],axis=0)
        meanTop=np.mean(boxPoints[4:],axis=0)
        direction=meanTop-meanBase
        
        #repeat direction for all points
        direction=np.tile(direction,(n,1))
        if depths is not None:
            direction=direction * depths[:, np.newaxis]
        else:
            direction=direction * self._depth  
            
        # Create rays [start_point, direction]
        rays = np.hstack((world_coordinates, direction))
       
        return rays 
    
    def pixel_to_world_coordinates(self, pixels: np.array, depths: np.array = None) -> np.ndarray:
        """Converts pixel coordinates in an image to 3D world coordinates.

        This function takes pixel coordinates and optional depths and converts them to 3D world coordinates. 
        
        Args:
            - pixels (np.array[n,2]) : Pixel coordinates in the image (row, column).
            - depths (np.array[n,1], optional) : Depths for the corresponding pixel coordinates. Defaults to 50m for each point.

        Returns:
            - A 2D array containing the 3D world coordinates.
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

    def world_to_pixel_coordinates(self,worldCoordinates) -> np.ndarray:
        """Converts 3D world coordinates to pixel coordinates in an image.

        This function takes 3D world coordinates and converts them to pixel coordinates in an image. It uses camera parameters such as the transformation matrix, focal length, image width, and image height.

        **NOTE**: the pixel coordinates have a (row, column) format. This fitts well with array indexing, but not with Matplotlib's imshow function.

        Args:
            - worldCoordinates (np.ndarray (n,3 or n,4 )): A set of 3D (homogeneous) points in world coordinates to be converted.

        Returns:
            - A 2D array containing the pixel coordinates (row, column) in the image.

        Note:
            - The function performs a series of transformations, including world to camera, camera to image, and image centering.
            - It returns the imageCoordinates as a 2D array.
        """
        worldCoordinates=gmu.convert_to_homogeneous_3d_coordinates(worldCoordinates)
        
        imageCoordinates= np.linalg.inv(self._cartesianTransform) @ worldCoordinates.T

        xy=imageCoordinates[0:2]
        xy[0]= imageCoordinates[0]/self._gsd
        xy[1]= imageCoordinates[1]/self._gsd

        uv=copy.deepcopy(xy)
        # uv[0]=self.imageHeight/2-xy[1]
        # uv[1]=xy[0]+self.imageWidth/2
        
        uv=copy.deepcopy(xy)
        uv[0]=xy[1]+self.imageHeight/2
        uv[1]=xy[0]+self.imageWidth/2
                
        return uv.T
    
    def project_lineset_on_image(self,linesets:List[o3d.geometry.LineSet],thickness:int=2,overwrite=True) ->np.ndarray:
        """Project Opend3D linesets onto the resource of the node.

        **NOTE**: this affects the original image if overwrite is True.
        
        .. image:: ../../../docs/pics/ortho_projection_1.png

        Args:
            - linesets (List[o3d.LineSet]) : List of linesets. Note that the color of the lines is stored in the lineset.
            - thickness (int) : Thickness of the projected lines
            - overwrite (bool) : If True, the original image is overwritten. If False, a new image is created.

        Returns:
            - The resource of the ImageNode with the projected lines.
        """
        if self.resource is None:
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
    
    def show(self, convertColorspace = False):
        super().show()
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
   