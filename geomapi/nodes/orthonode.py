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

from rdflib import Graph, URIRef
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import copy
from typing import List, Optional,Tuple,Union

#IMPORT MODULES
from geomapi.nodes import Node
import geomapi.utils as ut
import geomapi.utils.imageutils as iu
import geomapi.utils.geometryutils as gmu

class OrthoNode(Node):
    # class attributes
    
    def __init__(self,  graph : Graph = None, 
                        graphPath:Path=None,
                        subject : URIRef = None,
                        path : Path=None,                  
                        resource = None,
                        dxfPath : Path = None, 
                        tfwPath : Path = None,
                        gsd: float = None,
                        depth:float=10,
                        height:float=None,
                        imageWidth:int = None, #2000
                        imageHeight:int = None, #1000
                        getResource : bool = False,
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
        self._gsd=None
        self._imageWidth = None 
        self._imageHeight = None  
        self._depth=None
        self._height=None
        self._dxfPath=None
        self._tfwPath=None
   
        self.gsd=gsd
        self.imageWidth=imageWidth
        self.imageHeight=imageHeight
        self.depth=depth
        self.height=height
        self.dxfPath=dxfPath
        self.tfwPath=tfwPath

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            resource = resource,
                            getResource=getResource,
                            **kwargs) 
        
        #initialise functionality
        self.get_metadata_from_tfw_path() if self.tfwPath else None 
        self.get_metadata_from_dxf_path() if self.dxfPath else None # we don't do anything with units
 

#---------------------PROPERTIES----------------------------

    #---------------------dxfPath----------------------------
    @property
    def dxfPath(self): 
        """The path (Path) of the dxf file with the orthometadata from MetaShape."""
        return self._dxfPath

    @dxfPath.setter
    def dxfPath(self,value:Path):
        if value is None:
           pass
        elif Path(value).suffix.upper() in ut.CAD_EXTENSIONS:
            self._dxfPath=Path(value)
        else:
            raise ValueError('dxfPath invalid extension.')
        
    #---------------------tfwPath----------------------------
    @property
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
           pass
        elif Path(value).suffix.upper() in ut.CAD_EXTENSIONS:
            self._tfwPath=Path(value)
        else:
            raise ValueError('tfwPath invalid extension.')    
            
    #---------------------imageWidth----------------------------
    @property
    def imageWidth(self):
        """Get the imageWidth (int) or columns of the resource of the node."""
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
        
    #---------------------gsd----------------------------
    @property
    def gsd(self):
        """Get the Ground Sampling Distance (float) of the node."""
        if self._gsd is None :
            pass
        return self._gsd
    
    @gsd.setter
    def gsd(self,value:float):
        if value is None:
            pass
        elif type(float(value)) is float and float(value)>0:
            self._gsd=float(value)
        else:
            raise ValueError('self.gsd must be a float and greater than 0')
    
    #---------------------depth----------------------------
    @property
    def depth(self):
        """Get the average depth (float) of the Image. This is used for the convex hull and oriented bounding box."""
        return self._depth
    
    @depth.setter
    def depth(self,value:float):
        if value is None:
            pass
        elif type(float(value)) is float and float(value)>0:
            self._depth=float(value)
        else:
            raise ValueError('self.depth must be a float and greater than 0')
        
    #---------------------height----------------------------
    @property
    def height(self):
        """Get the average height (float) of cameras that generated the image. This is used for the convex hull and oriented bounding box."""
        return self._height
    
    @height.setter
    def height(self,value:float):
        if value is None:
            pass
        elif type(float(value)) is float:
            self._height=float(value)
        else:
            raise ValueError('self.height must be a float')

        
        
        
        
        
        
#---------------------METHODS----------------------------
   
    def set_resource(self,value):
        """Set the resource of the Node from various inputs.\n

        Args:
            - np.ndarray (OpenCV)
            - PIL Image
            - Open3D Image

        Raises:
            ValueError: Resource must be np.ndarray (OpenCV), PIL Image or Open3D Image.
        """

        if type(np.array(value)) is np.ndarray : #OpenCV
            self._resource = np.asarray(value)
        else:
            raise ValueError('Resource must be np.ndarray (OpenCV) or PIL Image')
            
    def get_resource(self)->np.ndarray: 
        """Returns the data in the node. If none is present, it will search for the data on using the attributes below.

        Args:
            - self.path

        Returns:
            - np.ndarray or None
        """
        if self._resource is not None :
            return self._resource
        elif self.get_path() and self._path.exists():
            self._resource   = np.array(Image.open(self._path)) # PIL is 5% faster than OpenCV cv2.imread(self.path)
        return self._resource  
        
    def set_path(self, value):
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

        Returns:
            - path 
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


    def save_resource(self, directory:Path |str=None,extension :str = '.jpg') ->bool:
        """Export the resource of the Node.

        Args:
            - directory (str, optional) : directory folder to store the data.
            - extension (str, optional) : file extension. Defaults to '.jpg'.

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.

        Returns:
            bool: return True if export was succesful
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
                directory=os.path.join(dir,'ORTHO')   
                self.path=os.path.join(dir,filename + extension)
            else:
                directory=os.path.join(os.getcwd(),'ORTHO')
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
    
    def get_image_width(self) -> int:
        if self._imageWidth:
            pass
        elif self._resource is not None:
            self._imageWidth=self._resource.shape[1]  
        else:
            self._imageWidth=2000
        return self._imageWidth
    
    def get_image_height(self) -> int:
        if self._imageHeight:
            pass
        elif self._resource is not None:
            self._imageHeight=self._resource.shape[0]  
        else:
            self._imageHeight=1000
        return self._imageHeight
    
    def get_height(self) -> int:
        if self._height:
            pass
        else:
            self._height=0
        return self._height
    
    def get_gsd(self) -> float:
        if self._gsd:
            pass
        elif self._imageWidth and self._orientedBoundingBox:
            #get most common value
            array1 = self._orientedBoundingBox.extent/self._imageHeight
            array2 = self._orientedBoundingBox.extent/self._imageWidth
            rounded_result = np.round(np.stack((array1, array2), axis=0),4)
            unique, counts = np.unique(rounded_result, return_counts=True)
            self._focalLength35mm = unique[np.argmax(counts)]
        else:
            self._gsd=0.01
        return self._gsd
    
    def get_cartesian_transform(self) -> np.ndarray:
        """Get the cartesianTransform of the node from various inputs. if no cartesianTransform is present, it is gathered from the following inputs. 
                
        Args:
            - self._convexHul: The cartesianTransform is set at the middle of the base of the convexHull, with the z-axis pointing upwards and the y-axis pointing towards the top of the image.
            - self._orientedBoundingBox: The same applies

        Returns:
            - cartesianTransform(np.ndarray(4x4))
        """
        if self._cartesianTransform is not None:
            return self._cartesianTransform
            
        if self._cartesianTransform is None and self.get_convex_hull() is not None:
            #get box points
            points = np.asarray(self._convexHull.vertices)
            #get topPoints and topCenter
            topCenter=np.mean(points[4:,:],axis=0)
            #compute vector
            vector=topCenter-self._convexHull.get_center()
            
            #get translation -> center is at the base
            translation = self._convexHull.get_center() - vector 
            
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
            self._cartesianTransform[2,3]=self.get_height()
        
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
        
        if self._convexHull is None and self._cartesianTransform is not None:
            #create box at origin
            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            vertices=np.array(box.vertices)
 
            #compute dimensions
            width=self.get_image_width()*self.get_gsd()
            height=self.get_image_height()*self.get_gsd()
            
            #modify vertices
            xmin=-width/2
            xmax=width/2
            ymin=-height/2
            ymax=height/2
            zmin=0
            zmax=self._depth
            vertices=np.array([[xmin,ymin,zmin],
                    [xmax,ymin,zmin],
                    [xmin,ymax,zmin],
                    [xmax,ymax,zmin],
                    [xmin,ymin,zmax],
                    [xmax,ymin,zmax],
                    [xmin,ymax,zmax],
                    [xmax,ymax,zmax]])
            box.vertices = o3d.utility.Vector3dVector(vertices)
            
            #transform box
            #rotate and translate to correct position     
            box.rotate(self._cartesianTransform[0:3,0:3],center=(0,0,0))
            translation=gmu.get_translation(self._cartesianTransform)
            box.translate(translation)  
            self._convexHull = box        
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
            #create box at origin
            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            box.translate((-0.5,-0.5,-0.5))  
            box = box.get_oriented_bounding_box() # u is x, v is y, w is z in this case

            #compute dimensions and expand box
            points=np.asarray(self._convexHull.vertices)
            width=np.linalg.norm(points[0]-points[1]) #self.get_image_width()*self.get_gsd()
            height=np.linalg.norm(points[0]-points[2]) #self.get_image_height()*self.get_gsd()
            depth=np.linalg.norm(points[0]-points[4])
            box=gmu.expand_box(box,width-1,height-1,depth-1)
            box.translate([0,0,depth/2]) #center at the base so move upwards
            
            #rotate and translate to correct position     
            box.rotate(self._cartesianTransform[0:3,0:3],center=(0,0,0)) #center (0,0,0) is at the base
            box.translate(gmu.get_translation(self._cartesianTransform))  
            self._orientedBoundingBox = box   
            #TEST
            test_vertices=np.asarray(self._orientedBoundingBox.get_box_points()) #TEST    
        return self._orientedBoundingBox
    
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
        
        # Open the file and read it line by line
        with open(self._tfwPath, 'r') as file:
            rows = file.readlines()
            
        self._name=Path(self._tfwPath).stem
        self.subject=URIRef(self._name)

        # Strip newline characters from each line
        rows = [float(line.strip()) for line in rows]
        
        #get gsd
        self._gsd=rows[0] #we assume that the gsd is the same in both directions
        
        #get translation -> this is not correct
        if self._imageWidth == 2000 or self._imageHeight == 1000:
            print('Warning: Image dimensions are not set. Please set the imageWidth (2000pix) and imageHeight(1000pix) to get the correct translation.')
        x=rows[4]+self._imageWidth/2*self._gsd # this is a little bit off
        y=rows[5]-self._imageHeight/2*self._gsd
        translation=np.array([x,y,self.get_height()])
        
        #get rotation -> we apply downwards rotation similar to pinhole camera coordinate systems
        rotation_matrix_180_x=   np.array( [[1,0,0],
                                        [ 0,-1,0],
                                        [ 0,0,-1]])  
        rotation_x=  R.from_euler('x',rows[2],degrees=False).as_matrix()
        rotation_y=  R.from_euler('y',rows[1],degrees=False).as_matrix()
        
        #unsure how to combine the rotations -> looks about right
        rotation_matrix=rotation_matrix_180_x*rotation_x*rotation_y
        self._cartesianTransform=gmu.get_cartesian_transform(translation=translation,rotation=rotation_matrix)
        
        #reset bounding box and convexhull
        self._orientedBoundingBox=None
        self._convexHull=None
        self.get_oriented_bounding_box()
        self.get_convex_hull()
    
    def get_metadata_from_dxf_path(self) -> bool:
        """Sets the metadata of the node from the dxf file.
        
        **BUG**: MetaShape Dxf are poorly formatted. resave the dxf with a CAD software like AutoCAD to fix the issue.

        Args:
            - self._dxfPath
            - self._name (should be identical to the orthomosaic name)
            - self._height
            - self._depth

        Args:
            - imageWidth 
            - imageHeight
            - cartesianTransform

        Returns:
            True if exif data is successfully parsed
        """        
        if (not self._dxfPath or 
            not os.path.exists(self._dxfPath)):
            return False
        
        if self._graph:
            return True
    
        dxf = ezdxf.readfile(self._dxfPath)
        #contours and names are in the same list as pairs
        entities=[entity for entity in dxf.modelspace()]
        
        def create_convex_hull_from_dxf_points():
            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            bottomLeftLow=points[2]
            bottomRightLow=points[3]
            topLeftLow=points[1]
            topRightLow=points[0]
            bottomLeftHigh=points[2]+normal*self._depth
            bottomRightHigh=points[3]+normal*self._depth
            topLeftHigh=points[1]+normal*self._depth
            topRightHigh=points[0]+normal*self._depth
            vertices=np.array([[bottomLeftLow],
                                [bottomRightLow],
                                [topLeftLow],
                                [topRightLow],
                                [bottomLeftHigh],
                                [bottomRightHigh],
                                [topLeftHigh],
                                [topRightHigh]])
            
            box.vertices = o3d.utility.Vector3dVector(np.reshape(vertices,(8,3)))                    
            self._convexHull = box  
        
        if len([entity for entity in entities if entity.dxftype() == 'INSERT' and Path(entity.attribs[0].dxf.text).stem==self._name])==0:
            print('Warning: No INSERT entity found with the name of the orthomosaic. taking first ...')
            entity=entities[0]
            self._name=Path(entity.attribs[0].dxf.text).stem
        
        #iterate through entities per two
        for i in range(0,len(entities),2):
            #entity1 are the entities with the name
            entity1=entities[i] 
            #entity2 are the entities with the geometry
            entity2=entities[i+1]
            name=Path(entity1.attribs[0].dxf.text).stem
            if name == self._name:        
                #get geometry
                g=cadu.ezdxf_entity_to_o3d(entity2)
                g.translate(np.array([0,0,self.get_height()]))
                #get points -> they are ordered counter clockwise starting from the top left
                points=np.asarray(g.points)
                #get the center of the geometry
                center=g.get_center()
                #get the vector 0-1 and 0-3
                vec1=points[1]-points[0]
                vec2=points[3]-points[0]
                #get the normal of the plane
                normal=np.cross(vec1,vec2)
                #normalize the normal
                normal=normal/np.linalg.norm(normal)
                
                #get the translation matrix
                translation=center#-normal*self._depth
                
                #get rotation matrix from this normal to the z-axis
                rotation_matrix=ut.get_rotation_matrix_from_forward_up(normal, vec2)
                
                cartesianTransform = gmu.get_cartesian_transform(translation=translation,rotation=rotation_matrix) 
                self._cartesianTransform=cartesianTransform    
                
                #create convexhull
                create_convex_hull_from_dxf_points()
                
                #reset bounding box
                self._orientedBoundingBox=None
                self.get_oriented_bounding_box()
                
                #get gsd
                self._gsd=np.linalg.norm(vec1[0])/self.get_image_width()
                      
                return True
    
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
        worldCoordinates=ut.convert_to_homogeneous_3d_coordinates(worldCoordinates)
        
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
   