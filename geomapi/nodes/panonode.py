"""
Panonode is a Python Class to govern the data and metadata of panoramic data (OpenCV, PIL). 

This node builds upon the OpenCV and PIL API for the image definitions.
It directly inherits from Node.
Be sure to check the properties defined in the above classes to initialise the Node.
"""
#IMPORT PACKAGES
from ast import Raise
from distutils import extension
from pathlib import Path
from typing import Tuple
import xml.etree.ElementTree as ET
import cv2
import PIL
from rdflib import Graph, URIRef
import numpy as np
import os
import open3d as o3d
import math
import uuid
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


#IMPORT MODULES
from geomapi.nodes import Node
from geomapi.nodes import ImageNode
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import geomapi.utils.imageutils as it




class PanoNode(ImageNode):
    # class attributes
    
    def __init__(self,  graph : Graph = None, 
                        graphPath:str=None,
                        subject : URIRef = None,
                        path : str=None, 
                        depthPath: str = None,
                        getResource : bool = False,
                        getDepth : bool = False,
                        **kwargs): 
        """Creates a Node from one or more of the following inputs. 
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to True.\n

        Args: \n
            1. graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the MeshNode)\n
            2. graphPath (str):  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the MeshNode)\n
            3. path (str) : path to image file (Note that this node will also contain the data) \n
            4. resource (ndarray, PIL Image,Open3D) : OpenCV, PIL (Note that this node will also contain the data)\n
            5. xmlPath (str) : Xml file path from Agisoft Metashape\n
            6. xmpPath (str) :  xmp file path from RealityCapture
            \n
            - getResource (bool, optional= False): If True, the node will search for its physical resource on drive \n
            - getMetaData (bool, optional= True): If True, the node will attempt to extract metadata from the resource if present \n

        Returns:
            An ImageNode with metadata 
        """  
        #private attributes 
        self.depthPath=None
        self._orientedBoundingBox=None
        self.imageWidth = None # (int) number of pixels
        self.imageHeight = None  # (int) number of pixels 
        self.depth = None
        self.linkedNodes = []

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            **kwargs)   

        #instance variables
        

        #initialisation functionality
        if getResource:
            self.get_resource()
        
        if self._resource is not None:
            self.get_metadata_from_resource()

        if getDepth and depthPath is None:
            depthPath = path.replace("pano", "pano_depth")
            self.depthPath = depthPath.replace("jpg", "PNG")
        elif depthPath is not None:
            self.depthPath = depthPath
        
        if getResource or self._resource is not None:
            self.get_metadata_from_resource()
            if getDepth and self.imageWidth is not None and self.imageHeight is not None:
                self.depth = self.get_depth()
            


# #---------------------PROPERTIES----------------------------

#     #---------------------xmlPath----------------------------
#     @property
#     def xmlPath(self): 
#         """Get the xmlPath (str) of the node."""
#         return ut.parse_path(self._xmlPath)

#     @xmlPath.setter
#     def xmlPath(self,value):
#         if value is None:
#             return None
#         elif (str(value).endswith('xml') ):
#             self._xmlPath=str(value)
#         else:
#             raise ValueError('self.xmlPath has invalid type, path or extension.')    

#   #---------------------xmpPath----------------------------
#     @property
#     def xmpPath(self): 
#         """Get the xmpPath (str) of the node."""
#         return ut.parse_path(self._xmpPath)

#     @xmpPath.setter
#     def xmpPath(self,value):
#         if value is None:
#             return None
#         elif (str(value).endswith('xmp') ):
#             self._xmpPath=str(value)
#         else:
#             raise ValueError('self.xmpPath has invalid type, path or extension.')    

# #---------------------orientedBoundingBox----------------------------
#     @property
#     def orientedBoundingBox(self): 
#         """Get the orientedBoundingBox of the Node from various inputs. \n

#         Args:
#             1. Open3D.geometry.OrientedBoundingBox \n
#             2. Open3D geometry\n

#         Returns:
#             orientedBoundingBox (o3d.geometry.OrientedBoundingBox) 
#         """
#         return self._orientedBoundingBox

#     @orientedBoundingBox.setter
#     def orientedBoundingBox(self,value):
#         if value is None:
#             return None
#         if 'orientedBoundingBox' in str(type(value)):
#             self._orientedBoundingBox=value
#         else:    
#             try: #geometry
#                 self._orientedBoundingBox=value.get_oriented_bounding_box()
#             except:
#                 raise ValueError('Input must be orientedBoundingBox (o3d.geometry.OrientedBoundingBox) or an Open3D Geometry.')

# #---------------------METHODS----------------------------

    def get_depth(self):
        """
        Function to decode the depthmaps generated by the navvis processing
        source: Location of the PNG files containing the depthmap
        resize(bool): If the resulting dethmap needs to be resized to match the size of the corresponding pano, by default True
        size: size of the corresponding pano, by default 8192x4096
        """

        depthmap = np.asarray(PIL.Image.open(self.depthPath)).astype(float)
        converted_depthmap = np.empty([np.shape(depthmap)[0], np.shape(depthmap)[1]])
        r = 0
        while r < np.shape(depthmap)[0]:
            c = 0
            while c < np.shape(depthmap)[1]:
                value = depthmap[r,c]
                depth_value = value[0] / 256 * 256 + value[1] / 256 * 256 * 256 + value[2] / 256 * 256 * 256 * 256 + value[3] / 256 * 256 * 256 * 256 * 256
                converted_depthmap[r,c] = depth_value
                c = c + 1
            r = r + 1

            # resized_depthmap = cv2.resize(converted_depthmap,(self.imageWidth, self.imageHeight))
        self.depth = converted_depthmap



    def set_resource(self,value):
        """Set the resource of the Node from various inputs.\n

        Args:
            1. np.ndarray (OpenCV) \n
            2. PIL Image\n
            3. Open3D Image\n

        Raises:
            ValueError: Resource must be np.ndarray (OpenCV), PIL Image or Open3D Image.
        """

        if type(value) is np.ndarray : #OpenCV
            self._resource = value
        elif 'Image' in str(type(value)): # PIL
            self._resource=  cv2.cvtColor(np.array(value), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError('Resource must be np.ndarray (OpenCV) or PIL Image')

    def get_resource(self)->np.ndarray: 
        """Returns the resource (image) in the node. 
        If none is present, it will search for the data on drive from the following inputs. \n

        Args:
            1. self.path\n
            2. self.graphPath\n
            3. self.name or self.subject

        Returns:
            np.ndarray or None
        """
        if self._resource is not None :
            return self._resource
        elif self.get_path(): 
            self._resource   = cv2.imread(self.path)
            self.get_metadata_from_resource()
        return self._resource 
    
    def save_resource(self, directory:str=None,extension :str = '.png') ->bool:
        """Export the resource of the Node.\n

        Args:
            1. directory (str, optional): directory folder to store the data.\n
            2. extension (str, optional): file extension. Defaults to '.png'.\n

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.\n

        Returns:
            bool: return True if export was succesful
        """     
        #check path
        if self.resource is None:
            return False
        
        #validate extension
        if extension not in ut.IMG_EXTENSIONS:
            raise ValueError('Invalid extension')

        # check if already exists
        if directory and os.path.exists(os.path.join(directory,self.get_name() + extension)):
            self.path=os.path.join(directory,self.get_name() + extension)
            return True
        elif not directory and self.get_path() and os.path.exists(self.path) and extension in ut.IMG_EXTENSIONS:
            return True    
          
        #get directory
        if (directory):
            pass    
        elif self.path is not None and os.path.exists(self.path):    
            directory=ut.get_folder(self.path)            
        elif(self.graphPath): 
            dir=ut.get_folder(self.graphPath)
            directory=os.path.join(dir,'IMG')   
        else:
            directory=os.path.join(os.getcwd(),'IMG')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory)   

        #write files
        if cv2.imwrite(self.path, self.resource):
            return True
        return False
    
    # def get_depth(self)->np.ndarray: 
    #     """Returns the resource (image) in the node. 
    #     If none is present, it will search for the data on drive from the following inputs. \n

    #     Args:
    #         1. self.path\n
    #         2. self.graphPath\n
    #         3. self.name or self.subject

    #     Returns:
    #         np.ndarray or None
    #     """
    #     if self._depth is not None :
    #         return self._depth
    #     elif self.get_path():
    #         self._depth   = cv2.imread(self.depthPath)
    #     return self._depth

    def save_depthmap(self, directory:str=None,extension :str = '.png') ->bool:
        """Export the resource of the Node.\n

        Args:
            1. directory (str, optional): directory folder to store the data.\n
            2. extension (str, optional): file extension. Defaults to '.png'.\n

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.\n

        Returns:
            bool: return True if export was succesful
        """     
        #check path
        if self.depth is None:
            return False
        
        #validate extension
        if extension not in ut.IMG_EXTENSION:
            raise ValueError('Invalid extension')

        # check if already exists
        if directory and os.path.exists(self.depthPath):
            return True
          
        #get directory
        if (directory):
            pass    
        elif self.depthPath is not None and os.path.exists(self.depthPath):    
            directory=ut.get_folder(self.depthPath)            
        elif(self.graphPath): 
            dir=ut.get_folder(self.graphPath)
            directory=os.path.join(dir,'IMG')   
        else:
            directory=os.path.join(os.getcwd(),'IMG')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory)
        #write files
        if plt.imsave(self.depthPath, self.depth):
            return True
        return False
       
#     def get_oriented_bounding_box(self)->o3d.geometry.OrientedBoundingBox:
#         """Gets the Open3D OrientedBoundingBox of the node from the conical mesh representation based on the 
#         cartesianTransform, the focal length at 35mm and a viewing range. \n

#         Returns:
#             o3d.geometry.orientedBoundingBox
#         """    
#         if self._orientedBoundingBox is not None:
#             pass
#         elif self._cartesianTransform is not None:
#             mesh=self.get_mesh_geometry()
#             self._orientedBoundingBox=mesh.get_oriented_bounding_box()            
#         else:
#             return None
#         return self._orientedBoundingBox

    def get_metadata_from_resource(self) ->bool:
        """Returns the metadata from a resource. \n

        Features:
            1. imageHeight\n
            2. imageWidth\n

        Returns:
            bool: True if exif data is successfully parsed
        """
        if self._resource is None:
            return False   
              
        try:
            if getattr(self,'imageHeight',None) is None:
                self.imageHeight=self.resource.shape[0]
            if getattr(self,'imageWidth',None) is None:
                self.imageWidth=self.resource.shape[1]
            return True
        except:
            raise ValueError('Metadata extraction from resource failed')
    
#     # def get_oriented_bounding_box(self)->o3d.geometry.OrientedBoundingBox:
#     #     """Gets the Open3D geometry from cartesianTransform

#     #     Returns:
#     #         o3d.geometry.orientedBoundingBox
#     #     """
#     #     if getattr(self,'orientedBoundingBox',None) is None:                
#     #         if getattr(self,'cartesianTransform',None) is not None:
#     #             box=o3d.geometry.create_mesh_box(width=1.0, height=1.0, depth=1.0)
#     #             self.orientedBoundingBox= box.transform(self.cartesianTransform)
#     #         else:
#     #             return None
#     #     return self.orientedBoundingBox

#     def get_metadata_from_exif_data(self) -> bool:
#         """Returns the metadata from a resource. \n

#         Features:
#             1. GPSInfo (geospatialTransform (np.array(3,1))
#             2. coordinateSystem (str) \n
#             2. DateTime ('%Y-%m-%dT%H:%M:%S')\n
#             3. XResolution (int)\n
#             4. YResolution (int)\n
#             5. ResolutionUnit (int)\n
#             6. ExifImageWidth (int)\n
#             7. ExifImageHeight (int)\n

#         Returns:
#             bool: True if meta data is successfully parsed
#         """
#         if  self.get_path() is None or not os.path.exists(self.get_path()) :
#             return False
        
#         if getattr(self,'timestamp',None) is None :
#             self.timestamp=ut.get_timestamp(self.path)
        
#         if getattr(self,'name',None) is None:
#             self.name=ut.get_filename(self.path)

#         if (getattr(self,'imageWidth',None) is not None and
#             getattr(self,'imageHeight',None) is not None and
#             getattr(self,'geospatialTransform',None) is not None):
#             return True

#         # pix = PIL.Image.open(self.path) 
#         with PIL.Image.open(self.path) as pix:
#             exifData=ut.get_exif_data(pix)

#         if exifData is not None:
#             self.timestamp=ut.get_if_exist(exifData, "DateTime")
#             self.resolutionUnit=ut.get_if_exist(exifData,"ResolutionUnit")
#             self.imageWidth=ut.get_if_exist(exifData,"ExifImageWidth")
#             self.imageHeight=ut.get_if_exist(exifData,"ExifImageHeight")
            
#             if 'GPSInfo' in exifData:
#                 gps_info = exifData["GPSInfo"]
#                 if gps_info is not None:
#                     # self.GlobalPose=GlobalPose # (structure) SphericalTranslation(lat,long,alt), Quaternion(qw,qx,qy,qz)
#                     latitude=ut.get_if_exist(gps_info, "GPSLatitude")
#                     latReference=ut.get_if_exist(gps_info, "GPSLatitudeRef")
#                     newLatitude=ut.filter_exif_gps_data(latitude,latReference)
#                     longitude= ut.get_if_exist(gps_info, "GPSLongitude")
#                     longReference=ut.get_if_exist(gps_info, "GPSLongitudeRef")
#                     newLongitude=ut.filter_exif_gps_data(longitude,longReference)
#                     self.geospatialTransform=[  newLatitude, 
#                                                 newLongitude,
#                                                 ut.get_if_exist(gps_info, "GPSAltitude")]
#                     self.coordinateSystem='geospatial-wgs84'
            
#             return True
#         else:
#             return False
    
    # def get_metadata_from_xmp_path(self)->bool:
    #     """Read Metadata from .xmp file generated by https://www.capturingreality.com/.

    #     Features:
    #         1. geospatialTransform (np.array(3x1))\n
    #         2. coordinateSystem (str)\n
    #         3. focalLength35mm (float)\n
    #         4. principalPointU (float)\n
    #         5. principalPointV (float)\n
    #         6. cartesianTransform (np.array(4x4))\n

    #     Returns:
    #         bool: True if metadata is sucesfully parsed
    #     """
    #     if self.xmpPath is None or not os.path.exists(self.xmpPath):
    #         return False

    #     if (getattr(self,'principalPointU',None) is not None and
    #         getattr(self,'principalPointV',None) is not None and
    #         getattr(self,'distortionCoeficients',None) is not None and
    #         getattr(self,'geospatialTransform',None) is not None ):         
    #         return True
        
    #     mytree = ET.parse(self.xmpPath)
    #     root = mytree.getroot()                       
        
    #     self.timestamp=ut.get_timestamp(self.xmpPath)
    #     self.name=ut.get_filename(self.xmpPath)
    #     self.subject=self.name
    #     for child in root.iter('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'):

    #         #Attributes
    #         for attribute in child.attrib:
    #             if ('latitude' in attribute and
    #                 'longitude'in attribute and
    #                 'altitude' in attribute):
    #                 lat=ut.xcr_to_lat(child.attrib['{http://www.capturingreality.com/ns/xcr/1.1#}latitude'])
    #                 long=ut.xcr_to_long(child.attrib['{http://www.capturingreality.com/ns/xcr/1.1#}longitude'])
    #                 alt=ut.xcr_to_alt(child.attrib['{http://www.capturingreality.com/ns/xcr/1.1#}altitude'])
    #                 self.geospatialTransform=np.array([lat, long, alt])
    #             if 'Coordinates' in attribute:
    #                 self.coordinateSystem=child.attrib[attribute]
    #             if 'FocalLength35mm' in attribute:
    #                 self.focalLength35mm=ut.xml_to_float(child.attrib[attribute])
    #             if 'PrincipalPointU' in attribute:
    #                 self.principalPointU=ut.xml_to_float(child.attrib[attribute])
    #             if 'PrincipalPointV' in attribute:
    #                 self.principalPointV=ut.xml_to_float(child.attrib[attribute])

    #         #Nodes
    #         rotationnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}Rotation')
    #         rotation=None
    #         if rotationnode is not None:
    #             rotation=ut.string_to_rotation_matrix(rotationnode.text)

    #         positionnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}Position')
    #         translation=None
    #         if positionnode is not None:
    #             translation=np.asarray(ut.string_to_list(positionnode.text))
             
    #         self.cartesianTransform=gmu.get_cartesian_transform(translation=translation,rotation=rotation)
            
    #         coeficientnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}DistortionCoeficients')
    #         if coeficientnode is not None:
    #             self.distortionCoeficients=ut.string_to_list(coeficientnode.text)  
    #     return True   

#     def get_metadata_from_xml_path(self) ->bool:
#         """Extract image metadata from XML Node generated by Agisoft Metashape (self.xmlData and self.subject should be present).

#         Features:
#             1. cartesianTransform (np.array(4x4))\n
#             2. sxy: accuracy in XY (m)\n
#             3. sz: accuracy in Z (m) \n

#         Returns:
#             bool: True if metadata is successfully parsed
#         """
#         if self.xmlPath is None or not os.path.exists(self.xmlPath):
#             return False

#         if (getattr(self,'cartesianTransform',None) is not None and
#             getattr(self,'sxy',None) is not None and
#             getattr(self,'sz',None) is not None ):
#             return True
        
#         self.timestamp=ut.get_timestamp(self.xmlPath)        
#         mytree = ET.parse(self.xmlPath)
#         root = mytree.getroot()          
#         xmlNode = next(cam for cam in root.findall('.//camera') if (ut.get_filename(cam.get('label')) == self.name or ut.get_filename(cam.get('label')) == ut.get_subject_name(self.subject) ))
        
#         if xmlNode:
#             #AGISOFT PARSING 1
#             for child in xmlNode.iter('reference'):  
#                 #get translation
#                 x =  child.get('x')
#                 y =  child.get('y')
#                 z =  child.get('z')
#                 if x and y and z:
#                     translation=np.array([float(x),float(y),float(z)])
#                     self.cartesianTransform= gmu.get_cartesian_transform(translation=translation)
#                 #get rotations
#                 yaw =  child.get('yaw')
#                 pitch =  child.get('pitch')
#                 roll =  child.get('roll')
#                 if yaw and pitch and roll:
#                     rotation = gmu.get_rotation_matrix(np.array([float(yaw),float(pitch),float(roll)]))
#                     self.cartesianTransform=gmu.get_cartesian_transform(translation=translation, rotation=rotation)
#                 #get accuracies
#                 sxy =  child.get('sxy')
#                 if sxy:
#                     self.sxy=float(sxy)
#                 sz =  child.get('sz')
#                 if sz:
#                     self.sz=float(sz)
            
#             #AGISOFT PARSING 2
#             transform=xmlNode.find('transform')
#             if transform is not None:
#                 self.cartesianTransform=ut.string_to_list(transform.text)
#         #! this exception breaks the code
#         # else:
#         #     raise ValueError ('subject not in xml file') 

#     def set_cartesian_transform(self,value):
#         """Set the cartesianTransform of the ImageNode from various inputs.
        
#         Args:
#             1. cartesianTransform(np.ndarray(4x4))\n
#             2. np.ndarray or Vector3dVector (1x3)  \n
#             3. cartesianBounds (np.ndarray (6x1))\n
#             4. np.ndarray or Vector3dVector (8x3 or nx3)\n
#             5. Open3D.geometry
#         """        
#         try: #np.ndarray (4x4) 
#             self._cartesianTransform=np.reshape(value,(4,4))
#         except:
#             try: #np.ndarray or Vector3dVector (1x3)  
#                 self._cartesianTransform=gmu.get_cartesian_transform(translation=np.asarray(value))
#             except:  
#                 try: # cartesianBounds (np.ndarray (6x1))
#                     self._cartesianTransform=gmu.get_cartesian_transform(cartesianBounds=np.asarray(value))
#                 except:
#                     try: # np.ndarray or Vector3dVector (8x3 or nx3)
#                         center=np.mean(np.asarray(value),0)
#                         self._cartesianTransform=gmu.get_cartesian_transform(translation=center)
#                     except:
#                         try: # Open3D.geometry
#                             self._cartesianTransform=gmu.get_cartesian_transform(translation=value.get_center())
#                         except:
#                             raise ValueError('Input must be np.ndarray(6x1,4x4,3x1,nx3), an Open3D geometry or a list of Vector3dVector objects.')


#     def get_cartesian_transform(self) -> np.ndarray:
#         """Get the cartesianTransform from various inputs.
        
#         Args:
#             1. self.cartesianBounds (np.array(6x1))  \n
#             2. self.orientedBounds (np.array(8x3)) or a list of Vector3dVector objects  \n
#             3. orientedBoundingBox\n
#             4. Open3D.geometry

#         Returns:
#             cartesianTransform(np.ndarray(4x4))
#         """
#         if self._cartesianTransform is not None:
#             pass
#         elif getattr(self,'cartesianTransform',None) is not None:
#             self._cartesianTransform = np.reshape(self.cartesianTransform, (4,4))
#         elif getattr(self,'_cartesianBounds',None) is not None:
#             self._cartesianTransform=gmu.get_cartesian_transform(cartesianBounds=self._cartesianBounds)
#         elif getattr(self,'_orientedBounds',None) is not None:
#             center=np.mean(self._orientedBounds,0)
#             self._cartesianTransform=gmu.get_cartesian_transform(translation=center)
#         elif getattr(self,'_orientedBoundingBox',None) is not None:
#             self._cartesianTransform=gmu.get_cartesian_transform(translation=self._orientedBoundingBox.get_center())
#         elif self._resource is not None:
#             self._cartesianTransform=gmu.get_cartesian_transform(translation=self._resource.get_center())
#         else:
#             return None
#         return self._cartesianTransform
    
    def create_rays(self,imagePoints:np.array,depths:np.array=None)->o3d.core.Tensor:
        """Generate a grid a rays from the camera location to a given set of imagePoints.\n
                
        **NOTE**: This function targets a subselection of imagePoints, use o3d.t.geometry.RaycastingScene.create_rays_pinhole if you want a dense raytracing for the full image.
        
        .. image:: ../../../docs/pics/Raycasting_1.PNG
        
        Args:
            imagePoints (np.array[n,2]): imagePoints are conform uv image coordinates system. so top left is (0,0). The camera intrinsic matrix is used to map it to the proper image coordinates.\n

        Returns:
            o3d.core.Tensor (n,6): [:,0:3] is the camera center and [:,3:6] are the directions of the rays towards the imagePoints.
        """
        points=imagePoints
        #validate inputs
        assert points.shape[-1]==2        
        points=np.reshape(points,(-1,2)) if len(points.shape) >2 else points
            
        f=self.focalLength35mm 
        k=self.get_intrinsic_camera_parameters().intrinsic_matrix
        m=self.cartesianTransform 
        t=gmu.get_translation(m)  
        n=points.shape[0]
        
        #transform pixels to image coordinates (rows are first)
        u=+points[:,1]-self.imageWidth/2
        v=+points[:,0]-self.imageHeight/2    
        camera_coordinates=np.vstack((u,v,np.ones(n)))
        
        #transform to world coordinates
        camera_coordinates=np.vstack((camera_coordinates[0:2,:],np.full(n, f).T,np.ones((n,1)).T))
        world_coordinates=m @ camera_coordinates
        
        #normalize direction
        displacement=world_coordinates[0:3,:].T-t
        direction=gmu.normalize_vectors(displacement)
  
        if depths is not None:
            direction=direction * depths[:, np.newaxis]
  
        
        #create rays [camera.center, direction]
        rays=np.hstack((np.full((n,3), t),direction))         
        return rays 


##==========================TO REMOVE
    # def get_path(self) -> str:
    #         """Returns the full path of the resource from this Node.\n

    #         Features:
    #             1. self.graphPath\n
    #             2. self._name or self._subject\n

    #         Returns:
    #             path (str)
    #         """      
    #         if self._path and os.path.exists(self._path):
    #             return self._path
            
    #         elif self._graphPath and (self._name or self._subject):
    #             folder=ut.get_folder_path(self._graphPath)
    #             nodeExtensions=ut.get_node_resource_extensions(str(type(self)))
    #             allSessionFilePaths=ut.get_list_of_files(folder) 
    #             for path in allSessionFilePaths:
    #                 if ut.get_extension(path) in nodeExtensions:
    #                     if self.get_name() in path or self.get_subject() in path :
    #                         self._path = path    
    #                         return self._path
    #             if self._name:
    #                 self._path=os.path.join(folder,self._name+nodeExtensions[0])
    #             else:
    #                 self._path=os.path.join(folder,self._subject+nodeExtensions[0])
    #             return self._path
    #         else:
    #             # print("No file containing this object's name and extension is found in the graphPath folder")
    #             return None
