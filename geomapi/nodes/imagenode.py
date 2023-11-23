"""
ImageNode is a Python Class to govern the data and metadata of image data (OpenCV, PIL). 

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


#IMPORT MODULES
from geomapi.nodes import Node
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import geomapi.utils.imageutils as it


class ImageNode(Node):
    # class attributes
    
    def __init__(self,  graph : Graph = None, 
                        graphPath:str=None,
                        subject : URIRef = None,
                        path : str=None, 
                        xmpPath: str = None,
                        xmlPath: str = None,
                        getResource : bool = False,
                        getMetaData : bool = True,
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
        self._xmlPath=None
        self._xmpPath=None
        self._orientedBoundingBox=None
        self.imageWidth = None # (int) number of pixels
        self.imageHeight = None  # (int) number of pixels
        self.focalLength35mm = None # (Float) focal length in mm     
        self.keypoints = None # (array) the image keypoints
        self.descriptors = None# (array) the image features

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            **kwargs)   

        #instance variables
        self.xmlPath=xmlPath
        self.xmpPath=xmpPath

        #initialisation functionality
        if getMetaData:
            if self.get_metadata_from_xmp_path():
                pass
            elif self.get_metadata_from_xml_path():
                pass

        if getResource:
            self.get_resource()

        if getMetaData:
            self.get_metadata_from_exif_data()
            if getResource or self._resource is not None:
                self.get_metadata_from_resource()

#---------------------PROPERTIES----------------------------

    #---------------------xmlPath----------------------------
    @property
    def xmlPath(self): 
        """Get the xmlPath (str) of the node."""
        return ut.parse_path(self._xmlPath)

    @xmlPath.setter
    def xmlPath(self,value):
        if value is None:
            return None
        elif (str(value).endswith('xml') ):
            self._xmlPath=str(value)
        else:
            raise ValueError('self.xmlPath has invalid type, path or extension.')    

  #---------------------xmpPath----------------------------
    @property
    def xmpPath(self): 
        """Get the xmpPath (str) of the node."""
        return ut.parse_path(self._xmpPath)

    @xmpPath.setter
    def xmpPath(self,value):
        if value is None:
            return None
        elif (str(value).endswith('xmp') ):
            self._xmpPath=str(value)
        else:
            raise ValueError('self.xmpPath has invalid type, path or extension.')    

#---------------------orientedBoundingBox----------------------------
    @property
    def orientedBoundingBox(self): 
        """Get the orientedBoundingBox of the Node from various inputs. \n

        Args:
            1. Open3D.geometry.OrientedBoundingBox \n
            2. Open3D geometry\n

        Returns:
            orientedBoundingBox (o3d.geometry.OrientedBoundingBox) 
        """
        return self._orientedBoundingBox

    @orientedBoundingBox.setter
    def orientedBoundingBox(self,value):
        if value is None:
            return None
        if 'orientedBoundingBox' in str(type(value)):
            self._orientedBoundingBox=value
        else:    
            try: #geometry
                self._orientedBoundingBox=value.get_oriented_bounding_box()
            except:
                raise ValueError('Input must be orientedBoundingBox (o3d.geometry.OrientedBoundingBox) or an Open3D Geometry.')

#---------------------METHODS----------------------------
   
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
        return self._resource  

    def get_path(self) -> str:
        """Returns the full path of the resource.
        If none is present, it will search for the data on drive from the following inputs.\n

        Args:
            1. self.graphPath \n
            2. self.name \n
            3. self.subject\n

        Returns:
            resource path (str)
        """      
        if self._path and os.path.exists(self._path):
            return self._path
        nodeExtensions=ut.get_node_resource_extensions(str(type(self)))
        if self._graphPath and (self._name or self._subject):
            folder=ut.get_folder_path(self._graphPath)
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if ut.get_extension(path) in nodeExtensions:
                    if self.get_name() in path or self.get_subject() in path :
                        self._path = path    
                        return self._path
            if self._name:
                self._path=os.path.join(folder,self._name+nodeExtensions[0])
            else:
                self._path=os.path.join(folder,self._subject+nodeExtensions[0])
            return self._path
        elif self._xmpPath and os.path.exists(self._xmpPath):
            folder=ut.get_folder_path(self._xmpPath)
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if ut.get_extension(path) in nodeExtensions:
                    if ut.get_filename(self._xmpPath) in path :
                        self.name=ut.get_filename(self._xmpPath)
                        self.subject=self._name
                        self.path = path    
                        return self._path
        elif self._xmlPath and os.path.exists(self._xmlPath):
            folder=ut.get_folder_path(self._xmlPath)
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if ut.get_extension(path) in nodeExtensions:
                    if self.get_name() in path or self.get_subject() in path :
                        self._path = path    
                        return self._path
        else:
            # print("No file containing this object's name and extension is found in the graphPath folder")
            return None

    def get_xmp_path(self)->str: 
        """Returns the xmpPath in the node. 
        If none is present, it will search for the data on drive from the following inputs.\n

        Args:
            1. self.graphPath \n
            2. self.name \n
            3. self.subject\n

        Returns:
            str or None
        """
        if self._xmpPath and os.path.exists(self._xmpPath):
            return self._xmpPath            
        elif self._graphPath and (self._name or self._subject):
            folder=ut.get_folder_path(self._graphPath)
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if ut.get_extension(path).endswith('xmp'):
                    if self.get_name() in path or self.get_subject() in path :
                        self._xmpPath = path    
                        return self._xmpPath
        else:
            return None

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
        if extension not in ut.IMG_EXTENSION:
            raise ValueError('Invalid extension')

        # check if already exists
        if directory and os.path.exists(os.path.join(directory,self.get_name() + extension)):
            self.path=os.path.join(directory,self.get_name() + extension)
            return True
        elif not directory and self.get_path() and os.path.exists(self.path) and extension in ut.IMG_EXTENSION:
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

        self.path=os.path.join(directory,ut.get_filename(self.subject.toPython()) + extension)

        #write files
        if cv2.imwrite(self.path, self.resource):
            return True
        return False
       
    def get_oriented_bounding_box(self)->o3d.geometry.OrientedBoundingBox:
        """Gets the Open3D OrientedBoundingBox of the node from the conical mesh representation based on the 
        cartesianTransform, the focal length at 35mm and a viewing range. \n

        Returns:
            o3d.geometry.orientedBoundingBox
        """    
        if self._orientedBoundingBox is not None:
            pass
        elif self._cartesianTransform is not None:
            mesh=self.get_mesh_geometry()
            self._orientedBoundingBox=mesh.get_oriented_bounding_box()            
        else:
            return None
        return self._orientedBoundingBox

    def get_image_features(self, featureType = "Orb", max = 1000) -> Tuple[np.array, np.array]:
        """Get the keypoints and the descriptors of this Nodes Image resource

        Args:
            featureType (str, optional): The featuretype to detect, use: orb, sift. Defaults to "Orb".
            max (int, optional): The max features to detect. Defaults to 1000.

        Returns:
            Tuple[np.array, np.array]: The keypoints and the descriptors
        """

        if(self.keypoints is None or self.descriptors is None):
            self.keypoints, self.descriptors = it.get_features(self.resource, featureType, max = max)
        return self.keypoints, self.descriptors

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
    
    # def get_oriented_bounding_box(self)->o3d.geometry.OrientedBoundingBox:
    #     """Gets the Open3D geometry from cartesianTransform

    #     Returns:
    #         o3d.geometry.orientedBoundingBox
    #     """
    #     if getattr(self,'orientedBoundingBox',None) is None:                
    #         if getattr(self,'cartesianTransform',None) is not None:
    #             box=o3d.geometry.create_mesh_box(width=1.0, height=1.0, depth=1.0)
    #             self.orientedBoundingBox= box.transform(self.cartesianTransform)
    #         else:
    #             return None
    #     return self.orientedBoundingBox

    def get_mesh_geometry(self, depth:float=10, focalLength35mm:float=24)->o3d.geometry.TriangleMesh:
        """Generate a concical mesh representation using the Image's cartesianTransform and focalLength35mm.\n
    
        .. image:: ../../../docs/pics/virtual_image2.PNG

        Args:
            1. depth (float, optional): Viewing depth of the image. Defaults to 10m.\n
            2. focalLength35mm (float,optional): standardised focal length on 35mm film (w=36mm, h = 24mm)\n

        Returns:
            o3d.geometry.TriangleMesh 
        """
        if self.cartesianTransform is not None:
            radius=35/(focalLength35mm*2)*depth        
            mesh= o3d.geometry.TriangleMesh.create_cone(radius=radius, height=depth, resolution=20, split=1)
            rotation=gmu.get_rotation_matrix(self.cartesianTransform)
            r=R.from_matrix(rotation)
            rz=R.from_euler('xyz' ,[0, 0, 0], degrees=True)
            t=gmu.get_translation(self.cartesianTransform)
            mesh=mesh.translate(t)
            r=rz*r
            # t2=r.as_matrix() * np.array([[1],[0],[0]]) *depth
            A = np.dot( r.as_matrix(),np.array([0,0,-1]) )*depth
            mesh=mesh.translate(A)
            rot=r.as_matrix()
            mesh=mesh.rotate(rot)
            return mesh
        else:
            return None

    def get_virtual_image(self, geometries: o3d.geometry, downsampling:int=2)-> o3d.geometry.Image:
        """Generates a virtual image of a set of geometries given the ImageNode's pose and piholeModel.

        .. image:: ../../../docs/pics/rendering3.PNG


        Args:
            1. geometries (o3d.geometry): geometries to include in the scene of the virtual image.\n
            2. downsampling (int, optional): pixel downsampling of the image both in height and width (each step reduces the density by factor 4). Defaults to 2.

        Returns:
            o3d.geometry.Image or None
        """
        pinholeCamera=self.get_pinhole_camera_parameters(downsampling)
        if pinholeCamera is not None:
            return gmu.generate_virtual_image(geometries,pinholeCamera)
        else:
            return None

    def get_pinhole_camera_parameters(self, downsampling:int=1) -> o3d.camera.PinholeCameraParameters():
        """Returns the intrinsic and extrinsix camera parameters based on the following attributes.

        .. image:: ../../../docs/pics/pinholemodel1.PNG

        Args:
            1. self.imageWidth: width of the image in pixels (u) \n
            2. self.imageHeight: height of the image in pixels (v) \n
            3. self.focalLength35mm: focal length with a standardised Field-of-View.\n 
            4. self.cartesianTransform: the inverted transform equals the external camera pose.\n
            2. downsampling (int, optional): pixel downsampling of the image both in height and width (each step reduces the density by factor 4). Defaults to 2.

        Returns:
            o3d.camera.PinholeCameraParameters()
        """
        param=o3d.camera.PinholeCameraParameters()
        if getattr(self,'cartesianTransform',None) is not None:
            # param.extrinsic=np.linalg.inv(self.cartesianTransform) #! unsure why this was inverted
            param.extrinsic=self.cartesianTransform            
            param.intrinsic=self.get_intrinsic_camera_parameters(downsampling)
            self.pinholeCamera=param
            return self.pinholeCamera
        else:
            return None

    def get_intrinsic_camera_parameters(self, downsampling:int=1) -> o3d.camera.PinholeCameraIntrinsic():
        """Returns the intrinsic camera parameters based on the following attributes.
        
        Args:
            1. self.imageWidth: width of the image in pixels (u). Defaults to 640p \n
            2. self.imageHeight: height of the image in pixels (v). Defaults to 480p  \n
            3. self.focalLength35mm: focal length with a standardised Field-of-View. Defaults to 25mm \n 
            4. self.PrincipalPointU: cx \n
            4. self.PrincipalPointV: cy \n

        Returns:
            o3d.camera.PinholeCameraIntrinsic(width,height,fx,fy,cx,cy)
        """
        #validate inputs
        width=int(self.imageWidth/downsampling) if getattr(self,'imageWidth',None) is not None else 640
        height=int(self.imageHeight/downsampling) if getattr(self,'imageHeight',None) is not None else 480
        f=self.focalLength35mm if getattr(self,'focalLength35mm',None) is not None else 2500

        #! deprecated
        # pixX=width/36 #these are standard 35mm film properties
        # pixY=height/24 #these are standard 35mm film properties
        # fx=pixX*f
        # fy=pixY*f        

        if (getattr(self,'principalPointU',None) is not None and
            getattr(self,'principalPointV',None) is not None ):
            cx=width/2-0.5+self.principalPointU
            cy=height/2-0.5+self.principalPointV
        else:
            cx=width/2-0.5
            cy=height/2-0.5
        pinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width,height,f,f,cx,cy)
        self.intrinsic_matrix = pinholeCameraIntrinsic.intrinsic_matrix
        return pinholeCameraIntrinsic

    def get_metadata_from_exif_data(self) -> bool:
        """Returns the metadata from a resource. \n

        Features:
            1. GPSInfo (geospatialTransform (np.array(3,1))
            2. coordinateSystem (str) \n
            2. DateTime ('%Y-%m-%dT%H:%M:%S')\n
            3. XResolution (int)\n
            4. YResolution (int)\n
            5. ResolutionUnit (int)\n
            6. ExifImageWidth (int)\n
            7. ExifImageHeight (int)\n

        Returns:
            bool: True if meta data is successfully parsed
        """
        if  self.get_path() is None or not os.path.exists(self.get_path()) :
            return False
        
        if getattr(self,'timestamp',None) is None :
            self.timestamp=ut.get_timestamp(self.path)
        
        if getattr(self,'name',None) is None:
            self.name=ut.get_filename(self.path)

        if (getattr(self,'imageWidth',None) is not None and
            getattr(self,'imageHeight',None) is not None and
            getattr(self,'geospatialTransform',None) is not None):
            return True

        # pix = PIL.Image.open(self.path) 
        with PIL.Image.open(self.path) as pix:
            exifData=ut.get_exif_data(pix)

        if exifData is not None:
            self.timestamp=exifData.get("DateTime")
            self.resolutionUnit=exifData.get("ResolutionUnit")
            self.imageWidth=exifData.get("ExifImageWidth")
            self.imageHeight=exifData.get("ExifImageHeight")
            
            if 'GPSInfo' in exifData:
                gps_info = exifData["GPSInfo"]
                if gps_info is not None:
                    # self.GlobalPose=GlobalPose # (structure) SphericalTranslation(lat,long,alt), Quaternion(qw,qx,qy,qz)
                    latitude=gps_info.get("GPSLatitude")
                    latReference=gps_info.get("GPSLatitudeRef")
                    newLatitude=ut.parse_exif_gps_data(latitude,latReference)
                    longitude=gps_info.get( "GPSLongitude")
                    longReference=gps_info.get("GPSLongitudeRef")
                    newLongitude=ut.parse_exif_gps_data(longitude,longReference)
                    self.geospatialTransform=[  newLatitude, 
                                                newLongitude,
                                                gps_info.get("GPSAltitude")]
                    self.coordinateSystem='geospatial-wgs84'
            
            return True
        else:
            return False
    
    def get_metadata_from_xmp_path(self)->bool:
        """Read Metadata from .xmp file generated by https://www.capturingreality.com/.

        Features:
            1. geospatialTransform (np.array(3x1))\n
            2. coordinateSystem (str)\n
            3. focalLength35mm (float)\n
            4. principalPointU (float)\n
            5. principalPointV (float)\n
            6. cartesianTransform (np.array(4x4))\n

        Returns:
            bool: True if metadata is sucesfully parsed
        """
        if self.xmpPath is None or not os.path.exists(self.xmpPath):
            return False

        if (getattr(self,'principalPointU',None) is not None and
            getattr(self,'principalPointV',None) is not None and
            getattr(self,'distortionCoeficients',None) is not None and
            getattr(self,'geospatialTransform',None) is not None ):         
            return True
        
        mytree = ET.parse(self.xmpPath)
        root = mytree.getroot()                       
        
        self.timestamp=ut.get_timestamp(self.xmpPath)
        self.name=ut.get_filename(self.xmpPath)
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
                    self.focalLength35mm=ut.xml_to_float(child.attrib[attribute])
                if 'PrincipalPointU' in attribute:
                    self.principalPointU=ut.xml_to_float(child.attrib[attribute])
                if 'PrincipalPointV' in attribute:
                    self.principalPointV=ut.xml_to_float(child.attrib[attribute])

            #Nodes
            rotationnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}Rotation')
            rotation=None
            if rotationnode is not None:
                rotation=ut.string_to_rotation_matrix(rotationnode.text)

            positionnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}Position')
            translation=None
            if positionnode is not None:
                translation=np.asarray(ut.string_to_list(positionnode.text))
             
            self.cartesianTransform=gmu.get_cartesian_transform(translation=translation,rotation=rotation)
            
            coeficientnode=child.find('{http://www.capturingreality.com/ns/xcr/1.1#}DistortionCoeficients')
            if coeficientnode is not None:
                self.distortionCoeficients=ut.string_to_list(coeficientnode.text)  
        return True   

    def get_metadata_from_xml_path(self) ->bool:
        """Extract image metadata from XML Node generated by Agisoft Metashape (self.xmlData and self.subject should be present).

        Features:
            1. cartesianTransform (np.array(4x4))\n
            2. sxy: accuracy in XY (m)\n
            3. sz: accuracy in Z (m) \n

        Returns:
            bool: True if metadata is successfully parsed
        """
        if self.xmlPath is None or not os.path.exists(self.xmlPath):
            return False

        if (getattr(self,'cartesianTransform',None) is not None and
            getattr(self,'sxy',None) is not None and
            getattr(self,'sz',None) is not None ):
            return True
        
        self.timestamp=ut.get_timestamp(self.xmlPath)        
        mytree = ET.parse(self.xmlPath)
        root = mytree.getroot()          
        xmlNode = next(cam for cam in root.findall('.//camera') if (ut.get_filename(cam.get('label')) == self.name or ut.get_filename(cam.get('label')) == ut.get_subject_name(self.subject) ))
        
        if xmlNode:
            #AGISOFT PARSING 1
            for child in xmlNode.iter('reference'):  
                #get translation
                x =  child.get('x')
                y =  child.get('y')
                z =  child.get('z')
                if x and y and z:
                    translation=np.array([float(x),float(y),float(z)])
                    self.cartesianTransform= gmu.get_cartesian_transform(translation=translation)
                #get rotations
                yaw =  child.get('yaw')
                pitch =  child.get('pitch')
                roll =  child.get('roll')
                if yaw and pitch and roll:
                    rotation = gmu.get_rotation_matrix(np.array([float(yaw),float(pitch),float(roll)]))
                    self.cartesianTransform=gmu.get_cartesian_transform(translation=translation, rotation=rotation)
                #get accuracies
                sxy =  child.get('sxy')
                if sxy:
                    self.sxy=float(sxy)
                sz =  child.get('sz')
                if sz:
                    self.sz=float(sz)
            
            #AGISOFT PARSING 2
            transform=xmlNode.find('transform')
            if transform is not None:
                self.cartesianTransform=ut.string_to_list(transform.text)
        #! this exception breaks the code
        # else:
        #     raise ValueError ('subject not in xml file') 

    def set_cartesianTransform(self,value):
        """Set the cartesianTransform of the ImageNode from various inputs.
        
        Args:
            1. cartesianTransform(np.ndarray(4x4))\n
            2. np.ndarray or Vector3dVector (1x3)  \n
            3. cartesianBounds (np.ndarray (6x1))\n
            4. np.ndarray or Vector3dVector (8x3 or nx3)\n
            5. Open3D.geometry
        """        
        try: #np.ndarray (4x4) 
            self._cartesianTransform=np.reshape(value,(4,4))
        except:
            try: #np.ndarray or Vector3dVector (1x3)  
                self._cartesianTransform=gmu.get_cartesian_transform(translation=np.asarray(value))
            except:  
                try: # cartesianBounds (np.ndarray (6x1))
                    self._cartesianTransform=gmu.get_cartesian_transform(cartesianBounds=np.asarray(value))
                except:
                    try: # np.ndarray or Vector3dVector (8x3 or nx3)
                        center=np.mean(np.asarray(value),0)
                        self._cartesianTransform=gmu.get_cartesian_transform(translation=center)
                    except:
                        try: # Open3D.geometry
                            self._cartesianTransform=gmu.get_cartesian_transform(translation=value.get_center())
                        except:
                            raise ValueError('Input must be np.ndarray(6x1,4x4,3x1,nx3), an Open3D geometry or a list of Vector3dVector objects.')


    def get_cartesian_transform(self) -> np.ndarray:
        """Get the cartesianTransform from various inputs.
        
        Args:
            1. self.cartesianBounds (np.array(6x1))  \n
            2. self.orientedBounds (np.array(8x3)) or a list of Vector3dVector objects  \n
            3. orientedBoundingBox\n
            4. Open3D.geometry

        Returns:
            cartesianTransform(np.ndarray(4x4))
        """
        if self._cartesianTransform is not None:
            pass
        elif getattr(self,'cartesianTransform',None) is not None:
            self._cartesianTransform = np.reshape(self.cartesianTransform, (4,4))
        elif getattr(self,'_cartesianBounds',None) is not None:
            self._cartesianTransform=gmu.get_cartesian_transform(cartesianBounds=self._cartesianBounds)
        elif getattr(self,'_orientedBounds',None) is not None:
            center=np.mean(self._orientedBounds,0)
            self._cartesianTransform=gmu.get_cartesian_transform(translation=center)
        elif getattr(self,'_orientedBoundingBox',None) is not None:
            self._cartesianTransform=gmu.get_cartesian_transform(translation=self._orientedBoundingBox.get_center())
        elif self._resource is not None:
            self._cartesianTransform=gmu.get_cartesian_transform(translation=self._resource.get_center())
        else:
            return None
        return self._cartesianTransform
    
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
    
    # def create_rays(self,imagePoints:np.array)->o3d.core.Tensor:
    #     """Generate a grid a rays from the camera location to a given set of imagePoints.\n
                
    #     **NOTE**: This function targets a subselection of imagePoints, use o3d.t.geometry.RaycastingScene.create_rays_pinhole if you want a dense raytracing for the full image.
        
    #     .. image:: ../../../docs/pics/Raycasting_1.PNG
        
    #     Args:
    #         imagePoints (np.array[n,2]): imagePoints are conform uv image coordinates system. so top left is (0,0). The camera intrinsic matrix is used to map it to the proper image coordinates.\n

    #     Returns:
    #         o3d.core.Tensor (n,6): [:,0:3] is the camera center and [:,3:6] are the directions of the rays towards the imagePoints.
    #     """
    #     points=imagePoints
    #     #validate inputs
    #     assert points.shape[-1]==2        
    #     points=np.reshape(points,(-1,2)) if len(points.shape) >2 else points
            
    #     f=self.focalLength35mm 
    #     k=self.get_intrinsic_camera_parameters().intrinsic_matrix
    #     m=self.cartesianTransform 
    #     t=gmu.get_translation(m)  
    #     n=points.shape[0]
        
    #     #transform pixels to image coordinates (rows are first)
    #     u=points[:,1]-self.imageWidth/2
    #     v=-points[:,0]+self.imageHeight/2    
    #     camera_coordinates=np.vstack((u,v,np.ones(n)))
        
    #     #transform to world coordinates
    #     camera_coordinates=np.vstack((camera_coordinates[0:2,:],np.full(n, f).T,np.ones((n,1)).T))
    #     world_coordinates=m @ camera_coordinates
    #     world_coordinates=gmu.normalize_vectors(world_coordinates[0:3,:].T)
  
    #     #create rays [camera.center, direction(world_coordinates)]
    #     rays=np.hstack((np.full((n,3), t),world_coordinates))         
    #     return rays 