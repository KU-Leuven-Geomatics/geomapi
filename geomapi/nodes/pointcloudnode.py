"""
PointCloudNode - a Python Class to govern the data and metadata of point cloud data (Open3D, E57).\n

This node builds upon the Open3D and PYE57 API for the point cloud definitions.\n
It inherits from GeometryNode which in turn inherits from Node.\n
Be sure to check the properties defined in those abstract classes to initialise the Node.

"""
#IMPORT PACKAGES
import xml.etree.ElementTree as ET
from xmlrpc.client import Boolean 
import open3d as o3d 
import numpy as np 
import os
from scipy.spatial.transform import Rotation as R
from rdflib import Graph, URIRef
import pye57 

#IMPORT MODULES
from geomapi.nodes import GeometryNode
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class PointCloudNode (GeometryNode):
    def __init__(self,  graph : Graph = None, 
                        graphPath: str = None,
                        subject : URIRef = None,
                        path : str = None, 
                        e57XmlPath:str = None,
                        e57Index : int =0, 
                        getResource : bool = False,
                        getMetaData : bool = True,
                        **kwargs):
        """
        Creates a PointCloudNode. Overloaded function.\n
        This Node can be initialised from one or more of the inputs below.\n
        By default, no data is imported in the Node to speed up processing.\n
        If you also want the data, call node.get_resource() or set getResource() to True.\n
        
        Args:\n
            0.graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the MeshNode)\n
            
            1.graphPath (str):  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the MeshNode)\n

            2.path (str) : path to .pcd file (Note that this node will also contain the data)\n

            3.path (str) + e57Index (int) : path to e57 file + index of the scan you want to import\n

            4.path (str) + e57Index (int) : path to e57 XML file + index of the scan you want to import\n

            5.resource (o3d.geometry.PointCloud) : Open3D point cloud file (Note that this node will also contain the data)\n
           
            getResource (bool, optional= False): If True, the node will search for its physical resource on drive \n
            getMetaData (bool, optional= True): If True, the node will attempt to extract geometric metadata from the resource if present (cartesianBounds, etc.) \n
                
        Returns:
            A pointcloudnode with metadata 
        """        
        #private attributes
        self._e57Index=0          
        self.pointCount=None
        self.e57XmlPath=None

        #instance variables
        self.e57Index = e57Index
        self.e57XmlPath = ut.parse_path(e57XmlPath)

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            **kwargs)    

        #initialisation functionality
        if getResource:
            self.get_resource()
        
        if getMetaData: 
            if self.get_metadata_from_e57xml():
                pass
            elif self.get_metadata_from_e57_header():
                pass
            elif getResource or self._resource is not None:
                self.get_metadata_from_resource()                

#---------------------PROPERTIES----------------------------

    #---------------------e57Index----------------------------
    @property
    def e57Index(self): 
        """Get the e57Index (int) of the node."""
        return self._e57Index

    @e57Index.setter
    def e57Index(self,value):
        if value is None:
            return 0
        try:
            if int(value) >=0:
                self._e57Index=int(value)
            else:
                raise ValueError('e57Index should be positive integer.')
        except:
            raise ValueError('e57Index should be integer.')

#---------------------METHODS----------------------------
    def get_e57Index(self):
        if self._e57Index:
            pass 
        else:
            self._e57Index=0
        return self._e57Index

    def set_resource(self,value): 
        """Set the self.resource (o3d.geometry.PointCloud) of the Node.\n

        Args:
            1. open3d.geometry.PointCloud
            2. pye57.e57.E57 instance

        Raises:
            ValueError: Resource must be an o3d.geometry.PointCloud with len(resource.points) >=3 or an pye57.e57.E57 instance.
        """
        if 'PointCloud' in str(type(value)) and len(value.points) >=3:
            self._resource = value
        elif 'e57' in str(type(value)):
            self._resource=gmu.e57_to_pcd(value,self.get_e57Index())
        else:
            raise ValueError('Resource must be an o3d.geometry.PointCloud with len(resource.points) >=3 or an pye57.e57.E57 instance.')

    def get_resource(self, percentage:float=0.1) -> o3d.geometry.PointCloud:
        """Returns the pointcloud data in the node. \n
        If none is present, it will search for the data on drive from path, graphPath, name or subject. 

        Args:
            1. self (pointCloudNode)\n
            2. percentage (float,optional): percentage of point cloud to load. Defaults to 1.0 (100%).\n

        Returns:
            o3d.geometry.PointCloud or None
        """
        if self._resource is not None and len(self._resource.points)>4:
            return self._resource
        elif self.get_path():
            if self.path.endswith('pcd'):
                resource =  o3d.io.read_point_cloud(self.path)
                resource = resource.random_down_sample(percentage)
                if len(resource.points)>3:
                    self._resource  =resource
            elif self.path.endswith('e57'):
                self._resource = gmu.e57path_to_pcd(self.path, self.get_e57Index(),percentage=percentage) 
        return self._resource  

    def save_resource(self, directory:str=None,extension :str = '.pcd') ->bool:
        """Export the resource of the Node.\n

        Args:
            directory (str, optional): directory folder to store the data.\n
            extension (str, optional): file extension. Defaults to '.pcd'.\n

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.\n

        Returns:
            bool: return True if export was succesful
        """        
        #check resource
        if self.resource is None:
            return False

        #validate extension
        if extension not in ut.PCD_EXTENSION:
            raise ValueError('Invalid extension')

        filename=ut.validate_string(self.name)

        #check if already exists
        if directory and os.path.exists(os.path.join(directory,filename + extension)):
            self.path=os.path.join(directory,filename + extension)
            return True
        elif not directory and self.subject and os.path.exists(self.path) and extension in ut.MESH_EXTENSION:
            return True        
                   
        #get path        
        if directory:
            self.path=os.path.join(directory,filename + extension)
        else:
            if self.get_path():
                directory = ut.get_folder(self.path)
                
            elif(self.graphPath): 
                dir=ut.get_folder(self.graphPath)
                directory=os.path.join(dir,'PCD')   
                self.path=os.path.join(dir,filename + extension)
            else:
                directory=os.path.join(os.getcwd(),'PCD')
                self.path=os.path.join(dir,filename + extension)
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        #write files
        if self.path.endswith('.e57'):
            data3D=gmu.get_data3d_from_pcd(self.resource)
            rotation=np.array([1,0,0,0])
            translation=np.array([0,0,0])
            with pye57.E57(self._path, mode="w") as e57_write:
                e57_write.write_scan_raw(data3D, rotation=rotation, translation=translation) 
                return True
        else:
            if o3d.io.write_point_cloud(self.path, self.resource):
                return True
        return False
   
    def get_metadata_from_resource(self) ->bool:
        """Returns the metadata from a resource. \n

        Features:
            PointCount\n
            cartesianTransform\n
            cartesianBounds\n
            orientedBounds \n

        Returns:
            bool: True if exif data is successfully parsed
        """
        if (not self.resource or
            len(self.resource.points) <=4):
            return False     
        try:
            if getattr(self,'pointCount',None) is None:
                self.pointCount=len(self.resource.points)

            if  getattr(self,'cartesianTransform',None) is None:
                center=self.resource.get_center() 
                self.cartesianTransform= np.array([[1,0,0,center[0]],
                                                [0,1,0,center[1]],
                                                [0,0,1,center[2]],
                                                [0,0,0,1]])

            if getattr(self,'cartesianBounds',None) is  None:
                self.cartesianBounds=gmu.get_cartesian_bounds(self.resource)
            if getattr(self,'orientedBoundingBox',None) is  None:
                self.orientedBoundingBox=self.resource.get_oriented_bounding_box()
            if getattr(self,'orientedBounds',None) is  None:
                box=self.resource.get_oriented_bounding_box()
                self.orientedBounds= np.asarray(box.get_box_points())
            return True
        except:
            raise ValueError('Metadata extraction from resource failed')
    
    def get_metadata_from_e57_header(self) -> bool:
        """Returns the metadata from a resource. \n

        Features:
            PointCount\n
            guid\n
            cartesianTransform\n
            cartesianBounds\n
            orientedBounds \n

        Returns:
            bool: True if meta data is successfully parsed
        """  
        if (not self._path or
            'e57' not in self._path or
            not os.path.exists(self._path)):
            return False
        
        if (getattr(self,'cartesianBounds',None) is not None and
            getattr(self,'cartesianTransform',None) is not None and
            getattr(self,'pointCount',None) is not None):
            return True

        pye57.e57.SUPPORTED_POINT_FIELDS.update({'nor:normalX' : 'd','nor:normalY': 'd','nor:normalZ': 'd'})
        try:
            e57 = pye57.E57(self.path)   
            header = e57.get_header(self._e57Index)
            if 'pose' in header.scan_fields:
                rotation_matrix=None
                translation=None
                if getattr(header,'rotation',None) is not None:
                    rotation_matrix=header.rotation_matrix
                if getattr(header,'translation',None) is not None:
                    translation=header.translation
                self.cartesianTransform=gmu.get_cartesian_transform(rotation=rotation_matrix,translation=translation)
            if 'name' in header.scan_fields:
                self.name=header['name'].value()
                string=ut.validate_string(self._name)
                if 'file:///' not in string and 'http://' not in string:
                    string='file:///'+string
                self.subject= URIRef(string) 
            if 'cartesianBounds' in header.scan_fields:
                c=header.cartesianBounds
                self.cartesianBounds=np.array([c["xMinimum"].value(),
                                                c["xMaximum"].value(), 
                                                c["yMinimum"].value(),
                                                c["yMaximum"].value(),
                                                c["zMinimum"].value(),
                                                c["zMaximum"].value()])   
            if 'points' in header.scan_fields:
                self.pointCount=header.point_count
            return True
        except:
            raise ValueError('e57 header parsing error. perhaps missing scan_fields/point_fields?')

    def get_metadata_from_e57xml(self) ->bool:
        """Returns the metadata from a resource. \n
        Specifically, an e57 XML file generated by .e57xmldump.exe.\n
        Note that the XML file should not contain the first rule <?xml version="1.0" encoding="UTF-8"?> 
        as this breaks the code

        Features:
            timestamp\n
            path\n
            PointCount\n
            guid\n
            cartesianTransform\n
            cartesianBounds\n

        Returns:
            bool: True if meta data is successfully parsed
        """
        if (getattr(self,'e57XmlPath',None) is None or
            '.xml' not in self.e57XmlPath):
            return False
               
        if (getattr(self,'cartesianBounds',None) is not None and
            getattr(self,'cartesianTransform',None) is not None and
            getattr(self,'pointCount',None) is not None):
            return True

        self.name=ut.get_filename(self.e57XmlPath) +'_'+str(self.e57Index)
        self.subject=self.name
        self.timestamp=ut.get_timestamp(self.e57XmlPath)
        try:
            mytree = ET.parse(self.e57XmlPath)
            root = mytree.getroot()         
            for idx,e57xml in enumerate(root.iter('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}vectorChild')):
                if idx == self.e57Index:
                    cartesianBoundsnode=e57xml.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}cartesianBounds') 
                    if cartesianBoundsnode is not None:
                        try:
                            cartesianBounds=np.array([ut.xml_to_float(cartesianBoundsnode[0].text),
                                                    ut.xml_to_float(cartesianBoundsnode[1].text),
                                                    ut.xml_to_float(cartesianBoundsnode[2].text),
                                                    ut.xml_to_float(cartesianBoundsnode[3].text),
                                                    ut.xml_to_float(cartesianBoundsnode[4].text),
                                                    ut.xml_to_float(cartesianBoundsnode[5].text)])
                            cartesianBounds=cartesianBounds.astype(float)
                            cartesianBounds=np.nan_to_num(cartesianBounds)
                        except:
                            cartesianBounds=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
                    self.cartesianBounds=cartesianBounds

                    #POSE
                    posenode=e57xml.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}pose')
                    if posenode is not None:
                        rotationnode=posenode.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}rotation')
                        if rotationnode is not None:               
                            try:
                                quaternion=np.array([ ut.xml_to_float(rotationnode[3].text),
                                                ut.xml_to_float(rotationnode[0].text),
                                                ut.xml_to_float(rotationnode[1].text),
                                                ut.xml_to_float(rotationnode[2].text) ])
                                quaternion=quaternion.astype(float)   
                                quaternion=np.nan_to_num(quaternion)                
                            except:
                                quaternion=np.array([0,0,0,1])
                            r = R.from_quat(quaternion)
                            rotationMatrix =r.as_matrix()

                        translationnode=posenode.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}translation')
                        if translationnode is not None: 
                            try:
                                translationVector= np.array([ut.xml_to_float(translationnode[0].text),
                                                            ut.xml_to_float(translationnode[1].text),
                                                            ut.xml_to_float(translationnode[2].text)])
                                translationVector=translationVector.astype(float)
                                translationVector=np.nan_to_num(translationVector)       
                            except:
                                translationVector=np.array([0.0,0.0,0.0])
                        self.cartesianTransform=gmu.get_cartesian_transform(rotationMatrix,translationVector)
                    # SET POSE FROM cartesianBounds
                    elif self.cartesianBounds is not None:            
                        self.cartesianTransform=gmu.get_cartesian_transform(self.cartesianBounds)

                    pointsnode=e57xml.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}points')
                    if not pointsnode is None:
                        self.pointCount=int(pointsnode.attrib['recordCount'])
            return True
        except:
            raise ValueError("Parsing e57 header failed (maybe some missing metadata?)!")
