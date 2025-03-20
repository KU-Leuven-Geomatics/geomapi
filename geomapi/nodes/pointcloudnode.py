"""
**PointCloudNode** is the Node class that governs the data and metadata of point cloud data (Open3D, E57, LAS).

.. image:: ../../../docs/pics/graph_pcds2.png

This node builds upon the [Open3D](https://www.open3d.org/), [PYE57](https://github.com/davidcaron/pye57) and [LASPY](https://laspy.readthedocs.io/en/latest/) API for the point cloud definitions. 
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
import laspy 
from pathlib import Path

#IMPORT MODULES
# from geomapi.nodes import GeometryNode
from geomapi.nodes import Node
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class PointCloudNode (Node):
    def __init__(self,  graph : Graph = None, 
                        graphPath: Path = None,
                        subject : URIRef = None,
                        path : Path = None, 
                        e57Index : int =None, 
                        resource = None,
                        getResource : bool = False,
                        **kwargs):
        """
        Creates a PointCloudNode. Overloaded function.
        
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to True.
        
        Args:
            - graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - graphPath (Path) :  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)

            - path (Path) : path to .pcd, .e57, .las or .laz file (data is not automatically loaded)
            
            - subject (URIRef, optional) : A subject to use as identifier for the Node. If a graph is also present, the subject should be part of the graph.

            - e57Index (int) : index of the scan you want to import from an e57 file. Defaults to 0.

            - resource (o3d.geometry.PointCloud) : Open3D point cloud data parsed from an e57, las or pcd class.
           
            - getResource (bool, optional= False) : If True, the node will search for its physical resource on drive 
                            
        Returns:
            pointcloudnode : A pointcloudnode with metadata 
        """        
        #private attributes
        self._e57Index=None         
        self.pointCount=None
        # self.e57XmlPath=None

        #instance variables
        self.e57Index = e57Index
        # self.e57XmlPath = ut.parse_path(e57XmlPath)

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            resource = resource,
                            getResource=getResource,
                            **kwargs)    
        #initialisation
        #  self.set_metadata_from_e57xml() if self.e57XmlPath else None
        self.get_point_count()
        if (self.path and self.path.suffix =='.e57'):
            self.set_metadata_from_e57_header()
        

#---------------------PROPERTIES----------------------------

    #---------------------e57Index----------------------------
    @property
    def e57Index(self): 
        """(int) value of the e57Index of an e57 file. Defaults to 0. 
        
        Raises:
            ValueError: e57Index should be positive integer.
        """
        return self._e57Index

    @e57Index.setter
    def e57Index(self,value:int):
        if value is None:
            pass
        elif int(value) >=0:
            self._e57Index=int(value)
        else:
            raise ValueError('e57Index should be positive integer.')


#---------------------METHODS----------------------------
    def get_e57Index(self):
        if self._e57Index:
            pass 
        else:
            self._e57Index=0
        return self._e57Index
    

    def set_resource(self,value): 
        """Set the self.resource (o3d.geometry.PointCloud) of the Node.

        Args:
            - open3d.geometry.PointCloud
            - pye57.e57.E57 instance
            - pye57 dict instance
            - laspy las file

        Raises:
            ValueError: Resource type not supported or len(resource.points) <3.
        """
        if isinstance(value,o3d.geometry.PointCloud) and len(value.points) >=3:
            self._resource = value
        elif isinstance(value,pye57.e57.E57):
            self._resource=gmu.e57_to_pcd(value,self.get_e57Index())
        elif isinstance(value,dict):
            self._resource=gmu.e57_dict_to_pcd(value)
        elif isinstance(value,laspy.lasdata.LasData):
            self._resource=gmu.las_to_pcd(value)
        else:
            raise ValueError('Resource type not supported or len(resource.points) <3.')

    def get_resource(self, percentage:float=1.0) -> o3d.geometry.PointCloud:
        """Returns the pointcloud data in the node. If none is present, it will search for the data on drive from path, graphPath, name or subject. 

        Args:
            - percentage (float,optional) : percentage of point cloud to load. Defaults to 1.0 (100%).

        Returns:
            o3d.geometry.PointCloud or None
        """
        if not self._resource and self.get_path():
            if self.path.suffix == '.pcd':
                resource =  o3d.io.read_point_cloud(str(self.path))
                resource = resource.random_down_sample(percentage)
                self.resource  =resource
            elif self.path.suffix == '.e57':
                self.resource = gmu.e57path_to_pcd(self.path, self.get_e57Index(),percentage=percentage) 
            elif self.path.suffix == '.las' or self.path.suffix == '.laz':
                las=laspy.read(self.path)
                self.resource = gmu.las_to_pcd(las,getColors=True,getNormals=True)                
        return self._resource  

    def save_resource(self, directory:Path | str=None,extension :str = '.pcd') ->bool:
        """Export the resource of the Node.

        Args:
            - directory (str, optional) : directory folder to store the data.
            - extension (str, optional) : file extension. Defaults to '.pcd'.

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.

        Returns:
            bool: return True if export was succesful
        """        
        #check resource
        if self.resource is None:
            return False

        #validate extension
        if extension.upper() not in ut.PCD_EXTENSIONS:
            raise ValueError('Invalid extension')

        filename=ut.validate_string(self.name)

        #check if already exists
        if directory and os.path.exists(os.path.join(directory,filename + extension)):
            self.path=os.path.join(directory,filename + extension)
            return True
        elif not directory and self.subject and os.path.exists(self.path) and extension.upper() in ut.MESH_EXTENSIONS:
            return True        
                   
        #get path        
        if directory:
            self.path=os.path.join(directory,filename + extension)
        else:
            if self.get_path():
                directory =self._path.parent
                
            elif self._graphPath: 
                dir=self.graphPath.parent
                directory=os.path.join(dir,'PCD')   
                self.path=os.path.join(dir,filename + extension)
            else:
                directory=os.path.join(os.getcwd(),'PCD')
                self.path=os.path.join(dir,filename + extension)
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        #write files
        if self.path.suffix == '.e57':
            data3D=gmu.get_data3d_from_pcd(self.resource)
            rotation=np.array([1,0,0,0])
            translation=np.array([0,0,0])
            with pye57.E57(self._path, mode="w") as e57_write:
                e57_write.write_scan_raw(data3D, rotation=rotation, translation=translation) 
        elif self.path.suffix == '.pcd':
            o3d.io.write_point_cloud(str(self.path), self.resource)
        elif self.path.suffix == '.las':
            las= gmu.pcd_to_las(self.resource)
            las.write(self.path)
        else:
            return False
        return True
    
    def set_path(self, value:Path):
        """sets the path for the Node type. 
        """
        if value is None:
            pass
        elif Path(value).suffix.upper() in ut.PCD_EXTENSIONS:
            self._path = Path(value) 
        else:
            raise ValueError('Invalid extension')
    
    def get_point_count(self) ->int:
        """Returns the number of points in the resource.
        """
        if self._resource:
            self.pointCount=len(self._resource.points)  
        return self.pointCount
    
    # def get_metadata_from_resource(self) ->bool:
    #     """Returns the metadata from a resource. \n

    #     Args:
    #         PointCount\n
    #         cartesianTransform\n
    #         cartesianBounds\n
    #         orientedBounds \n

    #     Returns:
    #         bool: True if exif data is successfully parsed
    #     """
    #     if (not self.resource or
    #         len(self.resource.points) <=4):
    #         return False     
    #     try:
    #         if getattr(self,'pointCount',None) is None:
    #             self.pointCount=len(self.resource.points)

    #         if  getattr(self,'cartesianTransform',None) is None:
    #             center=self.resource.get_center() 
    #             self.cartesianTransform= np.array([[1,0,0,center[0]],
    #                                             [0,1,0,center[1]],
    #                                             [0,0,1,center[2]],
    #                                             [0,0,0,1]])

    #         # if getattr(self,'cartesianBounds',None) is  None:
    #         #     self.cartesianBounds=gmu.get_cartesian_bounds(self.resource)
    #         # if getattr(self,'orientedBoundingBox',None) is  None:
    #         #     self.orientedBoundingBox=self.resource.get_oriented_bounding_box()
    #         # if getattr(self,'orientedBounds',None) is  None:
    #         #     box=self.resource.get_oriented_bounding_box()
    #         #     self.orientedBounds= np.asarray(box.get_box_points())
    #         return True
    #     except:
    #         raise ValueError('Metadata extraction from resource failed')
    
    def set_metadata_from_e57_header(self) -> bool:
        """Sets the metadata from an e57 header. 

        Args:
            - pointCount
            - name
            - subject
            - cartesianTransform
            - cartesianBounds
            - orientedBoundingBox

        Returns:
            bool: True if meta data is successfully parsed
        """ 

        #TODO dit zorgt voor conflicten aagezien deze waarden altijd een standaardwaarde krijgen in de Node class 
        #if self._graph:
        #    print("Graph is already defined, no need to parse values")
        #    return True
        
        #if self.cartesianTransform is not None and self.convexHull is not None and self.orientedBoundingBox is not None:
        #    return True

        # try:
        e57 = pye57.E57(str(self.path))   
        header = e57.get_header(self.get_e57Index())
        
        if 'name' in header.scan_fields:
            self.name=header['name'].value()
            self.subject=self.name
        
        if 'pose' in header.scan_fields:
            rotation_matrix=None
            translation=None
            if getattr(header,'rotation',None) is not None:
                rotation_matrix=header.rotation_matrix
            if getattr(header,'translation',None) is not None:
                translation=header.translation
            self.cartesianTransform=gmu.get_cartesian_transform(rotation=rotation_matrix,translation=translation)
       
        if 'cartesianBounds' in header.scan_fields:
            c=header.cartesianBounds
            cartesianBounds=np.array([c["xMinimum"].value(),
                                            c["xMaximum"].value(), 
                                            c["yMinimum"].value(),
                                            c["yMaximum"].value(),
                                            c["zMinimum"].value(),
                                            c["zMaximum"].value()])   
            #construct 8 bounding points from cartesianBounds
            points=gmu.get_oriented_bounds(cartesianBounds)
            self.orientedBoundingBox=points
            self.convexHull=points
            
        if 'points' in header.scan_fields:
            self.pointCount=header.point_count
        #     return True
        # except:
        #     raise ValueError('e57 header parsing error. perhaps missing scan_fields/point_fields?')

    def set_metadata_from_e57xml(self) ->bool:
        """Get the metadata from an e57 XML file generated by .e57xmldump.exe. Note that the XML file should not contain the first rule <?xml version="1.0" encoding="UTF-8"?> 
        as this breaks the code

        Args:
            - timestamp
            - path
            - pointCount
            - name
            - cartesianTransform
            - cartesianBounds

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
                        self.cartesianTransform=gmu.get_cartesian_transform(translation=translationVector,rotation=rotationMatrix)
                    # SET POSE FROM cartesianBounds
                    # elif self.cartesianBounds is not None:            
                    #     self.cartesianTransform=gmu.get_cartesian_transform(self.cartesianBounds)

                    pointsnode=e57xml.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}points')
                    if not pointsnode is None:
                        self.pointCount=int(pointsnode.attrib['recordCount'])
            return True
        except:
            raise ValueError("Parsing e57 header failed (maybe some missing metadata?)!")
