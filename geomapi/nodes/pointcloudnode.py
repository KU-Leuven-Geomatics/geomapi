"""
**PointCloudNode** is the Node class that governs the data and metadata of point cloud data (Open3D, E57, LAS).

.. image:: ../../../docs/pics/graph_pcds2.png

This node builds upon the [Open3D](https://www.open3d.org/), [PYE57](https://github.com/davidcaron/pye57) and [LASPY](https://laspy.readthedocs.io/en/latest/) API for the point cloud definitions. 
"""
#IMPORT PACKAGES
from typing import Optional
import xml.etree.ElementTree as ET
from xmlrpc.client import Boolean 
import open3d as o3d 
import numpy as np 
import os
from scipy.spatial.transform import Rotation as R
from rdflib import XSD, Graph, URIRef
import pye57 
import laspy 
from pathlib import Path

import trimesh

#IMPORT MODULES
# from geomapi.nodes import GeometryNode
from geomapi.nodes import Node
import geomapi.utils as ut
from geomapi.utils import rdf_property, GEOMAPI_PREFIXES
import geomapi.utils.geometryutils as gmu

class PointCloudNode (Node):
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
                e57Index: int = None,
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
        
        # properties
        self.e57Index = e57Index
        # self.e57XmlPath = ut.parse_path(e57XmlPath)

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

        
        

#---------------------PROPERTIES----------------------------

    #---------------------pointCount----------------------------
    @property
    @rdf_property(datatype=XSD.int)
    def pointCount(self):
        if self.resource:
            return len(np.asarray(self.resource.points))
        else: 
            return self._pointCount
    
    @pointCount.setter
    def pointCount(self, value):
        if self.resource:
            print("PointCount cannot be set directly when a resource is present")
        self._pointCount = value

    #---------------------e57Index----------------------------
    @property
    def e57Index(self): 
        """(int) value of the e57Index of an e57 file. Defaults to 0. 
        
        Raises:
            ValueError: e57Index should be positive integer.
        """
        if self._e57Index:
            pass 
        else:
            self._e57Index=0
        return self._e57Index

    @e57Index.setter
    def e57Index(self,value:int):
        if value is None:
            self._e57Index = None
        elif int(value) >=0:
            self._e57Index=int(value)
        else:
            raise ValueError('e57Index should be positive integer.')


#---------------------PROPERTY OVERRIDES----------------------------

    @Node.resource.setter
    def resource(self, value):  
        """Set the self.resource (o3d.geometry.PointCloud) of the Node.

        Args:
            - open3d.geometry.PointCloud
            - pye57.e57.E57 instance
            - pye57 dict instance
            - laspy las file

        Raises:
            ValueError: Resource type not supported or len(resource.points) <1.
        """
        if value is None:
            self._resource = None
        elif isinstance(value,o3d.geometry.PointCloud) and len(value.points) >=1:
            self._resource = value
        elif isinstance(value,pye57.e57.E57):
            self._resource=gmu.e57_to_pcd(value,self.e57Index)
        elif isinstance(value,dict):
            self._resource=gmu.e57_dict_to_pcd(value)
        elif isinstance(value,laspy.lasdata.LasData):
            self._resource=gmu.las_to_pcd(value)
        else:
            raise ValueError('Resource type not supported or len(resource.points) <3.')

#---------------------METHODS----------------------------

    def _set_metadata_from_e57_header(self, cartesianTransform = None, orientedBoundingBox = None, convexHull = None) -> bool:
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

        #TODO dit zorgt voor conflicten aangezien deze waarden altijd een standaardwaarde krijgen in de Node class 
        if self.graph:
            print("Graph is already defined, no need to parse values")
            return True
        
        #if self.cartesianTransform is not None and self.convexHull is not None and self.orientedBoundingBox is not None:
        #    return True

        # try:
        e57 = pye57.E57(str(self.path))   
        header = e57.get_header(self.e57Index)
        
        if 'name' in header.scan_fields:
            self.name=header['name'].value()
            self.subject=self.name
        
        if 'pose' in header.scan_fields and cartesianTransform is None:
            rotation_matrix=None
            translation=None
            if getattr(header,'rotation',None) is not None:
                rotation_matrix=header.rotation
            if getattr(header,'translation',None) is not None:
                translation=header.translation
            self.cartesianTransform=gmu.get_cartesian_transform(rotation=rotation_matrix,translation=translation)
        if 'cartesianBounds' in header.scan_fields and orientedBoundingBox is None:
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
            if(convexHull is None): self.convexHull=points
            
        if 'points' in header.scan_fields:
            self.pointCount=header.point_count
        #     return True
        # except:
        #     raise ValueError('e57 header parsing error. perhaps missing scan_fields/point_fields?')

    def _set_geometric_properties(self, _cartesianTransform=None, _convexHull=None, _orientedBoundingBox=None):
        
        #initialisation
        if (self.path and self.path.suffix =='.e57'):
            self._set_metadata_from_e57_header(cartesianTransform = _cartesianTransform, orientedBoundingBox = _orientedBoundingBox, convexHull = _convexHull)
        
        super()._set_geometric_properties(_cartesianTransform if self.cartesianTransform is None else self.cartesianTransform, 
                                          _convexHull if self.convexHull is None else self.convexHull, 
                                          _orientedBoundingBox if self.orientedBoundingBox is None else self.orientedBoundingBox)

    def _transform_resource(self, transformation: np.ndarray, rotate_around_center: bool):
        """
        Apply a transformation to the pointcloud resource.

        If rotate_around_center is True, the transformation is applied about the mesh's center.
        Otherwise, the transformation is applied as-is.

        Args:
            transformation (np.ndarray): A 4x4 transformation matrix.
            rotate_around_center (bool): Whether to rotate around the mesh's center.
        """
        if rotate_around_center:
            center = self.resource.get_center()
            t1 = np.eye(4)
            t1[:3, 3] = -center
            t2 = np.eye(4)
            t2[:3, 3] = center
            transformation = t2 @ transformation @ t1
        self.resource.transform(transformation)

    def load_resource(self, percentage:float=1.0) -> o3d.geometry.PointCloud:
        """Returns the pointcloud data in the node. If none is present, it will search for the data on drive from path, graphPath, name or subject. 

        Args:
            - percentage (float,optional) : percentage of point cloud to load. Defaults to 1.0 (100%).

        Returns:
            o3d.geometry.PointCloud or None
        """
        # Perform path checks
        super().load_resource()

        if self.path:
            if self.path.suffix == '.pcd':
                resource =  o3d.io.read_point_cloud(str(self.path))
                resource = resource.random_down_sample(percentage)
                self.resource = resource
            elif self.path.suffix == '.e57':
                self.resource = gmu.e57path_to_pcd(self.path, self.e57Index,percentage=percentage) 
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
        # perform the path check and create the directory
        if not super().save_resource(directory, extension):
            return False

        #write files
        if self.path.suffix == '.e57':
            data3D=gmu.get_data3d_from_pcd(self.resource)
            rotation=np.array([1,0,0,0])
            translation=np.array([0,0,0])
            with pye57.E57(self.path, mode="w") as e57_write:
                e57_write.write_scan_raw(data3D, rotation=rotation, translation=translation) 
        elif self.path.suffix == '.pcd':
            o3d.io.write_point_cloud(str(self.path), self.resource)
        elif self.path.suffix == '.las':
            las= gmu.pcd_to_las(self.resource)
            las.write(self.path)
        else:
            return False
        return True
    
    def show(self,  inline = False):
        super().show()
        if(inline):
            from IPython.display import display
            display(trimesh.Scene(gmu.mesh_to_trimesh(self.resource)).show())
        else:
            gmu.show_geometries([self.resource])