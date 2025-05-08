"""
**MeshNode** is the Node class that governs the data and metadata of mesh data (Open3D, Trimesh). 

.. image:: ../../../docs/pics/graph_meshes_2.png

This node builds upon the Open3D and Trimesh API for the geometry definitions.

"""
#IMPORT PACKAGES
# from multiprocessing.sharedctypes import Value
from typing import Optional
import open3d as o3d 
import numpy as np 
from rdflib import XSD, Graph, URIRef
import os
from pathlib import Path
import trimesh
#IMPORT MODULES
# from geomapi.nodes import GeometryNode
from geomapi.nodes import Node
import geomapi.utils as ut
from geomapi.utils import rdf_property, GEOMAPI_PREFIXES
import geomapi.utils.geometryutils as gmu

class MeshNode (Node):   
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
                **kwargs):
        """
        Creates a MeshNode. Overloaded function.
        
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to True.

        Args:
            - graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - graphPath (str) :  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)

            - path (Path) : Path to mesh .obj or .ply file (data is not automatically loaded)

            - resource (o3d.geometry.TriangleMesh) : Open3D Triangle mesh data from trimesh or open3d
                
            - getResource (bool, optional= False) : If True, the node will search for its physical resource on drive 
        
        Returns:
            MeshNode : A MeshNode with metadata
        """ 

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
            return len(np.asarray(self.resource.vertices))
        else: 
            return self._pointCount
    
    @pointCount.setter
    def pointCount(self, value):
        if self.resource:
            print("PointCount cannot be set directly when a resource is present")
        self._pointCount = value
        
    #---------------------faceCount----------------------------
    @property
    @rdf_property(datatype=XSD.int)
    def faceCount(self):
        if self.resource:
            return len(np.asarray(self.resource.triangles))
        else: 
            return self._faceCount

    @faceCount.setter
    def faceCount(self, value):
        if self.resource:
            print("FaceCount cannot be set directly when a resource is present")
        self._faceCount = value


#---------------------PROPERTY OVERRIDES----------------------------

    @Node.resource.setter
    def resource(self, value): 
        """Set the self.resource (o3d.geometry.TriangleMesh) of the Node.

        Args:
            - open3d.geometry.TriangleMesh
            - trimesh.Trimesh

        Raises:
            ValueError: Resource must be an o3d.geometry.TriangleMesh with len(resource.triangles) >=1 or an trimesh.Trimesh instance.
        """
        if value is None:
            self._resource = None
        elif isinstance(value,o3d.geometry.TriangleMesh) and len(value.triangles) >=1:
            self._resource = value
        elif isinstance(value,trimesh.base.Trimesh):
            vertices = o3d.utility.Vector3dVector(value.vertices)
            triangles = o3d.utility.Vector3iVector(value.faces)
            self._resource = o3d.geometry.TriangleMesh(vertices, triangles)
        else:
            raise ValueError('Resource must be an o3d.geometry.TriangleMesh with len(resource.triangles) >=1 or an trimesh.Trimesh instance.')

#---------------------METHODS----------------------------

    def _transform_resource(self, transformation: np.ndarray, rotate_around_center: bool):
        """
        Apply a transformation to the mesh resource.

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

    def load_resource(self)->o3d.geometry.TriangleMesh: 
        """Load the resource from the path.
            
        Returns:
            o3d.geometry.TriangleMesh or None
        """
        # Perform path checks
        if(not super().load_resource()):
            return None

        self.resource =  o3d.io.read_triangle_mesh(str(self.path))
        return self.resource 

    def save_resource(self, directory:str=None,extension :str = '.ply') ->bool:
        """Export the resource of the Node.

        Args:
            - directory (str, optional) : directory folder to store the data.
            - extension (str, optional) : file extension. Defaults to '.ply'.

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in the ontology.

        Returns:
            bool: return True if export was successful
        """
        # perform the path check and create the directory
        if not super().save_resource(directory, extension):
            return False

        #write files
        if o3d.io.write_triangle_mesh(str(self.path), self.resource):
            return True
        return False
    
    def show(self, inline = False):
        super().show()
        if(inline):
            from IPython.display import display
            display(gmu.mesh_to_trimesh(self.resource).show())
        else:
            gmu.show_geometries([self.resource])
       
    
