"""
**MeshNode** is the Node class that governs the data and metadata of mesh data (Open3D, Trimesh). 

.. image:: ../../../docs/pics/graph_meshes2.png

This node builds upon the Open3D and Trimesh API for the geometry definitions.

"""
#IMPORT PACKAGES
# from multiprocessing.sharedctypes import Value
import open3d as o3d 
import numpy as np 
from rdflib import Graph, URIRef
import os
from pathlib import Path
import trimesh
#IMPORT MODULES
# from geomapi.nodes import GeometryNode
from geomapi.nodes import Node
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class MeshNode (Node):   
    def __init__(self,  graph : Graph = None, 
                        graphPath:Path=None,
                        subject : URIRef = None,
                        path : Path=None, 
                        resource = None,
                        getResource: bool = False,
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
        #private attributes
        self.pointCount = None 
        self.faceCount = None        

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            resource = resource,
                            getResource=getResource,
                            **kwargs)    

        #initialisation functionality
        self.get_point_and_face_count()

#---------------------METHODS----------------------------

    def set_resource(self, value): 
        """Set the self.resource (o3d.geometry.TriangleMesh) of the Node.

        Args:
            - open3d.geometry.TriangleMesh
            - trimesh.Trimesh

        Raises:
            ValueError: Resource must be an o3d.geometry.TriangleMesh with len(resource.triangles) >=2 or an trimesh.Trimesh instance.
        """
        if isinstance(value,o3d.geometry.TriangleMesh) and len(value.triangles) >=1:
            self._resource = value
        elif isinstance(value,trimesh.base.Trimesh):
            self._resource=value.as_open3d
        else:
            raise ValueError('Resource must be an o3d.geometry.TriangleMesh with len(resource.triangles) >=2 or an trimesh.Trimesh instance.')

    def get_resource(self)->o3d.geometry.TriangleMesh: 
        """Returns the mesh data in the node. If none is present, it will search for the data on drive from path, graphPath, name or subject. 
        
        Args:
            - self.path
            
        Returns:
            o3d.geometry.TriangleMesh or None
        """
        if not self._resource and self.get_path() :
            resource =  o3d.io.read_triangle_mesh(str(self.path))
            if len(resource.triangles)>2:
                self._resource = resource
        return self._resource  
    
    def set_path(self, value:Path):
        """sets the path for the Node type. 
        """
        if value is None:
            pass
        elif Path(value).suffix.upper() in ut.MESH_EXTENSIONS:
            self._path = Path(value) 
        else:
            raise ValueError('Invalid extension')
        
    def save_resource(self, directory:str=None,extension :str = '.ply') ->bool:
        """Export the resource of the Node.

        Args:
            - directory (str, optional) : directory folder to store the data.
            - extension (str, optional) : file extension. Defaults to '.ply'.

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.

        Returns:
            bool: return True if export was succesful
        """         
        #check path
        if self.resource is None:
            return False
        
        #validate extension
        if extension.upper() not in ut.MESH_EXTENSIONS:
            raise ValueError('Invalid extension')

        # check if already exists
        if directory and os.path.exists(os.path.join(directory,self.name + extension)):
            self.path=os.path.join(directory,self.get_name() + extension)
            return True
        elif not directory and self.get_path() and os.path.exists(self.path) and extension.upper() in ut.MESH_EXTENSIONS:
            return True
                    
        #get directory
        if (directory):
            pass    
        elif self.path is not None:    
            directory=ut.get_folder(self.path)            
        elif(self.graphPath): 
            dir=ut.get_folder(self.graphPath)
            directory=os.path.join(dir,'MESH')   
        else:
            directory=os.path.join(os.getcwd(),'MESH')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        self.path=os.path.join(directory,Path(self.subject.toPython()).stem  + extension) #subject.toPython() replaced by get_name()

        #write files
        if o3d.io.write_triangle_mesh(str(self.path), self.resource):
            return True
        return False


    def get_point_and_face_count(self) ->int:
        """Returns the number of vertices and faces in the resource.
        """
        if self._resource:
            self.pointCount=len(np.asarray(self._resource.vertices))  
            self.faceCount=len(np.asarray(self._resource.triangles))
        return self.pointCount, self.faceCount
    
    
    # def get_metadata_from_resource(self) -> bool:
    #     """Returns the metadata from a resource. \n

    #     Args:
    #         - PointCount
    #         - faceCount 
    #         - orientedBoundingBox
    #         - cartesianTransform
    #         - cartesianBounds
    #         - orientedBounds 

    #     Returns:
    #         bool: True if exif data is successfully parsed
    #     """
    #     if (not self.resource or
    #         len(self.resource.triangles) <2):
    #         return False    

    #     if getattr(self,'pointCount',None) is None:
    #         self.pointCount=len(self.resource.vertices)

    #     if getattr(self,'faceCount',None) is None:
    #         self.faceCount=len(self.resource.triangles)

    #     if  getattr(self,'cartesianTransform',None) is None:
    #         center=self.resource.get_center()  
    #         self.cartesianTransform= np.array([[1,0,0,center[0]],
    #                                             [0,1,0,center[1]],
    #                                             [0,0,1,center[2]],
    #                                             [0,0,0,1]])

    #     if getattr(self,'cartesianBounds',None) is None:
    #         self.cartesianBounds=gmu.get_cartesian_bounds(self.resource)
    #     if getattr(self,'orientedBoundingBox',None) is  None:
    #         try:
    #             self.orientedBoundingBox=self.resource.get_oriented_bounding_box()
    #         except:
    #             pass
    #     if getattr(self,'orientedBounds',None) is None:
    #         try:
    #             box=self.resource.get_oriented_bounding_box()
    #             self.orientedBounds= np.asarray(box.get_box_points())
    #         except:
    #             pass
            

       
    
