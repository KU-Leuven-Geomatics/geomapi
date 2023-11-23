"""
MeshNode - a Python Class to govern the data and metadata of mesh data (Open3D, Trimesh). \n

This node builds upon the Open3D and Trimesh API for the point cloud definitions.\n
It inherits from GeometryNode which in turn inherits from Node.\n
Be sure to check the properties defined in those abstract classes to initialise the Node.

"""
#IMPORT PACKAGES
from multiprocessing.sharedctypes import Value
import open3d as o3d 
import numpy as np 
from rdflib import Graph, URIRef
import os

#IMPORT MODULES
from geomapi.nodes import GeometryNode
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class MeshNode (GeometryNode):   
    def __init__(self,  graph : Graph = None, 
                        graphPath:str=None,
                        subject : URIRef = None,
                        path : str=None, 
                        getResource : bool = False,
                        getMetaData : bool = True,
                        **kwargs): 
        """
        Creates a MeshNode. Overloaded function.\n
        This Node can be initialised from one or more of the inputs below.\n
        By default, no data is imported in the Node to speed up processing.\n
        If you also want the data, call node.get_resource() or set getResource() to True.\n

        Args:\n
            0.graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the MeshNode)\n
            
            1.graphPath (str):  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the MeshNode)\n

            2.path (str) : path to mesh file (Note that this node will also contain the data)\n

            3.resource (o3d.geometry.TriangleMesh) : Open3D point cloud file (Note that this node will also contain the data)\n
                
            getResource (bool, optional= False): If True, the node will search for its physical resource on drive \n
            getMetaData (bool, optional= True): If True, the node will attempt to extract geometric metadata from the resource if present (cartesianBounds, etc.) \n
        
        Returns:
            A MeshNode with metadata
        """   
        #private attributes
        self.pointCount = None 
        self.faceCount = None        

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            **kwargs)        

        #initialisation functionality
        if getResource:
            self.get_resource() 
        
        if getMetaData:
            if getResource or self._resource is not None:
                self.get_metadata_from_resource()

#---------------------METHODS----------------------------

    def set_resource(self, value): 
        """Set the self.resource (o3d.geometry.TriangleMesh) of the Node.\n

        Args:
            1. open3d.geometry.TriangleMesh
            2. trimesh.Trimesh

        Raises:
            ValueError: Resource must be an o3d.geometry.TriangleMesh with len(resource.triangles) >=2 or an trimesh.Trimesh instance.
        """
        if 'TriangleMesh' in str(type(value)) and len(value.triangles) >=1:
            self._resource = value
        elif 'Trimesh' in str(type(value)):
            self._resource=value.as_open3d
        else:
            raise ValueError('Resource must be an o3d.geometry.TriangleMesh with len(resource.triangles) >=2 or an trimesh.Trimesh instance.')

    def get_resource(self)->o3d.geometry.TriangleMesh: 
        """Returns the mesh data in the node. \n
        If none is present, it will search for the data on drive from path, graphPath, name or subject. 

        Returns:
            o3d.geometry.TriangleMesh or None
        """
        if self._resource is not None and len(self._resource.triangles)>=2:
            pass
        elif self.get_path():
            resource =  o3d.io.read_triangle_mesh(self.path)
            if len(resource.triangles)>2:
                self._resource = resource
        return self._resource  

    def save_resource(self, directory:str=None,extension :str = '.ply') ->bool:
        """Export the resource of the Node.\n

        Args:
            directory (str, optional): directory folder to store the data.\n
            extension (str, optional): file extension. Defaults to '.ply'.\n

        Raises:
            ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.\n

        Returns:
            bool: return True if export was succesful
        """         
        #check path
        if self.resource is None:
            return False
        
        #validate extension
        if extension not in ut.MESH_EXTENSION:
            raise ValueError('Invalid extension')

        # check if already exists
        if directory and os.path.exists(os.path.join(directory,self.name + extension)):
            self.path=os.path.join(directory,self.get_name() + extension)
            return True
        elif not directory and self.get_path() and os.path.exists(self.path) and extension in ut.MESH_EXTENSION:
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

        self.path=os.path.join(directory,ut.get_filename(self.subject.toPython()) + extension)

        #write files
        if o3d.io.write_triangle_mesh(self.path, self.resource):
            return True
        return False

    def get_metadata_from_resource(self) -> bool:
        """Returns the metadata from a resource. \n

        Features:
            PointCount\n
            faceCount \n
            orientedBoundingBox\n
            cartesianTransform\n
            cartesianBounds\n
            orientedBounds \n

        Returns:
            bool: True if exif data is successfully parsed
        """
        if (not self.resource or
            len(self.resource.triangles) <2):
            return False    

        if getattr(self,'pointCount',None) is None:
            self.pointCount=len(self.resource.vertices)

        if getattr(self,'faceCount',None) is None:
            self.faceCount=len(self.resource.triangles)

        if  getattr(self,'cartesianTransform',None) is None:
            center=self.resource.get_center()  
            self.cartesianTransform= np.array([[1,0,0,center[0]],
                                                [0,1,0,center[1]],
                                                [0,0,1,center[2]],
                                                [0,0,0,1]])

        if getattr(self,'cartesianBounds',None) is None:
            self.cartesianBounds=gmu.get_cartesian_bounds(self.resource)
        if getattr(self,'orientedBoundingBox',None) is  None:
            try:
                self.orientedBoundingBox=self.resource.get_oriented_bounding_box()
            except:
                pass
        if getattr(self,'orientedBounds',None) is None:
            try:
                box=self.resource.get_oriented_bounding_box()
                self.orientedBounds= np.asarray(box.get_box_points())
            except:
                pass
            

       
    
