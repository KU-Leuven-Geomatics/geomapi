"""
BIMNode is a Python Class to govern the data and metadata of BIM data. 
This node builds upon the Open3D and IFCOpenShell API for the BIM definitions.
It inherits from GeometryNode which in turn inherits from Node.
Be sure to check the properties defined in those abstract classes to initialise the Node.

**IMPORTANT**: The current BIMNode class is designed from a geospatial perspective to 
use in geometric analyses. As such, it's geometry is defined by Open3D.geometry.TriangleMesh objects 
and contains only a skeleton set of IFC Information. Users should use this class to conduct their analyses
and then combine it with the existing IFC files or IFCOWL RDF variants to integrate the results.

"""
#IMPORT PACKAGES
# from ast import Raise
from pathlib import Path
import open3d as o3d 
import numpy as np 
from rdflib import Graph, URIRef
import os
import ifcopenshell
import ifcopenshell.geom as geom
import ifcopenshell.util
from ifcopenshell.util.selector import Selector


#IMPORT MODULES
from geomapi.nodes import GeometryNode
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class BIMNode (GeometryNode):
    def __init__(self,  graph : Graph = None, 
                        graphPath: str= None,
                        subject : URIRef = None,
                        path : str= None, 
                        ifcPath : str = None,                        
                        globalId : str = None,
                        getResource : bool = False,
                        getMetaData : bool = True,
                        **kwargs): 
        """Creates a BIMNode from one or more of the following inputs. 
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to True.\n

        Args:\n 
            0.subject (RDFlib URIRef): subject to be used as the main identifier in the RDF Graph\n
            1.graph (RDFlib Graph) : RDF Graph with a single subject. if no subject is present, the first subject in the Graph will be used to initialise the Node.\n     
            2.graphPath (str):  RDF Graph file path. if no subject is present, the first subject in the Graph will be used to initialise the Node.\n
            3.path (str) : path to mesh file (Note that this node will also contain the data)\n
            4.resource (o3d.geometry.TriangleMesh, ifcopenshell.entity_instance): Warning, never attach an IfcElement to a node directly as this is very unstable! \n
            5.ifcPath (str) : path to IFC file\n
            6.globalId (str) : IFC globalId\n
            \n
            - getResource (bool, optional= False): If True, the node will search for its physical resource on drive \n
            - getMetaData (bool, optional= True): If True, the node will attempt to extract geometric metadata from the resource if present (cartesianBounds, etc.) \n
                
        Returns:
            A BIMNode with metadata 
        """           
        #private attributes 
        self._ifcPath=None
        self._globalId=None

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            **kwargs) 
                            
        #instance variables
        self.ifcPath=ifcPath
        self.globalId=globalId

        #initialisation functionality
        if getResource:
            self.get_resource() 
        
        if getMetaData:
            self.get_metadata_from_ifc_path()
            if getResource or self._resource is not None:
                self.get_metadata_from_resource()

#---------------------PROPERTIES----------------------------

    #---------------------ifcPath----------------------------
    @property
    def ifcPath(self): 
        """Get the ifcPath (str) of the node."""
        return ut.parse_path(self._ifcPath)

    @ifcPath.setter
    def ifcPath(self,value):
        if value is None:
            return None
        if (ut.get_extension(str(value)) =='.ifc'):
            self._ifcPath=str(value)
        else:
            raise ValueError('self.ifcPath has invalid type, path or extension')

    #---------------------globalId----------------------------
    @property
    def globalId(self): 
        """Get the globalId (str) of the node."""
        return self._globalId

    @globalId.setter
    def globalId(self,value):
        if value is None:
            return None
        try: 
            self._globalId=str(value)
        except:
            raise TypeError('self.globalId should be string compatible')

#---------------------METHODS----------------------------
    def set_resource(self,value):
        """Set self.resource (o3d.geometry.TriangleMesh) of the Node.\n

        Args:
            1. o3d.geometry.TriangleMesh \n
            2. trimesh.base.Trimesh\n
            3. ifcopenshell.entity_instance (this also sets the name, subject, etc.\n

        Raises:
            ValueError: Resource must be ao3d.geometry.TriangleMesh, trimesh.base.Trimesh or ifcopenshell.entity_instance with len(mesh.triangles) >=2.
        """
        if 'TriangleMesh' in str(type(value)) and len(value.triangles) >=2:
            self._resource = value
        elif 'Trimesh' in str(type(value)):
            self._resource=  value.as_open3d
        elif type(value) is ifcopenshell.entity_instance:
            self._resource= gmu.ifc_to_mesh(value)
            self.name=value.Name
            self.className=value.is_a()
            self.globalId=value.GlobalId
            if self.name and self.globalId:
                self.subject= self.name +'_'+self.globalId 
        else:
            raise ValueError('Resource must be ao3d.geometry.TriangleMesh, trimesh.base.Trimesh or ifcopenshell.entity_instance with len(mesh.triangles) >=2')

    def get_resource(self)->o3d.geometry.TriangleMesh: 
        """Returns the mesh data in the node.
        If none is present, it will search for the data on using the attributes below.

        Features:
            1. self.path\n
            2. self.graphPath\n
            3. self.name or self.subject\n

        Returns:
            o3d.geometry.TriangleMesh or None
        """
        if self._resource is not None and len(self._resource.triangles)>=2:
            return self._resource
        elif self.get_path():
            resource =  o3d.io.read_triangle_mesh(self.path)
            if len(resource.triangles)>2:
                self._resource  =resource
        elif self.ifcPath and os.path.exists(self.ifcPath):
            try:
                ifc = ifcopenshell.open(self.ifcPath)   
                ifcElement= ifc.by_guid(self.get_globalId())
                self._resource=gmu.ifc_to_mesh(ifcElement)
            except:
                print('mesh=gmu.ifc_to_mesh(ifcElement) error')
        return self._resource  

    def get_globalId(self):
        """Returns the ifc globalId (str).
        """
        if self._globalId:
            pass 
        elif os.path.exists(self.ifcPath):
            selector = Selector()
            ifc = ifcopenshell.open(self.ifcPath)  
            ifcElement=next(ifcElement for ifcElement in selector.parse(ifc, '.ifcObject') )
            self._globalId=ifcElement.GlobalId
        return self._globalId

    def save_resource(self, directory:str=None,extension :str = '.ply') ->bool:
        """Export the resource of the Node.\n

        Args:
            1. directory (str, optional): directory folder to store the data.\n
            2. extension (str, optional): file extension. Defaults to '.ply'.\n

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
        if directory and os.path.exists(os.path.join(directory,self.subject + extension)):
            self.path=os.path.join(directory,self.subject + extension)
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
            directory=os.path.join(dir,'BIM')   
        else:
            directory=os.path.join(os.getcwd(),'BIM')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        self.path=os.path.join(directory,ut.get_filename(self.subject) + extension) #subject.toPython() replaced by get_name()

        #write files
        if o3d.io.write_triangle_mesh(self.path, self.resource):
            return True
        return False
    
    
    def get_metadata_from_ifc_path(self) -> bool:
        """Returns the metadata from a resource. \n

        Args (self):
            ifcPath\n
            globalId\n

        Features:
            1. PointCount\n
            2. faceCount \n
            3. cartesianTransform\n
            4. cartesianBounds\n
            5. orientedBounds \n
            6. globalId \n
            7. name \n

        Returns:
            bool: True if exif data is successfully parsed
        """        
        if (not self.ifcPath or 
            not os.path.exists(self.ifcPath) or
            not self.get_globalId()):
            return False
        
        if (getattr(self,'name',None) is not None and
            getattr(self,'className',None) is not None):
            return True
        
        ifc = ifcopenshell.open(self.ifcPath)   
        ifcElement= ifc.by_guid(self.globalId)
        if ifcElement:
            self.name=ifcElement.Name 
            self.className=ifcElement.is_a()   
            if self.name and self.globalId:
                self.subject= self.name +'_'+self.globalId 
            return True
        else:
            return False
     
    def get_metadata_from_resource(self) -> bool:
        """Returns the metadata from a resource. \n

        Features:
            1. PointCount\n
            2. faceCount \n
            3. cartesianTransform\n
            4. cartesianBounds\n
            5. orientedBounds \n

        Returns:
            bool: True if exif data is successfully parsed
        """
        if (not self.resource or
            len(self.resource.triangles) <2):
            return False    

        try:
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
       