"""
**BIMNode** is a Python Class to govern the data and metadata of BIM data. 
This node builds upon the [Open3D](https://www.open3d.org/) and [ifcopenshell](https://ifcopenshell.org/) API for the BIM definitions.
Be sure to check the properties defined in those abstract classes to initialise the Node.

.. image:: ../../../docs/pics/graph_ifc1.png

**IMPORTANT**: The current BIMNode class is designed from a geospatial perspective to 
use in geometric analyses. As such, it's geometry is defined by Open3D.geometry.TriangleMesh objects 
and contains only a skeleton set of IFC Information. Users should use this class to conduct their analyses
and then combine it with the existing IFC files or IFCOWL RDF variants to integrate the results.

"""
#IMPORT PACKAGES
import os
from pathlib import Path
import open3d as o3d 
import numpy as np 
import ifcopenshell
import ifcopenshell.geom as geom
import ifcopenshell.util
import ifcopenshell.util.selector
import trimesh
import uuid
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

#IMPORT MODULES
# from geomapi.nodes import GeometryNode
from geomapi.nodes import Node
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class BIMNode (Node):
    def __init__(self,  graph : Graph = None, 
                        graphPath: Path= None,
                        subject : URIRef = None,
                        path : Path= None, 
                        resource = None,
                        ifcPath : Path = None,                        
                        globalId : str = None,
                        className : str = None,
                        objectType : str = None,
                        getResource : bool = False,
                        **kwargs): 
        """Creates a BIMNode. Overloaded function.
        
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to True.
        
        **Warning**: never attach an IfcElement to a node directly as this is very unstable!

        Args:
            - subject (RDFlib URIRef) : subject to be used as the main identifier in the RDF Graph
            
            - graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - graphPath (Path) :  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - path (Path) : Path to mesh .obj or .ply file (data is not automatically loaded)
            
            - resource (o3d.geometry.TriangleMesh, ifcopenshell.entity_instance) : Open3D Triangle mesh data from trimesh, open3d or ifcopenshell. 
            
            - ifcPath (str|Path) : path to IFC file
            
            - globalId (str) : IFC globalId
            
            - className (str) : IFC className e.g. IfcWall, IfcBeam, IfcSlab, etc.
            
            - objectType (str) : IFC object type e.g.  i.e. Floor:232_FL_Concrete CIP 400mm
            
            - getResource (bool, optional= False) : If True, the node will search for its physical resource on drive 
                            
        Returns:
            BIMNode : A BIMNode with metadata 
        """           
        #private attributes 
        self._ifcPath=None
        self._globalId=None
        self._className=None
        self._objectType=None
        
        #instance variables
        self.ifcPath=ifcPath
        self.globalId=globalId
        self.className=className
        self.objectType=objectType

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            resource = resource,
                            getResource=getResource,
                            **kwargs) 
                     

        #initialisation functionality
        self.get_metadata_from_ifc_path() if self.ifcPath else None
        self.get_class_name()
        
#---------------------PROPERTIES----------------------------

    #---------------------ifcPath----------------------------
    @property
    def ifcPath(self): 
        """The path (Path) of the ifc file."""
        return self._ifcPath

    @ifcPath.setter
    def ifcPath(self,value:Path):
        if value is None:
           pass
        elif Path(value).suffix.upper() in ut.BIM_EXTENSIONS:
            self._ifcPath=Path(value)
        else:
            raise ValueError('ifcPath invalid extension.')

    #---------------------globalId----------------------------
    @property
    def globalId(self): 
        """The GlobalId (str) of the node that originates from an ifc file."""
        return self._globalId

    @globalId.setter
    def globalId(self,value:str):
        if value is None:
            pass
        else:
            self._globalId=str(value)
            
    #---------------------className----------------------------
    @property
    def className(self): 
        """The IFC className (str) of the node that originates from an ifc file. 
        
        **Note**: This must be a IFC formatted class name e.g. IfcWall, IfcBeam, IfcSlab, etc.
        """
        return self._className

    @className.setter
    def className(self,value:str):
        if value is None:
            pass
        else:
            self._className=str(value)
            
    #---------------------objectType----------------------------
    @property
    def objectType(self): 
        """The IFC objectType (str) of the node that originates from an ifc file. 
        
        **Note**: In most software, this is the family or type name of the object, which contains information of the material composition i.e. Floor:232_FL_Concrete CIP 400mm
        """
        return self._objectType

    @objectType.setter
    def objectType(self,value:str):
        if value is None:
            pass
        else:
            self._objectType=str(value)


#---------------------METHODS----------------------------

    def get_subject(self) -> str:
        """Get the subject of the node. If no subject is present, it is gathered from the folowing parameters or given a unique GUID.
        
        Args:
            - self._graph
            - self._path
            - self._name
            - uuid.uuid1() guid
            
        Returns:
            - subject (URIREF)
        """
        #subject
        if self._subject:
            pass
        # self.graph
        elif self._graph:
            self._subject=next(self._graph.subjects(RDF.type))
        #self.path
        elif self._path:
            self.globalId=self._path.stem.split('_')[-1]
            name=self._path.stem
            self.name=name.replace('_'+self.globalId,'')
            self._subject=URIRef('http://'+ut.validate_string(self._path.stem))
        #self_name
        elif self._name:
            self._subject=URIRef('http://'+ut.validate_string(self._name))
        #guid
        else:
            self._name=str(uuid.uuid1())
            self._subject=URIRef('http://'+self._name)            
        return self._subject

    def set_resource(self,value):
        """Set self.resource (o3d.geometry.TriangleMesh) of the Node.

        Args:
            - o3d.geometry.TriangleMesh 
            - trimesh.base.Trimesh
            - ifcopenshell.entity_instance (this also sets the name, subject, etc.)

        Raises:
            ValueError: Resource must be ao3d.geometry.TriangleMesh, trimesh.base.Trimesh or ifcopenshell.entity_instance with len(mesh.triangles) >=1.
        """
        if isinstance(value,o3d.geometry.TriangleMesh) and len(value.triangles) >=1:
            self._resource = value
        elif isinstance(value,trimesh.base.Trimesh):
            self._resource=  value.as_open3d
        elif isinstance(value,ifcopenshell.entity_instance):
            self._resource= gmu.ifc_to_mesh(value)
            self.name=value.Name
            self.className=value.is_a()
            self.globalId=value.GlobalId
            self.objectType=value.ObjectType
            self.faceCount=len(self._resource.triangles)
            self.pointCount=len(self._resource.vertices)
            if self._name and self._globalId:
                self.subject= self.name +'_'+self.globalId 
        else:
            raise ValueError('Resource must be ao3d.geometry.TriangleMesh, trimesh.base.Trimesh or ifcopenshell.entity_instance with len(mesh.triangles) >=1')
    
    def set_path(self, value:Path):
        """sets the path for the Node type. 
        """
        if value is None:
            pass
        elif Path(value).suffix.upper() in ut.MESH_EXTENSIONS:
            self._path = Path(value) 
        else:
            raise ValueError('Invalid extension')

    def get_resource(self)->o3d.geometry.TriangleMesh: 
        """Returns the mesh data in the node. If none is present, it will search for the data on using the attributes below.
        
        **NOTE**: The resource is only loaded if len(resource.triangles) >2.
        
        **NOTE**: If the resource is an ifcopenshell.entity_instance, the mesh is generated from the IFC data. This is an error prone process and not all IFC data can be converted to a mesh.

        Args:
            - self.path
            - self.ifcPath

        Returns:
            o3d.geometry.TriangleMesh or None
        """
        if not self._resource and self.get_path() :
            resource =  o3d.io.read_triangle_mesh(str(self.path))
            if len(resource.triangles)>2:
                self._resource = resource
        elif not self._resource and self.ifcPath and os.path.exists(self.ifcPath):
            try:
                ifc = ifcopenshell.open(self.ifcPath)   
                ifcElement= ifc.by_guid(self.get_globalId())
                self._resource=gmu.ifc_to_mesh(ifcElement)
            except:
                print('mesh=gmu.ifc_to_mesh(ifcElement) error')
        if getattr(self,'faceCount',None) is None and self._resource:
            self.faceCount=len(self._resource.triangles)
        if getattr(self,'pointCount',None) is None and self._resource:
            self.pointCount=len(self._resource.vertices)
        return self._resource  

    def get_class_name(self) -> str:
        """Returns the IFC class name (str).
        """
        if self._className:
            pass 
        else:
            self._className='IfcBuildingElement'
        return self._className
            
    def get_globalId(self) -> str:
        """Returns the ifc globalId (str).
        """
        if self._globalId:
            pass 
        elif os.path.exists(self.ifcPath): # takes first element, not sure if this is good!
            ifc = ifcopenshell.open(self.ifcPath)  
            ifcElement=next((ifcElement for ifcElement in ifcopenshell.util.selector.filter_elements(ifc,"IfcElement")),None)
            self._globalId=ifcElement.GlobalId if ifcElement else None
        return self._globalId

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
        if directory and os.path.exists(os.path.join(directory,self.subject + extension)):
            self.path=os.path.join(directory,self.subject + extension)
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
            directory=os.path.join(dir,'BIM')   
        else:
            directory=os.path.join(os.getcwd(),'BIM')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        self.path=os.path.join(directory,Path(self.subject.toPython()).stem  + extension) #subject.toPython() replaced by get_name()

        #write files
        if o3d.io.write_triangle_mesh(str(self.path), self.resource):
            return True
        return False
    
    
    def get_metadata_from_ifc_path(self) -> bool:
        """Sets the metadata of the node from the ifc file.

        Args:
            - self._ifcPath
            - self._globalId

        Args:
            - globalId 
            - name 
            - className
            - objectType

        Returns:
            bool: True if exif data is successfully parsed
        """        
        if (not self.ifcPath or 
            not os.path.exists(self.ifcPath) or
            not self.get_globalId()):
            return False
        
        if self._graph:
            return True
        
        if self._name and self._className: #this is super dangerous!
            return True
        
        ifc = ifcopenshell.open(self.ifcPath)   
        ifcElement= ifc.by_guid(self.globalId)
        if ifcElement:
            self.name=ifcElement.Name 
            self.className=ifcElement.is_a()   
            self.objectType=ifcElement.ObjectType #get_info()['ObjectType']  #don't we want this full dictionary like ref height ?
            if self._name and self._globalId:
                self.subject= self.name +'_'+self.globalId 
            return True
        else:
            return False
     
    # def get_metadata_from_resource(self) -> bool:
    #     """Returns the metadata from a resource. \n

    #     Args:
    #         1. PointCount
    #         2. faceCount 
    #         3. cartesianTransform
    #         4. cartesianBounds\n
    #         5. orientedBounds \n

    #     Returns:
    #         bool: True if exif data is successfully parsed
    #     """
    #     if (not self.resource or
    #         len(self.resource.triangles) <2):
    #         return False    

    #     try:
    #         if getattr(self,'pointCount',None) is None:
    #             self.pointCount=len(self.resource.vertices)

    #         if getattr(self,'faceCount',None) is None:
    #             self.faceCount=len(self.resource.triangles)

    #         if  getattr(self,'cartesianTransform',None) is None:
    #             center=self.resource.get_center()  
    #             self.cartesianTransform= np.array([[1,0,0,center[0]],
    #                                                 [0,1,0,center[1]],
    #                                                 [0,0,1,center[2]],
    #                                                 [0,0,0,1]])

    #         if getattr(self,'cartesianBounds',None) is  None:
    #             self.cartesianBounds=gmu.get_cartesian_bounds(self.resource)
    #         if getattr(self,'orientedBoundingBox',None) is  None:
    #             self.orientedBoundingBox=self.resource.get_oriented_bounding_box()
    #         if getattr(self,'orientedBounds',None) is  None:
    #             box=self.resource.get_oriented_bounding_box()
    #             self.orientedBounds= np.asarray(box.get_box_points())
    #         return True
    #     except:
    #         raise ValueError('Metadata extraction from resource failed')
       