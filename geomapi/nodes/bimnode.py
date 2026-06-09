"""
**BIMNode** is a Python Class to govern the data and metadata of BIM data. 
This node builds upon the [Open3D](https://www.open3d.org/) and [ifcopenshell](https://ifcopenshell.org/) API for the BIM definitions.
Be sure to check the properties defined in those abstract classes to initialise the Node.

This Node is an extension of the MeshNode so it inherits all mesh-based transformation and visualization functionality.

.. image:: ../../../docs/pics/graph_ifc1.png

**IMPORTANT**: The current BIMNode class is designed from a geospatial perspective to 
use in geometric analyses. As such, it's geometry is defined by Open3D.geometry.TriangleMesh objects 
and contains only a skeleton set of IFC Information. Users should use this class to conduct their analyses
and then combine it with the existing IFC files or IFCOWL RDF variants to integrate the results.

"""
#IMPORT PACKAGES
import os
from pathlib import Path
from typing import Optional
import open3d as o3d 
import numpy as np 
import ifcopenshell
import ifcopenshell.geom as geom
import ifcopenshell.util
import ifcopenshell.util.selector
import trimesh
import uuid
from rdflib import XSD, Graph, URIRef
from rdflib.namespace import RDF

#IMPORT MODULES
# from geomapi.nodes import GeometryNode
from geomapi.nodes.meshnode import MeshNode
import geomapi.utils as ut
from geomapi.utils import GEOMAPI_PREFIXES, rdf_property
import geomapi.utils.geometryutils as gmu

class BIMNode (MeshNode):
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
                ifcPath : Path = None,                        
                globalId : str = None,
                className : str = None,
                objectType : str = None,
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

        
        #set properties
        self.ifcPath=ifcPath
        self.globalId=globalId
        self.className=className
        self.objectType=objectType

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

    #---------------------ifcPath----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES["ifc"].ifcPath ,datatype=XSD.string)
    def ifcPath(self): 
        """The path (Path) of the ifc file."""
        return self._ifcPath

    @ifcPath.setter
    def ifcPath(self,value:Path):
        if value is None:
           self._ifcPath = None
        elif Path(value).suffix.upper() in ut.get_node_resource_extensions(self):
            self._ifcPath=Path(value)
        else:
            raise ValueError('ifcPath invalid extension.')

    #---------------------globalId----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES['ifc'].IfcGloballyUniqueId, datatype=XSD.string)
    def globalId(self): 
        """The GlobalId (str) of the node that originates from an ifc file."""
        #if self._globalId:
        #    pass 
        #elif self.ifcPath is not None and os.path.exists(self.ifcPath): # takes first element, not sure if this is good!
        #    ifc = ifcopenshell.open(self.ifcPath)
        #    print(ifc)
        #    ifcElement=next((ifcElement for ifcElement in ifcopenshell.util.selector.filter_elements(ifc,"IfcElement")),None)
        #    print(ifcElement)
        #    self._globalId=ifcElement.GlobalId if ifcElement else None
        #    print("checking if file exists")
        return self._globalId

    @globalId.setter
    def globalId(self,value:str):
        if value is None:
            self._globalId = None
        else:
            self._globalId=str(value)
            
    #---------------------className----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES["ifc"].className)
    def className(self): 
        """The IFC className (str) of the node that originates from an ifc file. 
        
        **Note**: This must be a IFC formatted class name e.g. IfcWall, IfcBeam, IfcSlab, etc.
        """
       #if(self._className is None):
       #    self._className='IfcBuildingElement'
        return self._className

    @className.setter
    def className(self,value:str):
        if value is None:
            self._className = None
        else:
            self._className=str(value)
            
    #---------------------objectType----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES['ifc'].objectType_IfcObject, datatype=XSD.string)
    def objectType(self): 
        """The IFC objectType (str) of the node that originates from an ifc file. 
        
        **Note**: In most software, this is the family or type name of the object, which contains information of the material composition i.e. Floor:232_FL_Concrete CIP 400mm
        """
        return self._objectType

    @objectType.setter
    def objectType(self,value:str):
        if value is None:
            self._objectType = None
        else:
            self._objectType=str(value)

#---------------------PROPERTY OVERRIDES----------------------------
    
    @MeshNode.resource.setter
    def resource(self,value):
        """Set self.resource (o3d.geometry.TriangleMesh) of the Node.

        Args:
            - o3d.geometry.TriangleMesh 
            - trimesh.base.Trimesh
            - ifcopenshell.entity_instance (this also sets the name, subject, etc.)

        Raises:
            ValueError: Resource must be ao3d.geometry.TriangleMesh, trimesh.base.Trimesh or ifcopenshell.entity_instance with len(mesh.triangles) >=1.
        """
        if(value is None):
            self._resource = None
        elif isinstance(value,o3d.geometry.TriangleMesh) and len(value.triangles) >=1:
            self._resource = value
        elif isinstance(value,trimesh.base.Trimesh):
            self._resource = value.as_open3d
        elif isinstance(value,ifcopenshell.entity_instance):
            self._resource= gmu.ifc_to_mesh(value)
            self.name=value.Name
            self.className=value.is_a()
            self.globalId=value.GlobalId
            self.objectType=value.ObjectType
            self.subject= self.name +'_'+self.globalId

        else:
            raise ValueError('Resource must be ao3d.geometry.TriangleMesh, trimesh.base.Trimesh or ifcopenshell.entity_instance with len(mesh.triangles) >=1')
    
#---------------------METHODS----------------------------

