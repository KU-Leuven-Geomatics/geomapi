"""
**LineSetNode** is a Python Class to govern the data and metadata of DXF data. 
This node builds upon the [Open3D](https://www.open3d.org/) and [EZDXF](https://ezdxf.readthedocs.io/en/stable/) API for the Geometry definitions.
Be sure to check the properties defined in those abstract classes to initialise the Node.

.. image:: ../../../docs/pics/graph_cad_1.png

**IMPORTANT**: This Node class is designed to manage the geometry and metadata of a DXF file. Additional information can be linked through the RDF graph.

"""
#IMPORT PACKAGES
import os
from pathlib import Path
from typing import Optional
import open3d as o3d 
import numpy as np 
import ezdxf
#from ezdxf.entities.dxfentity import DXFEntity

from rdflib import XSD, Graph, URIRef
from rdflib.namespace import RDF

#IMPORT MODULES
from geomapi.nodes import Node
import geomapi.utils as ut
from geomapi.utils import rdf_property, GEOMAPI_PREFIXES
import geomapi.utils.geometryutils as gmu
import geomapi.utils.cadutils as cadu


class LineSetNode (Node):
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
                loadResource: Optional[bool] = False,
                dxfPath : Optional[Path] = None,                        
                handle : Optional[str] = None,
                layer : Optional[str] = None,
                dxfType : Optional[str] = None,
                **kwargs):
        """Creates a LineSetNode. Overloaded function.
        
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to True.
        
        Args:
            - subject (RDFlib URIRef) : subject to be used as the main identifier in the RDF Graph
            
            - graph (RDFlib Graph) : Graph with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - graphPath (Path) :  Graph file path with a single subject (if multiple subjects are present, only the first will be used to initialise the Node)
            
            - path (Path) : Path to a Lineset .ply file (data is not automatically loaded)
            
            - resource (o3d.geometry.Lineset, ezdxf.entities) : Open3D Lineset data from [Open3D](https://www.open3d.org/) or [EZDXF](https://ezdxf.readthedocs.io/en/stable/). 
            
            - dxfPath (str|Path) : path to DXF file
            
            - handle (str) : CAD handle
            
            - layer (str) : CAD layername e.g. IFC$1$BT8_Loofboom_Laag_WGI2, etc.
                        
            - getResource (bool, optional= False) : If True, the node will search for its physical resource on drive 
                            
        Returns:
            BIMNode : A BIMNode with metadata 
        """
        
        #instance variables
        self.dxfPath=dxfPath
        self.handle=handle
        self.layer=layer
        self.dxfType=dxfType 

        super().__init__( subject = subject,
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
        
    #---------------------lineCount----------------------------
    @property
    @rdf_property(datatype=XSD.int)
    def lineCount(self):
        if self.resource:
            return len(np.asarray(self.resource.lines))
        else: 
            return self._lineCount

    @lineCount.setter
    def lineCount(self, value):
        if self.resource:
            print("LineCount cannot be set directly when a resource is present")
        self._lineCount = value


    #---------------------dxfPath----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def dxfPath(self): 
        """The path (Path) of the dxf file. Autocad and BrisCAD files currently have confirmed compatibility."""
        return self._dxfPath

    @dxfPath.setter
    def dxfPath(self,value:Path):
        if value is None:
           self._dxfPath = None
        elif Path(value).suffix.upper() == ".DXF":
            self._dxfPath=Path(value)
        else:
            raise ValueError('dxfPath invalid extension.')
        
    #---------------------handle----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def handle(self): 
        """The handle (str) of the node that originates from a dxf file."""
        return self._handle

    @handle.setter
    def handle(self,value:str):
        if value is None:
            self._handle = None
        else:
            self._handle=str(value)
        
    #---------------------layer----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def layer(self): 
        """The CAD layername (str) of the node that originates from a dxf file. 
        
        """
        return self._layer

    @layer.setter
    def layer(self,value:str):
        if value is None:
            self._layer = None
        else:
            self._layer=str(value)
            
    #---------------------dxfType----------------------------
    @property
    @rdf_property(datatype=XSD.string)
    def dxfType(self): 
        """The DXF dxfType (str) of the node that originates from an ifc file. 
        
        **Note**: This must be a RDF formatted class name.
        """
        return self._dxfType

    @dxfType.setter
    def dxfType(self,value:str):
        if value is None:
            self._dxfType = None
        else:
            self._dxfType=str(value)


#---------------------PROPERTY OVERRIDES----------------------------

    @Node.resource.setter
    def resource(self, value): 
        """Set self.resource (o3d.geometry.Lineset) of the Node.

        Args:
            - o3d.geometry.Lineset 
            - ezdxf.entities

        Raises:
            ValueError: Resource must be ao3d.geometry.Lineset or ezdxf.entities with geometry.
        """
        if(value is None):
            self._resource = None
        elif isinstance(value,o3d.geometry.LineSet) and len(value.lines) >=1:
            self._resource = value
        elif  'ezdxf.entities' in str(type(value)):
            g=cadu.ezdxf_entity_to_o3d(value)
            if g is not None and isinstance(g,o3d.geometry.LineSet):
                # DXF specific properties
                self._resource=g
                self.layer=getattr(value.dxf,'layer',None)
                self.handle=value.dxf.handle
                self.dxfType=value.dxftype()

                #colorize
                if hasattr(value.dxf,'color'):
                    self.color=np.array(cadu.get_rgb_from_aci(value.dxf.color))/255 
                    self._resource.paint_uniform_color(np.repeat(value.dxf.color/256,3))

                # Node properties
                self.name=getattr(value.dxf,'name',None)    #maybe protect this with getattr
                if self._name and self._handle:
                    self.subject= self._name +'_'+self._handle 
                else:
                    self.subject= self._handle
                
            else:
                raise ValueError('No geometry found in DXF entity')
        else:
            raise ValueError('Resource must be ao3d.geometry.Lineset or ezdxf.entities with geometry.')
            
#---------------------METHODS----------------------------

    def _transform_resource(self, transformation: np.ndarray, rotate_around_center: bool):
        """
        Apply a transformation to the lineset resource.

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

    def load_resource(self)->o3d.geometry.LineSet: 
        """Load the resource from the path.
            
        Returns:
            o3d.geometry.LineSet or None
        """
        # Perform path checks
        if(not super().load_resource()):
            return None

        self.resource =  o3d.io.read_line_set(str(self.path))
        return self._resource  
    
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
        # perform the path check and create the directory
        if not super().save_resource(directory, extension):
            return False

        #write files
        if o3d.io.write_line_set(str(self.path), self.resource):
            return True
        return False
   
    def show(self, inline = False):
        super().show()
        if(inline):
            from IPython.display import display
            display(gmu.mesh_to_trimesh(self.resource).show())
        else:
            gmu.show_geometries([self.resource])
    