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
import open3d as o3d 
import numpy as np 
import ezdxf

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

#IMPORT MODULES
from geomapi.nodes import Node
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import geomapi.utils.cadutils as cadu


class LineSetNode (Node):
    def __init__(self,  graph : Graph = None, 
                        graphPath: Path= None,
                        subject : URIRef = None,
                        path : Path= None, 
                        resource = None,
                        dxfPath : Path = None,                        
                        handle : str = None,
                        layer : str = None,
                        dxfType : str = None,
                        getResource : bool = False,
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
        #private attributes 
        self._dxfPath=None 
        self._handle=None
        self._layer=None
        self._dxfType=None
        
        #instance variables
        self.dxfPath=dxfPath
        self.handle=handle
        self.layer=layer
        self.dxfType=dxfType 

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,
                            resource = resource,
                            getResource=getResource,
                            **kwargs) 
                            
        #initialise functionality
        self.get_metadata_from_dxf_path() if self.dxfPath else None # we don't do anything with units

#---------------------PROPERTIES----------------------------

    #---------------------dxfPath----------------------------
    @property
    def dxfPath(self): 
        """The path (Path) of the dxf file. Autocad and BrisCAD files currently have confirmed compatibility."""
        return self._dxfPath

    @dxfPath.setter
    def dxfPath(self,value:Path):
        if value is None:
           pass
        elif Path(value).suffix.upper() in ut.CAD_EXTENSIONS:
            self._dxfPath=Path(value)
        else:
            raise ValueError('dxfPath invalid extension.')
        
    #---------------------handle----------------------------
    @property
    def handle(self): 
        """The handle (str) of the node that originates from a dxf file."""
        return self._handle

    @handle.setter
    def handle(self,value:str):
        if value is None:
            pass
        else:
            self._handle=str(value)
        
    #---------------------layer----------------------------
    @property
    def layer(self): 
        """The CAD layername (str) of the node that originates from a dxf file. 
        
        """
        return self._layer

    @layer.setter
    def layer(self,value:str):
        if value is None:
            pass
        else:
            self._layer=str(value)
            
    #---------------------dxfType----------------------------
    @property
    def dxfType(self): 
        """The DXF dxfType (str) of the node that originates from an ifc file. 
        
        **Note**: This must be a RDF formatted class name.
        """
        return self._dxfType

    @dxfType.setter
    def dxfType(self,value:str):
        if value is None:
            pass
        else:
            self._dxfType=str(value)
            
#---------------------METHODS----------------------------

    def set_resource(self,value):
        """Set self.resource (o3d.geometry.Lineset) of the Node.

        Args:
            - o3d.geometry.Lineset 
            - ezdxf.entities

        Raises:
            ValueError: Resource must be ao3d.geometry.Lineset or ezdxf.entities with geometry.
        """
        if isinstance(value,o3d.geometry.LineSet) and len(value.lines) >=1:
            self._resource = value
        elif  'ezdxf.entities' in str(type(value)):
            g=cadu.ezdxf_entity_to_o3d(value)
            if g is not None and isinstance(g,o3d.geometry.LineSet):
                self._resource=g
                self.pointCount=len(g.points)
                self.lineCount=len(g.lines)
                self.layer=getattr(value.dxf,'layer',None)
                
                self.handle=value.dxf.handle
                self.dxfType=value.dxftype()
                self.orientedBoundingBox=gmu.get_oriented_bounding_box(g)
                self.cartesianTransform= gmu.get_cartesian_transform(translation=g.get_center())
                self.convexHull=gmu.get_convex_hull(g)
                self.name=getattr(value.dxf,'name',None)    #maybe protect this with getattr
                if self._name and self._handle:
                    self.subject= self._name +'_'+self._handle 
                else:
                    self.subject= self._handle
                #colorize
                if hasattr(value.dxf,'color'):
                    self.color=np.array(cadu.get_rgb_from_aci(value.dxf.color))/255 
                    self._resource.paint_uniform_color(np.repeat(value.dxf.color/256,3))

            else:
                raise ValueError('Resource must be ao3d.geometry.Lineset or ezdxf.entities with geometry.')
        else:
            raise ValueError('Resource must be ao3d.geometry.Lineset or ezdxf.entities with geometry.')
    
    def set_path(self, value:Path):
        """sets the path for the Node type. 
        """
        if value is None:
            pass
        elif Path(value).suffix.upper() in ut.CAD_EXTENSIONS:
            self._path = Path(value) 
        else:
            raise ValueError('Invalid extension')

    
    def get_resource(self)->o3d.geometry.TriangleMesh: 
        """Returns the data in the node. If none is present, it will search for the data on using the attributes below.
        
        **NOTE**: The resource is only loaded if it is valid.
        
        Args:
            - self.path
            - self.ifcPath

        Returns:
            o3d.geometry.TriangleMesh or None
        """
        if not self._resource and self.get_path() :
            resource =  o3d.io.read_line_set(str(self._path))
            if len(resource.lines) > 0:
                self.resource=resource
        elif not self._resource and self._dxfPath and os.path.exists(self._dxfPath) and self._handle:
            try:
                doc = ezdxf.readfile(self._dxfPath)
                entity = doc.entitydb.get(self._handle)
                self.resource=cadu.ezdxf_entity_to_o3d(entity)
            except:
                print('Dxf retrieval error')
        if getattr(self,'faceCount',None) is None and self._resource:
            self.lineCount=len(self._resource.lines) 
        if getattr(self,'pointCount',None) is None and self._resource:
            self.pointCount=len(self._resource.points)
        return self._resource  
    
    def get_layer(self):
        """Returns the layer name of the node.
        """
        if self._layer is None and self._dxfPath and os.path.exists(self._dxfPath):
            try:
                doc = ezdxf.readfile(self._dxfPath)
                entity = doc.entitydb.get(self.handle)
                self.layer=getattr(entity.dxf,'layer',None)
            except:
                print('Dxf retrieval error')
        return self._layer
        
    def get_handle(self):
        """Returns the dxf file handle of the node.
        """
        if self._handle is None and self._dxfPath and os.path.exists(self._dxfPath):
            try:
                doc = ezdxf.readfile(self._dxfPath) #this opens rather slow
                first_entity = next(entity for entity in doc.modelspace() if entity.dxftype() in ['LINE','ARC','CIRCLE','SPLINE','POLYLINE','LWPOLYLINE','ELLIPSE'])
                self.handle = first_entity.dxf.handle
            except:
                print('Dxf retrieval error')
        return self._handle    
    
    def get_class_name(self) -> str:
        """Returns the DXF class name (str).
        """
        if self._dxfType:
            pass 
        else:
            self._dxfType='LINE'
        return self._dxfType
        
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
        if extension.upper() not in ut.CAD_EXTENSIONS:
            raise ValueError('Invalid extension')

        # check if already exists
        if directory and os.path.exists(os.path.join(directory,self.subject + extension)):
            self.path=os.path.join(directory,self.subject + extension)
            return True
        elif not directory and self.get_path() and os.path.exists(self.path) and extension.upper() in ut.CAD_EXTENSIONS:
            return True
                    
        #get directory
        if (directory):
            pass    
        elif self.path is not None:    
            directory=Path(self.path).parent            
        elif(self.graphPath): 
            dir=Path(self.graphPath).parent
            directory=os.path.join(dir,'CAD')   
        else:
            directory=os.path.join(os.getcwd(),'CAD')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        self.path=os.path.join(directory,Path(self.subject.toPython()).stem  + extension) #subject.toPython() replaced by get_name()

        #write files
        if o3d.io.write_line_set(str(self.path), self.resource):
            return True
        return False
   
    def get_metadata_from_dxf_path(self) -> bool:
        """Sets the metadata of the node from the dxf file.

        Args:
            - self._dxfPath
            - self._handle

        Args:
            - layer 
            - name 
            - dxfType

        Returns:
            bool: True if exif data is successfully parsed
        """        
        if (not self._dxfPath or 
            not os.path.exists(self._dxfPath) or
            not self.get_handle()):
            return False
        
        if self._graph:
            return True
        
        if self._name and self._dxfType and self._layer :# and getattr(self,'color',None) is not None: #this is super dangerous!
            return True
        
        doc = ezdxf.readfile(self._dxfPath)
        entity = doc.entitydb.get(self._handle)
        
        if entity:
            self.layer=getattr(entity.dxf,'layer',None)
            self.dxfType=entity.dxftype()
            self.name=getattr(entity.dxf,'name',None)
            self.resource=entity 
            #colorize -> layer first
            if hasattr(entity.dxf,'layer'):
                self.color=np.array(cadu.get_rgb_from_aci(doc.layers.get(self._layer).dxf.color))/255
                self.resource.paint_uniform_color(self.color) if self.resource else None
            elif hasattr(entity.dxf,'color'):
                self.color=np.array(cadu.get_rgb_from_aci(entity.dxf.color))/255    
                self.resource.paint_uniform_color(self.color) if self.resource else None
            
            if self._name and self._handle: #this is super dangerous if a graph is already present!
                self.subject= self.name +'_'+self.handle
            return True
        else:
            return False   
        
    def show(self, inline = False):
        super().show()
        if(inline):
            from IPython.display import display
            display(gmu.mesh_to_trimesh(self.resource).show())
        else:
            gmu.show_geometries([self.resource])
    