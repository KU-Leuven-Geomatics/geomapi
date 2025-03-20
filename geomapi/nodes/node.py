"""
**Node** is an abstract Python Class to govern the data and metadata of remote sensing data (pcd, images, meshes, orthomosaics). It is the base class for all other node classes. It contains the base RDF graph functionality and I/O from and to RDF files.

.. image:: ../../../docs/pics/ontology_node.png

**IMPORTANT**: The Node class is an archetype class from which specific data classes (e.g., PointCloudNode) inherit. Do not use this class directly if you can use a child class with more functionality.

**Example**: Use create_node() with a resource or a graph and it will automatically create the correct node type.

Goals:
 - Govern the metadata and geospatial information of big data files in an accessible and lightweight format.
 - Serialize the metadata of remote sensing and geospatial files (BIM, Point clouds, meshes, etc.) as RDF Graphs.
 - Attach spatial and temporal analyses through RDF Graph navigation.
"""

#IMPORT PACKAGES
import os
import re
from pathlib import Path 
from typing import List, Optional,Tuple,Union
import uuid
import datetime
import numpy as np
from rdflib import Graph, URIRef, Literal,Namespace,XSD
from rdflib.namespace import RDF
import open3d as o3d 
import copy
from collections import Counter

#IMPORT MODULES
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
from geomapi.utils import GEOMAPI_PREFIXES

class Node:
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
                 getResource: bool = False,
                 **kwargs):
        """
        Creates a Node from one or more of the following inputs.
        
        Args:
            - subject (URIRef, optional) : A subject to use as identifier for the Node. If a graph is also present, the subject should be part of the graph.
            
            - graph (Graph, optional) : An RDF Graph to parse.
            
            - graphPath (Path, optional) : The path of an RDF Graph to parse. If no subject is provided, the first subject of the graph is retained.
            
            - name (str, optional) : A name of the Node. This is not a unique identifier but serves as non-functional description.
            
            - path (Path, optional) : A filepath to a resource.
            
            - timestamp (str, optional) : Timestamp for the node.
            
            - resource (optional) : Resource associated with the node.
            
            - cartesianTransform (np.ndarray, optional) : The (4x4) transformation matrix.
            
            - orientedBoundingBox (o3d.geometry.OrientedBoundingBox, optional) : The oriented bounding box of the node.
            
            - convexHull (o3d.geometry.TriangleMesh, optional) : The convex hull of the node.            
            
        Returns:
            Node: An instance of the Node class.
        """
        #private attributes 
        self._subject=None
        self._graph=None
        self._graphPath=None 
        self._path=None
        self._name=None
        self._timestamp=None 
        self._resource=None 
        self._cartesianTransform=None
        self._orientedBoundingBox=None
        self._convexHull=None

        #instance variables (protected inputs)       
        self.subject=subject
        self.graphPath=graphPath
        self.graph=graph
        self.path=path        
        self.name=name
        self.timestamp=timestamp
        self.resource=resource 
        self.orientedBoundingBox=orientedBoundingBox
        self.convexHull=convexHull
        self.cartesianTransform=cartesianTransform

        self.initialize(getResource,kwargs)

    def initialize(self,getResource, kwargs):
        """ Initializes the Node by setting the attributes from the graph, path, or name. It also sets the subject, name, and timestamp if they are not present. Finally, it updates the geometries from highest to lowest detailing."""
        
        #if graphPath is present, parse and set graph
        if not self._graph and self._graphPath and os.path.exists(self._graphPath):
            graph=Graph().parse(self._graphPath)
            self.graph=ut.get_subject_graph(graph,self._subject) 
            self.subject=next(self._graph.subjects(RDF.type))
        
        if self._graph:
            self.set_attributes_from_graph()
            
        #if path is present, set name and timestamp
        if getResource:
            self.get_resource()
            
        #update attributes if they are None
        self.get_subject()
        self.get_name()        
        self.get_timestamp()
        
        #update geometries from highest to lowest detailing
        self.get_cartesian_transform()
        self.get_convex_hull() 
        self.get_oriented_bounding_box()
                   

        self.__dict__.update(kwargs)

#---------------------PROPERTIES----------------------------

    #---------------------PATH----------------------------
    @property
    def path(self): 
        """Path (Path) of the resource of the node. If no path is present, you can use `get_path()` to reconstruct the path from either the graphPath or working directory.
        
        Args:
            - value (str or Path): The new path for the node.
        
        Raises:
            ValueError: If the path has an invalid type, path, or extension.
        
        """
        return self._path
    
    @path.setter 
    def path(self,value: Optional[Path]):        
        if value is None:
            pass
        else:
            self.set_path(value)   

    #---------------------NAME----------------------------
    @property
    def name(self):
        """The name (str) of the node. This can include characters that the operating
        system does not allow. If no name is present, you can use `get_name()` to construct a name from the subject or path.

        Args:
            - self.path
            - self.subject
        """        
        return self._name

    @name.setter
    def name(self, value: Optional[str]):
        if value is None:
            pass
        else:            
            self._name = str(value)

    #---------------------TIMESTAMP----------------------------
    @property
    def timestamp(self) -> str:
        """The timestamp (str(yyyy-MM-ddTHH:mm:ss)) of the node. If no timestamp is present, use `get_timestamp()` to gather the timestamp from the path or graphPath.

        Features:
            - self.path
            - self.graphPath
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self,timestamp):
        if timestamp is None:
            pass
        elif timestamp:
            self._timestamp=ut.validate_timestamp(timestamp)
        else:
            raise ValueError('timestamp should be str(yyyy-MM-ddTHH:mm:ss)')

    #---------------------GRAPHPATH----------------------------    
    @property
    def graphPath(self) -> str:
        """The path (Path) of graph of the node. This can be both a local file or a remote URL.""" 
        return self._graphPath #ut.parse_path(self._graphPath)

    @graphPath.setter
    def graphPath(self,value):
        if value is None:
            pass
        elif any(Path(value).suffix.upper()==extension for extension in ut.RDF_EXTENSIONS): 
            self._graphPath=Path(value)
        else:
            raise ValueError('GraphPath parsing error due to invalid extension or syntax. Only .ttl is currently proofed')    

    #---------------------GRAPH----------------------------    
    @property
    def graph(self) -> Graph:
        """The Graph (RDFLib.Graph) of the node. If no graph is present, you can use `get_graph()` to parse the graph from a graphPath. Alternatively, you can use `to_graph()` to serialize the Nodes attributes to RDF. 
        """       
        return self._graph

    @graph.setter
    def graph(self,value):
        if value is None:
            pass
        else:
            self.set_graph(value)

    #---------------------SUBJECT----------------------------    
    @property
    def subject(self) -> URIRef:
        """The subject (RDFLib.URIRef) of the node. If no subject is present, you can use `get_subject()` to construct it from a graph, name or path. Otherwise, a random guid is generated.
        
        Features:
            - self.name
            - self.graph
            - self.path
        """
        return self._subject

    @subject.setter
    def subject(self,value):
        if value is None:
            pass
        else:
            self.set_subject(value)            
            
    
    #---------------------RESOURCE----------------------------    
    @property
    def resource(self):
        """The resource (mesh, image, etc.) of the node. If no resource is present, you can use `get_resource()`, to load the resource from a path or search it through the name and graphpath. 

        Inputs:
            - self.path
            - self.name
            - self.graphPath
        """        
        return self._resource

    @resource.setter
    def resource(self,value):
        if value is None:
            pass
        else:
            self.set_resource(value)

    @resource.deleter
    def resource(self):
        self._resource=None
         
    #---------------------CARTESIANTRANSFORM----------------------------    
    @property
    def cartesianTransform(self) -> np.ndarray:
        """
        The (4x4) transformation matrix of the node containing the translation & rotation. If no matrix is present, you can use `get_cartesian_transform()`, to gather it from the resource, orientedBoundingBox, or convexHull.
        
        **NOTE**: The cartesianTransform is specific for every nodeType. For geometry Nodes, it is located at the center of the geometry and the axes are with the world coordinate system. For the image-based Nodes, it is located at the center of the image and the axes are aligned with the image coordinate system (see Figure). The default cartesianTransform is the identity matrix.
        
        .. image:: ../../../docs/pics/imagenode.png

        **ROTATIONS**: rotations are (+) counterclockwise and are applied in the following order: z,y,x if no rotation matrix is present.
        
        Examples:
            - The pose of a mesh is determined by the average of its bounding vertices.
            - The pose of an image is at the center of the lens complex.
        
        Returns:
            numpy.ndarray: The (4x4) transformation matrix.
        """
        return self._cartesianTransform

    @cartesianTransform.setter
    def cartesianTransform(self, value: Optional[Union[np.ndarray, List[float]]]):
        """
        Sets the cartesianTransform for the Node. 

        Args:
            - (4x4) full transformation matrix
            - (3x1) translation vector

        Raises:
            ValueError: If the input is not a valid numpy array of shape (4,4) or (3,).
        """
        if value is None:
            pass
        else:
            self.set_cartesian_transform(value)
            
    #---------------------ORIENTEDBOUNDINGBOX----------------------------
    @property
    def orientedBoundingBox(self) -> o3d.geometry.OrientedBoundingBox: 
        """
        The o3d.orientedBoundingBox of the Node containing the bounding box of the geometry. If no box is present, you can use `get_oriented_bounding_box()`, to gather it from the resource, cartesianTransform or convexHull.

        .. image:: ../../../docs/pics/GEOMAPI_metadata3.png
        
        **POINTORDER**:    [[xmin ymin zmin]
                            [xmax ymin zmin]
                            [xmin ymax zmin]
                            [xmin ymin zmax]
                            [xmax ymax zmax]
                            [xmin ymax zmax]
                            [xmax ymin zmax]
                            [xmax ymax zmin]]
        
        **NOTE**: The orientedBoundingBox is a 3D box that is aligned with the geometry's principal axes. It is used to determine the geometry's position, orientation, and scale.
        
        **Parameters**:
            - center (np.array(3)): The center of the box.
            - extent (np.array(3)): The length, width, and height of the box.
            - R (np.array(3x3)): The rotation matrix of the box.
        
        **Features**:
            - LineSetNode, MeshNode, BIMNode, PointCloudNode: The orientedBoundingBox encloses the geometry according to the resource's principal axes.
            - ImageNode: The orientedBoundingBox encloses the image centerpoint and it's 4 corner points projected up to a default distance (20m) from the centerpoint.
            - OrthomosaicNode: The orientedBoundingBox encloses the 4 corner points of the mosaic projected up and down up to a default distance of (20m)
            - PanoNode: The orientedBoundingBox the ellipsoide of the panorama projected up to a default distance of (20m) from the centerpoint.

        Inputs:
            - Open3D.geometry.OrientedBoundingBox
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector
            - 9 parameters $[x,y,z,e_x,e_y,e_z, R_x,R_y,R_z]$
            - custom parameters are gathered for the ImageNode, OrthomosaicNode, and PanoNode.
            

        Returns:
            o3d.geometry.OrientedBoundingBox: The oriented bounding box of the node.
        """
        return self._orientedBoundingBox

    @orientedBoundingBox.setter
    def orientedBoundingBox(self, value):
        if value is None:
            pass
        else:
            self.set_oriented_bounding_box(value)

    @orientedBoundingBox.deleter
    def orientedBoundingBox(self):
        self._orientedBoundingBox=None
#---------------------CONVEXHULL----------------------------
    @property
    def convexHull(self) -> o3d.geometry.TriangleMesh:
        """
        The convex hull of the Node containing the bounding hull of the geometry. If no convex hull is present, you can use `get_convex_hull()`, to gather it from the resource, cartesianTransform or orientedBoundingBox.

        .. image:: ../../../docs/pics/GEOMAPI_metadata3.png
        
        **NOTE**: The convex hull is a 3D mesh that encloses the geometry. It is used to determine the geometry's outer shape and volume.
        
        **Parameters**:
            - vertices (np.array): The vertices of the mesh.
            - triangles (np.array): The triangles of the mesh.
            
        **Features**:
            - LineSetNode, MeshNode, BIMNode, PointCloudNode: The convexHull encloses the geometry according to the resource's vertices or points.
            - ImageNode: The convexHull encloses the image centerpoint and it's 4 corner points projected up to a default distance (20m) from the centerpoint. As such, it has a pyramid shape.
            - OrthomosaicNode: The convexHull encloses the 4 corner points of the mosaic projected up and down up to a default distance of (20m). As such, it has a boxlike shape.
            - PanoNode: The convexHull the ellipsoide of the panorama projected up to a default distance of (20m) from the centerpoint. As such, it has a ellipsoid shape.

        Inputs:
            - Open3D.geometry.TriangleMesh
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector
            - custom parameters are gathered for the ImageNode, OrthomosaicNode, and PanoNode.

        Returns:
            o3d.geometry.TriangleMesh: The convex hull of the node.
        """
        return self._convexHull

    @convexHull.setter
    def convexHull(self, value):
        if value is None:
            pass
        else:
            self.set_convex_hull(value)

    @convexHull.deleter
    def convexHull(self):
        self._convexHull=None        
#---------------------METHODS----------------------------     
 
    def set_attributes_from_graph(self,overwrite:bool=False):
        """Helper function to convert graph literals to node attributes. 

        Args:
            - self._graph (RDFlib.Graph):  Graph to parse
            - self._subject (RDFlib.URIRef): The subject to parse the graph for
            - overwrite (bool, optional): Overwrite existing attributes or not. Defaults to False.
        """       
       
        def handle_path(attr, obj):
            path=Path(obj)
            
            # Handle relative path
            if not path.is_absolute():
                path = (self._graphPath.parent / path).resolve() if self._graphPath else path.resolve()
            setattr(self, attr, path) if getattr(self, attr,None) is None or overwrite  else None #don't overwrite existing paths

        attr_handlers = { #this is dealing with exceptions
            'type': lambda attr, obj: setattr(self, 'className', ut.get_ifcopenshell_class_name(obj)) if 'IFC' in obj else None, #not sure if this works
            'label': lambda attr, obj: setattr(self, 'name', obj),
            'created': lambda attr, obj: setattr(self, 'timestamp', obj),
            # 'objectType': lambda attr, obj: setattr(self, 'objectType', obj.split('/')[-1]), 
            'IfcGloballyUniqueId': lambda attr, obj: setattr(self, 'globalId', obj),
            'objectType_IfcObject': lambda attr, obj: setattr(self, 'objectType', obj),
            'imageLength': lambda attr, obj: setattr(self, 'imageHeight', obj), #not sure
            'hasPart': lambda attr, obj: setattr(self, 'linkedSubjects', [str(obj) for obj in self._graph.objects(subject=self._subject, predicate=GEOMAPI_PREFIXES['geomapi'].hasPart)]),
        }
        # Count occurrences of each predicate
        predicate_counts = Counter(predicate for predicate, _ in self._graph.predicate_objects(subject=self._subject))

        for predicate, object in self._graph.predicate_objects(subject=self._subject):
            #get attribute
            attr = ut.get_attribute_from_predicate(self._graph, predicate)
            
            #get datatype
            datatype=getattr(object,'datatype',None)
            
            #convert object to python datatype
            if datatype is not None:
                object=ut.apply_method_to_object( datatype, object)
            else:
                object=ut.literal_to_python(object)                 
            
            #set attribute
            if attr in attr_handlers:
                attr_handlers[attr](attr, object)
            elif re.search('path', attr, re.IGNORECASE):
                handle_path(attr, object)            
            else:
                # if predicate occurs multiple times, append to list
                existing_value = getattr(self, attr, None)
                if predicate_counts[predicate] > 1:
                    # If predicate occurs multiple times, use a list
                    if existing_value is None:
                        setattr(self, attr, [object])
                    elif isinstance(existing_value, list):
                        existing_value.append(object)
                    else:
                        setattr(self, attr, [existing_value, object])
                else:
                    setattr(self, attr, object)
                # setattr(self, attr,object ) #if getattr(self, attr,None) is None or overwrite else None #don't overwrite existing attributes
                    
    def set_subject(self, value: Union[URIRef, str]):
        """Set the subject for the node, ensuring it is a valid URIRef or string compatible with RDF and Windows."""
        if isinstance(value, URIRef):
            self._subject=value
        else: 
            string=str(value)
            prefix='http://'
            if 'file:///' in string:
                string=string.replace('file:///','')
                prefix='file:///'
            elif 'http://' in string:
                string=string.replace('http://','')
                prefix='http://' 
            elif 'https://' in string:
                string=string.replace('https://','')
                prefix='https://' 
            self._subject=URIRef(prefix+ut.validate_string(string))  
        
    def get_subject(self) -> str:
        """Get the subject of the node. If no subject is present, it is gathered from the folowing parameters or given a unique GUID.
        
        Features:
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
            self._name=self._path.stem 
            self._subject=URIRef('http://'+ut.validate_string(self._name))
        #self_name
        elif self._name:
            self._subject=URIRef('http://'+ut.validate_string(self._name))
        #guid
        else:
            self._name=str(uuid.uuid1())
            self._subject=URIRef('http://'+self._name)            
        return self._subject

    def get_timestamp(self):
        """Get the timestamp (str) of the Node. If no timestamp is present, it is gathered from the folowing parameters.

        Features:
            - self._timestamp
            - self._path
            - self._graphPath
            - datetime.datetime.now()
        
        Returns:
            timestamp (str): '%Y-%m-%dT%H:%M:%S'
        """
        if self._timestamp is None:
            if self._path and os.path.exists(self._path):
                self._timestamp=ut.get_timestamp(self._path)  
            elif self._graphPath and os.path.exists(self._graphPath):
                self._timestamp=ut.get_timestamp(self._graphPath)  
            else:
                self._timestamp=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        return self._timestamp

    def get_name(self) -> str:
        """Returns the name (str) of the Node. If no name is present, it is gathered from the following parameters.
        
        Features:
            - self._name
            - self._path
            - self._subject

        Returns:
           name (str)
        """
        if self._name is None:
            if self._path:                
                self._name=Path(self._path).stem 
            elif self._subject:                     
                self._name=ut.get_subject_name(self._subject)
        return self._name

    def set_cartesian_transform(self, value):
        """Set the cartesianTransform for the Node from various inputs. This is overwritten per node type.

        Args:
            - (4x4) full transformation matrix
            - (3x1) translation vector

        Raises:
            ValueError: If the input is not a valid numpy array of shape (4,4) or (3,).
        """
        if isinstance(value, np.ndarray) and value.shape == (4, 4):
            self._cartesianTransform = value
        else:
            value = np.reshape(np.asarray(value), (-1))
            if value.shape == (16,):
                self._cartesianTransform = value.reshape((4, 4))
            elif value.shape == (3,):
                self._cartesianTransform = gmu.get_cartesian_transform(translation=value)
            else:
                raise ValueError('Input must be a numpy array of shape (4,4) or (3,).')

    def get_cartesian_transform(self) -> np.ndarray:
        """Get the cartesianTransform of the node from various inputs. if no cartesianTransform is present, it is gathered from the following inputs. 
        
        **NOTE**: this function is overwritten for ImageNodes,PanoNodes and OrthoNodes to get the cartesianTransform from the image centerpoint.
        
        Args:
            - self._resource
            - self._orientedBoundingBox
            - self._convexHull

        Returns:
            cartesianTransform(np.ndarray(4x4))
        """
        if self._cartesianTransform is not None:
            return self._cartesianTransform
        
        if self._resource is not None:
            self._cartesianTransform = gmu.get_cartesian_transform(translation=self._resource.get_center()) # this only works for open3d resources
        if self._cartesianTransform is None and self._convexHull is not None:
            self._cartesianTransform = gmu.get_cartesian_transform(translation=self._convexHull.get_center())
        if self._cartesianTransform is None and self._orientedBoundingBox is not None:
            self._cartesianTransform = gmu.get_cartesian_transform(translation=self._orientedBoundingBox.get_center())
        if self._cartesianTransform is None:
            self._cartesianTransform = np.eye(4)
        
        return self._cartesianTransform
    
    def set_oriented_bounding_box(self, value):
        """Set the orientedBoundingBox for the Node from various inputs. This is overwritten per node type.
        
        Args:
            - orientedBoundingBox (o3d.geometry.OrientedBoundingBox)
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector
            - 9 parameters $[x,y,z,e_x,e_y,e_z, R_x,R_y,R_z]$
        """
        if isinstance(value, o3d.geometry.OrientedBoundingBox):
            self._orientedBoundingBox = value
        else:
            self._orientedBoundingBox=gmu.get_oriented_bounding_box(value)

    def get_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        """Gets the Open3D OrientedBoundingBox of the node. If no orientedBoundingBox is present, it is gathered from the following inputs.
        
        Features:
            1. self._resource
            2. self._convexHull
            3. self._cartesianTransform

        Returns:
            o3d.geometry.OrientedBoundingBox
        """
        if self._orientedBoundingBox is not None:
            return self._orientedBoundingBox
        
        if self._resource is not None:
            self._orientedBoundingBox = gmu.get_oriented_bounding_box(self._resource)
        if self._orientedBoundingBox is None and self._convexHull is not None:
            self._orientedBoundingBox = gmu.get_oriented_bounding_box(self._convexHull)
        if self._orientedBoundingBox is None and self._cartesianTransform is not None: 
            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            box.translate([-0.5, -0.5, -0.5])
            box.transform(self._cartesianTransform)
            self._orientedBoundingBox = box.get_oriented_bounding_box()
        return self._orientedBoundingBox
    
    def set_convex_hull(self, value):
        """Set the convex hull for the Node from various inputs. This is overwritten per node type.
        
        Args:
            - convexHull (o3d.geometry.TriangleMesh)
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVectord 
        """
        if isinstance(value, o3d.geometry.TriangleMesh):
            self._convexHull = value
        else:
            self._convexHull=gmu.get_convex_hull(value)
 
    def get_convex_hull(self) -> o3d.geometry.TriangleMesh:
        """Gets the Open3D Convex Hull of the node. If no convex hull is present, it is gathered from the following inputs.
        
        Features:
            - self._resource
            - self._orientedBoundingBox
            - self._cartesianTransform

        Returns:
            o3d.geometry.TriangleMesh
        """
        if self._convexHull is not None:
            return self._convexHull
        
        if self._resource is not None:
            self._convexHull = gmu.get_convex_hull(self._resource)
        if self._convexHull is None and self._orientedBoundingBox is not None:
            self._convexHull = gmu.get_convex_hull(self._orientedBoundingBox)
        if self._convexHull is None and self._cartesianTransform is not None:
            try:
                box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                box.translate([-0.5, -0.5, -0.5])
                box.transform(self._cartesianTransform)
                self._convexHull = box
            except Exception as e:
                print(f"Failed to compute convex hull from cartesian transform: {e}")
        
        return self._convexHull
        
    def get_resource(self):
        """Returns the resource from the Node type. Overwrite this function for each node type to access more utilities.
        """
        return self._resource

    def set_resource(self,value):
        """sets the resource for the Node type. Overwrite this function for each node type to access more utilities.
        """
        self._resource=value #copy.deepcopy(value) #copy is required to avoid reference issues

    def get_center(self) -> np.ndarray:
        """Returns the center of the node."""
        return self._cartesianTransform[:3, 3]

    def set_path(self, value):
        """sets the path for the Node type. Overwrite this function for each node type to access more utilities.
        """
        self._path = Path(value) 
            
    def get_path(self) -> Path:
        """Returns the full path of the resource from this Node. If no path is present, it is gathered from the following inputs.
        
        Args:
            - self._path
            - self._graphPath
            - self._name
            - self._subject

        Returns:
            path 
        """      
        if self._path is not None: #self._path.exists():
            return self._path
        
        elif self._graphPath and (self._name or self._subject):
            folder=self._graphPath.parent 
            nodeExtensions=ut.get_node_resource_extensions(str(type(self)))
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if path.suffix.upper() in nodeExtensions:
                    if self.get_name() in path.stem :
                        self.path = path    
                        return self._path
            # if self._name:
            #     self.path=os.path.join(folder,self._name+nodeExtensions[0])
            # else:
            #     self.path=os.path.join(folder,self._subject+nodeExtensions[0])
            # return self._path
        else:
            return None
        
    def set_graph(self, value: Graph,overwrite=True):
        """
        Helper method to set the graph attribute.

        Args:
            - Graph: The RDF graph to set.

        Raises:
            ValueError: If the input is not a rdf Graph.
        """
        if isinstance(value, Graph) :
            self._graph =  value if len(set(value.subjects()))==1 else ut.get_subject_graph(value,self._subject) 
        else:
            raise ValueError('Input must be a rdflib.Graph.')
        
    def get_graph(self, graphPath: Path = None, overwrite: bool = True, save: bool = False, base: URIRef = None) -> Graph:
        """
        Returns the graph of the Node. If no graph is present, it is gathered from the following inputs:
        
        **NOTE** that adding a base URI will change the graph's subject to the base URI/subject. URIRef('http://subject') -> URIRef('http://node#subject'). This is useful for readability and graph navigation.
        
        Args:
            - graphPath (Path) : The path of the graph to parse.
            - overwrite (bool) : Overwrite the existing graph or not.
            - base (str | URIRef) : BaseURI to match subjects to in the graph (improves readability) e.g. http://node#. Also, the base URI is used to set the subject of the graph. RDF rules and customs apply so the string must be a valid URI (http:// in front, and # at the end).
            - save (bool) : Save the graph to the self.graphPath or graphPath.
        
        Returns:
            Graph: The RDF graph
        """
        if graphPath:
            self.graphPath = graphPath          
            
        if self._graph is None or overwrite:
            #create graph
            self._graph = ut.bind_ontologies(Graph())
            
            #create graph scope
            if base:
                inst = Namespace(base)
                self._graph.bind("inst", inst)                
                self.subject = inst[self._subject.replace('#', '/').split('/')[-1]]
            
            nodeType = ut.get_node_type(self)
            self._graph.add((self._subject, RDF.type, nodeType))
            attributes = ut.get_variables_in_class(self)
            attributes = ut.clean_attributes_list(attributes) #this is a bit shitty
            pathlist = ut.get_paths_in_class(self)
            
            def handle_oriented_bounding_box(predicate, value, dataType):                        
                        value=gmu.get_oriented_bounding_box_parameters(self.get_oriented_bounding_box())
                        self._graph.add((self._subject, predicate,  Literal(value, datatype=dataType)))
                        
            def handle_convex_hull(predicate, value, dataType):
                        hull=self.get_convex_hull()
                        value=np.asarray(hull.vertices)
                        self._graph.add((self._subject, predicate,  Literal(value, datatype=dataType)))
            
            #watch out, because named attributes can unintentionally affect attributes in other node types
            attr_handlers = { #this is dealing with exceptions
                        'name': lambda predicate, value, dataType: self._graph.add((self._subject, GEOMAPI_PREFIXES['rdfs'].label,  Literal(value, datatype=dataType))),
                        'className': lambda predicate, value, dataType: self._graph.add((self._subject, GEOMAPI_PREFIXES['rdfs'].type,  ut.get_ifcowl_uri(value))),
                        'globalId': lambda predicate, value, dataType: self._graph.add((self._subject, GEOMAPI_PREFIXES['ifc'].IfcGloballyUniqueId,  Literal(value))), #this is not a valid URI so we make it a Literal, and in IFCOWL, the global id is contained in an express:hasString predicate                        
                        'objectType': lambda predicate, value, dataType: self._graph.add((self._subject, GEOMAPI_PREFIXES['ifc'].objectType_IfcObject,  Literal(value))), #this is a bit shitty 
                        'resource': lambda predicate, value, dataType: None,
                        'timestamp': lambda predicate, value, dataType: self._graph.add((self._subject, GEOMAPI_PREFIXES['dcterms'].created,  Literal(value, datatype=XSD.dateTime))),
                        'convexHull': lambda predicate, value, dataType: handle_convex_hull(predicate, value, dataType),
                        'orientedBoundingBox': lambda predicate, value, dataType: handle_oriented_bounding_box(predicate, value, dataType),
                        'imageWidth': lambda predicate, value, dataType: self._graph.add((self._subject, GEOMAPI_PREFIXES['exif'].imageWidth,  Literal(value, datatype=dataType))),
                        'imageHeight': lambda predicate, value, dataType: self._graph.add((self._subject, GEOMAPI_PREFIXES['exif'].imageLength,  Literal(value, datatype=dataType))),
                        'linkedSubjects': lambda predicate, value, dataType: [self._graph.add((self._subject, GEOMAPI_PREFIXES['geomapi'].hasPart,  URIRef(subject))) for subject in value],
                    }
 
            for attribute in attributes:
                # predicate,dataType = ut.match_uri(attribute)
                predicate,dataType =ut.get_predicate_and_datatype(attribute)
                value = getattr(self, attribute)    
                dataType = dataType if dataType else ut.get_data_type(value)
                            

                #     dataType = ut.get_data_type(value)
                #     if self._graph.value(self._subject, predicate, None) == str(value):
                #         continue
                #     elif overwrite:
                #         self._graph.remove((self._subject, predicate, None))
                    
                #     if 'linkedSubjects' in attribute:
                #         if len(value) != 0:
                #             value = [subject.toPython() for subject in self.linkedSubjects]
                #         else:
                #             continue
                                
                #set graph attributes

                #handle special cases
                if any(handler in predicate for handler in attr_handlers):
                    attr_handlers[attribute](predicate, value, dataType)     
                #handle URIRef elements
                # elif dataType not in ut.get_xsd_datatypes() and dataType not in ut.get_geomapi_data_types():
                #     self._graph.add((self._subject, predicate,  value))   
                #handle Path elements          
                elif attribute in pathlist:
                    if self._graphPath:
                        folderPath = ut.get_folder(self._graphPath)
                        value = Path(os.path.relpath(value, folderPath))
                    self._graph.add((self._subject, predicate,  Literal(value.as_posix(), datatype=dataType)))
                else:
                    #first try as URIREF -> this might be a list
                    try:
                        for n in ut.item_to_list(value):
                            self._graph.add((self._subject, predicate,  n))
                        # self._graph.add((self._subject, predicate,  value))
                    except:
                        #then as literal
                        self._graph.add((self._subject, predicate,  Literal(value, datatype=dataType)))
            if(save):
                self.save_graph(graphPath)      
        return self._graph
    
    # def to_graph(self, graphPath : str = None, overwrite:bool=True,save:bool=False) -> Graph: #merge this with get_graph
    #     """Converts the current Node variables to a graph and optionally save.

    #     Args:
    #         - graphPath (str, optional): The full path to write the graph to. Defaults to None.
    #         - overwrite (bool, optional=True): Overwrite current graph values or not
    #         - save (bool, optional=False): Save the graph to the self.graphPath or graphPath.
    #     """
    #     if graphPath and next(graphPath.endswith(extension) for extension in ut.RDF_EXTENSIONS) :
    #         self._graphPath=graphPath

    #     self._graph=Graph() 
    #     ut.bind_ontologies(self._graph)      
    #     nodeType=ut.get_node_type(str(type(self)))                
    #     self._graph.add((self.subject, RDF.type, nodeType ))  

    #     # enumerate attributes in node and write them to triples
    #     attributes = ut.get_variables_in_class(self)
    #     attributes = ut.clean_attributes_list(attributes)        
    #     pathlist = ut.get_paths_in_class(self)
                
    #     for attribute in attributes: 
    #         predicate = ut.match_uri(attribute)
    #         value=getattr(self,attribute)
            
    #         if value is not None:
    #             dataType=ut.get_data_type(value)

    #             if self._graph.value(self._subject, predicate, None)== str(value):
    #                 continue

    #             #check if exists
    #             elif overwrite:
    #                 self._graph.remove((self._subject, predicate, None))

    #             if 'linkedSubjects' in attribute:
    #                 if len(value) !=0:
    #                     value=[subject.toPython() for subject in self.linkedSubjects]
    #                 else:
    #                     continue
                
    #             elif attribute in pathlist:
    #                 if (self._graphPath):
    #                     folderPath=ut.get_folder_path(self.graphPath)
    #                     try:
    #                         value=os.path.relpath(value,folderPath)
    #                     except:
    #                         pass
    #             if 'string' not in dataType.toPython():        
    #                 self._graph.add((self._subject, predicate, Literal(value,datatype=dataType)))
    #             else:
    #                 self._graph.add((self._subject, predicate, Literal(value)))

    #     #Save graph
    #     if(save):
    #         self.save_graph(graphPath)            
    #     return self._graph
        
    def save_graph(self,graphPath : str = None) -> bool:
        """Serialize the graph in an RDF file on drive. The RDF graph will be stored in self.graphPath or provided graphPath (str).

        Args:
            - graphPath (str, optional)

        Raises:
            - ValueError: No valid graphPath if file/folder location is not found
            - ValueError: No valid extension if not in ut.RDF_EXTENSIONS
            - ValueError: Save failed despite valid graphPath and extension (serialization error).

        Returns:
            bool: True if file is succesfully saved.
        """
        #check path validity
        if(graphPath and ut.check_if_path_is_valid(graphPath)): 
            self.graphPath=graphPath
        elif ut.check_if_path_is_valid(self._graphPath):
            pass
        else: 
            raise ValueError(graphPath +  ' is no valid graphPath.')
        #check extension
        if (self._graphPath.suffix.upper() not in ut.RDF_EXTENSIONS):
            raise ValueError(''.join(ut.RDF_EXTENSIONS) + ' currently are only supported extensions.')

        try: 
            # f= open(self._graphPath, 'w') 
            # base=ut.get_folder(self.graphPath)
            self._graph.serialize(self._graphPath)#,base=base
            # f.close()
            if os.path.exists(self._graphPath):                
                return True

            return False
        except:
            raise ValueError('Save failed despite valid graphPath.') 

    def transform(self, 
                  transformation: Optional[np.ndarray] = None, 
                  rotation: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None, 
                  translation: Optional[np.ndarray] = None, 
                  rotate_around_center: bool = True):
        """
        Apply a transformation to the Node's cartesianTransform, resource, and convexHull.
        
        Args:
            - transformation (Optional[np.ndarray]): A 4x4 transformation matrix.
            - rotation (Optional[Union[np.ndarray, Tuple[float, float, float]]]): A 3x3 rotation matrix or Euler angles $(R_z,R_y,R_x)$ for rotation.
            - translation (Optional[np.ndarray]): A 3-element translation vector.
            - rotate_around_center (bool): If True, rotate around the object's center.
        """
        if self.cartesianTransform is None:
            self.get_cartesian_transform()
            
        if transformation is not None:
            transformation=np.reshape(np.asarray(transformation),(4,4))            
        elif translation is not None or rotation is not None:
            transformation = gmu.get_cartesian_transform(rotation=rotation, translation=translation)
            
   
        if rotate_around_center:
            #cartesian transformation                
            transform_to_center = gmu.get_cartesian_transform (translation=-self.cartesianTransform[:3,3] ) 
            transform_back = gmu.get_cartesian_transform (translation=self.cartesianTransform[:3,3] )
            self._cartesianTransform = transform_back @ transformation @ transform_to_center @ self.cartesianTransform
            
            #resource
            if self._resource is not None:
                transform_to_center = gmu.get_cartesian_transform (translation=-self.resource.get_center() )
                transform_back = gmu.get_cartesian_transform (translation=self.resource.get_center() )                
                self._resource.transform(transform_to_center)
                self._resource.transform(transformation) # this can be a problem if the resource center is different from the node center
                self._resource.transform(transform_back) 
                               
                                     
            #oriented bounding box
            if self._orientedBoundingBox is not None:
                transform_to_center = gmu.get_cartesian_transform (translation=-self._orientedBoundingBox.get_center() )
                transform_back = gmu.get_cartesian_transform (translation=self._orientedBoundingBox.get_center() )            
                points=self._orientedBoundingBox.get_box_points()
                pcd=o3d.geometry.PointCloud(points)      
                pcd.transform(transform_to_center)
                pcd.transform(transformation)      
                pcd.transform(transform_back )
                self._orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            
            #convex hull
            if self._convexHull is not None:
                transform_to_center = gmu.get_cartesian_transform (translation=-self._convexHull.get_center() )
                transform_back = gmu.get_cartesian_transform (translation=self._convexHull.get_center() )
                self._convexHull.transform(transform_to_center)
                self._convexHull.transform( transformation)
                self._convexHull.transform(transform_back)
                            
        else:
            #cartesian transformation                
            self._cartesianTransform = transformation @ self.cartesianTransform 
            
            #resource
            self._resource.transform(transformation) if self._resource is not None else None
            
            #oriented bounding box
            if self._orientedBoundingBox is not None:
                points=self._orientedBoundingBox.get_box_points()
                pcd=o3d.geometry.PointCloud(points)            
                pcd.transform(transformation )
                self._orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            
            #convex hull
            self._convexHull.transform( transformation ) if self._convexHull is not None else None
    
    
###############################################
