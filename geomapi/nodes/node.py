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
from collections import Counter
import inspect

#IMPORT MODULES
import geomapi.utils as ut
from geomapi.utils import rdf_property, RDFMAPPINGS
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
                loadResource: bool = False,
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
            
            - loadResource (bool, False) : Load the resource at initialization?
        Returns:
            Node: An instance of the Node class.
        """

        #set properties (protected inputs)       
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

        self._serializeAttributes = []

        #if graphPath is present, parse and set graph
        if not self.graph and self.graphPath:
            if not os.path.exists(self.graphPath):
                raise ValueError("Invalid graphPath does not exist")
            graph=Graph().parse(self.graphPath)
            if(subject is None):
                self.graph=ut.get_subject_graph(graph) 
                self.subject=next(self.graph.subjects(RDF.type))
            else:
                self.graph=ut.get_subject_graph(graph,self.subject)
        if self.graph:
            if(subject is None):
                self.subject=next(self.graph.subjects(RDF.type))
            self.graph=ut.get_subject_graph(self.graph, self.subject) 
                
            self._set_attributes_from_graph()
        # make sure the node has an identifier when initialised without any data
        if(not graph and not path and not self._name and not subject):
            self.name=uuid.uuid1()
        #if path is present, set name and timestamp
        if loadResource:
            self.load_resource()

        # load the geometric properties
        self._set_geometric_properties(self.cartesianTransform, self.convexHull, self.orientedBoundingBox)
        self.__dict__.update(kwargs)

#---------------------PROPERTIES----------------------------

    #---------------------PATH----------------------------
    @property
    @rdf_property(serializer=ut.get_relative_path)
    def path(self): 
        """Path (Path) of the resource of the node.
        
        Args:
            - value (str or Path): The new path for the node.
        
        """
        if self._path is not None: #self._path.exists():
            return self._path

    @path.setter 
    def path(self,value: Optional[Path]):        
        if value is None:
            self._path = None
        elif Path(value).suffix.upper() in ut.get_node_resource_extensions(self):
            self._path = Path(value) 
        else:
            raise ValueError('Invalid extension: ' + Path(value).suffix.upper())

    #---------------------NAME----------------------------
    @property
    @rdf_property(predicate= GEOMAPI_PREFIXES['rdfs'].label, datatype=XSD.string)
    def name(self):
        """The name (str) of the node. This can include characters that the operating
        system does not allow.

        Args:
            - self.path
            - self.subject
        """        
        if self._name is None:
            if self.path:                
                self._name=Path(self.path).stem 
            elif self.subject:                     
                self.name=ut.get_subject_name(self.subject)
        return self._name

    @name.setter
    def name(self, value: Optional[str]):
        if value is None:
            self._name = None
        else:            
            self._name = str(value)

    #---------------------TIMESTAMP----------------------------
    @property
    @rdf_property(predicate=GEOMAPI_PREFIXES['dcterms'].created, datatype=XSD.dateTime)
    def timestamp(self) -> str:
        """The timestamp (str(yyyy-MM-ddTHH:mm:ss)) of the node.

        Features:
            - self.path
            - self.graphPath
        """
        if self._timestamp is None:
            if self.path and os.path.exists(self.path):
                self._timestamp=ut.get_timestamp(self.path)  
            elif self.graphPath and os.path.exists(self.graphPath):
                self._timestamp=ut.get_timestamp(self.graphPath)  
            else:
                self._timestamp=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        return self._timestamp

    @timestamp.setter
    def timestamp(self,timestamp):
        if timestamp is None:
            self._timestamp = None
        else:
            self._timestamp=ut.literal_to_datetime(timestamp)

    #---------------------GRAPHPATH----------------------------    
    @property
    def graphPath(self) -> str:
        """The path (Path) of graph of the node. This can be both a local file or a remote URL.""" 
        return self._graphPath

    @graphPath.setter
    def graphPath(self,value):
        if value is None:
            self._graphPath = None
        elif any(Path(value).suffix.upper()==extension for extension in ut.RDF_EXTENSIONS): 
            self._graphPath=Path(value)
        else:
            self._graphPath = None
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
            self._graph = None
        else:
            if isinstance(value, Graph) :
                self._graph = value# =  value if len(set(value.subjects()))==1 else ut.get_subject_graph(value,self.subject) 
            else:
                self._graph = None
                raise ValueError('Input must be a rdflib.Graph.')

    #---------------------SUBJECT----------------------------    
    @property
    def subject(self) -> URIRef:
        """The subject (RDFLib.URIRef) of the node. If no subject is present, you can use `get_subject()` to construct it from a graph, name or path. Otherwise, a random guid is generated.
        
        Features:
            - self.name
            - self.graph
            - self.path
        """
        #subject
        if self._subject:
            pass
        # self.graph
        elif self.graph:
            self._subject=next(self.graph.subjects(RDF.type))
        #self.path
        elif self.path:
            self.name=self.path.stem 
            self._subject=URIRef('http://'+ut.validate_string(self.name))
        #self_name
        elif self.name:
            self._subject=URIRef('http://'+ut.validate_string(self.name))
        #guid
        else:
            self.name=str(uuid.uuid1())
            self._subject=URIRef('http://'+self.name)            
        return self._subject

    @subject.setter
    def subject(self,value):
        if value is None:
            self._subject = None
        else:
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
            
    
    #---------------------RESOURCE----------------------------    
    @property
    def resource(self):
        """The resource (mesh, image, etc.) of the node. If no resource is present, you can use `get_resource()`, to load the resource from a path or search it through the name and graphpath. 

        Inputs:
            - self.path
            - self.name
            - self.graphPath
        """        
        if(self._resource is None and self.path):
            print("Resource not loaded, but path is defined, call `load_resource()` to access it.")
        return self._resource

    @resource.setter
    def resource(self,value):
        if value is None:
            self._resource = None
        else:
            self._resource=value

    @resource.deleter
    def resource(self):
        self._resource=None
         
    #---------------------CARTESIANTRANSFORM----------------------------    
    @property
    @rdf_property()
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
            self._cartesianTransform = None
        else:
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
            
    #---------------------ORIENTEDBOUNDINGBOX----------------------------
    @property
    @rdf_property(serializer=gmu.get_oriented_bounding_box_parameters)
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
            self._orientedBoundingBox = None
        else:
            if isinstance(value, o3d.geometry.OrientedBoundingBox):
                self._orientedBoundingBox = value
            else:
                self._orientedBoundingBox=gmu.get_oriented_bounding_box(value)

    @orientedBoundingBox.deleter
    def orientedBoundingBox(self):
        self._orientedBoundingBox=None
#---------------------CONVEXHULL----------------------------
    @property
    @rdf_property(serializer=lambda v: np.asarray(v.vertices))
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
            self._convexHull = None
        else:
            if isinstance(value, o3d.geometry.TriangleMesh):
                self._convexHull = value
            else:
                self._convexHull=gmu.get_convex_hull(value)

    @convexHull.deleter
    def convexHull(self):
        self._convexHull=None  

#---------------------METHODS----------------------------     
 
    def _set_attributes_from_graph(self,overwrite:bool=False):
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
                path = (self.graphPath.parent / path).resolve() if self._graphPath else path.resolve()
            setattr(self, attr, path) if getattr(self, attr,None) is None or overwrite  else None #don't overwrite existing paths

        #loop over all predicates
        predicate_counts = Counter(predicate for predicate, _ in self.graph.predicate_objects(subject=self.subject))       
        for predicate, object in self.graph.predicate_objects(subject=self.subject):
            #get attribute based on the last part of the predicate
            attr = ut.get_attribute_from_predicate(self.graph, predicate)
            # check if the attribute is not defined by another name by using the rdf decorator
            for func_name, mapping in RDFMAPPINGS.items():
                if mapping["predicate"] == predicate:
                    attr = func_name
            
            # add the non pre-defined attributes to the list so they will be re serialized
            # don't add "type" because it is defined by the nodetype
            if(not hasattr(self, attr) and str(attr) != "type"):
                #print(attr)
                self._serializeAttributes.append(attr)
            
            #get datatype
            datatype=getattr(object,'datatype',None)

            #convert object to python datatype
            if datatype is not None:
                object=ut.apply_method_to_object( datatype, object)
            else:
                object=ut.literal_to_number(object)

            if re.search('path', attr, re.IGNORECASE):
                handle_path(attr, object)            
            else:
                if predicate_counts[predicate] > 1:
                    # check if the value is a list
                    existing_value = getattr(self, attr, None)
                    # If predicate occurs multiple times, use a list
                    if existing_value is None:
                        setattr(self, attr, [object])
                    elif isinstance(existing_value, list):
                        existing_value.append(object)
                    else: #create a list with the original value and the new value
                        setattr(self, attr, [existing_value, object])
                else:
                    setattr(self, attr, object)

    def _set_geometric_properties(self, _cartesianTransform = None, _convexHull = None, _orientedBoundingBox = None):
        
        # first try transform
        self.cartesianTransform = _cartesianTransform
        self.convexHull = _convexHull
        self.orientedBoundingBox = _orientedBoundingBox

        hasResource = self.resource is not None

        if self.cartesianTransform is None:
            if hasResource:
                self.cartesianTransform = gmu.get_cartesian_transform(translation=self.resource.get_center())
            elif self.convexHull is not None:
                self.cartesianTransform = gmu.get_cartesian_transform(translation=self.convexHull.get_center())
            elif self.orientedBoundingBox is not None:
                self.cartesianTransform = gmu.get_cartesian_transform(translation=self.orientedBoundingBox.get_center(), rotation=self.orientedBoundingBox.R) # the carthesian transform matches the rotation of the bounding box
            else:
                self.cartesianTransform = np.eye(4)

        if self.convexHull is None:
            if hasResource:
                self.convexHull = gmu.get_convex_hull(self.resource)
            elif self.orientedBoundingBox is not None:
                self.convexHull = gmu.get_convex_hull(self.orientedBoundingBox)
            else:
                box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                box.translate([-0.5, -0.5, -0.5])
                box.transform(self.cartesianTransform)
                self.convexHull = box

        if self.orientedBoundingBox is None:
            if hasResource:
                self.orientedBoundingBox = gmu.get_oriented_bounding_box(self.resource)
            else:
                self.orientedBoundingBox = gmu.get_oriented_bounding_box(self.convexHull)

    def get_center(self) -> np.ndarray:
        """Returns the center of the node."""
        return self._cartesianTransform[:3, 3]
    


    def load_resource(self)->bool:
        """Load the resource from the path. Overwrite this function for each node type to access more utilities.
        """
        if self.path is None:
            print("No path is defined, unable to load resource")
            return False
        if not os.path.exists(self.path):
            print("Path does not exist, unable to load resource")
            return False
        if not Path(self.path).suffix.upper() in ut.get_node_resource_extensions(self):
            print("Path has unsupported extension: " + Path(self.path).suffix.upper())
            return False
        return True
        
    def save_resource(self, directory:str=None,extension :str = '.txt') ->bool:
        """Export the resource of the Node. this base function checks for the validity of the path. 
        if no directory is given: checks for path and uses that, checks for 
        """         
        #check path
        if self.resource is None:
            print("This Node has no resource")
            #raise ValueError('No resource to save')
            return False
        
        #validate extension
        if extension and extension.upper() not in ut.get_node_resource_extensions(self):
            print('Invalid extension')
            return False
        
        #get directory
        if (directory):
            if(Path(directory).suffix == '' and not Path(directory).name.startswith('.')): # it is likely a folder
                directory = Path(directory)
            else: # else get the parent folder of the file path
                directory = Path(directory).parent
        elif self.path is not None:    
            directory=Path(self.path).parent            
        elif(self.graphPath): 
            dir=Path(self.graphPath).parent
            directory=os.path.join(dir,self.__class__.__name__[:-4])   
        else:
            directory=os.path.join(os.getcwd(),self.__class__.__name__[:-4])
        # create directory if not present
        if not os.path.exists(Path(directory)):                        
            os.mkdir(directory) 

        self.path=os.path.join(directory,Path(self.subject.toPython()).stem  + extension) #subject.toPython() replaced by get_name()
        return True



    def get_graph(self, graphPath: Path = None, overwrite: bool = True, save: bool = False, base: URIRef = None, serializeAttributes: List = None) -> Graph:
        """Serializes the object into an RDF graph using only its decorated properties.
        
        **NOTE** that adding a base URI will change the graph's subject to the base URI/subject. URIRef('http://subject') -> URIRef('http://node#subject'). This is useful for readability and graph navigation.
        **NOTE** that by default, only the pre-defined properties are serialized, if you want to serialize your own attributes use `serializeAttributes`
        
        Args:
            - graphPath (Path) : The path of the graph to parse.
            - overwrite (bool) : Overwrite the existing graph or not.
            - base (str | URIRef) : BaseURI to match subjects to in the graph (improves readability) e.g. http://node#. Also, the base URI is used to set the subject of the graph. RDF rules and customs apply so the string must be a valid URI (http:// in front, and # at the end).
            - save (bool) : Save the graph to the self.graphPath or graphPath.
            - serializeAttributes (List(str)) : a list of attributes defined in the node that also need to be serialized
        
        Returns:
            Graph: The RDF graph
        """

        if graphPath:
            self.graphPath = graphPath

        if self._graph is None or overwrite:
            # Create RDF graph
            self._graph = ut.bind_ontologies(Graph())

            # Set base URI scope
            if base:
                inst = Namespace(base)
                self._graph.bind("inst", inst)
                self.subject = inst[self.subject.replace('#', '/').split('/')[-1]]

            # Define node type
            nodeType = ut.get_node_type(self)
            self._graph.add((self.subject, RDF.type, nodeType))
            
            #set the default propertylist
            propertylist = RDFMAPPINGS.copy() # copy the mappings list to add the attributes for this specific graph

            # check for the attributes that were read from the initial graph
            if(len(self._serializeAttributes) > 0):
                for prop in self._serializeAttributes:
                    _predicate, _datatype = ut.get_predicate_and_datatype(prop)
                    propertylist[prop] = {"predicate": _predicate, "serializer": None, "datatype": _datatype}
            # check for any user specified attributes that they also want serialized
            if(serializeAttributes is not None):
                for prop in serializeAttributes:
                    _predicate, _datatype = ut.get_predicate_and_datatype(prop)
                    propertylist[prop] = {"predicate": _predicate, "serializer": None, "datatype": _datatype}

            # Process each property using its decorator-defined predicate and serializer
            for attr, metadata in propertylist.items():
                predicate = URIRef(metadata["predicate"])
                datatype = metadata["datatype"]
                serializer = metadata.get("serializer", None)
                value = getattr(self, attr, None)

                if value is not None:
                    
                    if serializer:
                        params = inspect.signature(serializer).parameters
                        try:
                            if len(params) == 1:
                                result = serializer(value)
                            elif len(params) == 2:
                                result = serializer(self, value)
                            else:
                                raise TypeError(f"Unexpected number of arguments in serializer: {serializer}")
                        except Exception as e:
                            print(f"Serializer error for {attr}: {e}")
                            continue  # optionally skip problematic attributes
                    else:
                        result = value
                    #if the result of the serializer is a List, add it multiple times as a separate triple
                    if isinstance(result, list):
                        for val in result:
                            serialized_value = Literal(val, datatype=datatype)
                            self._graph.add((self.subject, predicate, serialized_value))
                    else:
                        serialized_value = Literal(result, datatype=datatype)
                        self._graph.add((self.subject, predicate, serialized_value))

            # Save graph if requested
            if save:
                self.save_graph(graphPath)

        return self._graph
        
    def save_graph(self,graphPath : str = None) -> bool:
        """Serialize the graph in an RDF file on drive. The RDF graph will be stored in self.graphPath or provided graphPath (str).

        Args:
            - graphPath (str, optional)

        Raises:
            - ValueError: No valid graphPath if file/folder location is not found
            - ValueError: No valid extension if not in ut.RDF_EXTENSIONS
            - ValueError: Save failed despite valid graphPath and extension (serialization error).

        Returns:
            bool: True if file is successfully saved.
        """
        #check path validity
        if(graphPath and ut.validate_path(graphPath)): 
            self.graphPath=graphPath
        elif ut.validate_path(self.graphPath):
            pass
        else: 
            raise ValueError(graphPath +  ' is no valid graphPath.')
        #check extension
        if (self.graphPath.suffix.upper() not in ut.RDF_EXTENSIONS):
            raise ValueError(''.join(ut.RDF_EXTENSIONS) + ' currently are only supported extensions.')

        try: 
            # f= open(self._graphPath, 'w') 
            # base=ut.get_folder(self.graphPath)
            self.graph.serialize(self.graphPath)#,base=base
            # f.close()
            if os.path.exists(self.graphPath):                
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
        Apply a transformation to the Node's cartesianTransform, orientedBoundingBox, and convexHull.
        Subclasses should override this method to transform their specific resource types.
        
        Args:
            - transformation (Optional[np.ndarray]): A 4x4 transformation matrix.
            - rotation (Optional[Union[np.ndarray, np.array[float, float, float]]]): A 3x3 rotation matrix or Euler angles (Rz, Ry, Rx).
            - translation (Optional[np.ndarray]): A 3-element translation vector.
            - rotate_around_center (bool): If True, rotate around the object's center (handled by subclass if needed).
        """
        if transformation is None:
            transformation = np.eye(4)

            # Handle rotation
            if rotation is not None:
                if isinstance(rotation, (tuple, List)):
                    R = o3d.geometry.get_rotation_matrix_from_xyz( np.deg2rad(np.array(rotation)))
                elif isinstance(rotation, np.ndarray) and rotation.shape == (3,):
                    R = o3d.geometry.get_rotation_matrix_from_xyz( np.deg2rad(rotation))
                elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
                    R = rotation
                else:
                    raise ValueError("Rotation must be a 3x3 matrix or a 3-element array of Euler angles.")
                rot_transform = np.eye(4)
                rot_transform[:3, :3] = R
                transformation = rot_transform @ transformation

            # Handle translation
            if translation is not None:
                trans_transform = np.eye(4)
                trans_transform[:3, 3] = translation
                transformation = trans_transform @ transformation

        # Update cartesianTransform
        if self.cartesianTransform is None:
            self.cartesianTransform = transformation
        else:
            self.cartesianTransform = transformation @ self.cartesianTransform

        # Apply transformation to optional geometry elements
        if self.orientedBoundingBox is not None:
            #self.orientedBoundingBox = self.orientedBoundingBox.transform(transformation)
            scale = np.cbrt(np.linalg.det(transformation[:3, :3]))
            self.orientedBoundingBox.rotate(transformation[:3, :3]/scale, center=(0, 0, 0))
            self.orientedBoundingBox.scale(scale, center=(0,0,0))
            self.orientedBoundingBox.translate(transformation[:3, 3])
        if self.convexHull is not None:
            self.convexHull.transform(transformation)

        # Let subclass handle resource transformation
        if(self.resource is not None):
            self._transform_resource(transformation, rotate_around_center)

    def _transform_resource(self, transformation: np.ndarray, rotate_around_center: bool):
        """Optional hook for subclasses to implement their own resource transformation. The base works for all types of open3D geometry"""
        if(isinstance(self.resource, o3d.geometry.Geometry)):
            if rotate_around_center:
                center = self.resource.get_center()
                t1 = np.eye(4)
                t1[:3, 3] = -center
                t2 = np.eye(4)
                t2[:3, 3] = center
                transformation = t2 @ transformation @ t1
            self.resource.transform(transformation)
    
    def show(self):
        """Creates a visualization of the resource (if loaded)
        """
        # shows the resource of the node
        if(self.resource is None):
            print("No resource Present")
            return
        #print("Showing Resource")
        
###############################################
