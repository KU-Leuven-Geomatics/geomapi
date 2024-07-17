"""
**Node** is an abstract Python Class to govern the data and metadata of remote sensing data (pcd, images, meshes, orthomosaics). It is the base class for all other node classes. It contains the base RDF graph functionality and I/O from and to RDF files.

.. image:: ../../../docs/pics/ontology_node.png

**IMPORTANT**: The Node class is an archetype class from which specific data classes (e.g., PointCloudNode) inherit. Do not use this class directly if you can use a child class with more functionality.

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
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF
import open3d as o3d 
import copy

#IMPORT MODULES
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class Node:
    def __init__(self, 
                 subject: Optional[URIRef] = None,
                 graph: Optional[Graph] = None,
                 graphPath: Optional[str] = None,
                 name: Optional[str] = None,
                 path: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 resource = None,
                 cartesianTransform: Optional[np.ndarray] = None,
                 orientedBoundingBox: Optional[o3d.geometry.OrientedBoundingBox] = None,
                 convexHull: Optional[o3d.geometry.TriangleMesh] =None,
                 **kwargs):
        """
        Creates a Node from one or more of the following inputs.
        
        Args:
            - subject (URIRef, optional): A subject to use as identifier for the Node. If a graph is also present, the subject should be part of the graph.
            - graph (Graph, optional): An RDF Graph to parse.
            - graphPath (str, optional): The path of an RDF Graph to parse. If no subject is provided, the first subject of the graph is retained.
            - name (str, optional): A name of the Node. This is not a unique identifier but serves as non-functional description.
            - path (str, optional): A filepath to a resource.
            - timestamp (str, optional): Timestamp for the node.
            - resource (optional): Resource associated with the node.
            - cartesianTransform (np.ndarray, optional): The (4x4) transformation matrix.
            - orientedBoundingBox (o3d.geometry.OrientedBoundingBox, optional): The oriented bounding box of the node.
            - convexHull (o3d.geometry.TriangleMesh, optional): The convex hull of the node.            
            
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

        self.initialize(kwargs)

    def initialize(self, kwargs):
        if not self._timestamp and self._path:
            self._name = ut.get_filename(self._path)
            if os.path.exists(self._path):
                self._timestamp = ut.get_timestamp(self._path)

        if not self._graph and self._graphPath and os.path.exists(self._graphPath):
            self._graph = Graph().parse(self._graphPath)

        self.get_subject()

        if self._graph:
            if ut.check_if_subject_is_in_graph(self._graph, self._subject):
                self._graph = ut.get_subject_graph(self._graph, self._subject)
                self.get_metadata_from_graph(self._graph, self._subject)
            elif 'session' in str(type(self)):
                pass
            else:
                raise ValueError('Subject not in graph')

        self.__dict__.update(kwargs)

#---------------------PROPERTIES----------------------------

    #---------------------PATH----------------------------
    @property
    def path(self): 
        """Path (str) of the resource of the node. If no path is present, you can use `get_path()` to reconstruct the path from either the graphPath or working directory.
        
        Args:
            - value (str): The new path for the node.
        
        Raises:
            ValueError: If the path has an invalid type, path, or extension.
        
        """
        return self._path
    
    @path.setter #if i change this, the code breaks for no reason
    def path(self,value):        
        if value is None:
            self._path=None
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
        self._name = None if value is None else str(value)

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
            self._timestamp=None
        elif timestamp:
            self._timestamp=ut.validate_timestamp(timestamp)
        else:
            raise ValueError('timestamp should be str(yyyy-MM-ddTHH:mm:ss)')

    #---------------------GRAPHPATH----------------------------    
    @property
    def graphPath(self) -> str:
        """The path (str) of graph of the node. This can be both a local file or a remote URL.""" 
        return ut.parse_path(self._graphPath)

    @graphPath.setter
    def graphPath(self,value):
        if value is None:
            self._graphPath=None
        # elif any(str(value).endswith(extension) for extension in ut.RDF_EXTENSIONS):
        elif any(str(value).endswith(extension) for extension in ut.RDF_EXTENSIONS):
            self._graphPath=str(value)
        else:
            raise ValueError('self.graphPath has invalid type, path or extension')    

    #---------------------GRAPH----------------------------    
    @property
    def graph(self) -> Graph:
        """The Graph (RDFLib.Graph) of the node. If no graph is present, you can use `get_graph()` to parse the graph from a graphPath. Alternatively, you can use `to_graph()` to serialize the Nodes attributes to RDF. 
        """       
        return self._graph

    @graph.setter
    def graph(self,graph):
        if graph is None:
            self._graph = None
        elif isinstance(graph, Graph):
            self._graph=graph
        else:
            raise TypeError('The graph must be an instance of rdflib.Graph')

    #---------------------SUBJECT----------------------------    
    @property
    def subject(self) -> URIRef:
        """Get the subject (RDFLib.URIRef) of the node. If no subject is present, you can use `get_subject()` to construct it from a graph, name or path. Otherwise, a random guid is generated.
        
        Features:
            - self.name
            - self.graph
            - self.path
        """
        return self._subject

    @subject.setter
    def subject(self,subject):
        if subject is None:
            self._subject = None
        elif isinstance(subject, URIRef):
            self._subject=subject
        else:
            string=str(subject)
            prefix='file:///'
            if 'file:///' in string:
                string=string.replace('file:///','')
                prefix='file:///'
            elif 'http://' in string:
                string=string.replace('http://','')
                prefix='http://' 
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
        return self._resource

    @resource.setter
    def resource(self,value):
        if value is None:
            self._resource=None
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
            self.set_cartesian_transform(value)
            
    #---------------------ORIENTEDBOUNDINGBOX----------------------------
    @property
    def orientedBoundingBox(self) -> o3d.geometry.OrientedBoundingBox: 
        """
        The o3d.orientedBoundingBox of the Node containing the bounding box of the geometry. If no box is present, you can use `get_oriented_bounding_box()`, to gather it from the resource, cartesianTransform or convexHull.

        Inputs:
            - Open3D.geometry.OrientedBoundingBox
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector

        Returns:
            o3d.geometry.OrientedBoundingBox: The oriented bounding box of the node.
        """
        return self._orientedBoundingBox

    @orientedBoundingBox.setter
    def orientedBoundingBox(self, value):
        if value is None:
            self._orientedBoundingBox = None
        else:
            self.set_oriented_bounding_box(value)

#---------------------CONVEXHULL----------------------------
    @property
    def convexHull(self) -> o3d.geometry.TriangleMesh:
        """
        The convex hull of the Node containing the bounding hull of the geometry. If no convex hull is present, you can use `get_convex_hull()`, to gather it from the resource, cartesianTransform or orientedBoundingBox.

        Inputs:
            - Open3D.geometry.TriangleMesh
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector

        Returns:
            o3d.geometry.TriangleMesh: The convex hull of the node.
        """
        return self._convexHull

    @convexHull.setter
    def convexHull(self, value):
        if value is None:
            self._convexHull = None
        else:
            self.set_convex_hull(value)

            
#---------------------METHODS----------------------------     
    def get_metadata_from_graph(self, graph:Graph,subject:URIRef):
        """Convert the data contained in a graph to a set of node attributes. If the graph contains multiple subjects, it is reduced to the subject's triples.
        
        **NOTE**: The use of a SessionNode is advised when dealing with multi-subject graphs.

        Args:
            - self.graph (RDFlib.Graph):  Graph to parse
            - self.subject (RDFlib.URIRef): The subject to parse the graph for
        
        """
        if len([x for x in self._graph.subjects(RDF.type)])>1:
            self._graph=ut.get_subject_graph(graph,subject)

        for predicate, object in graph.predicate_objects(subject=subject):
            attr= ut.get_attribute_from_predicate(graph, predicate) 
            value=object.toPython()
            
            #GEOMETRY
            if attr == 'cartesianBounds':
                self.cartesianBounds=ut.literal_to_array(object) 
            elif attr == 'orientedBounds':
                self.orientedBounds=ut.literal_to_orientedBounds(object) 
            elif attr == 'cartesianTransform':
                self.cartesianTransform=ut.literal_to_cartesianTransform(object) 
            elif attr == 'geospatialTransform':
                self.geospatialTransform=ut.literal_to_array(object)    
            #PATHS
            elif re.search('path', attr, re.IGNORECASE):
                path=ut.literal_to_string(object)
                if path and self._graphPath:
                    path = path.replace("\\", os.sep)
                    if '..' in path:
                        path=path.strip(str('..' + os.sep))
                        folder=ut.get_folder_path(ut.get_folder_path(self._graphPath))
                    else:
                        folder=ut.get_folder_path(self._graphPath)
                    path=os.path.join(folder,path)
                setattr(self,attr,path)
                    
            #INT    
            elif attr in ut.INT_ATTRIBUTES:
                setattr(self,attr,ut.literal_to_int(object)) 
            #FLOAT
            elif attr in ut.FLOAT_ATTRIBUTES:
                setattr(self,attr,ut.literal_to_float(object)) 
            #LISTS
            elif attr in ut.LIST_ATTRIBUTES:
                setattr(self,attr,ut.literal_to_list(object)) 
            #LINKEDSUBEJCTS
            elif attr == 'linkedSubjects':
                # test=ut.literal_to_linked_subjects(object)
                # self.linkedSubjects=test
                setattr(self,attr,ut.literal_to_linked_subjects(object)) 
            
            #STRINGS
            else:
                setattr(self,attr,object.toPython()) 

    def get_subject(self) -> str:
        """Returns and validates the current subject. If empty, a new subject is created based on an unique GUID.

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
            self._name=ut.get_filename(self._path)
            self._subject=URIRef('file:///'+ut.validate_string(self._name))
        elif self._name:
            self._subject=URIRef('file:///'+ut.validate_string(self._name))
        #guid
        else:
            self._name=str(uuid.uuid1())
            self._subject=URIRef('file:///'+self._name)            
        return self._subject

    def get_timestamp(self):
        """Get the timestamp (str) of the Node. If no timestamp is present, it is gathered from the folowing parameters.

        Features:
            - self._timestamp
            - self._path
            - self._graphPath
            * datetime.datetime.now()
        
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
                self._name=ut.get_filename(self._path)
            else:                     
                self._name=ut.get_subject_name(self._subject)
        return self._name

    def set_cartesian_transform(self, value):
        """
        Helper method to set the cartesianTransform attribute.

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
        
        Features:
            - self._resource
            - self._orientedBoundingBox
            - self._convexHull

        Returns:
            cartesianTransform(np.ndarray(4x4))
        """
        if self._cartesianTransform is not None:
            return self._cartesianTransform
        
        if self._resource is not None:
            try:
                self._cartesianTransform = gmu.get_cartesian_transform(translation=self._resource.get_center())
            except Exception as e:
                print(f"Failed to get cartesian transform from resource: {e}")
        
        if self._cartesianTransform is None and self._convexHull is not None:
            try:
                self._cartesianTransform = gmu.get_cartesian_transform(translation=self._convexHull.get_center())
            except Exception as e:
                print(f"Failed to get cartesian transform from convex hull: {e}")
        
        if self._cartesianTransform is None and self._orientedBoundingBox is not None:
            try:
                self._cartesianTransform = gmu.get_cartesian_transform(translation=self._orientedBoundingBox.get_center())
            except Exception as e:
                print(f"Failed to get cartesian transform from oriented bounding box: {e}")
        
        if self._cartesianTransform is None:
            self._cartesianTransform = np.eye(4)
        
        return self._cartesianTransform
    
    def set_oriented_bounding_box(self, value):
        """Set the oriented bounding box for the Node.
        
        Args:
            - orientedBoundingBox (o3d.geometry.OrientedBoundingBox)
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector
        """
        if isinstance(value, o3d.geometry.OrientedBoundingBox):
            self._orientedBoundingBox = value
        elif isinstance(value, o3d.geometry.Geometry):
            self._orientedBoundingBox = value.get_oriented_bounding_box()
        elif isinstance(value, o3d.utility.Vector3dVector):
            self._orientedBoundingBox = o3d.geometry.OrientedBoundingBox.create_from_points(value)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            try:
                points = o3d.utility.Vector3dVector(np.reshape(np.asarray(value), (-1, 3)))
                self._orientedBoundingBox = o3d.geometry.OrientedBoundingBox.create_from_points(points)
            except:
                raise ValueError('Input must be orientedBoundingBox (o3d.geometry.OrientedBoundingBox), an Open3D Geometry or a list of Vector3dVector or np.array objects')

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
            try:
                self._orientedBoundingBox = self._resource.get_oriented_bounding_box()
            except Exception as e:
                print(f"Failed to get oriented bounding box from resource: {e}")
        
        if self._orientedBoundingBox is None and self._convexHull is not None:
            try:
                self._orientedBoundingBox = self._convexHull.get_oriented_bounding_box()
            except Exception as e:
                print(f"Failed to get oriented bounding box from convex hull: {e}")
        
        if self._orientedBoundingBox is None and self._cartesianTransform is not None:
            try:
                box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                box.transform(self._cartesianTransform)
                self._orientedBoundingBox = box.get_oriented_bounding_box()
            except Exception as e:
                print(f"Failed to get oriented bounding box from cartesian transform: {e}")
        
        return self._orientedBoundingBox

    def set_convex_hull(self, value):
        """Set the convex hull for the Node from various inputs.
        
        Args:
            - convexHull (o3d.geometry.TriangleMesh)
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector
        """
        if isinstance(value, o3d.geometry.TriangleMesh):
            self._convexHull = copy.deepcopy(value)
        elif isinstance(value, o3d.geometry.Geometry):
            self._convexHull = value.compute_convex_hull()[0]
        elif isinstance(value, o3d.utility.Vector3dVector):
            self._convexHull = o3d.geometry.PointCloud(value).compute_convex_hull()[0]
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            try:
                points = o3d.utility.Vector3dVector(np.reshape(np.asarray(value), (-1, 3)))
                self._convexHull = o3d.geometry.PointCloud(points).compute_convex_hull()[0]
            except:
                raise ValueError('Input must be a TriangleMesh (o3d.geometry.TriangleMesh), an Open3D Geometry or a list of Vector3dVector or np.array objects')
            
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
            try:
                self._convexHull = self._resource.compute_convex_hull()[0]
            except Exception as e:
                print(f"Failed to compute convex hull from resource: {e}")
        
        if self._convexHull is None and self._orientedBoundingBox is not None:
            try:
                points = self._orientedBoundingBox.get_box_points()
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                self._convexHull = pcd.compute_convex_hull()[0]
            except Exception as e:
                print(f"Failed to compute convex hull from oriented bounding box: {e}")
        
        if self._convexHull is None and self._cartesianTransform is not None:
            try:
                box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
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
        self._resource=copy.deepcopy(value) #copy is required to avoid reference issues

    def get_center(self) -> np.ndarray:
        """Returns the center of the node."""
        return self._cartesianTransform[:3, 3]

    def set_path(self, value):
        """sets the path for the Node type. Overwrite this function for each node type to access more utilities.
        """
        self._path = Path(value).as_posix()
            
    def get_path(self) -> str:
        """Returns the full path of the resource from this Node. If no path is present, it is gathered from the following inputs.
        
        Features:
            - self._path
            - self._graphPath
            - self._name
            - self._subject

        Returns:
            path (str)
        """      
        if self._path and os.path.exists(self._path):
            return self._path
        
        elif self._graphPath and (self._name or self._subject):
            folder=ut.get_folder_path(self._graphPath)
            nodeExtensions=ut.get_node_resource_extensions(str(type(self)))
            allSessionFilePaths=ut.get_list_of_files(folder) 
            for path in allSessionFilePaths:
                if ut.get_extension(path) in nodeExtensions:
                    if self.get_name() in path or self.get_subject() in path :
                        self._path = path    
                        return self._path
            if self._name:
                self._path=os.path.join(folder,self._name+nodeExtensions[0])
            else:
                self._path=os.path.join(folder,self._subject+nodeExtensions[0])
            return self._path
        else:
            # print("No file containing this object's name and extension is found in the graphPath folder")
            return None

    def get_graph(self):
        """Returns the graph of the Node. if no graph is present, it is gathered from the following inputs.
        
        Features:
            - self.graphPath
            - self._subject

        Returns:
            graph (RDFlib.GRAPH)
        """
        if self._graph is None:
            if self._graphPath and os.path.exists(self._graphPath):
                self._graph=Graph().parse(self._graphPath)
                if self._subject and ut.check_if_subject_is_in_graph(self._graph,self._subject):
                    self._graph=ut.get_subject_graph(self._graph,self._subject)
                else:
                    print( 'Subject not in Graph')
        return self._graph

    def to_graph(self, graphPath : str = None, overwrite:bool=True,save:bool=False) -> Graph:
        """Converts the current Node variables to a graph and optionally save.

        Args:
            - graphPath (str, optional): The full path to write the graph to. Defaults to None.
            - overwrite (bool, optional=True): Overwrite current graph values or not
            - save (bool, optional=False): Save the graph to the self.graphPath or graphPath.
        """
        if graphPath and next(graphPath.endswith(extension) for extension in ut.RDF_EXTENSIONS) :
            self._graphPath=graphPath

        self._graph=Graph() 
        ut.bind_ontologies(self._graph)      
        nodeType=ut.get_node_type(str(type(self)))                
        self._graph.add((self.subject, RDF.type, nodeType ))  

        # enumerate attributes in node and write them to triples
        attributes = ut.get_variables_in_class(self)
        attributes = ut.clean_attributes_list(attributes)        
        pathlist = ut.get_paths_in_class(self)
                
        for attribute in attributes: 
            predicate = ut.match_uri(attribute)
            value=getattr(self,attribute)
            
            if value is not None:
                dataType=ut.get_data_type(value)
                temp=dataType.toPython()
                predtemp=predicate.toPython()

                if self._graph.value(self._subject, predicate, None)== str(value):
                    continue

                #check if exists
                elif overwrite:
                    self._graph.remove((self._subject, predicate, None))

                if 'linkedSubjects' in attribute:
                    if len(value) !=0:
                        value=[subject.toPython() for subject in self.linkedSubjects]
                    else:
                        continue
                
                elif attribute in pathlist:
                    if (self._graphPath):
                        folderPath=ut.get_folder_path(self.graphPath)
                        try:
                            value=os.path.relpath(value,folderPath)
                        except:
                            pass
                if 'string' not in dataType.toPython():        
                    self._graph.add((self._subject, predicate, Literal(value,datatype=dataType)))
                else:
                    self._graph.add((self._subject, predicate, Literal(value)))

        #Save graph
        if(save):
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
            bool: True if file is succesfully saved.
        """
        #check path validity
        if(graphPath and ut.check_if_path_is_valid(graphPath)): 
            self._graphPath=graphPath
        elif ut.check_if_path_is_valid(self._graphPath):
            pass
        else: 
            raise ValueError(graphPath +  ' is no valid graphPath.')
        #check extension
        if (ut.get_extension(graphPath) not in ut.RDF_EXTENSIONS):
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
                self._resource.transform(transformation)
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
