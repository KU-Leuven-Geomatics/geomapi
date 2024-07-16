"""
**Node** is an abstract Python Class to govern the data and metadata of remote sensing data (pcd, images, meshes, orthomosaics).
It is the base class for all other node classes. It contains the base RDF graph functionality and I/O from and to RDF files.

![Node Image](../../docs/pics/ontology_node.png)

**IMPORTANT**: The Node class is an archetype class from which specific data classes (e.g., PointCloudNode) inherit.
Do not use this class directly if you can use a child class with more functionality.

Goals:
- Govern the metadata and geospatial information of big data files in an accessible and lightweight format.
- Serialize the metadata of remote sensing and geospatial files (BIM, Point clouds, meshes, etc.) as RDF Graphs.
- Attach spatial and temporal analyses through RDF Graph navigation.
"""

#IMPORT PACKAGES
import os
import re
from typing import List, Optional
import uuid
import datetime
import numpy as np
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF
import open3d as o3d 

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
                 orientedBoundingBox = None,
                 **kwargs):
        """
        Creates a Node from one or more of the following inputs.
        
        Args:
            subject (URIRef, optional): A subject to use as identifier for the Node. If a graph is also present, the subject should be part of the graph.
            graph (Graph, optional): An RDF Graph to parse.
            graphPath (str, optional): The path of an RDF Graph to parse. If no subject is provided, the first subject of the graph is retained.
            name (str, optional): A name of the Node. This is not a unique identifier but serves as non-functional description.
            path (str, optional): A filepath to a resource.
            timestamp (str, optional): Timestamp for the node.
            resource (optional): Resource associated with the node.
            cartesianTransform (np.ndarray, optional): The (4x4) transformation matrix.
            
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

        #instance variables        
        self.subject=subject
        self.graphPath=graphPath
        self.graph=graph
        self.path=path        
        self.name=name
        self.timestamp=timestamp
        self.resource=resource 
        self.cartesianTransform=cartesianTransform
        self.orientedBoundingBox=orientedBoundingBox

        #initialisation functionality
        if not self._timestamp and self._path:
            self.name=ut.get_filename(path)
            if os.path.exists(self._path):
                self.timestamp=ut.get_timestamp(path) 

        if not self._graph and (graphPath and os.path.exists(self._graphPath)):
            self._graph = Graph().parse(graphPath) 

        self.get_subject()
        
        if(self._graph):
            if ut.check_if_subject_is_in_graph(self._graph,self._subject):
                self._graph=ut.get_subject_graph(self._graph,self._subject)
                self.get_metadata_from_graph(self._graph,self._subject) 
            elif 'session' in str(type(self)):
                pass
            else:
                raise ValueError( 'Subject not in graph')
        self.__dict__.update(kwargs)

#---------------------PROPERTIES----------------------------

    #---------------------PATH----------------------------
    @property
    def path(self): 
        """Get the resource path (str) of the node. If no path is present, you can use get_path() to reconstruct the path from either 
        the graphPath or working directory
        
        Features:
            1. folder\n
            2. self.name\n
            3. self.graphPath\n
        """

        return ut.parse_path(self._path)
    
    @path.setter
    def path(self,value):
        if value is None:
            return None
        nodeExtensions=ut.get_node_resource_extensions(str(type(self)))
        if (ut.get_extension(str(value)) in nodeExtensions):
            self._path=str(value)
        else:
            raise ValueError('self.path has invalid type, path or extension')

    #---------------------NAME----------------------------
    @property
    def name(self):
        """Get the name (str) of the node. This can include characters that the operating
        system does not allow. If no name is present, you can use get_name() to construct a name from the subject or path.

        Features:
            1. self.path\n
            2. self.subject\n
        """        
        return self._name

    @name.setter
    def name(self,name):
        if name is None:
            return None
        try: 
            self._name=str(name)
        except:
            raise TypeError('self.name should be string compatible')

    #---------------------TIMESTAMP----------------------------
    @property
    def timestamp(self):
        """The timestamp (str(yyyy-MM-ddTHH:mm:ss)) of the node. If no timestamp is present, use get_timestamp() to gather the timestamp from the path or graphPath.

        Features:
            1. self.path\n
            2. self.graphPath\n
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self,timestamp):
        if timestamp is None:
            return None
        elif timestamp:
            self._timestamp=ut.validate_timestamp(timestamp)
        else:
            raise ValueError('timestamp should be str(yyyy-MM-ddTHH:mm:ss)')

    #---------------------GRAPHPATH----------------------------    
    @property
    def graphPath(self):
        """Get the path (str) of graph of the node, or the graphPath in which the subject is contained."""
        
        return ut.parse_path(self._graphPath)

    @graphPath.setter
    def graphPath(self,value):
        if value is None:
            return None
        elif (next(str(value).endswith(extension) for extension in ut.RDF_EXTENSIONS) ):
            self._graphPath=str(value)
        else:
            raise ValueError('self.graphPath has invalid type, path or extension')    

    #---------------------GRAPH----------------------------    
    @property
    def graph(self):
        """Get the graph (RDFLib.Graph) of the node. If no graph is present, you can use get_graph() to parse the graph from a graphPath. Alternatively,
        you can use to_graph() to serialize the Nodes attributes to RDF.
        
        Features:
            1. self.graphPath
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
    def subject(self):
        """Get the subject (RDFLib.URIRef) of the node. If no subject is present, you can use get_subject() to construct it from a graph, name or path.
        Otherwise, a random guid is generated.
        
        Features:
            1. self.name\n
            2. self.graph\n
            3. self.path\n
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
        """Get the resource (mesh, pcd, etc.) of the node. If no resource is present, you can use get_resource(),\n
        to load the resource from a path or search it through the name and graphpath. 

        Inputs:
            1. self.path\n
            2. self.name\n
            3. self.graphPath\n
        """        
        return self.get_resource()

    @resource.setter
    def resource(self,value):
        if value is None:
            return None
        else:
            self.set_resource(value)

    @resource.deleter
    def resource(self):
        self._resource=None
         
    #---------------------CARTESIANTRANSFORM----------------------------    
    @property
    def cartesianTransform(self):
        """
        The (4x4) transformation matrix of the node containing the translation & rotation.
        Depending on the type of Node and the different inputs, different matrix values will be obtained.
        
        Examples:
            - The pose of a mesh is determined by the average of its bounding vertices.
            - The pose of an image is at the center of the lens complex.
        
        Returns:
            numpy.ndarray: The (4x4) transformation matrix.
        """
        return self._cartesianTransform

    @cartesianTransform.setter
    def cartesianTransform(self, value):
        """
        Sets the cartesianTransform for the Node.

        Args:
            value (numpy.ndarray or list): Input value which can be:
                - (4x4) full transformation matrix
                - (3x1) translation vector

        Raises:
            ValueError: If the input is not a valid numpy array of shape (4,4) or (3,).
        """
        if value is None:
            self._cartesianTransform = None
        else:
            self._set_cartesianTransform(value)
            
    #---------------------ORIENTEDBOUNDINGBOX----------------------------
    @property
    def orientedBoundingBox(self): 
        """
        The o3d.orientedBoundingBox of the Node containing the bounding box of the geometry.

        Inputs:
            Open3D.geometry.OrientedBoundingBox\n
            Open3D geometry\n
            set of points (np.array(nx3)) or Vector3dVector\n

        Returns:
            o3d.geometry.OrientedBoundingBox: The oriented bounding box of the node.
        """
        return self._orientedBoundingBox

    @orientedBoundingBox.setter
    def orientedBoundingBox(self,value):
        if value is None:
            self._orientedBoundingBox = None
            return
        
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
            
#---------------------METHODS----------------------------     
    def get_metadata_from_graph(self, graph:Graph,subject:URIRef):
        """Convert the data contained in a graph to a set of node attributes.
        If the graph contains multiple subjects, it is reduced to the subject's triples. \n
        
        **NOTE**: The use of a SessionNode is advised when dealing with multi-subject graphs.\n

        Args:
            1. self.graph (RDFlib.Graph):  Graph to parse\n
            2. self.subject (RDFlib.URIRef): The subject to parse the graph for
        
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
        """Returns and validates the current subject. If empty, a new subject is created based on an unique GUID.\n

        Returns:
            subject (URIREF)
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
        """Get the timestamp (str) of the Node. 
        This can be retrieved from user input, the path, the graphPath or the current time.\n

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
        """Returns the name (str) of the Node.\n

        Returns:
           name (str)
        """
        if self._name is None:
            if self._path:
                self._name=ut.get_filename(self._path)
            else:                     
                self._name=ut.get_subject_name(self.subject)
        return self.name
    
    def get_subjectname(self) -> str:
        """Returns the subjectname (str) of the Node.\n

        Returns:
            name (str)
        """                      
        self._name=ut.get_subject_name(self.subject)
        return self.name

    def _set_cartesianTransform(self, value):
        """
        Helper method to set the cartesianTransform attribute.

        Args:
            value (numpy.ndarray or list): Input value which can be:
                - (4x4) full transformation matrix
                - (3x1) translation vector

        Raises:
            ValueError: If the input is not a valid numpy array of shape (4,4) or (3,).
        """
        value = np.reshape(np.asarray(value),(-1))

        if value.shape == (16,):
            self._cartesianTransform = value.reshape((4, 4))
        elif value.shape == (3,):
            self._cartesianTransform = gmu.get_cartesian_transform(translation=value)
        else:
            raise ValueError('Input must be a numpy array of shape (4,4) or (3,).')


    def get_cartesianTransform(self):
        """Returns the cartesianTransform from the Node type. Overwrite this function for each node type to access more utilities.
        """
        return self._cartesianTransform 

    def get_resource(self):
        """Returns the resource from the Node type. Overwrite this function for each node type to access more utilities.
        """
        return self._resource

    def set_resource(self,value):
        """sets the resource for the Node type. Overwrite this function for each node type to access more utilities.
        """
        self._resource=value

    def clear_resource(self):
        """Clear all resources (images, pcd, meshes, ortho's, etc.) in the Node.
        """
        self._resource=None
            
    def get_path(self) -> str:
        """Returns the full path of the resource from this Node.\n

        Features:
            1. self.graphPath\n
            2. self._name or self._subject\n

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
        """Returns the graph of the Node.

        Features:
            1. self.graphPath\n
            2. self._subject\n

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
            1. graphPath (str, optional): The full path to write the graph to. Defaults to None.\n
            2. overwrite (bool, optional=True): Overwrite current graph values or not\n
            3. save (bool, optional=False): Save the graph to the self.graphPath or graphPath.\n
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
        """Serialize the graph in an RDF file on drive.
        The RDF graph will be stored in self.graphPath or provided graphPath (str).

        Args:
            graphPath (str, optional)\n

        Raises:
            ValueError: No valid graphPath if file/folder location is not found\n
            ValueError: No valid extension if not in ut.RDF_EXTENSIONS\n
            ValueError: Save failed despite valid graphPath and extension (serialization error).\n

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

###############################################
