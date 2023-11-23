"""
SessionNode - a Python Class to govern the data and metadata of remote sensing data captured in the same epoch.

This node buids upon Open3D for its geometry definitions.
This node uses LinkedNodes (ImageNodes, PointCloudNodes, etc.) to describe resources gathered in the same epoch.
It directly inherits from Node.
Be sure to check the properties defined in the above classes to initialise the Node.

Goals:
- Given a path, import all the linked images, meshes, ect... into a session class
- Convert non-RDF metadata files (json, xml, ect..) to SessionsNodes and export them to RDF
- get the boundingbox of the whole session
- use URIRef() to reference the images, ect...

"""  
from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from typing import List
import open3d as o3d 
import cv2
import numpy as np
import os
from typing import Union
import uuid
import concurrent.futures

#IMPORT MODULES
from geomapi.nodes import *
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class SessionNode(GeometryNode):

    def __init__(self,  graph : Graph = None, 
                        graphPath: str=None,
                        subject : URIRef = None,
                        linkedNodes : Node = None,
                        linkedResources = None,
                        getResource : bool = False,
                        getMetaData : bool = True,
                        **kwargs):
        """
        Creates a SessionNode & all the child nodes. Overloaded function.\n
        This Node can be initialised from one or more of the inputs below.\n
        By default, no data is imported in the Node to speed up processing.\n
        If you also want the data, call node.get_resource() or set getResource() to True.\n

        Args:
            0.graph (Graph, optional): Upon initialisation, bigger graphs can be submitted and parsed i.e.: \n
                > The RDF Graph of only the sessionNode (1 subject)
                > AN RDF Graph with only resourceNodes
            1.graphPath (str, optional): The path of the Graph of only the session. \n
            2.subject (URIRef, optional): The subject to parse the Graph with. \n
            3.resources (Any, optional): A set of resources which will be converted to LinkedNodes \n
            4.LinkedNodes (Nodes, optional): A set of Nodes (ImageNodes, MeshNodes) to contain within the session \n
            
            getResource (bool, optional= False): If True, the node will search for its physical resource on drive \n
            getMetaData (bool, optional= True): If True, the node will attempt to extract geometric metadata from the resource if present (cartesianBounds, etc.) \n
        
        Returns:
            Args(None): Create an empty Graph and Node with an unique subject guid
            
            Args(subject): create a new Graph and Node with the given subject
            Args(subject & graphPath): parse the graph with the given subject
                - 1 or more are matched: Use that node as the SessionNode
                - 1 or more are found but not matched: Raise an error
                - None are found: Create a new node with the given subject

            OVERWRITE
            Args(Graph):  parse the graph and search for a SessionNode
                - 1 or more are found: Use that node as the SessionNode
                - None are found: Create a new SessionNode with an unique id containing all the other Nodes
            Args(grapPath): parse the graph and search for a SessionNode
                - 1 or more are found: Use that node as the SessionNode
                - None are found: Create a new SessionNode with an unique id containing all the other Nodes

            Args(nodeList): create a new Graph from the joint metadata
           
        """
        #private attributes 
        self._linkedNodes=[]
        self._linkedSubjects=[]
        self._subject=None
        self._graph=None
        self._graphPath=None
        self._path=None
        self._name=None

        #instance variables
        self.linkedNodes=linkedNodes    
        self.graphPath=graphPath     
        self.subject=subject

        #initialisation functionality
        if(graphPath and not graph):
            graph = Graph().parse(graphPath)

        if(graph):
            self.parse_session_graph(graph) 

        if (self._subject is None):
            self.subject=str(uuid.uuid1())

        super().__init__(graph=self._graph, 
                        graphPath=self._graphPath, 
                        subject=self._subject,
                        **kwargs)
        
        if(linkedResources):
            self.resources_to_linked_nodes(linkedResources)

        if(getResource):
            self.get_resource()

        if (getMetaData):
            self.get_metadata_from_linked_nodes()
#---------------------PROPERTIES----------------------------

    #---------------------linkedNodes----------------------------
    @property
    def linkedNodes(self): 
        """Get the linkedNodes (Node) of the node."""
        return self._linkedNodes

    @linkedNodes.setter
    def linkedNodes(self,list):
        list=ut.item_to_list(list)
        if not list or list[0] is None:
            return []
        elif all('Node' in str(type(value)) for value in list):
            self.add_linked_nodes(list)
        else:
            raise ValueError('Some elements in self.linkedNodes are not Nodes')    
    
    #---------------------linkedSubjects----------------------------
    @property
    def linkedSubjects(self): 
        """Get the linkedSubjects (URIRef) of the node."""
        if not self._linkedSubjects:
           self._linkedSubjects= [node.subject for node in self.linkedNodes]
        return self._linkedSubjects

    @linkedSubjects.setter
    def linkedSubjects(self,list):
        list=ut.item_to_list(list)
        if not list or list[0] is None:
            return []
        elif all('URIRef' in str(type(value)) for value in list):
            matches=[w for w in self._linkedSubjects if w in list]
            (self._linkedSubjects.remove(match) for match in matches)
            self._linkedSubjects.extend(list)
        else:
            for value in list:
                if not ut.check_if_uri_exists(self._linkedSubjects,URIRef(value)):
                    string=str(value)                
                    prefix='file:///'
                    if 'file:///' in string:
                        string=string.replace('file:///','')
                        prefix='file:///'
                    elif 'http://' in string:
                        string=string.replace('http://','')
                        prefix='http://'                     
                    self._linkedSubjects.append(URIRef(prefix+ut.validate_string(string)) )
#---------------------METHODS----------------------------

    def get_linked_nodes(self,resourceGraph:Graph,linkedSubjects:List[URIRef] = None):
        """Get self.linkedNodes based on a resourceGraph and list of linkedSubjects or self.linkedSubjects.

        Args:
            resourceGraph (Graph): contains the target nodes' metadata
            linkedSubjects (List[URIRef],optional)

        Returns:
            self.linkedNodes (Node)
        """
        #validate inputs
        if(not type(resourceGraph) ==Graph): 
            raise ValueError('resourceGraph is not an RDFlib Graph')
        if(not linkedSubjects and self.linkedSubjects): 
            linkedSubjects=self.linkedSubjects
        elif(not linkedSubjects and not self.linkedSubjects):
            print('No linkedSubjects present, taking all resource subjects')
            self.linkedSubjects=[s for s in resourceGraph.subjects(RDF.type)]

        nodeSubjectList=[node.subject for node in self.linkedNodes if (self.linkedNodes)]
        #create nodes
        for subject in linkedSubjects:   
            s=subject.toPython() #temp   
            if not ut.check_if_uri_exists(nodeSubjectList,subject):                
                g=ut.get_subject_graph(resourceGraph,subject=subject)  
                if (g):            
                    newNode=create_node(graph=g, subject=subject)
                    self.linkedNodes.append(newNode)
        return self.linkedNodes
    
    def add_linked_subjects(self,linkedSubjects : List[URIRef]=None):
        """Update self.linkedSubjects

        Args:
            linkedSubjects (List[URIRef], optional): 
        """   
        linkedSubjects=ut.item_to_list(linkedSubjects)
        [self.linkedSubjects.append(s) for s in linkedSubjects if s not in self.linkedSubjects]
      
    def add_linked_nodes(self,linkedNodes:List[Node]):
        """Update self.linkedNodes from a new linkedNodes list

        Args:
            linkedNodes (List[Node]): 
        """        
        linkedNodes=ut.item_to_list(linkedNodes) 
        for node in linkedNodes:
            if node not in self.linkedNodes:
                self.linkedNodes.append(node)
        self.add_linked_subjects([node.subject for node in linkedNodes])

    def parse_session_graph(self,graph:Graph)-> Union[URIRef , Graph, Node]:
        """Parse a graph to detect and split sessionNodes from resourceNodes

        Args:
            graph (Graph): Graph that either contains a sessionNode, resourceNodes or a combination of both. 

        Raises:
            ValueError: If self.subject is provided, it should match with a subject in the graph

        Returns:
            Union[subject ,Graph, Nodelist]
        """        
        # Step 1: extract the SessionNodes
        subjects = graph.subjects(RDF.type)
        sessionNodeSubjects = []
        resourceNodeSubjects = []
        for sub in subjects:
            nodeType = ut.literal_to_string(graph.value(subject=sub,predicate=RDF.type))
            if 'SessionNode' in nodeType:
                sessionNodeSubjects.append(sub)
            else:
                resourceNodeSubjects.append(sub)
        
        # Step 2: Get the SessionNode in the graph (if it exists)
        if (not sessionNodeSubjects): 
            print("no sessionSubjects found")
            self.get_subject()

        else: # there is 1 or more sessionNodes, search for a match
            if (not self._subject): # no subject was given, pick one from the list
                self.subject = sessionNodeSubjects[0]
                self.graph=ut.get_subject_graph(graph=graph,subject=self._subject)               
                if(len(sessionNodeSubjects) > 1): 
                    print("More than one SessionNode is present, while no subject was provided, picked:",self.subject,"out of", sessionNodeSubjects)
            else: # Check if the subject is in the list
                if (self._subject not in sessionNodeSubjects):
                    raise ValueError("The given subject is not in the Graph or is not a SessionNode")

        # Step 3: Parse all the other Nodes into the nodelist
        nodelist=[]
        for subject in resourceNodeSubjects:
            s=subject.toPython() #temp
            newGraph=ut.get_subject_graph(graph=graph,subject=subject)
            nodelist.append(create_node(graph = newGraph ,graphPath= self.graphPath,subject= subject))
        if nodelist:
            self.add_linked_nodes(nodelist)
        
        return self.subject, self.graph, self.linkedNodes
    
    def linked_nodes_to_graph(self,graphPath:str=None,save:bool=False) -> Graph:
        """Serialize the session's linkedNodes

        Args:
            1.graphPath (str,optional): defaults to linkednodesGraph.ttl in session graphPath location.\n
            2.save (bool, optional): Defaults to False.\n

        Returns:
            Graph with linkedNodes
        """
        if not graphPath:
            if self.graphPath:
                graphPath=os.path.join(ut.get_folder(self.graphPath),'linkednodesGraph.ttl')
            elif self.get_path() and os.path.exists(self.path):
                graphPath=os.path.join(self.path,'linkednodesGraph.ttl')

        g=Graph()
        g=ut.bind_ontologies(g)
        for node in self.linkedNodes:
                node.to_graph(graphPath)
                g+= node.graph
        if(graphPath and save):
            g.serialize(graphPath)     
        return g  

    def session_to_graph(self,graphPath:str=None,save:bool=False) -> Graph:
        """Serialize the session's linkedNodes and own metadata.\n

        Args:
            1.graphPath (str,optional): defaults to combinedGraph.ttl in session graphPath location.\n
            2.save (bool, optional): Defaults to False.\n

        Returns:
            Graph with linkedNodes and sessionNode
        """
        if graphPath and next(graphPath.endswith(extension) for extension in ut.RDF_EXTENSIONS) :
            self._graphPath=graphPath

        if not graphPath:
            if self.graphPath:
                graphPath=os.path.join(ut.get_folder(self.graphPath),'combinedGraph.ttl')
            elif self.get_path() and os.path.exists(self.path):
                graphPath=os.path.join(self.path,'combinedGraph.ttl')

        g=Graph()
        g=ut.bind_ontologies(g)
        for node in self.linkedNodes:
            node.to_graph(graphPath)
            g+= node.graph
        g+=self.to_graph(graphPath)
        if(graphPath and save):
            g.serialize(graphPath)     
        return g 

    def save_linked_resources(self,directory:str=None):
        """Export the resources of the linkedNodes.\n

        Args:
            directory (str, optional): directory folder to store the data.\n

        Returns:
            bool: return True if export was succesful
        """ 
        for node in self.linkedNodes:
            node.save_resource(directory)  

    def save_resource(self,directory:str=None,extension :str = '.ply') -> bool:
        """Export the resource (Convex hull) of the Node.\n

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
        if directory and os.path.exists(os.path.join(directory,self.get_name() + extension)):
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
            directory=os.path.join(dir,'SESSION')   
        else:
            directory=os.path.join(os.getcwd(),'SESSION')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        self.path=os.path.join(directory,ut.get_filename(self.subject.toPython()) + extension)

        #write files
        if o3d.io.write_triangle_mesh(self.path, self.resource):
            return True
        return False

    def resources_to_linked_nodes(self,linkedResources) -> None:
        """Create linked Nodes from a set of data resources.\n

        Args:
            resource (List[mesh,pcd,image,ortho,lineset,etc.])

        Returns:
            None 
        """
        nodelist=[]
        for resource in linkedResources:
            #check type
            if 'TriangleMesh' in str(type(resource)):
                nodelist.append(MeshNode(resource=resource))
            elif type(resource) is np.ndarray:
                nodelist.append( ImageNode(resource=resource))
            elif 'PointCloud' in str(type(resource)):
                nodelist.append( PointCloudNode(resource=resource))
            elif 'LineSet' in str(type(resource)):
                nodelist.append( LinesetNode(resource=resource))
            elif 'ifcelement' in str(type(resource)):
                nodelist.append( BIMNode(resource=resource))
        if nodelist:
            self.add_linked_nodes(nodelist)

    def get_metadata_from_linked_nodes(self):
        """Returns the sessionNode metadata from the linkedNodes. \n

        Features:
            cartesianTransform\n
            orientedBoundingBox \n
            orientedBounds\n
            cartesianBounds\n
            resource \n

        Returns:
            bool: True if exif data is successfully parsed
        """
        if (getattr(self,'cartesianTransform',None) is not None and
            getattr(self,'orientedBoundingBox',None) is not None and
            getattr(self,'orientedBounds',None) is not None and
            getattr(self,'cartesianBounds',None) is not None and
            getattr(self,'resource',None) is not None):
            return True

        if getattr(self,'timestamp',None) is None and self.linkedNodes:
            self.timestamp=self.linkedNodes[0].get_timestamp()

        points=o3d.utility.Vector3dVector()        
        for node in self.linkedNodes: 
            if getattr(node,'get_oriented_bounding_box',None) is not None:           
                box=node.get_oriented_bounding_box()
                if (box):
                    points.extend(box.get_box_points())    
                
            elif getattr(node,'cartesianTransform',None) is not None:
                t=gmu.get_translation(node.get_cartesian_transform())
                t=np.reshape(t,(1,3))
                p=o3d.utility.Vector3dVector(t)
                points.extend(p)

        if np.asarray(points).shape[0] >=5:    
            self.cartesianTransform=gmu.get_cartesian_transform(translation=np.mean( np.asarray(points),0)) 
            self.orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(points)
            self.orientedBounds=np.asarray(self.orientedBoundingBox.get_box_points())
            self.cartesianBounds=gmu.get_cartesian_bounds(o3d.geometry.AxisAlignedBoundingBox.create_from_points(points))
            pcd= o3d.geometry.PointCloud()
            pcd.points=points
            hull, _ =pcd.compute_convex_hull()
            self.resource=hull

    def get_linked_resources(self,percentage:float=1.0):
        """Returns the resources of the linkedNodes. \n
        If none is present, it will search for the data on drive from path, graphPath, name or subject. \n
        Otherwise, it will be reconstructed from the metadata present

        Args:
            1. self (sessionNode)
            2. percentage(float,optional): load percentage of point cloud resources in present PointCloudNodes.

        Returns:
            list[resource] or None
        """
        for node in self.linkedNodes:
            if 'PointCloud' in str(type(node)):
                node.get_resource(percentage)
            else:
                node.get_resource()
        return [n.resource for n in self.linkedNodes]
      
    def get_linked_resources_multiprocessing(self,percentage:float=1.0):
        """Returns the resources of the linkedNodes by multi-processing the imports. \n
        If none is present, it will search for the data on drive from path, graphPath, name or subject. \n

        **NOTE**: Starting parallel processing takes a bit of time. As such, this method will only outperform get_linked_resources with 10+ linkedNodes

        Args:
            1. self (sessionNode)
            2. percentage(float,optional): load percentage of point cloud resources in present PointCloudNodes.

        Returns:
            list[resource] or None
        """
        [n.get_path() for n in self.linkedNodes]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # first load all data and output it as np.arrays      
            results1=[executor.submit(gmu.e57_to_arrays,e57Path=n.path,e57Index=n.e57Index,percentage=percentage,tasknr=i) for i,n in enumerate(self.linkedNodes) 
                        if 'PointCloud' in str(type(n)) and 
                            n.resource is None and
                            n.path.endswith('.e57') and 
                            os.path.exists(n.path)]

            results2=[executor.submit(gmu.pcd_to_arrays,path=n.path,percentage=percentage,tasknr=i) for i,n in enumerate(self.linkedNodes) 
                        if 'PointCloud' in str(type(n)) and 
                            n.resource is None and
                            n.path.endswith('.pcd') and 
                            os.path.exists(n.path)]
            results3=[executor.submit(gmu.mesh_to_arrays,path=n.path,tasknr=i) for i,n in enumerate(self.linkedNodes) 
                        if ('MeshNode' in str(type(n)) or 'BIMNode' in str(type(n))) and 
                            n.resource is None and
                            os.path.exists(n.path)]
            results4=[executor.submit(gmu.img_to_arrays,path=n.path,tasknr=i) for i,n in enumerate(self.linkedNodes) 
                        if 'ImageNode' in str(type(n)) and 
                        n.resource is None and
                        os.path.exists(n.path)]

        # next, the arrays are assigned to point clouds outside the loop.
        # Note that these loops should be in parallel as they would otherwise obstruct one another.
        for r1 in concurrent.futures.as_completed(results1): 
            resource=gmu.arrays_to_pcd(r1.result())
            self.linkedNodes[r1.result()[-1]].resource=resource
        for r2 in concurrent.futures.as_completed(results2): 
            resource=gmu.arrays_to_pcd(r2.result())
            self.linkedNodes[r2.result()[-1]].resource=resource
        for r3 in concurrent.futures.as_completed(results3): 
            # print(len(r3.result()[0]))
            resource=gmu.arrays_to_mesh(r3.result())
            self.linkedNodes[r3.result()[-1]].resource=resource
        for r4 in concurrent.futures.as_completed(results4): 
            self.linkedNodes[r4.result()[-1]].resource=r4.result()[0]
        return [n.resource for n in self.linkedNodes]

    def get_resource(self)->o3d.geometry.TriangleMesh: 
        """Returns the convexhull of the node. \n
        If none is present, it will search for the data on drive from path, graphPath, name or subject. \n

        Returns:
            o3d.geometry.TriangleMesh or None
        """
        if self._resource is not None and len(self._resource.triangles)>=2:
            return self._resource
        elif self.get_path():
            self._resource   =  o3d.io.read_triangle_mesh(self.path)
        elif self.linkedNodes:
            points=o3d.utility.Vector3dVector()
            for node in self.linkedNodes: 
                if getattr(node,'get_bounding_box',None):           
                    box=node.get_bounding_box()
                    if (box):
                        points.extend(box.get_box_points()) 
            pcd= o3d.geometry.PointCloud()
            pcd.points=points
            self._resource=pcd.compute_convex_hull()   
        return self._resource  

    
    def get_metadata_from_resource(self) -> bool:
        """Returns the metadata from a resource. \n

        Features:
            cartesianTransform\n
            orientedBoundingBox\n
            cartesianBounds\n
            orientedBounds \n

        Returns:
            bool: True if exif data is successfully parsed
        """
        if (not self.resource or
            len(self.resource.triangles) <2):
            return False    

        try:
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
           
#################################
def create_node(graph: Graph = None, graphPath: str =None, subject: URIRef = None, resource = None, **kwargs)-> Node:
    """_summary_

    Args:
        graph (Graph, optional): _description_. Defaults to None.
        graphPath (str, optional): _description_. Defaults to None.
        subject (URIRef, optional): _description_. Defaults to None.

    Returns:
        Node (PointCloudNode,MeshNode,GeometryNode,ImageNode)
    """
    #input validation
    if(graphPath and not graph):
            graph = Graph().parse(graphPath)
    if(graph and not subject):
        subject=next(graph.subjects(RDF.type))
    if (subject and graph):    
        nodeType = ut.literal_to_string(graph.value(subject=subject,predicate=RDF.type))
    elif (resource):
        if type(resource) is o3d.geometry.PointCloud:
            nodeType='PointCloudNode'
        elif type(resource) is o3d.geometry.TriangleMesh:
            nodeType='MeshNode'
        elif type(resource) is o3d.geometry:
            nodeType='GeometryNode'
        elif type(resource) is np.ndarray:
            nodeType='ImageNode'        
    else:        
        nodeType = 'Node'

    #node creation
    if 'BIMNode' in nodeType:
        node=BIMNode(graph=graph, graphPath=graphPath, resource=resource,subject=subject, **kwargs)
    elif 'MeshNode' in nodeType:
        node=MeshNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'GeometryNode' in nodeType:
        node=GeometryNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'PointCloudNode' in nodeType:
        node=PointCloudNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'ImageNode' in nodeType:
        node=ImageNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'SessionNode' in nodeType:
        node=SessionNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)  
    else:
        node=Node(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs) 
    return node
