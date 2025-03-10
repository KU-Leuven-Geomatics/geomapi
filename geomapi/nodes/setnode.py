"""
**SetNode** - a Python Class to govern the data and metadata of remote sensing data captured in the same epoch, or that all code from the same file.

This node builds upon the [OpenCV](https://opencv.org/), [Open3D](https://www.open3d.org/) and [PIL](https://pillow.readthedocs.io/en/stable/) API for the image definitions.
Be sure to check the properties defined in those abstract classes to initialise the Node.

.. image:: ../../../docs/pics/graph_set_1.png

**NOTE**: This node does not represent sensory data, but rather a group of data. The resource is a convex hull of all the linkedNodes.

Goals:
- Given a path, import all the linked images, meshes, ect... into a set
- Convert non-RDF metadata files (json, xml, ect..) to setsNodes and export them to RDF
- get the boundingbox of the whole set
- use URIRef() to reference the images, ect...

"""  
from rdflib import Graph, URIRef, Literal,Namespace,XSD
from rdflib.namespace import RDF
import open3d as o3d 
import cv2
import numpy as np
import os
from typing import List, Optional,Tuple,Union
import uuid
import concurrent.futures
from pathlib import Path
import ifcopenshell
import open3d as o3d

#IMPORT MODULES
from geomapi.nodes import *
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu

class SetNode(Node):

    def __init__(self,  graph : Graph = None, 
                        graphPath: str=None,
                        subject : URIRef = None,
                        linkedNodes : Node = None,
                        resource = None,
                        cartesianTransform: Optional[np.ndarray] = None,
                        orientedBoundingBox: Optional[o3d.geometry.OrientedBoundingBox] = None,
                        convexHull: Optional[o3d.geometry.TriangleMesh] =None,
                        getResource : bool = False,
                        **kwargs):
        """
        Creates a SetNode & all the child nodes. Overloaded function.
        This Node can be initialised from one or more of the inputs below.
        By default, no data is imported in the Node to speed up processing.
        If you also want the data, call node.get_resource() or set getResource() to True.

        Args:
            - subject (RDFlib URIRef) : subject to be used as the main identifier in the RDF Graph
            
            - graph (RDFlib Graph) : Two possible graphs are parsed:
                - The RDF Graph of only the setNode (1 subject)
                - AN RDF Graph with only resourceNodes
                
            - graphPath (Path) :  The path of the Graph of only the set.
            
            - path (Path) : Path to the convexhull resource of the setNode

            - resource (o3d.geometry) : o3d.geometry.TriangleMesh of the convexhull of the setNode or a list of resources to be converted to LinkedNodes (PointCloudNode, MeshNode, ImageNode, etc.)
                        
            - LinkedNodes (Nodes, optional) : A set of Nodes (ImageNodes, MeshNodes) to contain within the set 
            
            - getResource (bool, optional= False) : If True, the node will search for its physical resource on drive 
            
            - getMetaData (bool, optional= True) : If True, the node will attempt to extract geometric metadata from the resource if present (cartesianBounds, etc.) 
        
        Returns:
           
            - Args(subject) : create a new Graph and Node with the given subject
            
            - Args(Graph) : parse the graph with the given subject
                - 1 or more are matched: Use that node as the SetNode
                - 1 or more are found but not matched: Raise an error
                - None are found: Create a new node with the given subject
            
            - Args(resources) : create a setNode and linkedNodes from the resources
            
            - Args(linkedNodes) : create a new Graph from the joint metadata
        """
        #private attributes 
        self._linkedNodes=[]
        self._linkedSubjects=[]
        self._subject=None
        self._graph=None
        self._graphPath=None
        self._path=None
        self._name=None
        
        #initialise these attributes as they are otherwise overwritten by the super().__init__ method
        self._resource=None
        self._orientedBoundingBox=None
        self._convexHull=None
        self._cartesianTransform=None        
        self.orientedBoundingBox=orientedBoundingBox
        self.convexHull=convexHull
        self.cartesianTransform=cartesianTransform

        #instance variables
        self.resource=resource
        self.linkedNodes=linkedNodes    
        self.graphPath=graphPath     
        self.subject=subject

        #initialisation functionality
        if(graphPath and not graph):
            graph = Graph().parse(graphPath)

        if(graph):
            self.parse_set_graph(graph) 

        if (self._subject is None):
            self.subject=str(uuid.uuid1())
            
        super().__init__(   graph= self._graph,
                            graphPath= self._graphPath,
                            subject= self._subject,
                            resource=self._resource,
                            getResource=getResource,
                            orientedBoundingBox=self._orientedBoundingBox,
                            convexHull=self._convexHull,
                            cartesianTransform=self._cartesianTransform,
                            **kwargs) 
        
        # self.get_metadata_from_linked_nodes()
        
#---------------------PROPERTIES----------------------------

    #---------------------linkedNodes----------------------------
    @property
    def linkedNodes(self): 
        """Get the linkedNodes (Node) of the node."""
        return self._linkedNodes

    @linkedNodes.setter
    def linkedNodes(self,list:List[Node]):
        list=ut.item_to_list(list)
        if not list or list[0] is None:
            return []
        if list is None:
            pass
        elif all('Node' in str(type(list)) for list in ut.item_to_list(list)):
            self.set_linked_nodes(list)
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
    def linkedSubjects(self,list:List[URIRef]):
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
                    self._linkedSubjects.append(URIRef(prefix+ut.validate_string(string)) )
                    
                    
                    
                    
                    
#---------------------METHODS----------------------------

    
    
    
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
        if self._cartesianTransform is None and self.get_convex_hull() is not None:
            self._cartesianTransform = gmu.get_cartesian_transform(translation=self._convexHull.get_center())
        # if self._cartesianTransform is None and self._orientedBoundingBox is not None:
        #     self._cartesianTransform = gmu.get_cartesian_transform(translation=self._orientedBoundingBox.get_center())
        # if self._cartesianTransform is None and len(self.linkedNodes) >0:
        #     points=o3d.utility.Vector3dVector()
        #     for node in self.linkedNodes: 
        #         transform=node.get_cartesian_transform()
        #         #take the mean of this transform
        #         t=gmu.get_translation(transform)
        #         t=np.reshape(t,(1,3))
        #         p=o3d.utility.Vector3dVector(t)
        #         points.extend(p)
        #     self._cartesianTransform = gmu.get_cartesian_transform(translation=np.mean( np.asarray(points),0))
        if self._cartesianTransform is None:
            self._cartesianTransform = np.eye(4)
        
        return self._cartesianTransform
    
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
        if self._orientedBoundingBox is None and self.get_convex_hull() is not None:
            self._orientedBoundingBox = gmu.get_oriented_bounding_box(self._convexHull)
        # if self._orientedBoundingBox is None and len(self.linkedNodes) >0:
        #     points=o3d.utility.Vector3dVector()
        #     for node in self.linkedNodes: 
        #         box=node.get_oriented_bounding_box()
        #         points.extend(box.get_box_points()) 
        #     self._orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(points)
        # if self._orientedBoundingBox is None and self._cartesianTransform is not None: 
        #     box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        #     box.translate([-0.5, -0.5, -0.5])
        #     box.transform(self._cartesianTransform)
        #     self._orientedBoundingBox = box.get_oriented_bounding_box()
        return self._orientedBoundingBox
     
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
        if self._convexHull is None and len(self.linkedNodes) >0:
            points=o3d.utility.Vector3dVector()
            for node in self.linkedNodes: 
                hull=node.get_convex_hull()
                points.extend(hull.vertices)
            pcd= o3d.geometry.PointCloud()
            pcd.points=points
            hull, _ =pcd.compute_convex_hull()
            self._convexHull = hull
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
            
    def get_linked_nodes(self,resourceGraph:Graph,linkedSubjects:List[URIRef] = None):
        """Get self.linkedNodes based on a resourceGraph and list of linkedSubjects or self.linkedSubjects.

        Args:
            - resourceGraph (Graph): contains the target nodes' metadata
            - linkedSubjects (List[URIRef],optional)

        Returns:
            linkedNodes (List[Node])
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
    
    def set_linked_nodes(self,linkedNodes:List[Node]):
        """Update self.linkedNodes from a new linkedNodes list

        Args:
            linkedNodes (List[Node]): 
        """        
        linkedNodes=ut.item_to_list(linkedNodes) 
        
        #filter out the nodes that are already in the list
        linkedNodes=[node for node in linkedNodes if node.subject not in self.linkedSubjects]        
        if len(linkedNodes) == 0:
            return
        
        #add the new nodes to the list
        [self._linkedNodes.append(node) for node in linkedNodes]
        
        #update the linkedSubjects
        if self._graph is None:
            self.set_convex_hull(linkedNodes)
            self.set_oriented_bounding_box(linkedNodes)
            self.set_cartesian_transform(linkedNodes)
            self.set_resource(self.get_convex_hull())
        self.set_linked_subjects([node.subject for node in linkedNodes]) #these two lists are kept up to date

    def set_convex_hull(self, value):
        """Set the convex hull for the Node from various inputs. 
        
        Args:
            - convexHull (o3d.geometry.TriangleMesh)
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVectord 
        """
        if isinstance(value, o3d.geometry.TriangleMesh):
            self._convexHull = value
        elif isinstance(value, List):
            points=o3d.utility.Vector3dVector()
            for node in value:
                points.extend(node.get_convex_hull().vertices)
            pcd= o3d.geometry.PointCloud()
            pcd.points=points
            hull, _ =pcd.compute_convex_hull()
            self._convexHull = hull    
        else:
            self._convexHull=gmu.get_convex_hull(value)
            
    def set_oriented_bounding_box(self, value):
        """Set the oriented bounding box for the Node.
        
        Args:
            - orientedBoundingBox (o3d.geometry.OrientedBoundingBox)
            - Open3D geometry
            - set of points (np.array(nx3)) or Vector3dVector
            - 9 parameters $[x,y,z,e_x,e_y,e_z, R_x,R_y,R_z]$
        """
        if isinstance(value, o3d.geometry.OrientedBoundingBox):
            self._orientedBoundingBox = value
        elif isinstance(value, List):
            # points=o3d.utility.Vector3dVector()
            # for node in value:
            #     points.extend(node.get_oriented_bounding_box().get_box_points())
            # pcd= o3d.geometry.PointCloud()
            # pcd.points=points
            self._orientedBoundingBox =self.get_convex_hull().get_oriented_bounding_box()
        else:
            self._orientedBoundingBox=gmu.get_oriented_bounding_box(value)
            
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
        elif isinstance(value, List):
            # points=o3d.utility.Vector3dVector()
            # for node in value:
            #     points.append(node.get_center())
            # pcd= o3d.geometry.PointCloud()
            # pcd.points=points
            self._cartesianTransform = gmu.get_cartesian_transform(translation=self.get_convex_hull().get_center())
        else:
            value = np.reshape(np.asarray(value), (-1))
            if value.shape == (16,):
                self._cartesianTransform = value.reshape((4, 4))
            elif value.shape == (3,):
                self._cartesianTransform = gmu.get_cartesian_transform(translation=value)
            else:
                raise ValueError('Input must be a numpy array of shape (4,4) or (3,).')
    
                            
    def set_linked_subjects(self,linkedSubjects : List[URIRef]=None):
        """Update self.linkedSubjects

        Args:
            linkedSubjects (List[URIRef], optional): 
        """   
        linkedSubjects=ut.item_to_list(linkedSubjects)
        [self.linkedSubjects.append(s) for s in linkedSubjects if s not in self.linkedSubjects]
      
    def parse_set_graph(self,graph:Graph)-> Union[URIRef , Graph, Node]:
        """Parse a graph to detect and split setNodes from resourceNodes

        Args:
            graph (Graph): Graph that either contains a setNode, resourceNodes or a combination of both. 

        Raises:
            ValueError: If self.subject is provided, it should match with a subject in the graph

        Returns:
            Union[subject ,Graph, Nodelist]
        """        
        # Step 1: extract the setNodes
        subjects = graph.subjects(RDF.type)
        setNodeSubjects = []
        resourceNodeSubjects = []
        for sub in subjects:
            nodeType = ut.literal_to_string(graph.value(subject=sub,predicate=RDF.type))
            if 'SetNode' in nodeType:
                setNodeSubjects.append(sub)
            else:
                resourceNodeSubjects.append(sub)
        
        # Step 2: Get the setNode in the graph (if it exists)
        if setNodeSubjects:
            self.subject = setNodeSubjects[0]
            self.graph=ut.get_subject_graph(graph=graph,subject=self._subject)               
            if(len(setNodeSubjects) > 1): 
                print("More than one setNode is present, while no subject was provided, picked:",self.subject,"out of", setNodeSubjects)
        # if (not setNodeSubjects): 
        #     print("no setSubjects found")
        #     self.get_subject()

        # else: # there is 1 or more setNodes, search for a match
        #     if (not self._subject): # no subject was given, pick one from the list
        #         self.subject = setNodeSubjects[0]
        #         self.graph=ut.get_subject_graph(graph=graph,subject=self._subject)               
        #         if(len(setNodeSubjects) > 1): 
        #             print("More than one setNode is present, while no subject was provided, picked:",self.subject,"out of", setNodeSubjects)
        #     else: # Check if the subject is in the list
        #         if (self._subject not in setNodeSubjects):
        #             raise ValueError("The given subject is not in the Graph or is not a setNode")

        # Step 3: Parse all the other Nodes into the nodelist
        nodelist=[]
        for subject in resourceNodeSubjects:
            s=subject.toPython() #temp
            newGraph=ut.get_subject_graph(graph=graph,subject=subject)
            nodelist.append(create_node(graph = newGraph ,graphPath= self.graphPath,subject= subject))
        if nodelist:
            self.set_linked_nodes(nodelist)
        
        return self.subject, self.graph, self.linkedNodes
    
    def linked_nodes_to_graph(self,graphPath:Path=None,overwrite: bool = True, base: URIRef = None,save:bool=False) -> Graph:
        """Serialize the set's linkedNodes

        Args:
            - graphPath (Path) : The path of the graph to parse.
            - overwrite (bool) : Overwrite the existing graph or not.
            - base (str | URIRef) : BaseURI to match subjects to in the graph (improves readability) e.g. http://node#. Also, the base URI is used to set the subject of the graph. RDF rules and customs apply so the string must be a valid URI (http:// in front, and # at the end).
            - save (bool) : Save the graph to the self.graphPath or graphPath.
        
        Returns:
            Graph with linkedNodes
        """ 
        
        if len(self.linkedNodes) == 0:
            print('No linkedNodes present')
            return
        else:
            #create graph
            g = ut.bind_ontologies(Graph())
            #create graph scope
            if base:
                inst = Namespace(base)
                g.bind("inst", inst)            
            for node in self.linkedNodes:
                node.get_graph(graphPath,base=base)
                g+= node.graph    
            if(graphPath and save):
                g.serialize(graphPath)   
        return g  

    def set_to_graph(self,graphPath:Path=None,overwrite: bool = True,base: URIRef = None,save:bool=False) -> Graph:
        """Serialize the set's linkedNodes and own metadata.

        Args:
            - graphPath (str,optional) : defaults to combinedGraph.ttl in set graphPath location.
            - save (bool, optional) : Defaults to False.

        Returns:
            Graph with linkedNodes and setNode
        """
        if graphPath:
            self.graphPath = graphPath   
            
        if self._graph is None or overwrite:
            #create graph
            g = ut.bind_ontologies(Graph())
            
            for node in self.linkedNodes:
                node.get_graph(graphPath,base=base)
                g+= node.graph
            g+=self.get_graph(graphPath,base=base)
            if(graphPath and save):
                g.serialize(graphPath)     
        return g 
    

    def save_linked_resources(self,directory:str=None):
        """Export the resources of the linkedNodes.

        Args:
            directory (str, optional) : directory folder to store the data.

        Returns:
            bool: return True if export was succesful
        """ 
        for node in self.linkedNodes:
            node.save_resource(directory)  

    def save_resource(self,directory:str=None,extension :str = '.ply') -> bool:
        """Export the resource (Convex hull) of the Node.

        Args:
            directory (str, optional) : directory folder to store the data.
            extension (str, optional) : file extension. Defaults to '.ply'.

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
        if directory and os.path.exists(os.path.join(directory,self.get_name() + extension)):
            self.path=os.path.join(directory,self.get_name() + extension)
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
            directory=os.path.join(dir,'SET')   
        else:
            directory=os.path.join(os.getcwd(),'SET')
        # create directory if not present
        if not os.path.exists(directory):                        
            os.mkdir(directory) 

        self.path=os.path.join(directory,Path(self.subject.toPython()).stem  + extension) 

        #write files
        if o3d.io.write_triangle_mesh(str(self.path), self.resource):
            return True
        return False

    def resources_to_linked_nodes(self,resources) -> None:
        """Create linked Nodes from a set of data resources.

        **NOTE**: Images, ortho and panos are not processed as their resources don't have a geometry.
        
        Args:
            - resources (List[mesh,pcd,lineset,etc.])

        Returns:
            None 
        """
        nodelist=[]
        for resource in ut.item_to_list(resources):
            #check type
            if isinstance(resource,o3d.geometry.TriangleMesh): 
                nodelist.append(MeshNode(resource=resource))
            elif isinstance(resource,o3d.geometry.PointCloud):
                nodelist.append( PointCloudNode(resource=resource))
            elif isinstance(resource,o3d.geometry.LineSet):
                nodelist.append( LineSetNode(resource=resource))
            elif isinstance(resource,ifcopenshell.entity_instance):
                nodelist.append( BIMNode(resource=resource))
            # elif isinstance(np.asarray(resource),np.ndarray) and len(np.asarray(resource).shape) == 3:
            #     nodelist.append( ImageNode(resource=resource)) # we can't diffirentiate between ortho and image and panorama
            else:
                print('Resource type not supported')
        if nodelist:
            self.set_linked_nodes(nodelist)

    # def get_metadata_from_linked_nodes(self):
    #     """Returns the setNode metadata from the linkedNodes. 

    #     Returns:
    #         - cartesianTransform
    #         - orientedBoundingBox 
    #         - convexHull
    #     """
    #     if self._graph:
    #         return True

    #     if getattr(self,'timestamp',None) is None and self.linkedNodes:
    #         self.timestamp=self.linkedNodes[0].get_timestamp()

    #     points=o3d.utility.Vector3dVector()        
    #     for node in self.linkedNodes: 
    #         if getattr(node,'get_oriented_bounding_box',None) is not None:           
    #             box=node.get_oriented_bounding_box()
    #             if (box):
    #                 points.extend(box.get_box_points())    
                
    #         elif getattr(node,'cartesianTransform',None) is not None:
    #             t=gmu.get_translation(node.get_cartesian_transform())
    #             t=np.reshape(t,(1,3))
    #             p=o3d.utility.Vector3dVector(t)
    #             points.extend(p)

    #     if np.asarray(points).shape[0] >=5:    
    #         self.cartesianTransform=gmu.get_cartesian_transform(translation=np.mean( np.asarray(points),0)) 
    #         self.orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(points)
    #         self.orientedBounds=np.asarray(self.orientedBoundingBox.get_box_points())
    #         self.cartesianBounds=gmu.get_cartesian_bounds(o3d.geometry.AxisAlignedBoundingBox.create_from_points(points))
    #         pcd= o3d.geometry.PointCloud()
    #         pcd.points=points
    #         hull, _ =pcd.compute_convex_hull()
    #         self.resource=hull

    def get_linked_resources(self,percentage:float=1.0):
        """Returns the resources of the linkedNodes. 
        If none is present, it will search for the data on drive from path, graphPath, name or subject. 
        Otherwise, it will be reconstructed from the metadata present

        Args:
            - self (setNode)
            - percentage(float,optional) : load percentage of point cloud resources in present PointCloudNodes.

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
        """Returns the resources of the linkedNodes by multi-processing the imports. 
        If none is present, it will search for the data on drive from path, graphPath, name or subject. 

        **NOTE**: Starting parallel processing takes a bit of time. As such, this method will only outperform get_linked_resources with 10+ linkedNodes

        Args:
            - self (setNode)
            - percentage(float,optional) : load percentage of point cloud resources in present PointCloudNodes.

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
        elif isinstance(value,List):
            self.resources_to_linked_nodes(value) 
            self._resource=  self.get_convex_hull()
        else:
            raise ValueError('Resource must be ao3d.geometry.TriangleMesh or a list of nodes')
    
    def get_resource(self)->o3d.geometry.TriangleMesh: 
        """Returns the convexhull of the node. If none is present, it will search for the data on drive from path, graphPath, name or subject. 
        
        **NOTE**: The resource is only loaded if len(resource.triangles) >2.

        Returns:
            o3d.geometry.TriangleMesh or None
        """
        if self._resource is None and self.get_path() :
            resource =  o3d.io.read_triangle_mesh(str(self.path))
            if len(resource.triangles)>2:
                self._resource = resource
        else:
            self._resource=self.get_convex_hull()   
        return self._resource  



#################################
def create_node(graph: Graph = None, graphPath: str =None, subject: URIRef = None, resource = None, **kwargs)-> Node:
    """Create a Node from a graph, graphPath, subject or resource.

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
        if isinstance(resource,o3d.geometry.TriangleMesh): 
            nodeType = 'MeshNode'
        elif isinstance(resource,o3d.geometry.PointCloud):
            nodeType = 'PointCloudNode'
        elif isinstance(resource,o3d.geometry.LineSet):
            nodeType = 'LineSetNode'
        elif isinstance(resource,ifcopenshell.entity_instance):
            nodeType = 'BIMNode'
        # elif isinstance(np.asarray(resource),np.ndarray) and len(np.asarray(resource).shape) == 3:
        #     nodelist.append( ImageNode(resource=resource)) # we can't diffirentiate between ortho and image and panorama
        else:
            print('Resource type not supported') 
    else:        
        nodeType = 'Node'

    #node creation
    if 'BIMNode' in nodeType:
        node=BIMNode(graph=graph, graphPath=graphPath, resource=resource,subject=subject, **kwargs)
    elif 'MeshNode' in nodeType:
        node=MeshNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'PointCloudNode' in nodeType:
        node=PointCloudNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'ImageNode' in nodeType:
        node=ImageNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'setNode' in nodeType:
        node=SetNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)  
    elif 'LineSetNode' in nodeType:
        node=LineSetNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'OrthoNode' in nodeType:
        node=OrthoNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'PanoNode' in nodeType:
        node=PanoNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
        
    else:
        node=Node(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs) 
    return node
