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
#IMPORT PACKAGES
import concurrent
import ifcopenshell
import numpy as np
import os
import open3d as o3d

from rdflib import RDF, XSD, Graph, Namespace, URIRef
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import List, Optional,Tuple,Union

#IMPORT MODULES
from geomapi.nodes import Node
from geomapi.nodes.imagenode import ImageNode
import geomapi.utils as ut
from geomapi.utils import rdf_property, GEOMAPI_PREFIXES
import geomapi.utils.geometryutils as gmu

class SetNode(Node):

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
                linkedNodes: List = None,
                linkedSubjects: List = None,
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

        #set properties (protected inputs)   
        self.linkedNodes=linkedNodes
        self.linkedSubjects= linkedSubjects


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

    #---------------------linkedNodes----------------------------
    @property
    def linkedNodes(self): 
        """Get the linkedNodes (Node) of the node."""
        return self._linkedNodes

    @linkedNodes.setter
    def linkedNodes(self,list:List[Node]):
        if list is None:
            self._linkedNodes = None
            return
        list=ut.item_to_list(list)
        if all('Node' in str(type(value)) for value in list):
            self._linkedNodes = list
        else:
            raise ValueError('Some elements in self.linkedNodes are not Nodes')    
    
    #---------------------linkedSubjects----------------------------
    @property
    @rdf_property(predicate=GEOMAPI_PREFIXES['geomapi'].hasPart, serializer = lambda uris: list(map(str, uris)))
    def linkedSubjects(self): 
        """Get the linkedSubjects (URIRef) of the node."""
        if not self._linkedSubjects and self.linkedNodes:
            self._linkedSubjects= [node.subject for node in self.linkedNodes]
        return self._linkedSubjects

    @linkedSubjects.setter
    def linkedSubjects(self,list:List[URIRef]):
        if list is None:
            self._linkedSubjects = None
            return
        list=ut.item_to_list(list)
        if all('URIRef' in str(type(value)) for value in list):
            self._linkedSubjects = list
        else:
            raise ValueError('Some elements are not URIRefs')
                    
                    
                    
                    
                    
#---------------------METHODS----------------------------

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
            elif self.linkedNodes is not None:
                all_geometries = o3d.geometry.TriangleMesh()
                for node in self.linkedNodes:
                    all_geometries += node.convexHull
                self.convexHull =  gmu.get_convex_hull(all_geometries)
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

    def get_graph(self, graphPath: Path = None, overwrite: bool = True, save: bool = False, base: URIRef = None, serializeAttributes: List = None, addLinkedNodes: bool = True) -> Graph:
        """Serialize the set's linkedNodes

        Args:
            - graphPath (Path) : The path of the graph to parse.
            - overwrite (bool) : Overwrite the existing graph or not.
            - base (str | URIRef) : BaseURI to match subjects to in the graph (improves readability) e.g. http://node#. Also, the base URI is used to set the subject of the graph. RDF rules and customs apply so the string must be a valid URI (http:// in front, and # at the end).
            - save (bool) : Save the graph to the self.graphPath or graphPath.
            - serializeAttributes (List(str)) : a list of attributes defined in the node that also need to be serialized
        
        Returns:
            Graph with linkedNodes
        """ 
        # Create the base graph of this Node, don't save yet
        self._graph = super().get_graph(graphPath=graphPath, 
                                      overwrite=overwrite, 
                                      save=False, base=base, 
                                      serializeAttributes=serializeAttributes)

        print(self._graph.serialize())
        if(addLinkedNodes and overwrite and self.linkedNodes):
            #Add the linked nodes to the graph
            for node in self.linkedNodes:
                node.get_graph(graphPath=graphPath, 
                               overwrite=overwrite, 
                               save=False, 
                               base=base, 
                               serializeAttributes=serializeAttributes)
                self._graph += node.graph

        # Save graph if requested
            if save:
                self.save_graph(graphPath)

        return self._graph

    

    def save_linked_resources(self,directory:str=None):
        """Export the resources of the linkedNodes.

        Args:
            directory (str, optional) : directory folder to store the data.

        Returns:
            bool: return True if export was succesful
        """ 
        if not self.linkedNodes:
            print("No linked Nodes defined")
            return
        for node in self.linkedNodes:
            node.save_resource(directory)  


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
        
        super().transform(transformation = transformation, 
                          rotation = rotation, 
                          translation = translation,
                          rotate_around_center=rotate_around_center)
        
        if(self.linkedNodes):
            for node in self.linkedNodes:
                # First set the node to the center in relation to the setNode
                #node.transform(np.linalg.inv(self.cartesianTransform))
                # Perform the transformation
                node.transform(transformation = transformation, 
                               rotation = rotation, 
                               translation = translation,
                               rotate_around_center=rotate_around_center)
                # Set the node back
                #node.transform(self.cartesianTransform)

    def show(self):
        if(self.linkedNodes is None or len(self.linkedNodes) == 0):
            print("No linkedNodes present")
            return
        geometries = []
        for node in self.linkedNodes:
            if(node.resource is None):
                geometries.append(gmu.mesh_get_lineset(node.convexHull))
            elif(isinstance(node.resource, (o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, o3d.geometry.LineSet))):
                geometries.append(node.resource)
            elif(isinstance(node, ImageNode)):
                frustum_lines, image_plane = gmu.create_camera_frustum_mesh_with_image(
                node.cartesianTransform,
                node.imageWidth, 
                node.imageHeight, 
                node.focalLength35mm, 
                node.depth,
                image_cv2=self.resource)
                geometries.append(frustum_lines)
                geometries.append(image_plane)
            else:
                geometries.append(gmu.mesh_get_lineset(node.convexHull))
        gmu.show_geometries(geometries)




    #def save_resource(self,directory:str=None,extension :str = '.ply') -> bool:
    #    """Export the resource (Convex hull) of the Node.
#
    #    Args:
    #        directory (str, optional) : directory folder to store the data.
    #        extension (str, optional) : file extension. Defaults to '.ply'.
#
    #    Raises:
    #        ValueError: Unsuitable extension. Please check permitted extension types in utils._init_.
#
    #    Returns:
    #        bool: return True if export was succesful
    #    """ 
    #    # perform the path check and create the directory
    #    if not super().save_resource(directory, extension):
    #        return False
#
    #    #write files
    #    if o3d.io.write_triangle_mesh(str(self.path), self.resource):
    #        return True
    #    return False



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

    #def get_linked_resources(self,percentage:float=1.0):
    #    """Returns the resources of the linkedNodes. 
    #    If none is present, it will search for the data on drive from path, graphPath, name or subject. 
    #    Otherwise, it will be reconstructed from the metadata present
#
    #    Args:
    #        - self (setNode)
    #        - percentage(float,optional) : load percentage of point cloud resources in present PointCloudNodes.
#
    #    Returns:
    #        list[resource] or None
    #    """
    #    for node in self.linkedNodes:
    #        if 'PointCloud' in str(type(node)):
    #            node.get_resource(percentage)
    #        else:
    #            node.get_resource()
    #    return [n.resource for n in self.linkedNodes]
    #  
    #def get_linked_resources_multiprocessing(self,percentage:float=1.0):
    #    """Returns the resources of the linkedNodes by multi-processing the imports. 
    #    If none is present, it will search for the data on drive from path, graphPath, name or subject. 
#
    #    **NOTE**: Starting parallel processing takes a bit of time. As such, this method will only outperform get_linked_resources with 10+ linkedNodes
#
    #    Args:
    #        - self (setNode)
    #        - percentage(float,optional) : load percentage of point cloud resources in present PointCloudNodes.
#
    #    Returns:
    #        list[resource] or None
    #    """
    #    [n.get_path() for n in self.linkedNodes]
#
    #    with concurrent.futures.ProcessPoolExecutor() as executor:
    #        # first load all data and output it as np.arrays      
    #        results1=[executor.submit(gmu.e57_to_arrays,e57Path=n.path,e57Index=n.e57Index,percentage=percentage,tasknr=i) for i,n in enumerate(self.linkedNodes) 
    #                    if 'PointCloud' in str(type(n)) and 
    #                        n.resource is None and
    #                        n.path.endswith('.e57') and 
    #                        os.path.exists(n.path)]
#
    #        results2=[executor.submit(gmu.pcd_to_arrays,path=n.path,percentage=percentage,tasknr=i) for i,n in enumerate(self.linkedNodes) 
    #                    if 'PointCloud' in str(type(n)) and 
    #                        n.resource is None and
    #                        n.path.endswith('.pcd') and 
    #                        os.path.exists(n.path)]
    #        results3=[executor.submit(gmu.mesh_to_arrays,path=n.path,tasknr=i) for i,n in enumerate(self.linkedNodes) 
    #                    if ('MeshNode' in str(type(n)) or 'BIMNode' in str(type(n))) and 
    #                        n.resource is None and
    #                        os.path.exists(n.path)]
    #        results4=[executor.submit(gmu.img_to_arrays,path=n.path,tasknr=i) for i,n in enumerate(self.linkedNodes) 
    #                    if 'ImageNode' in str(type(n)) and 
    #                    n.resource is None and
    #                    os.path.exists(n.path)]
#
    #    # next, the arrays are assigned to point clouds outside the loop.
    #    # Note that these loops should be in parallel as they would otherwise obstruct one another.
    #    for r1 in concurrent.futures.as_completed(results1): 
    #        resource=gmu.arrays_to_pcd(r1.result())
    #        self.linkedNodes[r1.result()[-1]].resource=resource
    #    for r2 in concurrent.futures.as_completed(results2): 
    #        resource=gmu.arrays_to_pcd(r2.result())
    #        self.linkedNodes[r2.result()[-1]].resource=resource
    #    for r3 in concurrent.futures.as_completed(results3): 
    #        # print(len(r3.result()[0]))
    #        resource=gmu.arrays_to_mesh(r3.result())
    #        self.linkedNodes[r3.result()[-1]].resource=resource
    #    for r4 in concurrent.futures.as_completed(results4): 
    #        self.linkedNodes[r4.result()[-1]].resource=r4.result()[0]
    #    return [n.resource for n in self.linkedNodes]
