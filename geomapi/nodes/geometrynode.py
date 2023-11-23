"""
**GeometryNode** is an abstract Python geometry class to govern the data and metadata of geometric data (Mesh, BIM, PointClouds).
It is the base class for the MeshNode, PointCloudNode and BIMNode classes. It contains the base concepts to manipulate cartesianBounds, orientedBounds, orientedBoundingBoxes, etc.
\n
**IMPORTANT**: the GeometryNode class is an archytype class from which specific data classes (e.g. PointCloudNode) inherit.
Do not use this class directly if you can use a child class with more functionality.

"""
#IMPORT MODULES
import numpy as np
from geomapi.nodes import Node
import geomapi.utils.geometryutils as gmu
import open3d as o3d 
from rdflib import Graph, URIRef

class GeometryNode (Node):  
    def __init__(self,  graph: Graph = None, 
                        graphPath: str = None,
                        subject: URIRef = None,
                        path:str=None,
                        cartesianBounds:np.array = None,
                        orientedBounds:np.array = None,
                        orientedBoundingBox:o3d.geometry.OrientedBoundingBox = None,
                        **kwargs):
        """Creates a GeometryNode from one or more of the following inputs. If successful, it will set the following attributes.\n

        Features:
            1. cartesianBounds(np.array [6x1]): [xMin,xMax,yMin,yMax,zMin,zMax]\n
            2. orientedBounds (np.ndarray [8x3]): bounding points of OrientedBoundingBox \n
            3. orientedBoundingBox (o3d.geometry.OrientedBoundingBox)\n

        Args:
            1.graph (Graph, optional): An RDF Graph to parse.\n
            2.graphPath (str, optional): The path of an RDF Graph to parse. If no subject is provided, the first subject of the graph is retained. \n
            3.path(str,optional): A filepath to a resource. \n
            4.subject (URIRef, optional): A subject to use as identifier for the Node. If a graph is also present, the subject should be part of the graph.\n
            5.name (str, optional): A name of the Node. This is not a unique identifier but serves as non-functional description.\n

        """
        #private attributes
        self._cartesianBounds=None      
        self._orientedBounds=None      
        self._orientedBoundingBox=None 

        super().__init__(   graph= graph,
                            graphPath= graphPath,
                            subject= subject,
                            path=path,        
                            **kwargs) 
        #instance variables
        self.cartesianBounds=cartesianBounds      
        self.orientedBounds=orientedBounds      
        self.orientedBoundingBox=orientedBoundingBox 

#---------------------PROPERTIES----------------------------

    #---------------------cartesianBounds----------------------------
    @property
    def cartesianBounds(self): 
        """Get the cartesianBounds of the node from various inputs.

        Args:
            1.np.array(6x1), list (6 elements) \n
            2.Vector3dVector (n elements)\n
            3.orientedBounds (np.array(8x3))\n 
            4.Open3D.geometry.OrientedBoundingBox\n
            5.Open3D geometry\n
        
        Returns:
            cartesianBounds (np.array [6x1]) [xMin,xMax,yMin,yMax,zMin,zMax]
        """
        return self._cartesianBounds

    @cartesianBounds.setter
    def cartesianBounds(self,value):
        if value is None:
            return None
        try: #lists, np.arrays
            self._cartesianBounds=np.reshape(value,6)
        except:
            try: #orientedBounds
                box=gmu.get_oriented_bounding_box(value)
                min=box.get_min_bound()
                max=box.get_max_bound()
                self._cartesianBounds=np.array([min[0],max[0],min[1],max[1],min[2],max[2]])
            except:
                try: #orientedBoundingBox
                    min=value.get_min_bound()
                    max=value.get_max_bound()
                    self._cartesianBounds=np.array([min[0],max[0],min[1],max[1],min[2],max[2]])
                except:
                    try: #Vector3dVector
                        box=gmu.get_oriented_bounding_box(np.asarray(value))
                        min=box.get_min_bound()
                        max=box.get_max_bound()
                        self._cartesianBounds=np.array([min[0],max[0],min[1],max[1],min[2],max[2]])
                    except:
                        try:#resource
                            self._cartesianBounds=gmu.get_cartesian_bounds(self._resource)
                        except:
                            raise ValueError('Input must be cartesianBounds (np.array [6x1]): [xMin,xMax,yMin,yMax,zMin,zMax], list like object, orientedBounds (np.Array(8x3)), or Open3D Bounding Box.')

#---------------------orientedBounds----------------------------
    @property
    def orientedBounds(self): 
        """Get the 8 bounding points of the Node from various inputs.\n
        
        Args:
            1. orientedBounds (np.array(nx3)), list (24 elements) or Vector3dVector (8 elements)\n
            2. Open3D.geometry.OrientedBoundingBox\n
            3. Open3D geometry\n
        
        Returns:
            orientedBounds (np.ndarray [8x3])
        """
        return self._orientedBounds

    @orientedBounds.setter
    def orientedBounds(self,value):
        if value is None:
            return None
        try: #array or list
            self._orientedBounds=np.reshape(value,(8,3))
        except:
            try: #orientedBoundingBox
                self._orientedBounds=np.asarray(value.get_box_points())
            except:
                try: #Vector3dVector
                    array=np.asarray(value)
                    self._orientedBounds=np.reshape(array,(8,3))
                except:
                    try:#resource
                        self._orientedBounds=np.asarray(value.get_oriented_bounding_box().get_box_points())
                    except:
                        raise ValueError('Input must be orientedBounds (np.ndarray [8x3]), list like object (len(24)), Vector3dVector (8 elements), or Open3D geometry required')

#---------------------orientedBoundingBox----------------------------
    @property
    def orientedBoundingBox(self): 
        """Get the orientedBoundingBox of the Node from various inputs.

        Args:
            1. Open3D.geometry.OrientedBoundingBox\n
            2. Open3D geometry\n
            3. orientedBounds (np.array(nx3)) or Vector3dVector\n

        Returns:
            orientedBoundingBox (o3d.geometry.OrientedBoundingBox)
        """
        return self._orientedBoundingBox

    @orientedBoundingBox.setter
    def orientedBoundingBox(self,value):
        if value is None:
            return None
        if 'orientedBoundingBox' in str(type(value)):
            self._orientedBoundingBox=value
        else:    
            try: #geometry
                self._orientedBoundingBox=value.get_oriented_bounding_box()
            except:
                try: #np.array(nx3)
                    points=o3d.utility.Vector3dVector(value)                    
                    self._orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(points)
                except:
                    try: #Vector3dVector
                        self._orientedBoundingBox=o3d.geometry.OrientedBoundingBox.create_from_points(points)
                    except:
                        raise ValueError('Input must be orientedBoundingBox (o3d.geometry.OrientedBoundingBox), an Open3D Geometry or a list of Vector3dVector objects')

#---------------------Methods----------------------------
        
    def set_resource(self,value):
        """Set the resource of the node.\n

        Args:
            value (Open3D.geometry.PointCloud or Open3D.geometry.TriangleMesh)

        Raises:
            ValueError: Resource must be Open3D.geometry
        """
        if 'open3d' in str(type(value)):
            self._resource=value
        else:
            raise ValueError('Resource must be Open3D.geometry')

    def get_resource(self):
        """Returns the geometry data in the node. \n
        If none is present, it will search for the data on drive from path, graphPath, name or subject. 

        Args:
            1. self (Geometry)\n
            2. percentage (float,optional): percentage of point cloud to load. Defaults to 1.0 (100%).\n

        Returns:
            o3d.geometry.PointCloud or None
        """
        if self._resource is not None:
            return self._resource
        elif self.get_path():
            if self.path.endswith('pcd'):
                resource =  o3d.io.read_point_cloud(self.path)
                self._resource  =resource
            elif self.path.endswith('e57'):
                self._resource = gmu.e57path_to_pcd(self.path) 
        return self._resource  

    def get_oriented_bounding_box(self)->o3d.geometry.OrientedBoundingBox:
        """Gets the Open3D OrientedBoundingBox of the node from various inputs.

        Args:
            1. cartesianBounds\n
            2. orientedBounds\n
            3. cartesianTransform\n
            4. Open3D geometry\n

        Returns:
            o3d.geometry.orientedBoundingBox
        """
        if self._orientedBoundingBox is not None:
            pass
        elif self._orientedBounds is not None:
            self._orientedBoundingBox=gmu.get_oriented_bounding_box(self._orientedBounds)
        elif self.cartesianBounds is not None:
            self._orientedBoundingBox=gmu.get_oriented_bounding_box(self._cartesianBounds)   
        elif self._cartesianTransform is not None:
                box=o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                boundingbox= box.get_oriented_bounding_box()
                translation=gmu.get_translation(self._cartesianTransform)
                self._orientedBoundingBox= boundingbox.translate(translation)
        elif self._resource is not None:
            try:
                self._orientedBoundingBox=self._resource.get_oriented_bounding_box()
            except:
                return None
        else:
            return None
        return self._orientedBoundingBox

    def set_cartesianTransform(self,value):
        """Set the cartesianTransform of the node from various inputs.\n
        
        Args:
            1. cartesianTransform(np.ndarray(4x4)) \n
            2. np.ndarray or Vector3dVector (1x3)  \n
            3. cartesianBounds (np.ndarray (6x1))\n
            4. np.ndarray or Vector3dVector (8x3 or nx3)\n
            5. Open3D.geometry\n
        """        
        try: #np.ndarray (4x4) 
            self._cartesianTransform=np.reshape(value,(4,4))
        except:
            try: #np.ndarray or Vector3dVector (1x3)  
                self._cartesianTransform=gmu.get_cartesian_transform(translation=np.asarray(value))
            except:  
                try: # cartesianBounds (np.ndarray (6x1))
                    self._cartesianTransform=gmu.get_cartesian_transform(cartesianBounds=np.asarray(value))
                except:
                    try: # np.ndarray or Vector3dVector (8x3 or nx3)
                        center=np.mean(np.asarray(value),0)
                        self._cartesianTransform=gmu.get_cartesian_transform(translation=center)
                    except:
                        try: # Open3D.geometry
                            self._cartesianTransform=gmu.get_cartesian_transform(translation=value.get_center())
                        except:
                            raise ValueError('Input must be np.ndarray(6x1,4x4,3x1,nx3), an Open3D geometry or a list of Vector3dVector objects.')

    def get_cartesian_transform(self) -> np.ndarray:
        """Get the cartesianTransform of the node from various inputs.\n

        Args:
            1. cartesianBounds\n
            2. orientedBounds\n
            3. orientedBoundingBox\n
            4. Open3D geometry\n
            5. list of Vector3dVector objects\n

        Returns:
            cartesianTransform(np.ndarray(4x4))
        """
        if self._cartesianTransform is not None:
            pass
        elif self._cartesianBounds is not None:
            self._cartesianTransform=gmu.get_cartesian_transform(cartesianBounds=self._cartesianBounds)
        elif self._orientedBounds is not None:
            center=np.mean(self._orientedBounds,0)
            self._cartesianTransform=gmu.get_cartesian_transform(translation=center)
        elif self._orientedBoundingBox is not None:
            self._cartesianTransform=gmu.get_cartesian_transform(translation=self._orientedBoundingBox.get_center())
        elif self._resource is not None:
            self._cartesianTransform=gmu.get_cartesian_transform(translation=self._resource.get_center())
        else:
            return None
        return self._cartesianTransform

    def get_cartesian_bounds(self) -> np.ndarray:
        """Get the cartesianBounds of the node from various inputs.\n

        Args:
            1. orientedBounds\n
            2. orientedBoundingBox\n
            3. resource (Open3D.geometry)\n

        Returns:
            cartesianBounds (np.array [6x1])        
        """
        if self._cartesianBounds is not None:
            pass
        elif self._orientedBounds is not None:
            box=gmu.get_oriented_bounding_box(self._orientedBounds)
            self._cartesianBounds= gmu.get_cartesian_bounds(box)
        elif self._orientedBoundingBox is not None:
            self._cartesianBounds=  gmu.get_cartesian_bounds(self._orientedBoundingBox)
        elif self._resource is not None:
             self._cartesianBounds=  gmu.get_cartesian_bounds(self._resource)
        else:
            return None
        return self._cartesianBounds

    def get_oriented_bounds(self) -> np.ndarray:
        """Get the 8 bounding points of the node from various inputs.\n

        Args: 
            1. cartesianBounds\n
            2. orientedBoundingBox\n
            3. resource (Open3D.geometry)\n

        Returns:
            OrientedBounds (np.ndarray(8x3))
        """
        if self._orientedBounds is not None:
            pass
        elif self._cartesianBounds is not None:
            self._orientedBounds=np.asarray(gmu.get_oriented_bounds(self._cartesianBounds))
        elif self._orientedBoundingBox is not None:
            self._orientedBounds=  np.asarray(self._orientedBoundingBox.get_box_points())
        elif self._resource is not None:
            box=self.resource.get_oriented_bounding_box()
            self._orientedBounds=  np.asarray(box.get_box_points())
        else:
            return None
        return self._orientedBounds

    def get_center(self) -> np.ndarray:
        """Returns the center of the node.

        **NOTE** the result may vary based on the metadata in the node.\n

        Args:        
            1. cartesianBounds\n
            2. orientedBoundingBox\n
            3. resource (Open3D.geometry)\n
            4. cartesianTransform\n
            5. orientedBoundingBox\n

        Returns:
            center (np.ndarray(3x1))
        """        
        if self._cartesianBounds is not None:
            return gmu.get_translation(self._cartesianBounds)
        elif self._orientedBounds is not None:
            return gmu.get_translation(self._orientedBounds)
        elif self._cartesianTransform is not None:
            return gmu.get_translation(self._cartesianTransform)
        elif self._orientedBoundingBox is not None:
            return self._orientedBoundingBox.get_center()
        elif self._resource is not None:
            return self._resource.get_center()
        else:
            return None    

    def visualize(self):
        """Visualise the node's geometry (PointCloud or TriangleMesh).
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if getattr(self,'mesh',None) is not None:
            vis.add_geometry(self.mesh)
        elif getattr(self,'pcd',None) is not None:
            vis.add_geometry(self.pcd)
        else:
            return None
        vis.run()
        vis.destroy_window()