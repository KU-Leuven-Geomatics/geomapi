---
marp: false
paginate : true
headingDivider: 4
---
# Nodes

GEOMAPI manages geospatial data as linked data resources, represented as RDF Graphs. The framework inputs include various types of close-range sensing data, such as geolocated imagery, point clouds, polygonal meshes, as well as construction data comprising BIM and CAD geometries. From these inputs, a series of metadata is extracted and serialized into RDF Graphs, that are compatible with efficient retrieval functions. 

GEOMAPI currently defines a general Node type with 7 data classes governed by 3 supertypes: **Node** is the most general one, **GeometryNode** for geometric inputs and **ImageNode** for image inputs. Each data class inherits functionality from the supertypes and extracts the metadata of its respective resource.

![bg vertical right:50% h:80%](../../pics/GEOMAPI_metadata3.png)


## Node
The general **Node** class serves as a template and stores the metric information in accordance with the conceptual framework outlined in the [OpenLabel Standard](https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f), including:

- cartesianTransform: 4x4 transformation matrix with pose information T = [R t; 0^T 1]
- cartesianBounds: tupple with (xmin, xmax, ymin, ymax, zmin, zmax) 
- Oriented Bounding Box:  9x1 matrix (x, y, z, Rx(θ), Ry(φ), Rz(ψ), sx, sy, sz) which includes:
  - The location (x, y, z)
  - The rotation around the three cardinal axes applied in the order $R_z$ (yaw), $R_y$ (pitch), $R_x$ (roll)
  - The size (sx, sy, sz) in each direction.
- Convex Hull: Bounding volume stored by its bounding points as an nx3 matrix: [x1 y1 z1;x2 y2 z2; ...]

Non-metric information includes:
- The timestamp
- subject: RDF conform URI
- name: an rdfs label describing the resource. This can be any text i.e. 'mynode&)!'
- The type of coordinate system
- The path to the resource
- Additional resource-specific details such as image width.

### Initilisation
Nodes can be initialised from a range of inputs i.e. the name or a subject. This is only used when no more specific node class can be used e.g. when its a new sort of data.

```py
from geomapi.nodes import Node
node= Node(subject='myNode',
            cartesianTransform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])) # create an empty node with a subject

{'_subject': rdflib.term.URIRef('file:///myNode'), '_graph': None, '_graphPath': None, '_path': None, '_name': None, '_timestamp': None, '_resource': None, '_cartesianTransform': None}
```


An important aspect of the Node system is its **RDF Graph** functionality. Each Node can be translated into a Graph and vice versa. This allows to store information about a resource in a database system, and efficiently retrieve information without having to load the actual resources. 

```py
from geomapi.nodes import Node
# Create a node from a graph 
node= Node(graph=rdflib.Graph()) 

# Create a series of nodes from a graph file
nodes=tl.graph_path_to_nodes('../tests/testfiles/graphs/graph.ttl') 

# Write a node to a Graph
graph=node.to_graph()
```

See the tutorial on [Nodes](../tutorial/tutorial_nodes.ipynb) for more examples on how to create nodes.


## GeometryNode

The geometry classes in GEOMAPI specify data and metadata for resources characterized by exact geometric properties, i.e., those whose boundaries are precisely defined by their geometric resource. Currently, we include

1. **PointCloudNode**: For point cloud data, capturing detailed spatial information from LiDAR and photogrammetry.
2. **MeshNode**: For polygonal meshes, representing 3D surfaces and structures.
3. **BIMNode**: For Building Information Models, linking to detailed architectural and structural data.
4. **LineSetNode**: For CAD geometries, representing lines and simple geometric shapes.

![geometrynode](../../pics/geometrynode.png)

```py
from geomapi.nodes import  PointCloudNode, MeshNode, BIMNode
pcd=o3d.io.read_point_cloud('../tests/testfiles/pcd/parking.pcd')
pcdNode = PointCloudNode(resource=pcd) # built from resource or data
meshNode= MeshNode (path='../tests/testfiles/mesh/parking.obj') # .stl and .obj are supported
bimNodes=tl.graph_path_to_nodes('../tests/testfiles/graphs/graph.ttl') #loads nodes from a graph file representing an IFCfile with BIM objects.
```

## ImageNode

The image classes in GEOMAPI are designed to define data and metadata for geolocated image-based resources, such as those processed by Structure-from-Motion (SfM) pipelines or Mobile Mapping Systems (MMS).

1. **CameraNode**: For conventional imagery, capturing images from standard cameras.
2. **PanoNode**: For panoramic imagery, capturing 360-degree views.
3. **OrthoNode**: For orthomosaic imagery, representing top-down images often used in mapping and surveying.

![alt text](../../pics/imagenode.png)

```py
from geomapi.nodes import ImageNode, PointCloudNode, MeshNode, BIMNode, Node
pcd=o3d.io.read_point_cloud('../tests/testfiles/pcd/parking.pcd')
pcdNode = PointCloudNode(resource=pcd) # built from resource or data
meshNode= MeshNode (path='../tests/testfiles/mesh/parking.obj') # .stl and .obj are supported
imgNode=ImageNode(xmpPath='../tests/testfiles/img/DJI_0085.xmp') # .xmp contains pose information from CapturingReality software. MetaShape .xml is also supported.
bimNodes=tl.graph_path_to_nodes('../tests/testfiles/graphs/graph.ttl') #loads nodes from a graph file representing an IFCfile with BIM objects.
```