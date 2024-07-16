---
marp: false
paginate : true
headingDivider: 4
---
# Functionality

The cool thing about these Nodes is that they can be jointly querried. We can find neighbouring nodes way faster by not using the actual data!

```py
import geomapi.tools as tl
tl.select_nodes_k_nearest_neighbors(pcdNode,[meshNode,imgNode,bimNodes],k=1) #selects the k nearest neighbors of a point cloud node from a list of nodes

([<geomapi.nodes.meshnode.MeshNode at 0x1d6ea7c2170>], DoubleVector[2.09905]) # the meshNode is the closest Node 2m away!
```

GEOMAPI divides functions into three layers.
1. [**Utilities**](../tutorial/tutorial_functionality.ipynb): Base functions for point clouds and images that support the Node system
```py
import geomapi.utils as ut
from geomapi.utils import geometryutils as gmu
```
2. [**Nodes**](../tutorial/tutorial_functionality.ipynb): Node specific functions such as projecting rays from an ImageNode in 3D, mutations, surface sampling, etc.
```py
imgNode=ImageNode(xmlPath='../tests/testfiles/img/road.xml', path='../tests/testfiles/img/101_0367_0007.JPG') 
imgNode.create_rays(imagePoints=[[0,0],[0,1],[1,0],[1,1]],depths=25) #creates rays from image points
```
3. [**Tools**](../tutorial/tutorial_functionality.ipynb): Functions that combine nodes for distance calculations, intersections, analyses, etc.
```py
import geomapi.tools as tl
import geomapi.tools.progresstools as pgt
pgt.project_pcd_to_rgbd_images (pointClouds, imgNodes, depth_max=15)
```

Look at the [Function Tutorial](../tutorial/tutorial_nodes.ipynb) notebooks to learn more about how to use the different functions.