---
marp: true
paginate : true
headingDivider: 4
---
# Getting started

GEOMAPI manages close-range sensing resources like images and point clouds. It greatly epands the functionality of Open Source projects such as OpenCV and Open3D to create homogeneous and easy to use resources. It has three levels.


1. Utilities




## Installation

You will need a python environment $ 3.7 \leq x \leq 3.10$ as Open3D currently doesn't support python $\leq 3.10$. Use the package manager [pip](https://pypi.org/project/geomapi) to install geomapi.

```bash
pip install geomapi
```

## Creating your first node

All Node types inherit the Base Node so they can be created in a very similar way.
A Node Can be initialised using a number of different parameters

```py
from geomapi.node import Node
newNode = Node()
```

### Without a Graph

If no Graph or -path is provided, You can create an empty node of any type with a (random) subject ID and no metadata.

```py
newNode = Node(subject = "myNode")
```

### With Graph

You can create a Node using a Graph or -Path, this will parse all the variables inside the Graph. If a subject is provided, it will use that specific subject for the Graph.

```py
newNode = Node(graphPath = "http://www.w3.org/People/Berners-Lee/card")
```

## Node Types

Currently, the API supports 6 Specialised Nodes:

```py
from geomapi.nodes import ImageNode()
from geomapi.nodes import GeometryNode()
from geomapi.nodes import PointcloudNode()
from geomapi.nodes import MeshNode()
from geomapi.nodes import BIMNode()
from geomapi.nodes import LinesetNode()
from geomapi.nodes import OrthoNode()
from geomapi.nodes import SessionNode()
```

Each Node Can be created just like a regular node, but it has extra inputs to assing variables.
Check out the tutorial section for info about each specific node.

## Working with Sessions

Since each node Only points to 1 resource, we use Sessions to group them.
A Session is a collection of Resources that are taken in a single recording session. This means that all they are all in the same coordinate system and are documenting the same information.
Check out the tutorial section about SessionNodes to get started.

## Importing sub-packages

The packages are typically abriviated using the first letter of each word: validationtools -> vt, in case there are doubles, insert the first different letter inbetween: geometryutils -> gmu. Packages with a single word become the first letter + the first constant: tools -> tl, utils -> ut

## Further Reading

Check out the testcases to learn about practical uses for the package
