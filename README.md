

![geomapiLogo](docs/source/_static/geomapi_logo.png?width=64)

# GEOMAPI
[![PyPI version](https://badge.fury.io/py/geomapi.svg)](https://badge.fury.io/py/geomapi)
[![Coverage Status](https://coveralls.io/repos/github/KU-Leuven-Geomatics/geomapi/badge.svg?branch=main)](https://coveralls.io/github/KU-Leuven-Geomatics/geomapi?branch=main)

This innovative toolbox, developped by [KU Leuven Geomatics](https://iiw.kuleuven.be/onderzoek/geomatics), jointly processes close-range sensing resources (point clouds, images) and BIM models for the AEC industry. 
More specifically, we combine [semantic web technologies](https://en.wikipedia.org/wiki/Semantic_Web) with state-of-the-art open source geomatics APIs
to process and analyse big data in construction applications.

## Installation

Use the package manager [pip](https://pypi.org/project/geomapi) to install geomapi.

```bash
pip install geomapi
```

## Documentation

You can read the full API reference here:
[Documentation](https://ku-leuven-geomatics.github.io/geomapi/index.html)


## Quickstart

```py
import geomapi
from geomapi.nodes import Node

newNode = Node()
```

## Contributing

The master branch is protected and you can only make changes by submitting a merge request. 
Please create a new branch if you would like to make changes and submit them for approval.

## Citation
If you want to cite us, refer to the following publication (accepted). 
```
@article{geomapi,
    title={Processing 3D data with semantic web technologies},
    author={Bassier M., Vermandere J., De Geyter S. and De Winter H.},
    booktitle={Automation in Construction},
    year={2024}
}
```
## TEAM
- maarten.bassier@kuleuven.be (PI)
- jelle.vermandere@kuleuven.be
- sam.degeyter@kuleuven.be
- heinder.dewinter@kuleuven.be
---
![team](docs/source/_static/geomapi_team.PNG?width=64)

## Licensing
The code in this project is licensed under MIT license.
