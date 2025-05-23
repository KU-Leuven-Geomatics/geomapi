

![geomapiLogo](https://raw.githubusercontent.com/KU-Leuven-Geomatics/geomapi/main/docs/source/_static/geomapi_logo_Blue.png?width=64)

# GEOMAPI
![Python](https://img.shields.io/pypi/pyversions/geomapi.svg?logo=python&logoColor=FBE072)
![License](https://img.shields.io/pypi/l/geomapi)
[![PyPI version](https://badge.fury.io/py/geomapi.svg)](https://badge.fury.io/py/geomapi)
[![Coverage Status](https://coveralls.io/repos/github/KU-Leuven-Geomatics/geomapi/badge.svg?branch=main)](https://coveralls.io/github/KU-Leuven-Geomatics/geomapi?branch=main)

This innovative toolbox, developped by [KU Leuven Geomatics](https://iiw.kuleuven.be/onderzoek/geomatics), jointly processes close-range sensing resources (point clouds, images) and BIM models for the AEC industry. 
More specifically, we combine [semantic web technologies](https://en.wikipedia.org/wiki/Semantic_Web) with state-of-the-art open source geomatics APIs
to process and analyse big data in construction applications.

## Installation

Use the package manager [pip](https://pypi.org/project/geomapi) to install geomapi as a user.

```bash
conda create --name geomapi_user python=3.11
pip install geomapi
```

Or as a developer, install the dependencies from the root folder through the command line.

```bash
pip install -r requirements.txt
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
If you want to cite us, refer to the following publication (published). 
```
@article{GEOMAPI,
title = {GEOMAPI: Processing close-range sensing data of construction scenes with semantic web technologies},
journal = {Automation in Construction},
volume = {164},
pages = {105454},
year = {2024},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2024.105454},
url = {https://www.sciencedirect.com/science/article/pii/S0926580524001900},
author = {Maarten Bassier and Jelle Vermandere and Sam De Geyter and Heinder De Winter},
keywords = {Geomatics, Semantic Web Technologies, Construction, Close-range sensing, BIM, Point clouds, Photogrammetry}
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
