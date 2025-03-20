# Change Log
All notable changes to this project will be documented in this file.

## Template
Use this template to document all future changes

```md
## [Version] - yyyy-mm-dd
 
'summary'

### Added

### Removed
 
### Changed
 
### Fixed
```

<!-------CHANGES-------->
## [0.0.9] - 2022-06-08
 
Documentation update

### Added
- Automatic versioning

### Removed
 
### Changed
 
### Fixed

## [0.0.8] - 2022-05-30
 
Node refactoring

### Added
- `get_resource()`functions in all nodes

### Removed
 
### Changed
- proper initialisation of all the Nodes
- `Node` **kwargs inheritance
- Path variables cleanUp
  - `sessionPath`: The path of the folder containg the session
  - `path`: The relative path of the resource
 
### Fixed


## [Unreleased] - 2022-05-24
 
Finishing The node
 
### Added
- `Node` `get_resource()` & `get_resource_path()` to get the files from the different nodes
- `Test`folder to store unit tests

### Removed

 
### Changed
- `Node` `get_folder_path()` to be more robust and raise errors where needed
- `SessionNode` Added full node parsing of a Graph
 
### Fixed
- Incorrect parsing of `cartesianBounds` in utils


## [Unreleased] - 2022-05-20
 
Updating the Node
 
### Added
- `get_attribute_from_predicate()` to reliabilly get all teh attributes

### Removed
- `predicate_to_attribute()`
 
### Changed
- `SessionNode` Parsing from Graph
 
### Fixed
- Incorrect parsing of attributes


## [Unreleased] - 2022-05-19
 
Updating the Node
 
### Added
- first test file in `test`

### Removed
- 'label' from `INT_ATTRIBUTES` 
 
### Changed
- Ordering `linkeddatatools`
 
### Fixed
- Incorrect Node constructor


## [Unreleased] - 2022-05-18
 
Updating the Node
 
### Added
- numpy.quaternion in `PointcloudNode`
- supported extensions in `utils`

### Removed
- scipy.spacial.rotation in `PointcloudNode`
- Obsolete double inits in node child classes
- SUPPORTED_POINT_FIELDS
 
### Changed
- Node init function parses the graphs with a subject or path
- option to choose subject in node creation
- started sorting utils
 
### Fixed


## [0.0.7] - 2022-05-18
 
Documentation Update
 
### Added

### Removed
- Removed the public folder, source is now in `docs/`
 
### Changed
- renamed the `GEOMAPI.txt` to `CHANGELOG.md`
- the location of the pics folder is now `docs/pics/`
- `.gitlab-ci.yml` refactored to build to public folder
 
### Fixed

## [2.0.0] - 2024-03-11
 
Test fixes
 
### Added

### Removed
- Removed the public folder, source is now in `docs/`
 
### Changed
- PointcloudNode carthesian check
- the location of the pics folder is now `docs/pics/`
- `.gitlab-ci.yml` refactored to build to public folder
 
### Fixed

<!-------EXTRA NOTES-------->

# Functionalities

## Work method

### input protection/ error handling
- try catch blokken
- zoveel mogelijk protection binnen functie
- type hinting input + output

### documentatie (uitleg, eenheid , datatype)
- uitleg bij functie
- uitleg bij parameters
- type hinting input + output
	
### standardisatie
- beperk custom classes waar mogelijk 
- beperk aantal argumenten
- gebruik zoveel mogelijk standaardconcepten
- name is zonder speciale tekens of spaties (door windows en RDFlIB)
	
### PYTHON
- module/files: allemaal kleine letters
- classes: Capital elk woord
- functie: allemaal kleine letters met underscores
- variables: kleine letters, allemaal aan elkaar, 2de woord capital
- niet teveel afkortingen
	
## FUNCTIONS
- histogram in a certain direction
- split horizontal from vertical points
- create virtual image
- triplestore
- documentation
- fix graphPath constructor
- crop point clouds
- collision point clouds
- make imagenodes from vlx metadata

### KWALITEITSANALYSE
- rendering van KWALITEITSANALYSE

### CROP MESHES
- check mesh collision functions
- crop_geometry first run a check if the Bboxes intersect => saves alot of intersection calculations
- crop_geometry add non_overlapping bool parameter that culls mesh with each selection => should speed up process
- crop geometry functions have a lot of deepcopies which hurt performance
- intersection based on convex hull

- crop geometry 1m30 on 600 nodes with 40k mesh => for 2M mesh and 6000 boxes this would take 1h30
- trimesh mesh intersection 2m17 for 600 nodes on 40k mesh =>for 2M mesh and 6000 boxes 1000min

### CROP PCDS
- Python has to memory allocation protection :'(
- create CC script thats loads pcd's and outputs pcd
- segment large pcd's into regions
- BBoxes pcd are to large

- crop BB 4m30s for 6x10M pcd 600 boxes => for 45 clouds and 6000 boxes (normal project) this would take 5h30
- segmentation based on Eucl. distance 5ms for 6x10M pcd 600 meshes => for 45 clouds and 6000 boxes (normal project) this would take 6h

### GEOMETRY
- protect oriented bounding box en bounding box tegen puntenwolken met te weinig punten (coplanair of 1pnt)
- e57 scanheader also contains oriented bounding box?

### LINKED DATA 
- functie node.get_oriented_bounding_box()

### IMAGENODE
- implementeer xml import

### SESSIONNODE
- path variable is useless here
- nodes or sessionNode should contain links to session
-  