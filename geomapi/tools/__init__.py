"""Different tools to Manage RDF data."""

#IMPORT PACKAGES
import csv
import inspect
import math
import cv2
import numpy as np 
import open3d as o3d 
import os 
import re
import pye57 
import pandas as pd
import xml.etree.ElementTree as ET 
from typing import List,Tuple, Union
from pathlib import Path
import ezdxf
from scipy.spatial.transform import Rotation as R

# import APIs
import rdflib
from rdflib import URIRef, Literal,Namespace,Graph
from rdflib.namespace import RDF
import ifcopenshell
import ifcopenshell.util
import ifcopenshell.util.selector

import multiprocessing
import concurrent.futures

#IMPORT MODULES 
import geomapi
from geomapi.nodes import *
import geomapi.utils as ut
from geomapi.utils import GEOMAPI_PREFIXES
import geomapi.utils.geometryutils as gmu
import geomapi.utils.cadutils as cadu
from warnings import warn

#ONTOLOGIES
loa=rdflib.Namespace('https://docplayer.net/131921614-Usibd-level-of-accuracy-loa-specification-guide.html#')
ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')

#### NODE CREATION ####

### GRAPH ###
def graph_to_nodes(graphPath : str = None, graph: Graph = None, subjects: List = None, **kwargs) -> List[Node]:
    """Convert a graphPath to a set of Nodes.

    Args:
        0. graphPath (str):  absolute path to .ttl RDF Graph\n
        1. kwargs (Any) \n

    Returns:
        A list of pointcloudnodes, imagenodes, meshnodes, bimnodes, orthonodes with metadata 
    """    
    """
    Parses RDF graph, finds subjects of type geomapi:*Node, and instantiates corresponding Python classes.
    
    :param graph: rdflib.Graph object
    :return: dict mapping subject URIs to class instances
    """
    instances = []
    if(not graph):
        if(not graphPath):
            raise ValueError("No graph or graphPath provided")
        else:
            graph = Graph().parse(graphPath)
    # Build a map of class names available in the given module
    class_map = {name: cls for name, cls in inspect.getmembers(geomapi.nodes, inspect.isclass)}

    for subject in graph.subjects(RDF.type, None):
        if(subjects is not None): # Check the subject filter
            if(str(subject) not in subjects):
                continue
        for rdf_type in graph.objects(subject, RDF.type):
            if str(rdf_type).startswith(str(GEOMAPI_PREFIXES["geomapi"])):
                class_name = rdf_type.split("#")[-1]
                cls = class_map.get(class_name)
                if cls:
                    try: instances.append(cls(graph = graph, graphPath = graphPath, subject = subject,**kwargs))
                    except: print("Failed to create Node: ", subject)
                else:
                    print(f"⚠️ No matching class found for RDF type: {class_name}")

    return instances

### XML ###
def e57xml_to_pointcloud_nodes(xmlPath :str, **kwargs) -> List[PointCloudNode]:
    """Parse XML file that is created with E57lib e57xmldump.exe.
        E57 XML file structure
        e57Root
           >data3D
               >vectorChild
                   >pose
                       >rotation
                       >translation
                   >cartesianBounds
                   >guid
                   >name
                   >points recordCount
           >images2D

    Args:
        path (string):  e57 xml file path e.g. "D:\\Data\\2018-06 Werfopvolging Academiestraat Gent\\week 22\\PCD\\week 22 lidar_CC.xml"
        **kwargs: All extra arguments are applied to all nodes
            
    Returns:
        A list of pointcloudnodes with the xml metadata 
    """
    
    path=Path(xmlPath)
    mytree = ET.parse(path)
    root = mytree.getroot()
    nodelist=[]
    e57Path=xmlPath.with_suffix('.e57')
    # loop over every vectorchild
    for idx,e57xml in enumerate(root.iter('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}vectorChild')):
        # OrientedBoundingBox
        # TODO switch to OBB 
        cartesianBoundsnode=e57xml.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}cartesianBounds') 
        if cartesianBoundsnode is not None:
            try:
                OrientedBoundingBox=np.array([ut.xml_to_float(cartesianBoundsnode[0].text),
                                        ut.xml_to_float(cartesianBoundsnode[1].text),
                                        ut.xml_to_float(cartesianBoundsnode[2].text),
                                        ut.xml_to_float(cartesianBoundsnode[3].text),
                                        ut.xml_to_float(cartesianBoundsnode[4].text),
                                        ut.xml_to_float(cartesianBoundsnode[5].text)])
                OrientedBoundingBox=OrientedBoundingBox.astype(float)
                OrientedBoundingBox=np.nan_to_num(OrientedBoundingBox)
            except:
                OrientedBoundingBox=None

        # CartesianTransform
        posenode=e57xml.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}pose')
        if posenode is not None:
            rotationnode=posenode.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}rotation')
            if rotationnode is not None:               
                try:
                    quaternion=np.array([ ut.xml_to_float(rotationnode[3].text),
                                    ut.xml_to_float(rotationnode[0].text),
                                    ut.xml_to_float(rotationnode[1].text),
                                    ut.xml_to_float(rotationnode[2].text) ])
                    quaternion=quaternion.astype(float)   
                    quaternion=np.nan_to_num(quaternion)                
                except:
                    quaternion=np.array([0,0,0,1])
                r = R.from_quat(quaternion)
                rotationMatrix =r.as_matrix()

            translationnode=posenode.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}translation')
            if translationnode is not None: 
                try:
                    translationVector= np.array([ut.xml_to_float(translationnode[0].text),
                                                ut.xml_to_float(translationnode[1].text),
                                                ut.xml_to_float(translationnode[2].text)])
                    translationVector=translationVector.astype(float)
                    translationVector=np.nan_to_num(translationVector)       
                except:
                    translationVector=np.array([0.0,0.0,0.0])
            cartesianTransform = gmu.get_cartesian_transform(translation=translationVector,rotation=rotationMatrix)
            #print(cartesianTransform)

        pointsnode=e57xml.find('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}points')
        if not pointsnode is None:
            pointCount=int(pointsnode.attrib['recordCount'])
        else: pointCount = None
        nodelist.append(PointCloudNode(cartesianTransform=cartesianTransform, orientedBoundingBox=OrientedBoundingBox, pointCount= pointCount, e57XmlPath=path,e57Index=idx,path=e57Path,**kwargs))
    return nodelist

def xml_to_image_nodes(path :str,subjects:List = None, skip:int=None, filterByFolder:bool=False,**kwargs) -> List[ImageNode]:
    """Parse XML file that is created with https://www.agisoft.com/.

    Args:
        1.xmlPath (string or Path): xml file path e.g. "D:/Data/cameras.xml"
        2.subjects (List[str]): a list of labels, matching with the camera labels, that you want filtered.
        2.skip (int, Optional): select every nth image from the xml. Defaults to None.
        3.filterByFolder (bool, Optional): Filter imgNodes based on the images in the folder or not. Defaults to False.
            
    Returns:
        A list of ImageNodes with the xml metadata 
    """
    path=Path(path) if path else None
    assert skip == None or skip >0, f'skip == None or skip '
    
    #open xml
    mytree = ET.parse(path)
    root = mytree.getroot()  

    #get reference
    chunk=root.find('chunk')
    globalTransform=gmu.get_cartesian_transform(rotation=ut.literal_to_matrix(chunk.find('transform').find('rotation').text),
                                                translation= ut.literal_to_matrix(chunk.find('transform').find('translation').text))
    globalScale=float(chunk.find('transform').find('scale').text)

    #get components -> in some xml files, there are no components.
    components=[]
    for component in root.iter('component'):       
        try:
            transform=component.find('transform')
            region=component.find('region')
            scale=float(transform.find('scale').text)
            components.append({'componentid':  int(component.get('id')),        
                            'refTransform': gmu.get_cartesian_transform(rotation=ut.literal_to_matrix(transform.find('rotation').text),
                                                translation= ut.literal_to_matrix(transform.find('translation').text)),
                            'scale': scale,
                            'center': gmu.get_cartesian_transform( translation=ut.literal_to_matrix(region.find('center').text)),
                            'size': ut.literal_to_matrix(region.find('size').text),
                            'R': ut.literal_to_matrix(region.find('R').text)})     
        except:
            components.append(None)
            continue

    #get sensors
    sensors=[]
    for sensor in root.iter('sensor'):       
        try:            
            # Extract resolution
            resolution = sensor.find('resolution')
            image_width = int(resolution.attrib['width'])
            image_height = int(resolution.attrib['height'])

            # Extract sensor properties to get the focal length
            properties = {prop.attrib['name']: prop.attrib['value'] for prop in sensor.findall('property')}
            if('pixel_width' in properties and 'pixel_height' in properties and 'focal_length' in properties):
                pixel_width_mm = float(properties['pixel_width'])
                pixel_height_mm = float(properties['pixel_height'])
                focal_length_mm = float(properties['focal_length'])

                # Calculate sensor dimensions in mm
                sensor_width_mm = image_width * pixel_width_mm
                sensor_height_mm = image_height * pixel_height_mm
                sensor_diagonal_mm = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
                full_frame_diagonal_mm = math.sqrt(36**2 + 24**2)
                crop_factor = full_frame_diagonal_mm / sensor_diagonal_mm
                focal_length_35mm = focal_length_mm / crop_factor
            else:
                focal_length_35mm = None

            # find the calibration
            calibration=sensor.find('calibration')
            if(calibration is not None):
                f = float(calibration.find('f').text if calibration.find('f') is not None else calibration.find('fx').text) # sometimes the focal length is named diffirently
                cx = image_width / 2 + float(calibration.find('cx').text)
                cy = image_height / 2 + float(calibration.find('cy').text)
            else:
                # If no calibration found, fallback to defaults
                f = focal_length_mm / pixel_width_mm
                cx = image_width / 2
                cy = image_height / 2
            # Construct the intrinsic matrix
            K = np.array([
                [f,   0,  cx],
                [0,   f,  cy],
                [0,    0,   1]
            ])
            
            sensors.append({'sensorid':  int(sensor.get('id'))   ,        
                            'imageWidth': image_width,
                            'imageHeight': image_height,
                            'intrinsicMatrix': K,
                            'focalLength35mm': focal_length_35mm})     
        except Exception as error:
            print("An error occurred in the sensor parsing:", error) # An error occurred: name 'x' is not defined
            sensors.append(None)
            continue
    
    #get image names in folder
    files=ut.get_list_of_files(path.parent)
    # files=[f for f in files if (f.endswith('.JPG') or 
    #                             f.endswith('.PNG') or 
    #                             f.endswith('.jpg') or
    #                             f.endswith('.png'))] #! deprecated
    # files = [f for f in files if any(f.endswith(ext).upper() for ext in ut.IMG_EXTENSIONS)]
    files = [file for file in files if any(file.suffix.upper() == ext.upper() for ext in ut.IMG_EXTENSIONS)]

    names=[file.stem for file in files]


    #get cameras
    nodelist=[]   
    for cam in root.iter('camera'):
        try:
            #get name
            name=cam.get('label')

            # Skip over al not matching labels
            if(subjects is not None):
                if(name not in subjects):
                    continue
            
            #get component
            componentid=cam.get('component_id')  
            if componentid:
                componentInformation= next(c for c in components if c['componentid']==int(componentid))  
                refTransform=componentInformation['refTransform']
                scale=componentInformation['scale']
            else:
                refTransform=globalTransform
                scale=globalScale
                
            #get transform
            transform=np.reshape(ut.literal_to_matrix(cam.find('transform').text),(4,4))
            #apply scale and reference transformation
            transform=gmu.get_cartesian_transform(rotation=transform[0:3,0:3],
                                        translation=transform[0:3,3]*scale)
            transform=refTransform  @ transform

            #get sensor information
            sensorid=int(cam.get('sensor_id'))      
            sensorInformation= next(s for s in sensors if s is not None and s.get('sensorid')==sensorid)

            #create image node 
            node=ImageNode(
                        name=name, 
                        cartesianTransform = transform,
                        imageWidth =  sensorInformation['imageWidth'],
                        imageHeight = sensorInformation['imageHeight'],
                        focalLength35mm = sensorInformation['focalLength35mm'], 
                        **kwargs)
            # node.xmlPath=xmlPath
            
            #assign node to nodelist depending on whether it's in the folder    
            try:
                i=names.index(Path(node.name).stem)
                node.path=files[i]
                nodelist.append(node)   
            except:
                None if filterByFolder else nodelist.append(node) 

        except Exception as error:
            print("Parsing error:", error) # An error occurred: name 'x' is not defined
            continue
    return nodelist[0::skip] if skip else nodelist

### E57 ###
def e57_to_pointcloud_nodes(e57Path, subjects = None, percentage: float = None, loadResource = False, multiProcessingTreshold = 10, **kwargs) -> List[PointCloudNode]:

    
    nodelist=[]   
    e57 = pye57.E57(str(e57Path))   
    gmu.e57_update_point_field(e57)

    # If you need to load a large amount of pointclouds, use multiprocessing
    if(e57.scan_count >= multiProcessingTreshold and loadResource):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # first load all e57 data and output it as np.arrays
            results=[executor.submit(gmu.e57_to_arrays,e57Path=e57Path,e57Index=s,percentage=percentage) for s in range(e57.scan_count)]
            # next, the arrays are assigned to point clouds outside the loop.
            for s,r in enumerate(concurrent.futures.as_completed(results)):
                resource=gmu.arrays_to_pcd(r.result())
                nodelist.append(PointCloudNode(path=Path(e57Path),e57Index=s,resource=resource, loadResource=loadResource, **kwargs))
    else:
        for idx in range(e57.scan_count):
            if(percentage and loadResource):
                resource = gmu.e57_to_pcd(e57,e57Index=idx,percentage=percentage)
            else: resource = None
            nodelist.append(PointCloudNode(path=Path(e57Path),e57Index=idx,resource=resource, loadResource=loadResource, **kwargs))
    return nodelist

### DXF ###

def dxf_to_lineset_nodes(dxfPath:str |Path, **kwargs) -> List[LineSetNode]:
    """Parse a dxf file to a list of LineSetNodes. created by CAD software

    **NOTE**: be aware of the scale factor!

    Args:
        - dxfPath(str): absolute path to .dxf file
        - **kwargs: additional arguments to pass to the Line

    Returns:
        List[LineSetNode]
    """    
    dxfPath=Path(dxfPath) 
    print(f"Reading DXF file from {dxfPath}...")
    dxf = ezdxf.readfile(str(dxfPath))
    
    # check units
    if dxf.header.get('$INSUNITS') is not None and dxf.header.get('$INSUNITS') !=6:
        units=ezdxf.units.decode(dxf.header.get('$INSUNITS'))
        print(f'Warning: dxf has {units} units while meters are expected!') 

    #create entities
    nodelist=[]
    counter=0
    for entity in dxf.modelspace():    
        #get geometry
        g=cadu.ezdxf_entity_to_o3d(entity)
        #filter on linesets
        if isinstance(g,o3d.geometry.LineSet): 
            layer=entity.dxf.layer
            color=np.array(cadu.get_rgb_from_aci(dxf.layers.get(layer).dxf.color))/255
            g.paint_uniform_color(color)
            handle=entity.dxf.handle
            dxfType=entity.dxftype()
            name=getattr(entity.dxf,'name',None)
            node=LineSetNode(resource=g,
                            dxfPath=dxfPath,
                            layer=layer,
                            color=color,
                            handle=handle,
                            dxfType=dxfType,
                            name=name,
                            **kwargs)
            nodelist.append(node)
        else:
            counter+=1
            continue
    print(f'{counter} entities were not LineSets. Skipping for now...')
    print(f'    loaded {len(nodelist)} lineSetNodes from dxf file')
    return nodelist

def dxf_to_ortho_nodes(dxfPath: Union[str, Path], name_filter: str = None, **kwargs) -> List[OrthoNode]:
    """
    Parse a DXF file into a list of OrthoNode objects., created by Metashape

    Args:
        dxfPath (str | Path): Path to the DXF file.
        name_filter (str, optional): If provided, only include entities matching this name.
        **kwargs: Additional arguments passed to OrthoNode.

    Returns:
        List[OrthoNode]
    """
    dxfPath = Path(dxfPath)
    print(f"Reading DXF file from {dxfPath}...")
    dxf = ezdxf.readfile(str(dxfPath))
    entities = [entity for entity in dxf.modelspace()]
    orthonodes = []

    # Try to auto-detect name_filter if not provided
    if name_filter is None:
        for e in entities:
            if e.dxftype() == 'INSERT':
                try:
                    name_filter = Path(e.attribs[0].dxf.text).stem
                    break
                except Exception:
                    continue
        if name_filter is None:
            print("Warning: No valid INSERT entity with a name found. Skipping name filtering.")

    for i in range(0, len(entities) - 1, 2):
        entity1 = entities[i]
        entity2 = entities[i + 1]

        try:
            name = Path(entity1.attribs[0].dxf.text).stem
        except Exception:
            continue

        if name_filter and name != name_filter:
            continue

        g = cadu.ezdxf_entity_to_o3d(entity2)
        if not hasattr(g, "points"):
            continue

        arg_height = kwargs.get("height", 0)
        g.translate(np.array([0, 0, arg_height]))

        points = np.asarray(g.points)
        center = g.get_center()
        vec1 = points[1] - points[0]
        vec2 = points[3] - points[0]
        normal = np.cross(vec1, vec2)
        normal = normal / np.linalg.norm(normal)

        rotation_matrix = gmu.get_rotation_matrix_from_forward_up(normal, vec2)
        translation = center
        cartesian_transform = gmu.get_cartesian_transform(translation=translation, rotation=rotation_matrix)

        image_width = kwargs.get("imageWidth", 1)
        gsd = np.linalg.norm(vec1[0]) / image_width

        node = OrthoNode(
            name=name,
            cartesianTransform=cartesian_transform,
            dxfPath=dxfPath,
            imageWidth=image_width,
            imageHeight=kwargs.get("imageHeight"),
            gsd=gsd,
            **kwargs
        )
        orthonodes.append(node)

    print(f"Loaded {len(orthonodes)} OrthoNodes from DXF.")
    return orthonodes

#### IFC ########

def ifc_to_bim_nodes(path:str, classes:str = None,  guids:list = None, types:List = None, getResource : bool=True,**kwargs)-> List[BIMNode]:
    """
    Parse ifc file to a list of BIMNodes, one for each ifcElement.\n

    **NOTE**: classes are not case sensitive. It is advised to solely focus on IfcBuildingElement classes or inherited classes as these typically have geometry representations that can be used by GEOMAPI.

    **NOTE**: If you intend to parse 1000+ elements, use the multithreading of the entire file instead and filter the BIMNodes afterwards as it will be faster. 

    **WARNING**: IfcOpenShell struggles with some ifc serializations. In our experience, IFC4 serializations is more robust.

    .. image:: ../../../docs/pics/ifc_inheritance.PNG


    Args:
        1. ifcPath (string):  absolute ifc file path e.g. "D:/myifc.ifc"\n
        2. classes (string, optional): ifcClasses e.g. 'IfcBeam, IfcColumn, IfcWall, IfcSlab'. Defaults to 'IfcBuildingElement'.   
    
    Raises:
        ValueError: 'No valid ifcPath.'

    Returns:
        List[BIMNode]
    """   
    path=Path(path)
    
    nodelist=[]   
    ifc = ifcopenshell.open(path)
    timestamp=ut.get_timestamp(path)
    
    if(classes is not None):
        ifcElements = ifcopenshell.util.selector.filter_elements(ifc, str(classes))
    elif(guids is not None):
        ifcElements = [ut.item_to_list(ifc.by_id(guid)) for guid in guids]
    elif(types is not None):
        ifcElements = [ifc.by_type(type) for type in types]
    else:
        ifcElements = ifcopenshell.util.selector.filter_elements(ifc, 'IfcBuildingElement')

    for ifcElement in ifcElements:
        nodelist.append(BIMNode(timestamp=timestamp, ifcPath=path, resource=ifcElement,getResource=getResource, **kwargs))

    return nodelist

def ifc_to_nodes_multiprocessing(path:str, **kwargs)-> List[BIMNode]:
    """Returns the contents of geometry elements in an ifc file as BIMNodes.
    This method is 3x faster than other parsing methods due to its multi-threading.
    However, only the entire ifc can be parsed.

    **WARNING**: IfcOpenShell strugles with some ifc serializations. In our experience, IFC4 serializations is more robust.

    Args:
        ifcPath (str | Path): ifc file path e.g. "D:/myifc.ifc"

    Raises:
        ValueError: 'No valid ifcPath.'

    Returns:
        List[BIMNode]
    """
    path=Path(path)
    
    try:
        ifc_file = ifcopenshell.open(path)
    except:
        print(ifcopenshell.get_log())
    else: 
        nodelist=[]   
        timestamp=ut.get_timestamp(path)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True) 
        iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
        if iterator.initialize():
            while True:
                shape = iterator.get()
                ifcElement = ifc_file.by_guid(shape.guid) 
                faces = shape.geometry.faces # Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
                verts = shape.geometry.verts # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
                # materials = shape.geometry.materials # Material names and colour style information that are relevant to this shape
                # material_ids = shape.geometry.material_ids # Indices of material applied per triangle face e.g. [f1m, f2m, ...]

                # Since the lists are flattened, you may prefer to group them per face like so depending on your geometry kernel
                grouped_verts = [[verts[i], verts[i + 1], verts[i + 2]] for i in range(0, len(verts), 3)]
                grouped_faces = [[faces[i], faces[i + 1], faces[i + 2]] for i in range(0, len(faces), 3)]

                #Convert grouped vertices/faces to Open3D objects 
                o3dVertices = o3d.utility.Vector3dVector(np.asarray(grouped_verts))
                o3dTriangles = o3d.utility.Vector3iVector(np.asarray(grouped_faces))

                # Create the Open3D mesh object
                mesh=o3d.geometry.TriangleMesh(o3dVertices,o3dTriangles)

                #if mesh, create node
                if len(mesh.triangles)>1:
                    # node=BIMNode(**kwargs)
                    name=ifcElement.Name
                    className=ifcElement.is_a()
                    globalId=ifcElement.GlobalId
                    subject= name +'_'+globalId 
                    resource=mesh
                    timestamp=timestamp
                    ifcPath=path
                    objectType =ifcElement.ObjectType
                    nodelist.append(BIMNode(name=name,
                                            className=className,
                                            globalId=globalId,
                                            subject=subject,
                                            resource=resource,
                                            timestamp=timestamp,
                                            ifcPath=ifcPath,
                                            objectType=objectType,
                                            faceCount=len(mesh.triangles),
                                            pointCount=len(mesh.vertices),
                                            **kwargs))
                if not iterator.next():
                    break
        return nodelist

### CSV ###

def navvis_csv_to_pano_nodes(csvPath :Path, 
                        directory : Path = None, 
                        includeDepth : bool = True, 
                        depthPath : Path = None, 
                        skip:int=1, **kwargs) -> List[PanoNode]:
    """Parse Navvis csv file and return a list of PanoNodes with the csv metadata.
    
    Args:
        - csvPath (Path): csv file path e.g. "D:/Data/pano/pano-poses.csv"
        - skip (int, Optional): select every nth image from the xml. Defaults to None.
        - Path (Path, Optional): path to the pano directory. Defaults to None.
        - includeDepth (bool, Optional): include depth images. Defaults to True.
        - depthPath (Path, Optional): path to the depth images. Defaults to None.
        - kwargs: additional keyword arguments for the PanoNode instances
                
    Returns:
        - A list of PanoNodes with the csv metadata
        
    """
    assert skip == None or skip >0, f'skip == None or skip '
    assert os.path.exists(csvPath), f'File does not exist.'
    assert csvPath.suffix == '.csv', f'File does not end with csv.'

    with open(csvPath, 'r') as f:
        first_line = f.readline()
        header_line = first_line.split(':', 1)[-1].strip()  # keep only after ':'

    # Now use pandas to read, but without skipping anything
    df = pd.read_csv(
        csvPath,
        sep=';',
        skiprows=1,  # skip the first weird comment line
        names=[h.strip() for h in header_line.split(';')],  # use cleaned header
        skipinitialspace=True  # handle spaces after ;
    )
    # Take every nth row
    df = df.iloc[::skip].reset_index(drop=True)

    if(directory is None):
        directory = Path(csvPath).parent

    nodes = []

    for idx, row in df.iterrows():
        pos = (row['pano_pos_x'], row['pano_pos_y'], row['pano_pos_z'])
        ori = (row['pano_ori_x'], row['pano_ori_y'], row['pano_ori_z'], row['pano_ori_w'])

        cartesian_transform = gmu.get_cartesian_transform(pos, ori)
        timestamp = ut.literal_to_datetime(row['timestamp'])
        path = directory / row['filename']

        node = PanoNode(
            name=str(row['ID']),
            cartesianTransform=cartesian_transform,
            timestamp=timestamp,
            path=path,
            **kwargs
        )

        nodes.append(node)
    return nodes

##### NODE SELECTION #####

def select_nodes_k_nearest_neighbors(nodes:List[Node], center: np.ndarray = [0,0,0],k:int=10):
    """Select k nearest nodes based on Euclidean distance between centroids."""
    
    assert k>0, f'k is {k}, but k should be >0.'
    # create a pointcloud te perform nearest neighbors search on
    pcd = o3d.geometry.PointCloud()
    array=np.empty(shape=(len(nodes),3))
    for idx,node in enumerate(nodes):
        array[idx]=gmu.get_translation(node.cartesianTransform)
    pcd.points = o3d.utility.Vector3dVector(array)

    #Create KDTree from pcd
    pcdTree = o3d.geometry.KDTreeFlann(pcd)

    #Find k nearest neighbors
    _, idxList, distances = pcdTree.search_knn_vector_3d(np.array(center), k)
    selectedNodeList=[node for idx,node in enumerate(nodes) if idx in idxList]

    if any(selectedNodeList):        
        return selectedNodeList, distances
    return None, None
    
def select_nodes_within_radius(nodes, center, radius):
    
    if(radius <= 0):
        raise ValueError("Radius must be > 0")
    
    centers=np.empty(shape=(len(nodes),3),dtype=float)
    for idx,node in enumerate(nodes):
        centers[idx]=gmu.get_translation(node.cartesianTransform)
    
    # Create a point cloud of centers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)

    #Create KDTree from pcd
    pcdTree = o3d.geometry.KDTreeFlann(pcd)

    #Find k nearest neighbors
    [_, idxList, distances] = pcdTree.search_radius_vector_3d(np.array(center), radius)
    selectedNodeList = [node for idx,node in enumerate(nodes) if idx in idxList ]
    selectedNodeList = [node for i,node in enumerate(selectedNodeList) if distances[i] <= radius]
    distances = [dist for dist in distances if dist <=radius]

    if any(selectedNodeList):        
        return selectedNodeList, distances
    return None, None

def select_nodes_within_bounding_box(nodes: List[Node], 
    bbox: o3d.geometry.OrientedBoundingBox, 
    margin: List[float] = [0, 0, 0]
    ) -> List[Node]:
    """
    Select the nodes whose centers lie inside the given bounding box.
    
    Args:
        nodes: List of node objects.
        bbox: An open3d.geometry.OrientedBoundingBox (or AxisAlignedBoundingBox).
        margin: List of margins [u, v, w] to expand the bounding box.
    
    Returns:
        List of nodes inside the bounding box, or None if no nodes are inside.
    """

    # Expand the bounding box by the given margin
    box = gmu.expand_box(bbox, u=margin[0], v=margin[1], w=margin[2])

    # Get node centers
    centers = np.empty((len(nodes), 3), dtype=float)
    for idx, node in enumerate(nodes):
        centers[idx] = gmu.get_translation(node.cartesianTransform)

    # Directly check which centers are inside
    points = o3d.utility.Vector3dVector(centers)
    idxList = box.get_point_indices_within_bounding_box(points)

    # Select nodes based on index list
    selectedNodeList = [node for idx, node in enumerate(nodes) if idx in idxList]

    if selectedNodeList:
        return selectedNodeList
    return None

def select_nodes_within_convex_hull(nodes: List[Node], 
    hull: o3d.geometry.TriangleMesh, 
    ) -> List[Node]:
    """
    Select the nodes whose centers lie inside the given convex hull.
    
    Args:
        nodes: List of node objects.
        hull: An open3d.geometry.TriangleMesh representing a convex hull.
    
    Returns:
        List of nodes inside the convex hull, or None if no nodes are inside.
    """    
    if not hull.is_watertight():
        raise ValueError("Input convex hull is not watertight.")

    # Get node centers
    centers = np.empty((len(nodes), 3), dtype=float)
    for idx, node in enumerate(nodes):
        centers[idx] = gmu.get_translation(node.cartesianTransform)

    # Directly check if centers are inside
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hull))  # we do not need the geometry ID for mesh
    occupancy = scene.compute_occupancy(o3d.cpu.pybind.core.Tensor(centers, dtype=o3d.core.Dtype.Float32))

    # Select nodes based on the mask
    selectedNodeList = [node for idx, node in enumerate(nodes) if occupancy[idx] == 1]

    if selectedNodeList:
        return selectedNodeList
    return None

def select_nodes_intersecting_bounding_box(nodes: List[Node], 
    bbox: o3d.geometry.OrientedBoundingBox, 
    margin: List[float] = [0, 0, 0]
    ) -> List[Node]:
    """
    Select the nodes whose bounding box intersects with the given bounding box.
    
    Args:
        nodes: List of node objects.
        bbox: An open3d.geometry.OrientedBoundingBox.
        margin: List of margins [u, v, w] to expand the bounding box.
    
    Returns:
        List of nodes whose bounding box intersects the given bounding box.
    """
    
    # Expand the bounding box by the given margin
    box = gmu.expand_box(bbox, u=margin[0], v=margin[1], w=margin[2])

    selectedNodeList = []

    for node in nodes:
        # Get the bounding box of the node
        node_box = node.orientedBoundingBox
        
        # Check if it intersects with the given bounding box
        if box.intersects(node_box):
            selectedNodeList.append(node)

    if selectedNodeList:
        return selectedNodeList
    return None

def select_nodes_intersecting_convex_hull(nodes: List[Node], 
    hull: o3d.geometry.TriangleMesh, 
    ) -> List[Node]:
    """
    Select the nodes whose convex hull intersects with the given convex hull.
    
    Args:
        nodes: List of node objects.
        hull: An open3d.geometry.TriangleMesh.
    
    Returns:
        List of nodes whose convex hull intersects the given convex hull.
    """
    hulls=[None]*len(nodes)
    for idx,node in enumerate(nodes):
            hulls[idx]=node.convexHull

    # Find the nodes of which the geometry intersects with the source node box
    idxList=gmu.get_mesh_inliers(reference=hull,sources=hulls)
    #print(idxList)

    # idxList=gmu.get_mesh_collisions_trimesh(mesh,meshes)
    selectedNodeList=[node for idx,node in enumerate(nodes) if idx in idxList]
    if any(selectedNodeList):        
        return selectedNodeList
    return None

#### GRAPH CREATION ####

def nodes_to_graph(nodelist : List[Node], path:str =None, overwrite: bool =False,save: bool =False,base: URIRef = None) -> Graph:
    """Convert list of nodes to an RDF graph.

    Args:
        - nodelist (List[Node])
        - graphPath (str, optional): path that serves as the basepath for all path information in the graph. This is also the storage location of the graph.
        - overwrite (bool, optional): Overwrite the existing graph triples. Defaults to False.
        - save (bool, optional): Save the Graph to file. Defaults to False.

    Returns:
        Graph 
    """
    path=Path(path) if path else None
    g=Graph()
    g=ut.bind_ontologies(g)
    
    for node in nodelist:
            node.get_graph(path,base=base)
            g+= node.graph
    if(path and save):
        g.serialize(path)     
    return g  

#### NAVVIS TOOLS ####

def navvis_decode_depthmap(depth_image):
    """
    Function to decode the depthmaps generated by the navvis processing

    Args:
        - None
        
    Returns:
        - np.array: Depthmap
    """
    if not isinstance(depth_image,np.ndarray):
        raise ValueError("Depth_image should be a 4 channel rgbd np.array")
    
    # Vectorized calculation for the depth values
    depth_value = (depth_image[:, :, 0] / 256) * 256 + \
                (depth_image[:, :, 1] / 256) * 256 ** 2 + \
                (depth_image[:, :, 2] / 256) * 256 ** 3 + \
                (depth_image[:, :, 3] / 256) * 256 ** 4

    # Assign the computed depth values to the class attribute _depthMap
    depthMap = depth_value/1000 # Convert to meters
    return depthMap 
























##OBSOLETE
#def create_selection_box_from_image_boundary_points(n:ImageNode,roi:Tuple[int,int,int,int],mesh:o3d.geometry.TriangleMesh,z:float=5)->o3d.geometry.OrientedBoundingBox:
#    """Create a selection box from an ImageNode, a region of interest (roi) and a mesh to raycast.
#    A o3d.geometry.OrientedBoundingBox will be created on the location of the intersection of the rays with the mesh.
#    The height of the box is determined by the offset of z in both positive and negative Z-direction
#
#    Args:
#        n (ImageNode): Imagenode used for the raycasting (internal and external camera paramters)
#        roi (Tuple[int,int,int,int]): region of interest (rowMin,rowMax,columnMin,columnMax)
#        mesh (o3d.geometry.TriangleMesh): mesh used for the raycasting
#        z (float, optional): offset in height of the bounding box. Defaults to [-5m:5m].
#
#    Returns:
#        o3d.geometry.OrientedBoundingBox or None (if not all rays hit the mesh)
#    """
#    box=None
#    
#    #create rays for boundaries
#    uvCoordinates=np.array([[roi[0],roi[2]], # top left
#                            [roi[0],roi[3]], # top right
#                            [roi[1],roi[2]], # bottom left
#                            [roi[1],roi[3]] # bottom right
#                            ])
#    # transform uvcoordinates  to world coordinates to rays   
#    rays=n.create_rays(uvCoordinates)
#    
#    # cast rays to 3D mesh 
#    distances,_=gmu.compute_raycasting_collisions(mesh,rays)
#    
#    if all(np.isnan(distances)==False): #if all rays hit
#        #compute endpoints 
#        _,endpoints=gmu.rays_to_points(rays,distances)
#        
#        #create box of projected points
#        points=np.vstack((gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z],[0,0,0,1]])),
#                        gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-z],[0,0,0,1]]))))
#        box=o3d.geometry.OrientedBoundingBox.create_from_points(o3d.cpu.pybind.utility.Vector3dVector(points))
#        box.color=[1,0,0]     
#    return box 
#
##### OBSOLETE #####
#
#def ifc_to_nodes_by_guids(path:str, guids:list,getResource : bool=True,**kwargs)-> List[BIMNode]:
#    """
#    Parse ifc file to a list of BIMNodes, one for each ifcElement.\n
#
#    .. image:: ../../../docs/pics/ifc_inheritance.PNG
#
#    Args:
#        1. ifcPath (string):  absolute ifc file path e.g. "D:\\myifc.ifc"\n
#        2. guids (list of strings): IFC guids you want to parse e.g. [''3saOMnutrFHPtEwspUjESB'',''3saOMnutrFHPtEwspUjESB']. \n  
#    
#    Returns:
#        List[BIMNode]
#    """ 
#    path=Path(path)
#    
#    ifc_file = ifcopenshell.open(path)
#    nodelist=[]   
#    for guid in guids:
#        ifcElements = ifc_file.by_id(guid)
#        ifcElements=ut.item_to_list(ifcElements)
#        for ifcElement in ifcElements:
#            node=BIMNode(resource=ifcElement,getResource=getResource, **kwargs)          
#            node.ifcPath=path
#            nodelist.append(node)
#    return nodelist
#
#def ifc_to_nodes_by_type(path:str, types:list=['IfcBuildingElement'],getResource : bool=True,**kwargs)-> List[BIMNode]:
#    """
#    Parse ifc file to a list of BIMNodes, one for each ifcElement.\n
#
#    **NOTE**: classes are not case sensitive. It is advised to solely focus on IfcBuildingElement classes or inherited classes as these typically have geometry representations that can be used by GEOMAPI.
#
#    **WARNING**: IfcOpenShell strugles with some ifc serializations. In our experience, IFC4 serializations is more robust.
#
#    .. image:: ../../../docs/pics/ifc_inheritance.PNG
#
#    Args:
#        1. ifcPath (string):  absolute ifc file path e.g. "D:\\myifc.ifc"\n
#        2. types (list of strings, optional): ifcClasses you want to parse e.g. ['IfcWall','IfcSlab','IfcBeam','IfcColumn','IfcStair','IfcWindow','IfcDoor']. Defaults to ['IfcBuildingElement']. \n  
#    
#    Raises:
#        ValueError: 'No valid ifcPath.'
#
#    Returns:
#        List[BIMNode]
#    """   
#    path=Path(path) 
#    
#    try:
#        ifc_file = ifcopenshell.open(path)
#    except:
#        print(ifcopenshell.get_log())
#    else:
#        nodelist=[]   
#        for type in types:
#            ifcElements = ifc_file.by_type(type)
#            ifcElements=ut.item_to_list(ifcElements)
#            for ifcElement in ifcElements:
#                node=BIMNode(resource=ifcElement,getResource=getResource, **kwargs)          
#                node.ifcPath=path
#                nodelist.append(node)
#        return nodelist
#
#
#
#
#def e57path_to_nodes(path:str,percentage:float=1.0) ->List[PointCloudNode]:
#    """Load an e57 file and convert all data to a list of PointCloudNodes.\n
#
#    **NOTE**: lowering the percentage barely affects assignment performance (numpy array assignments are extremely efficient). \n 
#    Only do this to lower computational complexity or free up memory.
#
#    Args:
#        1. e57path(str): absolute path to .e57 file\n
#        2. percentage(float,optional): percentage of points to load. Defaults to 1.0 (100%)\n
#
#    Returns:
#        o3d.geometry.PointCloud
#    """
#    path=Path(path) if path else None
#    
#    e57 = pye57.E57(str(path))
#    gmu.e57_update_point_field(e57)
#    nodes=[]
#    for s in range(e57.scan_count):
#        resource=gmu.e57_to_pcd(e57,e57Index=s,percentage=percentage)
#        node=PointCloudNode(resource=resource,
#                            path=path,
#                            e57Index=s,
#                            percentage=percentage)
#        node.pointCount=len(resource.points)
#        nodes.append(node)
#    return nodes
#
#def e57path_to_nodes_mutiprocessing(path:str,percentage:float=1.0) ->List[PointCloudNode]:
#    """Load an e57 file and convert all data to a list of PointCloudNodes.\n
#
#    **NOTE**: Complex types cannot be pickled (serialized) by Windows. Therefore, a two step parsing is used where e57 data is first loaded as np.arrays with multi-processing.
#    Next, the arrays are passed to o3d.geometry.PointClouds outside of the loop.\n  
#
#    **NOTE**: starting parallel processing takes a bit of time. This method will start to outperform single-core import from 3+ pointclouds.\n
#
#    **NOTE**: lowering the percentage barely affects assignment performance (numpy array assignments are extremely efficient). \n 
#    Only do this to lower computational complexity or free up memory.
#
#    Args:
#        1. e57path(str): absolute path to .e57 file\n
#        2. percentage(float,optional): percentage of points to load. Defaults to 1.0 (100%)\n
#
#    Returns:
#        o3d.geometry.PointCloud
#    """   
#    path=Path(path) if path else None
# 
#    e57 = pye57.E57(path)
#    gmu.e57_update_point_field(e57)
#
#    nodes=[]
#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        # first load all e57 data and output it as np.arrays
#        results=[executor.submit(gmu.e57_to_arrays,e57Path=path,e57Index=s,percentage=percentage) for s in range(e57.scan_count)]
#        # next, the arrays are assigned to point clouds outside the loop.
#        for s,r in enumerate(concurrent.futures.as_completed(results)):
#            resource=gmu.arrays_to_pcd(r.result())
#            node=PointCloudNode(resource=resource,
#                                path=path,
#                                e57Index=s,
#                                percentage=percentage)
#            node.pointCount=len(resource.points)
#            nodes.append(node)
#    return nodes
#
#def e57header_to_nodes(path:str, **kwargs) -> List[PointCloudNode]:
#    """Parse e57 file header that is created with E57lib e57xmldump.exe.
#
#    Args:
#        path (string):  e57 xml file path e.g. "D:\\Data\\2018-06 Werfopvolging Academiestraat Gent\\week 22\\PCD\\week 22 lidar_CC.xml"
#            
#    Returns:
#        A list of pointcloudnodes with the xml metadata 
#    """
#    path=Path(path) if path else None
#    
#    nodelist=[]   
#    e57 = pye57.E57(str(path))   
#    gmu.e57_update_point_field(e57)
#
#    for idx in range(e57.scan_count):
#        nodelist.append(PointCloudNode(path=path,e57Index=idx, **kwargs))
#    return nodelist



































#def metashape_dxf_to_orthonodes(dxfPath:str |Path, **kwargs) -> List[OrthoNode]:
#    dxf = ezdxf.readfile(self._dxfPath)
#    #contours and names are in the same list as pairs
#    entities=[entity for entity in dxf.modelspace()]
#    
#    def create_convex_hull_from_dxf_points():
#        box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
#        bottomLeftLow=points[2]
#        bottomRightLow=points[3]
#        topLeftLow=points[1]
#        topRightLow=points[0]
#        bottomLeftHigh=points[2]+normal*self._depth
#        bottomRightHigh=points[3]+normal*self._depth
#        topLeftHigh=points[1]+normal*self._depth
#        topRightHigh=points[0]+normal*self._depth
#        vertices=np.array([[bottomLeftLow],
#                            [bottomRightLow],
#                            [topLeftLow],
#                            [topRightLow],
#                            [bottomLeftHigh],
#                            [bottomRightHigh],
#                            [topLeftHigh],
#                            [topRightHigh]])
#        
#        box.vertices = o3d.utility.Vector3dVector(np.reshape(vertices,(8,3)))                    
#        self._convexHull = box  
#    
#    if len([entity for entity in entities if entity.dxftype() == 'INSERT' and Path(entity.attribs[0].dxf.text).stem==self._name])==0:
#        print('Warning: No INSERT entity found with the name of the orthomosaic. taking first ...')
#        entity=entities[0]
#        self._name=Path(entity.attribs[0].dxf.text).stem
#    
#    #iterate through entities per two
#    for i in range(0,len(entities),2):
#        #entity1 are the entities with the name
#        entity1=entities[i] 
#        #entity2 are the entities with the geometry
#        entity2=entities[i+1]
#        name=Path(entity1.attribs[0].dxf.text).stem
#        if name == self._name:        
#            #get geometry
#            g=cadu.ezdxf_entity_to_o3d(entity2)
#            g.translate(np.array([0,0,self.get_height()]))
#            #get points -> they are ordered counter clockwise starting from the top left
#            points=np.asarray(g.points)
#            #get the center of the geometry
#            center=g.get_center()
#            #get the vector 0-1 and 0-3
#            vec1=points[1]-points[0]
#            vec2=points[3]-points[0]
#            #get the normal of the plane
#            normal=np.cross(vec1,vec2)
#            #normalize the normal
#            normal=normal/np.linalg.norm(normal)
#            
#            #get the translation matrix
#            translation=center#-normal*self._depth
#            
#            #get rotation matrix from this normal to the z-axis
#            rotation_matrix=ut.get_rotation_matrix_from_forward_up(normal, vec2)
#            
#            cartesianTransform = gmu.get_cartesian_transform(translation=translation,rotation=rotation_matrix) 
#            self._cartesianTransform=cartesianTransform    
#            
#            #create convexhull
#            create_convex_hull_from_dxf_points()
#            
#            #reset bounding box
#            self._orientedBoundingBox=None
#            self.get_oriented_bounding_box()
#            
#            #get gsd
#            self._gsd=np.linalg.norm(vec1[0])/self.get_image_width()

# OBSOLETE, every node has a convex hull, and image nodes have frustrums
#def get_mesh_representation(node: Node)->o3d.geometry.TriangleMesh:
#    """Returns the mesh representation of a node resource\n
#    Returns the convex hull if it is a PointCloudNode.\n
#    For ImageNodes, a virtual mesh cone is used with respect to the field of view.
#
#    Args:
#        Node (Node): geomapi node such as a PointCloudNode
#
#    Returns:
#        o3d.geometry.TriangleMesh 
#    """
#    nodeType=str(type(node))
#    resource= node.resource
#   
#    if 'PointCloudNode' in str(type(node)):
#        hull, _ =resource.compute_convex_hull()
#        return hull
#    elif 'ImageNode' in nodeType:
#        return node.convexHull
#    elif 'OrthoNode' in nodeType:
#        #print('not implemented')
#        return node.convexHull
#    else:
#        return resource
#    
#
#def select_nodes_k_nearest_neighbors(node:Node,nodelist:List[Node],k:int=10) -> Tuple[List [Node], o3d.utility.DoubleVector]:
#    """ Select k nearest nodes based on Euclidean distance between centroids.\n
#
#    .. image:: ../../../docs/pics/selection_k_nearest.PNG
#
#    Args:
#        0. node (Node): node to search from\n
#        1. nodelist (List[Node])\n
#        2. k (int, optional): number of neighbors. Defaults to 10.\n
#
#    Returns:
#        List of Nodes
#    """
#    assert k>0, f'k is {k}, but k should be >0.'
#
#    #get node center
#    if node.cartesianTransform is not None:
#        point=gmu.get_translation(node.cartesianTransform)
#        #create pcd from nodelist centers
#        pcd = o3d.geometry.PointCloud()
#        array=np.empty(shape=(len(nodelist),3))
#        for idx,node in enumerate(nodelist):
#            if node.cartesianTransform is not None:
#                array[idx]=gmu.get_translation(node.cartesianTransform)
#            else:
#                array[idx]=[-10000.0,-10000.0,-10000.0]
#        pcd.points = o3d.utility.Vector3dVector(array)
#
#        #Create KDTree from pcd
#        pcdTree = o3d.geometry.KDTreeFlann(pcd)
#
#        #Find 200 nearest neighbors
#        _, idxList, distances = pcdTree.search_knn_vector_3d(point, k)
#        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
#
#        if any(selectedNodeList):        
#            return selectedNodeList, distances
#    else:
#        return None,None
#
#def select_nodes_with_centers_in_radius(node:Node,nodelist:List[Node],r:float=0.5) -> Tuple[List [Node] ,List[float]]:
#    """Select nodes within radius of the node centroid based on Euclidean distance between node centroids.\n
#
#    .. image:: ../../../docs/pics/selection_radius_nearest.PNG
#    
#    Args:
#        0. node (Node): node to search from\n
#        1. nodelist (List[Node])\n
#        2. r (float, optional): radius to search. Defaults to 0.5m.\n
#
#    Returns:
#        List of Nodes, List of Distances
#    """
#    
#    assert r >0, f'r is {r}, while it should be >0.'
#    
#    #get node center
#    if node.cartesianTransform is not None:
#        point=gmu.get_translation(node.cartesianTransform)
#        #create pcd from nodelist centers
#        pcd = o3d.geometry.PointCloud()
#        array=np.empty(shape=(len(nodelist),3))
#        for idx,node in enumerate(nodelist):
#            if node.cartesianTransform is not None:
#                array[idx]=gmu.get_translation(node.cartesianTransform)
#            else:
#                array[idx]=[-10000.0,-10000.0,-10000.0]
#        pcd.points = o3d.utility.Vector3dVector(array)
#
#        #Create KDTree from pcd
#        pcdTree = o3d.geometry.KDTreeFlann(pcd)
#
#        #Find 200 nearest neighbors
#        [_, idxList, distances] = pcdTree.search_radius_vector_3d(point, r)
#        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList ]
#        selectedNodeList=[node for i,node in enumerate(selectedNodeList) if distances[i]<=r ]
#        distances = [dist for dist in distances if dist <=r]
#        
#        if any(selectedNodeList):        
#            return selectedNodeList,distances
#    else:
#        return None,None
#
#def select_nodes_with_centers_in_bounding_box(node:Node,nodelist:List[Node],u:float=0.5,v:float=0.5,w:float=0.5) -> List [Node]: 
#    """Select the nodes of which the center lies within the oriented Bounding Box of the source node given an offset.\n
#
#    .. image:: ../../../docs/pics/selection_box_inliers.PNG
#    
#    Args:
#        0. node (Node): source Node \n
#        1. nodelist (List[Node]): target nodelist\n
#        2. u (float, optional): Offset in X. Defaults to 0.5m.\n
#        3. v (float, optional): Offset in Y. Defaults to 0.5m.\n
#        4. w (float, optional): Offset in Z. Defaults to 0.5m.\n
#
#    Returns:
#        List [Node]
#    """
#    #get box source node
#    if node.orientedBoundingBox is not None:
#        box=node.orientedBoundingBox
#        box=gmu.expand_box(box,u=u,v=v,w=w)
#
#        # get centers
#        centers=np.empty((len(nodelist),3),dtype=float)
#        for idx,node in enumerate(nodelist):
#            if node.cartesianTransform is not None:
#                centers[idx]=gmu.get_translation(node.cartesianTransform)
#
#        #points are the centers of all the nodes
#        pcd = o3d.geometry.PointCloud()
#        points = o3d.utility.Vector3dVector(centers)
#        pcd.points=points
#
#        # Find the nodes that lie within the index box 
#        idxList=box.get_point_indices_within_bounding_box(points)
#        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
#        if any(selectedNodeList):        
#            return selectedNodeList
#    else:
#        return None
#
#def select_nodes_with_bounding_points_in_bounding_box(node:Node,nodelist:List[Node],u:float=0.5,v:float=0.5,w:float=0.5) -> List [Node]: 
#    """Select the nodes of which atleast one of the bounding points lies within the oriented Bounding Box of the source node given an offset.\n
#
#    .. image:: ../../../docs/pics/selection_BB_intersection.PNG
#    
#    Args:
#        0. node (Node): source Node \n
#        1. nodelist (List[Node]): target nodelist\n
#        2. u (float, optional): Offset in X. Defaults to 0.5m.\n
#        3. v (float, optional): Offset in Y. Defaults to 0.5m.\n
#        4. w (float, optional): Offset in Z. Defaults to 0.5m.\n
#
#    Returns:
#        List [Node]
#    """
#    #get box source node
#    if node.orientedBoundingBox is not None:
#        box=node.orientedBoundingBox
#        box=gmu.expand_box(box,u=u,v=v,w=w)
#
#        # get boxes nodelist
#        boxes=np.empty((len(nodelist),1),dtype=o3d.geometry.OrientedBoundingBox)
#        for idx,node in enumerate(nodelist):
#            boxes[idx]=node.orientedBoundingBox
#
#        # Find the nodes of which the bounding points lie in the source node box
#        idxList=gmu.get_box_inliers(box,boxes)
#        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
#        if any(selectedNodeList):        
#            return selectedNodeList
#    else:
#        return None
#    
#def select_nodes_with_intersecting_bounding_box(node:Node,nodelist:List[Node],u:float=0.5,v:float=0.5,w:float=0.5) -> List [Node]: 
#    """Select the nodes of which the bounding boxes intersect.\n
#
#    .. image:: ../../../docs/pics/selection_BB_intersection2.PNG
#
#    Args:
#        0. node (Node): source Node \n
#        1. nodelist (List[Node]): target nodelist\n
#        2. u (float, optional): Offset in X. Defaults to 0.5m.\n
#        3. v (float, optional): Offset in Y. Defaults to 0.5m.\n
#        4. w (float, optional): Offset in Z. Defaults to 0.5m.\n
#
#    Returns:
#        List [Node]
#    """
#    #get box source node
#    if node.orientedBoundingBox is not None:
#        box=node.orientedBoundingBox
#        box=gmu.expand_box(box,u=u,v=v,w=w)
#
#        # get boxes nodelist
#        boxes=np.empty((len(nodelist),1),dtype=o3d.geometry.OrientedBoundingBox)
#        for idx,node in enumerate(nodelist):
#            boxes[idx]=node.orientedBoundingBox
#        
#        # Find the nodes of which the bounding box itersects with the source node box
#        idxList=gmu.get_box_intersections(box,boxes)
#        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
#        if any(selectedNodeList):        
#            return selectedNodeList
#    else:
#        return None
#
#def select_nodes_with_intersecting_resources(node:Node,nodelist:List[Node]) -> List [Node]: 
#    """Select the nodes of which the o3d.geometry.TriangleMeshes intersect.\n
#    This method relies on trimesh and fcl libraries for collision detection.\n
#    For PointCloudNodes, the convex hull is used.\n
#    For ImageNodes, a virtual mesh cone is used with respect to the field of view.\n
#
#    .. image:: ../../../docs/pics/collision_5.PNG
#
#    Args:
#        0. node (Node): source Node \n
#        1. nodelist (List[Node]): target nodelist\n
#
#    Returns:
#        List [Node] 
#    """
#    #get geometry source node
#    if node.resource is not None: 
#        mesh=get_mesh_representation(node)
#        # get geometries nodelist        
#        # meshes=np.empty((len(nodelist),1),dtype=o3d.geometry.TriangleMesh)
#        
#        meshes=[None]*len(nodelist)
#        for idx,testnode in enumerate(nodelist):
#            if testnode.resource is not None: 
#                    meshes[idx]=get_mesh_representation(testnode)
#
#        # Find the nodes of which the geometry intersects with the source node box
#        idxList=gmu.get_mesh_inliers(reference=mesh,sources=meshes)
#        print(idxList)
#
#        # idxList=gmu.get_mesh_collisions_trimesh(mesh,meshes)
#        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
#        if any(selectedNodeList):        
#            return selectedNodeList
#    return None