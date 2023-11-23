"""Different tools to Manage RDF data."""

#IMPORT PACKAGES
from dis import dis
from lib2to3.pytree import Node
import numpy as np 
import cv2 
import open3d as o3d 
import os 
import re
import pye57 
import pandas as pd
import xml.etree.ElementTree as ET 
from typing import List,Tuple

# import APIs
import rdflib
from rdflib import Graph
from rdflib import Graph
from rdflib import URIRef, Literal
from rdflib.namespace import CSVW, DC, DCAT, DCTERMS, DOAP, FOAF, ODRL2, ORG, OWL, \
                           PROF, PROV, RDF, RDFS, SDO, SH, SKOS, SOSA, SSN, TIME, \
                           VOID, XMLNS, XSD
import ifcopenshell
import ifcopenshell.util
import ifcopenshell.geom as geom
from ifcopenshell.util.selector import Selector
import multiprocessing
import concurrent.futures

#IMPORT MODULES 
from geomapi.nodes import *
from geomapi.nodes.sessionnode import create_node 
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
from warnings import warn

#ONTOLOGIES
loa=rdflib.Namespace('https://docplayer.net/131921614-Usibd-level-of-accuracy-loa-specification-guide.html#')
ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')

#### NODE CREATION ####
def e57xml_to_nodes(e57XmlPath :str, **kwargs) -> List[PointCloudNode]:
    """Parse XML file that is created with E57lib e57xmldump.exe.

    Args:
        path (string):  e57 xml file path e.g. "D:\\Data\\2018-06 Werfopvolging Academiestraat Gent\\week 22\\PCD\\week 22 lidar_CC.xml"
            
    Returns:
        A list of pointcloudnodes with the xml metadata 
    """
    if os.path.exists(e57XmlPath) and e57XmlPath.endswith('.xml'):    
        #E57 XML file structure
        #e57Root
        #   >data3D
        #       >vectorChild
        #           >pose
        #               >rotation
        #               >translation
        #           >cartesianBounds
        #           >guid
        #           >name
        #           >points recordCount
        #   >images2D
        mytree = ET.parse(e57XmlPath)
        root = mytree.getroot()  
        nodelist=[]   
        e57Path=e57XmlPath.replace('.xml','.e57')       

        for idx,child in enumerate(root.iter('{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}vectorChild')):
            nodelist.append(PointCloudNode(e57XmlPath=e57XmlPath,e57Index=idx,path=e57Path,**kwargs))
        return nodelist
    else:
        raise ValueError('No valid e57XmlPath.')

def img_xml_to_nodes(xmlPath :str,skip:int=None, filterByFolder:bool=False,**kwargs) -> List[ImageNode]:
    """Parse XML file that is created with https://www.agisoft.com/.

    Args:
        1.xmlPath (string): xml file path e.g. "D:/Data/cameras.xml"
        2.skip (int, Optional): select every nth image from the xml. Defaults to None.
        3.filterByFolder (bool, Optional): Filter imgNodes based on the images in the folder or not. Defaults to False.
            
    Returns:
        A list of ImageNodes with the xml metadata 
    """
    assert skip == None or skip >0, f'skip == None or skip '
    assert os.path.exists(xmlPath), f'File does not exist.'
    assert xmlPath.endswith('.xml'), f'File does not end with xml.' 
    
    #open xml
    mytree = ET.parse(xmlPath)
    root = mytree.getroot()  

    #get reference
    chunk=root.find('chunk')
    globalTransform=gmu.get_cartesian_transform(rotation=ut.literal_to_array(chunk.find('transform').find('rotation').text),
                                                translation= ut.literal_to_array(chunk.find('transform').find('translation').text))
    # globalScale = np.identity(4)*float(chunk.find('transform').find('scale').text)
    # globalScale[-1,-1]=1  
    #! test
    globalScale=float(chunk.find('transform').find('scale').text)

    #get components -> in some xml files, there are no components.
    components=[]
    for component in root.iter('component'):       
        try:
            transform=component.find('transform')
            region=component.find('region')
            # scale = np.identity(4)*float(transform.find('scale').text)
            # scale[-1,-1]=1
            #! test
            scale=float(transform.find('scale').text)
            components.append({'componentid':  int(component.get('id')),        
                            'refTransform': gmu.get_cartesian_transform(rotation=ut.literal_to_array(transform.find('rotation').text),
                                                translation= ut.literal_to_array(transform.find('translation').text)),
                            'scale': scale,
                            'center': gmu.get_cartesian_transform( translation=ut.literal_to_array(region.find('center').text)),
                            'size': ut.literal_to_array(region.find('size').text),
                            'R': ut.literal_to_array(region.find('R').text)})     
        except:
            components.append(None)
            continue

    #get sensors
    sensors=[]
    for sensor in root.iter('sensor'):       
        try:
            calibration=sensor.find('calibration')
            focalLength35mm= calibration.find('f').text if calibration.find('f') is not None else calibration.find('fx').text # sometimes the focal length is named diffirently
            sensors.append({'sensorid':  int(sensor.get('id'))   ,        
                            'imageWidth': int(calibration.find('resolution').get('width')),
                            'imageHeight': int(calibration.find('resolution').get('height')),
                            'focalLength35mm': float(focalLength35mm)})     
        except:
            sensors.append(None)
            continue
    
    #get image names in folder
    files=ut.get_list_of_files(ut.get_folder(xmlPath))
    files=[f for f in files if (f.endswith('.JPG') or 
                                f.endswith('.PNG') or 
                                f.endswith('.jpg') or
                                f.endswith('.png'))]
    names=[ut.get_filename(file) for file in files]

    #get cameras
    nodelist=[]   
    for cam in root.iter('camera'):
        try:
            #get name
            name=cam.get('label')
            
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
            transform=np.reshape(ut.literal_to_array(cam.find('transform').text),(4,4))
            #apply scale and reference transformation
            
            #! test
            transform=gmu.get_cartesian_transform(rotation=transform[0:3,0:3],
                                        translation=transform[0:3,3]*scale)
            
            
            transform=refTransform  @ transform
            #transform=refTransform  @ scale  @ transform

            #get sensor information
            sensorid=int(cam.get('sensor_id'))      
            sensorInformation= next(s for s in sensors if s is not None and s.get('sensorid')==sensorid)

            #create image node 
            import uuid          
            node=ImageNode(subject='file:///'+str(uuid.uuid1()),
                        name=name, 
                         cartesianTransform=transform,
                        imageWidth =  int(sensorInformation['imageWidth']),
                        imageHeight = int(sensorInformation['imageHeight'] ),
                        focalLength35mm = sensorInformation['focalLength35mm'], 
                        **kwargs)
            # node.xmlPath=xmlPath
            
            #assign node to nodelist depending on whether it's in the folder    
            try:
                test1=ut.get_filename(node.name)
                i=names.index(ut.get_filename(node.name))
                node.path=files[i]
                nodelist.append(node)   
            except:
                
                None if filterByFolder else nodelist.append(node) 

        except:
            continue
    return nodelist[0::skip] if skip else nodelist

def e57path_to_nodes(e57Path:str,percentage:float=1.0) ->List[PointCloudNode]:
    """Load an e57 file and convert all data to a list of PointCloudNodes.\n

    **NOTE**: lowering the percentage barely affects assignment performance (numpy array assignments are extremely efficient). \n 
    Only do this to lower computational complexity or free up memory.

    Args:
        1. e57path(str): absolute path to .e57 file\n
        2. percentage(float,optional): percentage of points to load. Defaults to 1.0 (100%)\n

    Returns:
        o3d.geometry.PointCloud
    """    
    e57 = pye57.E57(e57Path)
    gmu.e57_update_point_field(e57)
    nodes=[]
    for s in range(e57.scan_count):
        resource=gmu.e57_to_pcd(e57,e57Index=s,percentage=percentage)
        node=PointCloudNode(resource=resource,
                            path=e57Path,
                            e57Index=s,
                            percentage=percentage)
        node.pointCount=len(resource.points)
        nodes.append(node)
    return nodes
    
def e57path_to_nodes_mutiprocessing(e57Path:str,percentage:float=1.0) ->List[PointCloudNode]:
    """Load an e57 file and convert all data to a list of PointCloudNodes.\n

    **NOTE**: Complex types cannot be pickled (serialized) by Windows. Therefore, a two step parsing is used where e57 data is first loaded as np.arrays with multi-processing.
    Next, the arrays are passed to o3d.geometry.PointClouds outside of the loop.\n  

    **NOTE**: starting parallel processing takes a bit of time. This method will start to outperform single-core import from 3+ pointclouds.\n

    **NOTE**: lowering the percentage barely affects assignment performance (numpy array assignments are extremely efficient). \n 
    Only do this to lower computational complexity or free up memory.

    Args:
        1. e57path(str): absolute path to .e57 file\n
        2. percentage(float,optional): percentage of points to load. Defaults to 1.0 (100%)\n

    Returns:
        o3d.geometry.PointCloud
    """    
    e57 = pye57.E57(e57Path)
    gmu.e57_update_point_field(e57)

    nodes=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # first load all e57 data and output it as np.arrays
        results=[executor.submit(gmu.e57_to_arrays,e57Path=e57Path,e57Index=s,percentage=percentage) for s in range(e57.scan_count)]
        # next, the arrays are assigned to point clouds outside the loop.
        for s,r in enumerate(concurrent.futures.as_completed(results)):
            resource=gmu.arrays_to_pcd(r.result())
            node=PointCloudNode(resource=resource,
                                path=e57Path,
                                e57Index=s,
                                percentage=percentage)
            node.pointCount=len(resource.points)
            nodes.append(node)
    return nodes

def e57header_to_nodes(e57Path:str, **kwargs) -> List[PointCloudNode]:
    """
    Parse e57 file header that is created with E57lib e57xmldump.exe.

    Args:
        path (string):  e57 xml file path e.g. "D:\\Data\\2018-06 Werfopvolging Academiestraat Gent\\week 22\\PCD\\week 22 lidar_CC.xml"
            
    Returns:
        A list of pointcloudnodes with the xml metadata 
    """
    if os.path.exists(e57Path) and e57Path.endswith('.e57'):    

        nodelist=[]   
        e57 = pye57.E57(e57Path)   
        gmu.e57_update_point_field(e57)

        for idx in range(e57.scan_count):
            nodelist.append(PointCloudNode(path=e57Path,e57Index=idx, **kwargs))
        return nodelist
    else:
        raise ValueError('No valid e57Path.')

def get_loaclasses_from_ifcclass(ifcClass:str)->URIRef:
    """ Return the matching LOA class given a ifcClass e.g. IfcWall -> URIRef('https://B2010_EXTERIOR_WALLS').
    The returned subjects can be used to retreive the LOAm and LOAr values from the LOA graph.  

    Args:
        ifcClass (str): class names e.g. IfcWall

    Returns:
        URIRef: subjects of LOA graph 
    """
    if (ifcClass == 'IfcFooting' or
        ifcClass == 'IfcPile' or
        ifcClass == 'IfcPlate'):
        return URIRef('https://A_SUBSTRUCTURE')
    if (ifcClass == 'IfcWall' or
        ifcClass == 'IfcCurtainWall' or
        ifcClass == 'IfcWallStandardCase'):
        return URIRef('https://B2010_EXTERIOR_WALLS')
    if (ifcClass == 'IfcBuildingElement' or
        ifcClass == 'IfcSite' or
        ifcClass == 'IfcBuilding'):
        return URIRef('https://B_SHELL')
    if (ifcClass == 'IfcRoof'):
        return URIRef('https://B1020_ROOF_CONSTRUCTION')
    if (ifcClass == 'IfcSlab'):
        return URIRef('https://B1010_FLOOR_CONSTRUCTION')
    if (ifcClass == 'IfcBuildingStorey'or 
        ifcClass == 'IfcSpace' or
        ifcClass == 'IfcSpatialZone' ):
        return URIRef('https://C_INTERIORS')
    if (ifcClass == 'IfcWindow'):
        return URIRef('https://B2020_EXTERIOR_WINDOWS')
    if (ifcClass == 'IfcRailing'or 
        ifcClass == 'IfcStair' or
        ifcClass == 'IfcStairFlight' ):
        return URIRef('https://B1080_STAIRS')
    if (ifcClass == 'IfcDoor'):
        return URIRef('https://B2020_EXTERIOR_WINDOWS')
    if (ifcClass == 'IfcOpening'):
        return URIRef('https://B3060_HORIZONTAL_OPENINGS')
    if (ifcClass == 'IfcCeiling'):
        return URIRef('https://C1070_SUSPENDED_CEILING_CONSTRUCTION')

    return URIRef('https://C_INTERIORS')


def get_ifcclasses_from_loaclass(loaClass:str)->Literal:
    """_summary_

    Args:
        loaClass (str): _description_

    Returns:
        Literal: _description_
    """
    #A-SUBSTRUCTURE
    if (loaClass == 'A_SUBSTRUCTURE' or 
        loaClass == 'A10_FOUNDATIONS' or 
        loaClass == 'A1010_STANDARD_FOUNDATIONS' or 
        loaClass == 'A1020_SPECIAL_FOUNDATIONS'):
        return Literal(['IfcFooting','IfcPile','IfcPlate'])
    if (loaClass == 'A20_SUBGRADE_ENCLOSURES' or 
        loaClass == 'A2010_WALLS_FOR_SUBGRADE_ENCLOSURES'):
        return Literal(['IfcFooting','IfcPile','IfcPlate','IfcWall','IfcCurtainWall'])
    if (loaClass == 'A40_SLABS_ON_GRADE' or 
        loaClass == 'A4010_STANDARD_SLABS_ON_GRADE' or
        loaClass == 'A4020_STRUCTURAL_SLABS_ON_GRADE' ):
        return Literal(['IfcSlab'])
    
    #B-SHELL
    if (loaClass == 'B_SHELL'  ):
        return Literal(['IfcBuildingElement','IfcSite','IfcRoof','IfcBuildingStorey','IfcSpace','IfcBuilding','IfcBuildingElementProxy','IfcSpatialZone','IfcExternalSpatialStructureElement'])
    if loaClass == 'B10_SUPERSTRUCTURE':
        return Literal(['IfcBuildingElement','IfcBuildingStorey','IfcSpace','IfcBuilding','IfcBuildingElementProxy','IfcSpatialZone'])
    if loaClass == 'B1010_FLOOR_CONSTRUCTION':
        return Literal(['IfcSlab'])
    if loaClass == 'B1020_ROOF_CONSTRUCTION':
        return Literal(['IfcRoof'])
    if loaClass == 'B1080_STAIRS':
        return Literal(['IfcRailing','IfcStair','IfcStairFlight'])
    if loaClass == 'B20_EXTERIOR_VERTICAL_ENCLOSURES':
        return Literal(['IfcWall','IfcWindow','IfcDoor','IfcChimney','IfcCurtainWall','IfcWallStandardCase'])
    if (loaClass == 'B2010_EXTERIOR_WALLS' or
        loaClass == 'B2080_EXTERIOR_WALLS_AND_APPURTENANCES' or
        loaClass == 'B2090_EXTERIOR_WALLS_SPECIALTIES' ):
        return Literal(['IfcWall','IfcCurtainWall','IfcWallStandardCase'])
    if loaClass == 'B2020_EXTERIOR_WINDOWS':
        return Literal(['IfcWindow'])
    if loaClass == 'B2050_EXTERIOR_DOORS_AND_GRILLES':
        return Literal(['IfcDoor'])
    if loaClass == 'B30_EXTERIOR_HORIZONTAL_ENCLOSURES':
        return Literal(['IfcSlab','IfcRoof'])
    if (loaClass == 'B3010_ROOFING' or
        loaClass ==  'B3020_ROOF_APPERURTENANCES'):
        return Literal(['IfcRoof'])
    if loaClass == 'B3040_TRAFFIC_BEARING_HORIZONTAL_ENCLOSURES':
        return Literal(['IfcSlab'])
    if loaClass == 'B3060_HORIZONTAL_OPENINGS':
        return Literal(['IfcOpening'])
    if loaClass == 'B3080_OVERHEAD_EXTERIOR_ENCLOSURES':
        return Literal(['IfcSlab','IfcCeiling','IfcCovering'])
    
    #C-INTERIOR
    if loaClass == 'C_INTERIORS':
        return Literal(['IfcFurniture','IfcCeiling','IfcDoor','IfcWindow','IfcWall'])
    if loaClass == 'C10_INTERIOR_CONSTRUCTION':
        return Literal(['IfcFurniture','IfcCeiling','IfcDoor','IfcWindow','IfcWall'])
    if loaClass == 'C1010_INTERIOR_PARTITIONS':
        return Literal(['IfcRoom','IfcSpace'])
    if loaClass == 'C1020_INTERIOR_WINDOWS':
        return Literal(['IfcWindow'])
    if loaClass == 'C1030_INTERIOR_DOORS':
        return Literal(['IfcDoor'])
    if loaClass == 'C1040_INTERIOR_GRILLES_AND_GATES':
        return Literal(['IfcFurniture'])
    if loaClass == 'C1060_RAISED_FLOOR_CONSTRUCTION':
        return Literal(['IfcSlab'])
    if loaClass == 'C1070_SUSPENDED_CEILING_CONSTRUCTION':
        return Literal(['IfcCeiling'])
    if loaClass == 'C1090_INTERIOR_SPECIALTIES':
        return Literal(['IfcFurniture'])
    if loaClass == 'C20_INTERIOR_FINISHES':
        return Literal(['IfcFurniture'])
    if loaClass == 'C2010_WALL_FINISHES':
        return Literal(['IfcWall','IfcCurtainWall'])
    if loaClass == 'C2020_INTERIOR_FABRICATIONS':
        return Literal(['IfcFurniture'])
    if loaClass == 'C2030_FLOORING':
        return Literal(['IfcSlab'])
    if loaClass == 'C2040_STAIR_FINISHES':
        return Literal(['IfcRailing'])
    if loaClass == 'C2050_CEILING_FINISHES':
        return Literal(['IfcCeiling'])
    
    return Literal(['IfcBuildingElement'])

def create_default_loa_graph(LOAPath:str=None)->Graph:
    """Generates a Graph from the default USIBD_SPC-LOA_C220_2016_ver0_1 specification. This specification contains information on the accuraycy
    of building documentation and representation. \n

    Example:

        <https://A1010_STANDARD_FOUNDATIONS> a "LOA" ;\n
        ifc:classes "['IfcFooting', 'IfcPile', 'IfcPlate']" ;\n
        loa:CSI "A1010" ;\n
        loa:LOAm 10 ;\n
        loa:LOAr 20 ;\n
        loa:validation "B" .\n

    More documentation can be found on https://docplayer.net/131921614-Usibd-level-of-accuracy-loa-specification-guide.html# on how to use this specification.

    Args:
        LOAPath (str, optional): path to CSV with USIBD values

    Returns:
        Graph: graph with serialized accuracies, to be used in validation procedures
    """
    #load default dataframe
    if not LOAPath:
        LOAPath=os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"geomapi",'tools','validationtools','LOA.csv')
    LOAdataFrame = pd.read_csv(LOAPath,
                        sep=';')
    graph=Graph()
    graph=ut.bind_ontologies(graph)        
    graph.bind('loa', loa)    
    for index,row in LOAdataFrame.iterrows():
        subject=URIRef('https://'+row[0])
        graph.add((subject, RDF.type, Literal('LOA') ))  
        graph.add((subject, loa['CSI'], Literal(row[1]) ))  
        graph.add((subject, loa['LOAm'], Literal(row[2]) ))  
        graph.add((subject, loa['LOAr'], Literal(row[3]) ))  
        graph.add((subject, loa['validation'], Literal(row[4]) )) 
        graph.add((subject, ifc['classes'], get_ifcclasses_from_loaclass(row[0]))) 
    return graph

def parse_loa_excel(excelPath:str) -> Graph:
    """Parse an USIBD_SPC-LOA_C220_2016_ver0_1.xlsx spreadsheet that contains meaured/represented accuracy parameters for building documentation procedures.
    The returned graph can be used by GEOMAPI or other linked data processes to validate remote sensing/BIM models. \n

    More documentation can be found on https://docplayer.net/131921614-Usibd-level-of-accuracy-loa-specification-guide.html# on how to use this specification.
    If no excel is presented, a graph with standard values will be obtained.

    .. image:: ../../../docs/pics/USIBD.PNG

    Args:
        excelPath (str): file path to the spreadsheet

    Returns:
        Graph: graph 
    """
    #read standard LOA graph
    graph=Graph().parse(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"geomapi",'tools','validationtools','loaGraph.ttl')) 
    subjects=[s for s in graph.subjects(RDF.type)]

    #read excel
    dataFrame = pd.read_excel(excelPath,
                    header=None,
                    na_filter=False)

    #change LOA graph
    for index, row in dataFrame.iterrows():
        #get excell name
        s=None
        if row[3]:
            s=row[3]
        elif row[2]:
            s=row[2]            
        elif row[1]:
            s=row[1]       
        else:
            continue
        # get corresponding subject
        list=[subject for subject in subjects if re.search(s, subject, re.IGNORECASE)]
     
        if len(list)>0:
            subject=subjects[0]
        else:
            continue

        # modify graph LOAm value
        list=[(i+1)*10 for i,value in enumerate(row[5:9]) if value]          
        if len(list)>0:
            graph.set((subject,loa['LOAm'], Literal(list[0]) ))
        # modify graph LOAr value
        list=[(i+1)*10 for i,value in enumerate(row[13:17]) if value]           
        if len(list)>0:
            graph.set((subject,loa['LOAr'], Literal(list[0]) ))
        # modify graph validation value
        list=[value for value in [row[10]] if value]            
        if len(list)>0:
            graph.set((subject,loa['validation'], Literal(list[0]) ))
    return graph

def get_loa_class_per_bimnode(BIMNodes:List[BIMNode] , ExcelPath:str=None):
    """Assigns the accuracy properties of an LOA Excel spreadsheet to the list of BIMNodes. 
    The assignment is based on the ifc classNames which are mapped to LOA classes. 

    Features:
        1. LOAm (measured accuracy)
        2. LOAr (represented accuracy)
        3. validation (A, B or C)

    Args:
        BIMNodes (List[BIMNode]): List of nodes to assign the propteries to. 
        ExcelPath (str, optional): Path to Excel spreadsheet. If None, the default LOA properties are assigned.
    """
    #parse Excel if present
    loaGraph=parse_loa_excel(ExcelPath)

    #assign LOA properties
    for n in BIMNodes:
        loaClass=get_loaclasses_from_ifcclass(n.className)
        for p,o in loaGraph.predicate_objects(subject=loaClass):
            attr= ut.get_attribute_from_predicate(loaGraph, p) 
            if attr not in ['classes','type']:
                setattr(n,attr,o.toPython()) 

def ifc_to_nodes(ifcPath:str, classes:str='.IfcBuildingElement',getResource : bool=True,**kwargs)-> List[BIMNode]:
    """
    Parse ifc file to a list of BIMNodes, one for each ifcElement.\n

    **NOTE**: classes are not case sensitive. It is advised to solely focus on IfcBuildingElement classes or inherited classes as these typically have geometry representations that can be used by GEOMAPI.

    **NOTE**: If you intend to parse 1000+ elements, use the multithreading of the entire file instead and filter the BIMNodes afterwards as it will be faster. 

    **WARNING**: IfcOpenShell strugles with some ifc serializations. In our experience, IFC4 serializations is more robust.

    .. image:: ../docs/pics/ifc_inheritance.PNG

    Args:
        1. ifcPath (string):  absolute ifc file path e.g. "D:\\myifc.ifc"\n
        2. classes (string, optional): ifcClasses seperated by | e.g. '.IfcBeam | .IfcColumn '#'.IfcWall | .IfcSlab | .IfcBeam | .IfcColumn | .IfcStair | .IfcWindow | .IfcDoor'. Defaults to '.IfcBuildingElement'.   
    
    Raises:
        ValueError: 'No valid ifcPath.'

    Returns:
        List[BIMNode]
    """   
    if os.path.exists(ifcPath) and ifcPath.endswith('.ifc'):    
        nodelist=[]   
        ifc = ifcopenshell.open(ifcPath)   
        selector = Selector()
        for ifcElement in selector.parse(ifc, classes):
            node=BIMNode(resource=ifcElement,getResource=getResource, **kwargs)          
            node.ifcPath=ifcPath
            nodelist.append(node)
        return nodelist
    else:
        raise ValueError('No valid ifcPath.')

def ifc_to_nodes_by_guids(ifcPath:str, guids:list,getResource : bool=True,**kwargs)-> List[BIMNode]:
    """
    Parse ifc file to a list of BIMNodes, one for each ifcElement.\n

    .. image:: ../docs/pics/ifc_inheritance.PNG

    Args:
        1. ifcPath (string):  absolute ifc file path e.g. "D:\\myifc.ifc"\n
        2. guids (list of strings): IFC guids you want to parse e.g. [''3saOMnutrFHPtEwspUjESB'',''3saOMnutrFHPtEwspUjESB']. \n  
    
    Returns:
        List[BIMNode]
    """ 
    assert os.path.exists(ifcPath) and ifcPath.endswith('.ifc')
    ifc_file = ifcopenshell.open(ifcPath)
    nodelist=[]   
    for guid in guids:
        ifcElements = ifc_file.by_id(guid)
        ifcElements=ut.item_to_list(ifcElements)
        for ifcElement in ifcElements:
            node=BIMNode(resource=ifcElement,getResource=getResource, **kwargs)          
            node.ifcPath=ifcPath
            nodelist.append(node)
    return nodelist

def ifc_to_nodes_by_type(ifcPath:str, types:list=['IfcBuildingElement'],getResource : bool=True,**kwargs)-> List[BIMNode]:
    """
    Parse ifc file to a list of BIMNodes, one for each ifcElement.\n

    **NOTE**: classes are not case sensitive. It is advised to solely focus on IfcBuildingElement classes or inherited classes as these typically have geometry representations that can be used by GEOMAPI.

    **WARNING**: IfcOpenShell strugles with some ifc serializations. In our experience, IFC4 serializations is more robust.

    .. image:: ../docs/pics/ifc_inheritance.PNG

    Args:
        1. ifcPath (string):  absolute ifc file path e.g. "D:\\myifc.ifc"\n
        2. types (list of strings, optional): ifcClasses you want to parse e.g. ['IfcWall','IfcSlab','IfcBeam','IfcColumn','IfcStair','IfcWindow','IfcDoor']. Defaults to ['IfcBuildingElement']. \n  
    
    Raises:
        ValueError: 'No valid ifcPath.'

    Returns:
        List[BIMNode]
    """   
    #validate types

    if os.path.exists(ifcPath) and ifcPath.endswith('.ifc'):    
        try:
            ifc_file = ifcopenshell.open(ifcPath)
        except:
            print(ifcopenshell.get_log())
        else:
            nodelist=[]   
            for type in types:
                ifcElements = ifc_file.by_type(type)
                ifcElements=ut.item_to_list(ifcElements)
                for ifcElement in ifcElements:
                    node=BIMNode(resource=ifcElement,getResource=getResource, **kwargs)          
                    node.ifcPath=ifcPath
                    nodelist.append(node)
            return nodelist
    else:
        raise ValueError('No valid ifcPath.')

def ifc_to_nodes_multiprocessing(ifcPath:str, **kwargs)-> List[BIMNode]:
    """Returns the contents of geometry elements in an ifc file as BIMNodes.\n
    This method is 3x faster than other parsing methods due to its multi-threading.\n
    However, only the entire ifc can be parsed.\n

    **WARNING**: IfcOpenShell strugles with some ifc serializations. In our experience, IFC4 serializations is more robust.


    Args:
        ifcPath (str): path (string):  absolute ifc file path e.g. "D:\\myifc.ifc"\n

    Raises:
        ValueError: 'No valid ifcPath.'

    Returns:
        List[BIMNode]
    """
    if os.path.exists(ifcPath) and ifcPath.endswith('.ifc'):  
        try:
            ifc_file = ifcopenshell.open(ifcPath)
        except:
            print(ifcopenshell.get_log())
        else: 
            nodelist=[]   
            timestamp=ut.get_timestamp(ifcPath)
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
                        node=BIMNode(**kwargs)
                        node.name=ifcElement.Name
                        node.className=ifcElement.is_a()
                        node.globalId=ifcElement.GlobalId
                        if node.name and node.globalId:
                            node.subject= node.name +'_'+node.globalId 
                        node.resource=mesh
                        node.get_metadata_from_resource()
                        node.timestamp=timestamp
                        node.ifcPath=ifcPath
                        node.objectType =ifcElement.ObjectType
                        nodelist.append(node)
                        
                    if not iterator.next():
                        break
            return nodelist
    else:
        raise ValueError('No valid ifcPath.') 


##### NODE SELECTION #####

def select_nodes_k_nearest_neighbors(node:Node,nodelist:List[Node],k:int=10) -> Tuple[List [Node], o3d.utility.DoubleVector]:
    """ Select k nearest nodes based on Euclidean distance between centroids.\n

    .. image:: ../docs/pics/selection_k_nearest.PNG

    Args:
        0. node (Node): node to search from\n
        1. nodelist (List[Node])\n
        2. k (int, optional): number of neighbors. Defaults to 10.\n

    Returns:
        List of Nodes
    """
    assert k>0, f'k is {k}, but k should be >0.'

    #get node center
    if node.get_cartesian_transform() is not None:
        point=gmu.get_translation(node.cartesianTransform)
        #create pcd from nodelist centers
        pcd = o3d.geometry.PointCloud()
        array=np.empty(shape=(len(nodelist),3))
        for idx,node in enumerate(nodelist):
            if node.get_cartesian_transform() is not None:
                array[idx]=gmu.get_translation(node.cartesianTransform)
            else:
                array[idx]=[-10000.0,-10000.0,-10000.0]
        pcd.points = o3d.utility.Vector3dVector(array)

        #Create KDTree from pcd
        pcdTree = o3d.geometry.KDTreeFlann(pcd)

        #Find 200 nearest neighbors
        _, idxList, distances = pcdTree.search_knn_vector_3d(point, k)
        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]

        if any(selectedNodeList):        
            return selectedNodeList, distances
    else:
        return None,None


def create_selection_box_from_image_boundary_points(n:ImageNode,roi:Tuple[int,int,int,int],mesh:o3d.geometry.TriangleMesh,z:float=5)->o3d.geometry.OrientedBoundingBox:
    """Create a selection box from an ImageNode, a region of interest (roi) and a mesh to raycast.
    A o3d.geometry.OrientedBoundingBox will be created on the location of the intersection of the rays with the mesh.
    The height of the box is determined by the offset of z in both positive and negative Z-direction

    Args:
        n (ImageNode): Imagenode used for the raycasting (internal and external camera paramters)
        roi (Tuple[int,int,int,int]): region of interest (rowMin,rowMax,columnMin,columnMax)
        mesh (o3d.geometry.TriangleMesh): mesh used for the raycasting
        z (float, optional): offset in height of the bounding box. Defaults to [-5m:5m].

    Returns:
        o3d.geometry.OrientedBoundingBox or None (if not all rays hit the mesh)
    """
    box=None
    
    #create rays for boundaries
    uvCoordinates=np.array([[roi[0],roi[2]], # top left
                            [roi[0],roi[3]], # top right
                            [roi[1],roi[2]], # bottom left
                            [roi[1],roi[3]] # bottom right
                            ])
    # transform uvcoordinates  to world coordinates to rays   
    rays=n.create_rays(uvCoordinates)
    
    # cast rays to 3D mesh 
    distances,_=gmu.compute_raycasting_collisions(mesh,rays)
    
    if all(np.isnan(distances)==False): #if all rays hit
        #compute endpoints 
        _,endpoints=gmu.rays_to_points(rays,distances)
        
        #create box of projected points
        points=np.vstack((gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z],[0,0,0,1]])),
                        gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-z],[0,0,0,1]]))))
        box=o3d.geometry.OrientedBoundingBox.create_from_points(o3d.cpu.pybind.utility.Vector3dVector(points))
        box.color=[1,0,0]     
    return box 



def select_nodes_with_centers_in_radius(node:Node,nodelist:List[Node],r:float=0.5) -> Tuple[List [Node] ,List[float]]:
    """Select nodes within radius of the node centroid based on Euclidean distance between node centroids.\n

    .. image:: ../docs/pics/selection_radius_nearest.PNG
    
    Args:
        0. node (Node): node to search from\n
        1. nodelist (List[Node])\n
        2. r (float, optional): radius to search. Defaults to 0.5m.\n

    Returns:
        List of Nodes, List of Distances
    """
    
    assert r >0, f'r is {r}, while it should be >0.'
    
    #get node center
    if node.get_cartesian_transform() is not None:
        point=gmu.get_translation(node.cartesianTransform)
        #create pcd from nodelist centers
        pcd = o3d.geometry.PointCloud()
        array=np.empty(shape=(len(nodelist),3))
        for idx,node in enumerate(nodelist):
            if node.get_cartesian_transform() is not None:
                array[idx]=gmu.get_translation(node.cartesianTransform)
            else:
                array[idx]=[-10000.0,-10000.0,-10000.0]
        pcd.points = o3d.utility.Vector3dVector(array)

        #Create KDTree from pcd
        pcdTree = o3d.geometry.KDTreeFlann(pcd)

        #Find 200 nearest neighbors
        [_, idxList, distances] = pcdTree.search_radius_vector_3d(point, r)
        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList ]
        selectedNodeList=[node for i,node in enumerate(selectedNodeList) if distances[i]<=r ]
        distances = [dist for dist in distances if dist <=r]
        
        if any(selectedNodeList):        
            return selectedNodeList,distances
    else:
        return None,None

def select_nodes_with_centers_in_bounding_box(node:Node,nodelist:List[Node],u:float=0.5,v:float=0.5,w:float=0.5) -> List [Node]: 
    """Select the nodes of which the center lies within the oriented Bounding Box of the source node given an offset.\n

    .. image:: ../docs/pics/selection_box_inliers.PNG
    
    Args:
        0. node (Node): source Node \n
        1. nodelist (List[Node]): target nodelist\n
        2. u (float, optional): Offset in X. Defaults to 0.5m.\n
        3. v (float, optional): Offset in Y. Defaults to 0.5m.\n
        4. w (float, optional): Offset in Z. Defaults to 0.5m.\n

    Returns:
        List [Node]
    """
    #get box source node
    if node.get_oriented_bounding_box() is not None:
        box=node.orientedBoundingBox
        box=gmu.expand_box(box,u=u,v=v,w=w)

        # get centers
        centers=np.empty((len(nodelist),3),dtype=float)
        for idx,node in enumerate(nodelist):
            if node.get_cartesian_transform() is not None:
                centers[idx]=gmu.get_translation(node.cartesianTransform)

        #points are the centers of all the nodes
        pcd = o3d.geometry.PointCloud()
        points = o3d.utility.Vector3dVector(centers)
        pcd.points=points

        # Find the nodes that lie within the index box 
        idxList=box.get_point_indices_within_bounding_box(points)
        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
        if any(selectedNodeList):        
            return selectedNodeList
    else:
        return None

def select_nodes_with_bounding_points_in_bounding_box(node:Node,nodelist:List[Node],u:float=0.5,v:float=0.5,w:float=0.5) -> List [Node]: 
    """Select the nodes of which atleast one of the bounding points lies within the oriented Bounding Box of the source node given an offset.\n

    .. image:: ../docs/pics/selection_BB_intersection.PNG
    
    Args:
        0. node (Node): source Node \n
        1. nodelist (List[Node]): target nodelist\n
        2. u (float, optional): Offset in X. Defaults to 0.5m.\n
        3. v (float, optional): Offset in Y. Defaults to 0.5m.\n
        4. w (float, optional): Offset in Z. Defaults to 0.5m.\n

    Returns:
        List [Node]
    """
    #get box source node
    if node.get_oriented_bounding_box() is not None:
        box=node.orientedBoundingBox
        box=gmu.expand_box(box,u=u,v=v,w=w)

        # get boxes nodelist
        boxes=np.empty((len(nodelist),1),dtype=o3d.geometry.OrientedBoundingBox)
        for idx,node in enumerate(nodelist):
            boxes[idx]=node.get_oriented_bounding_box()

        # Find the nodes of which the bounding points lie in the source node box
        idxList=gmu.get_box_inliers(box,boxes)
        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
        if any(selectedNodeList):        
            return selectedNodeList
    else:
        return None
    
def select_nodes_with_intersecting_bounding_box(node:Node,nodelist:List[Node],u:float=0.5,v:float=0.5,w:float=0.5) -> List [Node]: 
    """Select the nodes of which the bounding boxes intersect.\n

    .. image:: ../docs/pics/selection_BB_intersection2.PNG

    Args:
        0. node (Node): source Node \n
        1. nodelist (List[Node]): target nodelist\n
        2. u (float, optional): Offset in X. Defaults to 0.5m.\n
        3. v (float, optional): Offset in Y. Defaults to 0.5m.\n
        4. w (float, optional): Offset in Z. Defaults to 0.5m.\n

    Returns:
        List [Node]
    """
    #get box source node
    if node.get_oriented_bounding_box() is not None:
        box=node.orientedBoundingBox
        box=gmu.expand_box(box,u=u,v=v,w=w)

        # get boxes nodelist
        boxes=np.empty((len(nodelist),1),dtype=o3d.geometry.OrientedBoundingBox)
        for idx,node in enumerate(nodelist):
            boxes[idx]=node.get_oriented_bounding_box()
        
        # Find the nodes of which the bounding box itersects with the source node box
        idxList=gmu.get_box_intersections(box,boxes)
        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
        if any(selectedNodeList):        
            return selectedNodeList
    else:
        return None

def select_nodes_with_intersecting_resources(node:Node,nodelist:List[Node]) -> List [Node]: 
    """Select the nodes of which the o3d.geometry.TriangleMeshes intersect.\n
    This method relies on trimesh and fcl libraries for collision detection.\n
    For PointCloudNodes, the convex hull is used.\n
    For ImageNodes, a virtual mesh cone is used with respect to the field of view.\n

    .. image:: ../docs/pics/collision_5.PNG

    Args:
        0. node (Node): source Node \n
        1. nodelist (List[Node]): target nodelist\n

    Returns:
        List [Node] 
    """
    #get geometry source node
    if node.get_resource() is not None: 
        mesh=get_mesh_representation(node)
        # get geometries nodelist        
        # meshes=np.empty((len(nodelist),1),dtype=o3d.geometry.TriangleMesh)
        
        meshes=[None]*len(nodelist)
        for idx,testnode in enumerate(nodelist):
            if testnode.get_resource() is not None: 
                    meshes[idx]=get_mesh_representation(testnode)

        # Find the nodes of which the geometry intersects with the source node box
        idxList=gmu.get_mesh_inliers(reference=mesh,sources=meshes)
        print(idxList)

        # idxList=gmu.get_mesh_collisions_trimesh(mesh,meshes)
        selectedNodeList=[node for idx,node in enumerate(nodelist) if idx in idxList]
        if any(selectedNodeList):        
            return selectedNodeList
    return None

def get_mesh_representation(node: Node)->o3d.geometry.TriangleMesh:
    """Returns the mesh representation of a node resource\n
    Returns the convex hull if it is a PointCloudNode.\n
    For ImageNodes, a virtual mesh cone is used with respect to the field of view.

    Args:
        Node

    Returns:
        o3d.geometry.TriangleMesh 
    """
    nodeType=str(type(node))
    resource= node.get_resource()
   
    if 'PointCloudNode' in str(type(node)):
        hull, _ =resource.compute_convex_hull()
        return hull
    elif 'ImageNode' in nodeType:
        return node.get_mesh_geometry()
    elif 'OrthoNode' in nodeType:
        print('not implemented')
        return None
    else:
        return resource

def nodes_to_graph(nodelist : List[Node], graphPath:str =None, overwrite: bool =False,save: bool =False) -> Graph:
    """Convert list of nodes to an RDF graph.\n

    Args:
        0. nodelist (List[Node])\n
        1. graphPath (str, optional): path that serves as the basepath for all path information in the graph. This is also the storage location of the graph.\n
        2. overwrite (bool, optional): Overwrite the existing graph triples. Defaults to False.\n
        3. save (bool, optional): Save the Graph to file. Defaults to False.\n

    Returns:
        Graph 
    """
    g=Graph()
    g=ut.bind_ontologies(g)
    for node in nodelist:
            node.to_graph(graphPath,overwrite=overwrite)
            g+= node.graph
    if(graphPath and save):
        g.serialize(graphPath)     
    return g  

#### OBSOLETE #####

def graph_path_to_nodes(graphPath : str,**kwargs) -> List[Node]:
    """Convert a graphPath to a set of Nodes.

    Args:
        0. graphPath (str):  absolute path to .ttl RDF Graph\n
        1. kwargs (Any) \n

    Returns:
        A list of pointcloudnodes, imagenodes, meshnodes, bimnodes, orthonodes with metadata 
    """    
    if os.path.exists(graphPath) and graphPath.endswith('.ttl'):
        nodelist=[]
        graph=Graph().parse(graphPath)
        for subject in graph.subjects(RDF.type):
            myGraph=ut.get_subject_graph(graph,subject)
            nodelist.append(create_node(graph=myGraph,graphPath=graphPath,subject=subject,**kwargs) )
        return nodelist
    else:
        raise ValueError('No valid graphPath (only .ttl).')

def graph_to_nodes(graph : Graph,**kwargs) -> List[Node]:
    """Convert a graph to a set of Nodes.

    Args:
        0. graph (RDFlib.Graph):  Graph to parse\n
        1. kwargs (Any) \n

    Returns:
        A list of pointcloudnodes, imagenodes, meshnodes, bimnodes, orthonodes with metadata 
    """    
    nodelist=[]
    for subject in graph.subjects(RDF.type):
        node=create_node(graph=graph,subject=subject,**kwargs) 
        nodelist.append(node)
    return nodelist

# def subject_to_node_type(graph: Graph , subject:URIRef, **kwargs)-> Node:
#     # warn("This function is depricated use a SessionNode instead")

#     nodeType = ut.literal_to_string(graph.value(subject=subject,predicate=RDF.type))
#     g = Graph()
#     g += graph.triples((subject, None, None))
#     if 'BIMNode' in nodeType:
#         node=BIMNode(graph=g,**kwargs)
#     elif 'MeshNode' in nodeType:
#         node=MeshNode(graph=g,**kwargs)
#     elif 'PointCloudNode' in nodeType:
#         node=PointCloudNode(graph=g,**kwargs)
#     elif 'ImageNode' in nodeType:
#         node=ImageNode(graph=g,**kwargs)
#     elif 'SessionNode' in nodeType:
#         node=SessionNode(graph=g,**kwargs)  
#     else:
#         node=Node(graph=g,**kwargs) 
#     return node
def create_node(graph: Graph = None, graphPath: str =None, subject: URIRef = None, resource = None, **kwargs)-> Node:
    """_summary_

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
        if type(resource) is o3d.geometry.PointCloud:
            nodeType='PointCloudNode'
        elif type(resource) is o3d.geometry.TriangleMesh:
            nodeType='MeshNode'
        elif type(resource) is o3d.geometry:
            nodeType='GeometryNode'
        elif type(resource) is np.ndarray:
            nodeType='ImageNode'        
    else:        
        nodeType = 'Node'

    #node creation
    if 'BIMNode' in nodeType:
        node=BIMNode(graph=graph, graphPath=graphPath, resource=resource,subject=subject, **kwargs)
    elif 'MeshNode' in nodeType:
        node=MeshNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'GeometryNode' in nodeType:
        node=GeometryNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'PointCloudNode' in nodeType:
        node=PointCloudNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'ImageNode' in nodeType:
        node=ImageNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)
    elif 'SessionNode' in nodeType:
        node=SessionNode(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs)  
    else:
        node=Node(graph=graph, graphPath=graphPath, resource=resource, subject=subject, **kwargs) 
    return node

def get_linked_nodes(node: Node ,graph:Graph, getResource=False, **kwargs) -> List[Node]:
    """Get related nodes based on linkedNodes variable.\n

    Args:
        0. node (Node): source node to evaluate. \n
        1. graph (Graph): Graph that contains the linkedNodes. \n
        2. getResource (bool, optional): Retrieve the reources. Defaults to False.\n

    Returns:
        List[Node]
    """
    warn("This function is depricated use a SessionNode instead")
    nodelist=[]
    if getattr(node,'linkedNodes',None) is not None:  
        for subject in node.linkedNodes:
            if graph.value(subject=subject,predicate=RDF.type) is not None:
                nodelist.append(create_node(graph=graph,subject=subject, getResource=getResource, **kwargs)) 
    return nodelist
