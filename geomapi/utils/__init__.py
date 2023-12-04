"""
General Basic functions to support other modules.
"""
import datetime
import ntpath
import os
from pathlib import Path, PureWindowsPath
import re
import time
import math
from typing import List
import random

import numpy as np
import PIL.Image
import rdflib
from PIL.ExifTags import GPSTAGS, TAGS
from rdflib import RDF, XSD, Graph, Literal, URIRef, FOAF

#### GLOBAL VARIABLES ####

RDF_EXTENSIONS = [".ttl"]
IMG_EXTENSION = [".jpg", ".png", ".JPG", ".PNG", ".JPEG"]
MESH_EXTENSION = [".obj",".ply",".fbx" ]
PCD_EXTENSION = [".pcd", ".e57",".pts", ".ply", '.xml','.csv']
INT_ATTRIBUTES = ['pointCount','faceCount','e57Index'] #'label'
FLOAT_ATTRIBUTES = ['xResolution','yResolution','imageWidth','imageHeight','focalLength35mm','principalPointU','principalPointV','accuracy']
LIST_ATTRIBUTES =  ['distortionCoeficients']
TIME_FORMAT = "%Y-%m-%d %H-%M-%S"

exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns#')
geo=rdflib.Namespace('http://www.opengis.net/ont/geosparql#') #coordinate system information
gom=rdflib.Namespace('https://w3id.org/gom#') # geometry representations => this is from mathias
omg=rdflib.Namespace('https://w3id.org/omg#') # geometry relations
fog=rdflib.Namespace('https://w3id.org/fog#')
v4d=rdflib.Namespace('https://w3id.org/v4d/core#')
openlabel=rdflib.Namespace('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#')
e57=rdflib.Namespace('http://libe57.org#')
xcr=rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')

#### BASIC OPERATIONS ####

def time_funtion(func, *args):
    """Measures how long the functions takes to run and returns the result 

    Args:
        func (function): The funtion to measure, write without ()
        *args (Tuple) : The arguments for the funtion, pass as a tuple with a * in front to pass the arguments seperatly

    Returns:
        object: The result of the function
    """

    start = time.time()
    result = func(*args)
    end = time.time()
    print("Completed function `" + func.__name__ + "()` in", np.round(end - start,3), "seconds")
    return result

def get_extension(path:str) -> str:
    """Returns the file extension.\n
    E.g. D://myData//test.txt -> .txt

    Args:
        path (str): file path

    Returns:
        extension (str) e.g. '.txt'
    """
    _,extension = os.path.splitext(path)
    return extension

#NOTE why does this exist?
def get_min_average_and_max_value(arrays:List[np.array],ignoreZero:bool=True,threshold:float=0.0)->List[float]:
    """Return an min,average and max values of a list of arrays. 

    **NOTE** Nan is automatically ignored.\n
    
    Args:
        1.arrays (List[np.array]): arrays are automatically flattened.\n
        2.ignoreZero (bool, optional): Ignores 0.0 in the arrays array. Defaults to True.\n
        3.Threshhold (float, optional): Return Nan if less than a threhold is not Nan. Defaults to 0.0.\n

    Returns:
        List[float]: average values
    """
    arrays=item_to_list(arrays)
    minarr=np.full(len(arrays),np.nan)
    meanarr=np.full(len(arrays),np.nan)
    maxarr=np.full(len(arrays),np.nan)
    
    for i,a in enumerate(arrays):
        a=a.flatten()
        if ignoreZero:
            a=a[np.nonzero(a)]
            if any(a) and a.size > threshold*minarr.size:                
                minarr[i]=np.nanmin(a)        
                meanarr[i]=np.nanmean(a)
                maxarr[i]=np.nanmax(a) 
    return minarr  ,meanarr  ,maxarr

def replace_str_index(text:str,index:int,replacement:str='_'):
    """Replace a string character at the location of the index with the replacement.\n

    Args:
        1. text (str)\n
        2. index (int): index to replace. \n
        3. replacement (str, optional): replacement character. Defaults to '_'.\n

    Returns:
        text (str) with updated characters
    """
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def random_color(range:int=1)->np.array:
    """Generate random color (either [0-1] or [0-255]).\n

    Args:
        range (int, optional): 1 or 255. Defaults to 1.

    Raises:
        ValueError: Range should be either 1 or 255.

    Returns:
        np.array[3x1]
    """
    color=np.array([random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
    if int(range)==255:
        pass
    elif int(range)==1:
       color=color/255
    else:
        raise ValueError('Range should be either 1 or 255.')
    return color

def split_list(list, n:int=None,l:int=None):
    """Split list into approximately equal chunks. Last list might have an unequal number of elements.

    Args:
        list (object): list to split.\n.
        n (int,optional): number of splits.\n
        l:(int,optional): length of each chunk.\n

    Returns:
        List[List]: 
    """
    if n:
        k, m = divmod(len(list), n)
        return [list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
    elif l:
        n=int(math.ceil(len(list)/l))
        return [list[i*l:(i+1)*l] for i in range(n)]
    else:
        raise ValueError('No input provided. Enter n or l.')

def get_folder_path(path :str) -> str:
    """Returns the folderPath.

    Args:
        path(str): file path

    Raises:
        FileNotFoundError: If the given graphPath or sessionPath does not lead to a valid systempath

    Returns:
        The path (str) to the folder
    """
    folderPath= os.path.dirname(path)
    if not os.path.isdir(folderPath):
        print("The given path is not a valid folder on this system.")    
    return folderPath

def get_variables_in_class(cls) -> List[str]: 
    """Returns a list of class attributes in the class.\n

    Args:
        cls(class): class e.g. MeshNode
    
    Returns:
        list of class attribute names
    """  
    return [i.strip('_') for i in cls.__dict__.keys() ] 

def get_list_of_files(directoryPath:str) -> list:
    """Get a list of all filepaths in the directory and subdirectories.\n

    Args:
        directoryPath: directory path e.g. \n
            "D:\\Data\\2018-06 Werfopvolging Academiestraat Gent\\week 22\\"
            
    Returns:
        A list of filepaths
    """
    # names in the given directory 
    directoryPath = str(directoryPath)
    listOfFile = os.listdir(directoryPath)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = Path(directoryPath) / entry
        # If entry is a directory then get the list of files in this directory 
        if fullPath.is_dir():
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath.as_posix())                
    return allFiles

def get_subject_graph(graph:Graph, subject:URIRef = None) -> Graph:
        """Returns a subselection of the full Graph that only contains the triples of one subject.\n

        Args:
            1. graph (Graph) \n
            2. subject (URIRef, optional): If no subject is provided, the first one is picked. Defaults to first subject in the graph.
        
        Returns:
            Graph with the triples of one subject
        """
        #input validation       
        if(subject and subject not in graph.subjects()): 
            raise ValueError('subject not in graph')
        elif (not subject): # No subject is defined yet, pick the first one
            subject=next(graph.subjects())        

        #create graph
        newGraph = Graph()
        newGraph += graph.triples((subject, None, None)) 
        newGraph.namespace_manager = graph.namespace_manager
        #newGraph._set_namespace_manager(graph._get_namespace_manager())

        #validate output
        if (len(newGraph) !=0):
            return newGraph
        else:
            return None

def get_paths_in_class(cls) -> list: 
    """Returns list of path attributes in the class.\n

    Args:
        cls (class): class e.g. MeshNode

    Rerturns:
        class atttributes that contain 'path' information e.g. cls.ifcPath, cls.path, etc.
    """  
    from re import search
    return [i.strip('_') for i in cls.__dict__.keys() if search('Path',i) or search('path',i)] 

# NOTE just use get() function
#def get_value_from_dict(dict:dict, key:str):
#    """Return data of a certain key in a dictionary:
#
#    Args:
#        1. data (dict): dictionay with data \n
#        2. key (str): keyword attribute in dictionary e.g. data['myattribute'] \n
#
#    Returns:
#        data (python values)
#    """
#    if key in dict:
#        return dict[key]
#    return None

# NOTE use different word than filter, because you are just inverting the negative coordinates
def parse_exif_gps_data(data, reference:str) -> float:
    """Returns decimal degrees from world angles in exif data.\n

    Args:
        data (float or tuple): decimal degrees or (degrees, minutes, seconds) notation of world angles\n
        reference (str): direction character i.e. 'S','W','s' or 'w'.\n

    Returns:
        decimal degrees (float)
    """
    if type(data) is float:
        if reference in ['S','W','s','w']:
            return data*-1
        else:
            return data
    elif type(data) is tuple and len(data) == 3:
        value=dms_to_dd(data[0],data[1],data[2],reference)
        return value    
     
# NOTE use consistent naming sceme
def dms_to_dd(degrees:float, minutes:float, seconds:float, direction:str) -> float:
    """Convert world angles (degrees, minutes, seconds, direction) to decimal degrees.\n

    Args:
        1. degrees (float) \n
        2. minutes (float) \n
        3. seconds (float) \n
        4. direction (str): 'N' and 'E' are positive, 'W' and 'S' are negative  \n

    Returns:
        decimal degrees (float)
    """
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd

def dd_to_dms(deg) -> List[float]:
    """Convert decimal degrees to List[degrees, minutes, seconds] notation. \n

    **NOTE**: Direction is not determined (should be easy based on sign).

    Args:
        decimal degrees (float)

    Returns:
        List[degrees(float),minutes(float),seconds(float)]
    """
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]

def parse_dms(dms) :
    """Returns decimal degrees (float) notation of geospatial coordinates from various (degrees, minutes, seconds, direction) notations.\n

    Args (degrees, minutes, seconds,direction):
        1. dms (np.array): \n
        2. dms (tuple[float,float,float,float])\n
        3. dms (str): stringed list with values\n

    Raises:
        ValueError: 'dms.size!=4'

    Returns:
        decimal degrees (float)
    """
    try:
        if type(dms) is np.ndarray and dms.size==4:
            return dms_to_dd(dms[0],dms[1],dms[2], dms[3] )
        elif type(dms) is tuple and len(dms)==4:
            return dms_to_dd(dms[0],dms[1],dms[2], dms[3] )
        elif type(dms) is str and 'None' not in dms:
            temp=validate_string(dms, ' ')
            temp=temp.replace("\n","")
            temp=temp.replace("\r","")
            temp=temp.split(' ')
            temp=[x for x in temp if x]
            if temp:
                res=np.asarray(temp) 
                return dms_to_dd(res[0],res[1],res[2], res[3] )
        return None  
    except:
        raise ValueError ('dms.size!=4')

def get_exif_data(img:PIL.Image)-> dict:
    """Returns a dictionary from the exif data of an Image (PIL) item. Also
    converts the GPS Tags.\n

    https://pillow.readthedocs.io/en/stable/reference/Image.html
    
    Args:
        Img (PIL.image)
    
    Returns:
        exifData (dict): dictionary containing all meta data of the PIL Image.
    """
    exifData = {}
    info = img._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exifData[decoded] = gps_data
            else:
                exifData[decoded] = value        
        return exifData      
    else:
        return None    

# TODO fix the path shit, only windows works with backslashes for file paths but is also compatible with POSIX style, so we should just use the Pathlib as much as possible
def get_filename(path :str,splitter:str='.') -> str:
    """ Deconstruct filepath and return filename.
    
    Args:
        path (str): filepath
        
    Returns:
        filename (str)
    """   
    path=ntpath.basename(path)
    _, tail = ntpath.split(path)
    array=tail.split(splitter)
    return array[0]

def get_folder(path :str) -> str:
    """ Deconstruct path and return folder
    
    Args:
        path (str): filepath
        
    Returns:
        folderPath (str)
    """
    return os.path.dirname(os.path.abspath(path))

# TODO when do we want a timestamp based on the last modified time of the metafile?
def get_timestamp(path : str) -> str:
    """Returns the timestamp ('%Y-%m-%dT%H:%M:%S') from a filepath.

    Args:
        path (str): filepath

    Returns:
        dateTime (str): '%Y-%m-%dT%H:%M:%S'
    """
    ctime=os.path.getctime(path)
    dtime=datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%dT%H:%M:%S')
    return dtime

#### CONVERSIONS ####

def literal_to_cartesianTransform(literal:Literal) -> np.array:
    """Returns cartesianTransform from rdflib.literal. This function is used to convert a serialized (str) to a usable np.array. 

    Args:
        literal (URIRef):  stringed list or np array of the cartesianTransform e.g. \n
            e57:cartesianTransform ""[[ 3.48110207e-01  9.37407536e-01  9.29487057e-03  2.67476305e+01]
                                    [-9.37341584e-01  3.48204779e-01 -1.20077869e-02  6.17326932e+01]
                                    [-1.44927083e-02 -4.53243552e-03  9.99884703e-01  4.84636987e+00]
                                    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]""

    Returns:
        cartesianTransform (np.array([4x4])): geometric transform of a Node/Resource.   
    """   
    temp=str(literal)
    try:
        if 'None' not in temp:
            temp=validate_string(temp, ' ')
            temp=temp.replace("\n","")
            temp=temp.replace("\r","")
            temp=temp.split(' ')
            temp=[x for x in temp if x]
            if temp:
                res = list(map(float, temp))   
                res=np.reshape(res,(4,4))
                return np.asarray(res)  
        return None  
    except:
        raise ValueError

def literal_to_array(literal: Literal) -> np.array:
    """Returns array from rdflib.literal.\n

    Args:
        literal (rdflib): literal containing serialized list or array 

    Raises:
        ValueError: 'Conversion error'

    Returns:
        np.array 
    """
    temp=str(literal)
    return np.asarray(string_to_list(temp))

def literal_to_orientedBounds(literal: Literal)-> np.array:
    """Returns orientedBounds from rdflib.literal. This function is used to convert a serialized (str) to a usable np.array. 

    Args:
        literal (URIRef):  stringed list or np array of the orientedBounds e.g. \n
            v4d:orientedBounds  ""[[-1.96025758e+01  1.65884155e+02  2.22874746e+01]\n
                                [ 1.22465470e+02  1.23859440e+02  2.29468276e+01]\n
                                [-5.26111779e+01  5.43129133e+01  2.33762930e+01]\n
                                [-1.95654774e+01  1.65648750e+02 -7.09825603e-01]\n
                                [ 8.94939663e+01  1.20527928e+01  1.03834566e+00]\n
                                [-5.25740795e+01  5.40775081e+01  3.78992731e-01]\n
                                [ 1.22502568e+02  1.23624035e+02 -5.04726756e-02]\n
                                [ 8.94568679e+01  1.22881979e+01  2.40356459e+01]]""

    Returns:
        orientedBounds (np.array([8x3])): 8 bounding coordinates of a geometric asset.   
    """   
    temp=str(literal)
    try:
        if 'None' not in temp:
            temp=validate_string(temp, ' ')
            temp=temp.replace("\n","")
            temp=temp.replace("\r","")
            temp=temp.split(' ')
            temp=[x for x in temp if x]
            if temp:
                res = list(map(float, temp))   
                res=np.reshape(res,(8,3))
                return np.asarray(res)  
        return None  
    except:
        raise ValueError

def literal_to_float(literal: Literal) -> float:
    """Returns float from rdflib.literal 

    Args:
        literal (rdflib): literal containing value

    Raises:
        ValueError: 'Conversion error'

    Returns:
        float 
    """
    try:
        if 'None' in literal:
            return None
        return float(literal.toPython())
    except:
        raise ValueError('Conversion error')

def literal_to_string(literal: Literal)->str:
    """Returns string from rdflib.literal.\n

    Args:
        literal (rdflib): literal containing value

    Returns:
        string
    """
    string=str(literal)
    try:
        if 'None' in string:
            return None
        else:
            return string
    except:
        raise ValueError

def literal_to_list(literal: Literal)->list:
    """Returns list from rdflib.literal.

    Args:
        literal (rdflib): literal containing value

    Raises:
        ValueError: 'Conversion error'

    Returns:
        int 
    """
    string=str(literal)
    try:
        if 'None' in string:
            return None
        else: 
            return string_to_list(string)
    except:
        raise ValueError

def literal_to_int(literal: Literal) -> int:
    """Returns int from rdflib.literal.

    Args:
        literal (rdflib): literal containing value

    Raises:
        ValueError: 'Conversion error'

    Returns:
        int 
    """
    try:
        if 'None' in literal:
            return None
        return int(literal.toPython())
    except:
        raise ValueError ('Conversion error')

def literal_to_uriref(literal: Literal)->URIRef:
    """Returns rdflib.URIRef from rdflib.literal.

    Args:
        literal (rdflib): literal containing value

    Raises:
        ValueError: 'float causes errors'

    Returns:
        URIRef: 
    """
    try:
        temp=literal.toPython()
        if type(temp) is float or type(temp) is int:
            raise ValueError('float causes errors')
        elif 'None' not in literal:            
            return URIRef(literal.toPython())      
        return None
    except:
        raise ValueError

def check_if_subject_is_in_graph(graph:Graph,subject:URIRef) ->bool:
    """Returns True if a subject is present in the Graph. 

    Args:
        graph (Graph): Graph to parse.\n
        subject (URIRef): subject to search. The function only uses the main body of the subject so no prefix mistakes can be made.\n

    Returns:
        bool: True if subject is present.
    """
    testSubject=subject.split('/')[-1]
    for s in graph.subjects():
        graphSubject= s.split('/')[-1]
        if testSubject==graphSubject:
            return True
    return False

def get_graph_intersection(graphs:List[Graph]) -> Graph:
    """Returns the intersection of multiple graphs i.e. all triples of common subjects.\n

    Args:
        graphs (List[Graph]):

    Returns:
        Graph of the intersection
    """
    #retrieve common subjects
    intersectionGraph=graphs[0]
    for i in range(len(graphs)-1):
        intersectionGraph=intersectionGraph & graphs[i+1]
    subjects=[s for s in intersectionGraph.subjects(RDF.type)]
    
    # if no overlap, return None
    if len(subjects)==0:
        return None

    #merge graphs
    joinedGraph=Graph()
    joinedGraph=bind_ontologies(joinedGraph)
    for graph in graphs:
        joinedGraph+=graph

    #select all relevant graphs
    selectGraph=Graph()
    selectGraph=bind_ontologies(selectGraph)
    for s in subjects:
        selectGraph+=get_subject_graph(joinedGraph,s)

    return selectGraph

def get_graph_subject(graph:Graph,subject:URIRef) ->URIRef:
    """Returns graph subject with its appropriate prefix given the main body of a subject. This function is mainly used for local file serializations that mud the graph subjects. You can then use this function to use the correct graph subject regardless of the prefix.\n

    Args:
        1. graph (Graph): graph with subjects with prefix e.g. URIRef(D:\\DATA\\Subject1)\n
        2. subject (URIRef): subject to search for. e.g. URIREF(Subject1)\n

    Raises:
        ValueError: 'subject not in graph.'

    Returns:
        URIRef(subject): e.g.  URIRef(D:\\DATA\\Subject1)
    """
    testSubject=subject.split('/')[-1]
    for s in graph.subjects():
        graphSubject= s.split('/')[-1]
        if testSubject==graphSubject:
            return s
    raise ValueError ('subject not in graph.')

def literal_to_linked_subjects(string:str) -> list:
    """Returns list of URI subjects that were serialized as a string.\n

    Args:
        string (str): e.g. '[ file:///Subject1, http://Subject2]'

    Returns:
        list[items] e.g. [ file:///Subject1, http://Subject2]
    """
    temp=string.split(',')
    temp=[re.sub(r"[\[ \' \]]",'',s) for s in temp]
    return temp

# TODO fix deze warboel van een functie
def string_to_list(string:str)->list:
    """Convert string of items to a list of their respective types. Function deals with both np.array and list encodings of the stringed values.\n

    Args:
        string (str): string to parse. 

    Returns:
        List[items in string]
    """
    try:
        if 'None' not in string:
            temp=validate_string(string, ' ')
            temp=temp.replace("\n","")
            temp=temp.replace("\r","")
            temp=temp.split(' ')
            temp=[x for x in temp if x]
            # res = list(map(float, temp))  
            if temp:
                res=[]
                for item in temp:      
                    if is_float(item): 
                        res.append(float(item))
                    elif is_int(item): 
                        res.append(int(item))
                    elif is_string(item): 
                        res.append(str(item))
                    elif is_uriref(item): 
                        res.append(URIRef(item)) 
                return res
        return None  
    except:
        raise ValueError

def string_to_rotation_matrix(string :str) -> np.array:
    """Returns rotation matrix (np.array(3x3)) from string.

    Args:
        string (str): string cast list or np.array that contains the values of the rotation matrix.\n

    Raises:
        ValueError: 'array.size!=9'

    Returns:
        np.array (3x3)
    """
    array=np.asarray(string_to_list(string))
    if array.size==9:
        return np.reshape(array,(3,3))
    else:
        raise ValueError('array.size!=9')

def xml_to_float(xml) -> float:
    """Cast XML string value to float if possible.

    Args:
        xml value

    Returns:
        float of value
    """
    if xml is None:
        return None
    else:
        return float(xml)

def xcr_to_lat(xcr:str) -> float:
    """Returns latitude from XCR serialization. This includes interpretation 'N' and 'S' geospatial values.

    Args:
        xcr (str)

    Returns:
        float of value
    """
    if 'None' in xcr:
        return None
    else:
        list=list=re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?', xcr)
        if 'N' in list[-1]:
            return float(list[0])
        elif 'S' in list[-1]:
            return - float(list[0])

def xcr_to_long(xcr:str) -> float:
    """Returns longitude from XCR serialization. This includes interpretation 'E' and 'W' geospatial values.

    Args:
        xcr (str): 

    Returns:
        float of value
    """
    if 'None' in xcr:
        return None
    else:        
        list=list=re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?', xcr)
        if 'E' in list[-1]:
            return float(list[0])
        elif 'W' in list[-1]:
            return - float(list[0])

def xcr_to_alt(xcr:str) -> float:
    """Returns altitude from XCR serialized height value. This value is sometimes encoded as a fracture 10000/1600.\n

    Args:
        xcr (str): value to process

    Returns:
        float of value
    """
    if 'None' in xcr:
        return None
    else:
        list=list=re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?', xcr)
        if list:
            return float(list[0])/float(list[-1])       

def cartesianTransform_to_literal(matrix : np.array) -> str:
    """ convert nparray [4x4] to str([16x1]).

    Args:
        nparray [4x4]

    Returns:
        List[16]
    """
    if matrix.size == 16: 
        return str(matrix.reshape(16,1))
    else:
        Exception("wrong array size!")    
#TODO 
def featured3d_to_literal(value) -> str:
    "No feature implementation yet"
#TODO
def featured2d_to_literal(value) -> str:
    "No feature implementation yet"

def item_to_list(item)-> list:
    """Returns [item] if item is not yet a list. This function protects functions that rely on list functionality. 

    Args:
        item (Python value) 

    Returns:
        list[item]
    """
    if type(item) is np.ndarray:
        item=item.flatten()
        return item.tolist()
    elif type(item) is np.array:
        item=item.flatten()
        return item.tolist()
    elif type(item) is list:
        return item
    else:
        return [item]

#### VALIDATION ####

def check_if_uri_exists(list:List[URIRef], subject:URIRef) ->bool:
    """Returns True if a subject occurs in a list of URIRefs.

    Args:
        list (List[URIRef]): reference list
        subject (URIRef): subject to test

    Returns:
        bool: True if subject occurs in list
    """
    list=item_to_list(list)
    list=[item.toPython() for item in list]
    subject=subject.toPython()
      
    if any(subject in s for s in list):
        return True
    else: 
        return False

def get_subject_name(subject:URIRef) -> str:
    """Get the main body of a URIRef graph subject

    Args:
        subject (URIRef)

    Returns:
        str
    """
    string=subject.toPython()
    return string.split('/')[-1]   

def validate_string(string:str, replacement ='_') -> str:
    """Checks path validity. A string is considered invalid if it cannot be serialized by rdflib or is not Windows subscribable.\n
    If not valid, The function adjusts path naming to be Windows compatible.\n

    Features (invalid characters):
        "()^[^<>{}[] ~`],|*$ /\:"\n

    Args:
        path (str): string to check
        replacement (character): characters to replace invalid charters in the string 

    Returns:
        str: cleansed string
    """
    prefix=''
    if 'file:///' in string:
        string=string.replace('file:///','')
        prefix='file:///'
    elif 'http://' in string:
        string=string.replace('http://','')
        prefix='http://'
    for idx,element in enumerate(string):
        if element in "()^[^<>{}[] ~`],|*$ /\:": #
            string=replace_str_index(string,index=idx,replacement=replacement)
    string=prefix+string
    return string

def parse_path(path : str)-> str:
    """Reads a path and converts it to posix universal styling

    Args:
        path (str): the input path

    Returns:
        str: the converted string
    """
    if(path):
        return PureWindowsPath(path).as_posix()
    return None

#NOTE this returns true if the argument can be cast to a float, so int work here to.
def is_float(element) -> bool:
    """Returns True if Python value is a float.

    Args:
        element (Python value)

    Returns:
        bool: True if Python value is a float
    """
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_int(element) -> bool:
    """Returns True if Python value is an int.

    Args:
        element (Python value)

    Returns:
        bool: True if Python value is an int
    """
    try:
        int(element)
        return True
    except ValueError:
        return False

def is_string(element) -> bool:
    """Returns True if Python value is a str.

    Args:
        element (Python value)

    Returns:
        bool: True if Python value is a str
    """
    #special_characters = "[!@#$%^&*()-+?_= ,<>/]'"
    try:
        str(element)
        return True
    except ValueError:
        return False

# NOTE this function does not confirm it is an URI ref
def is_uriref(element) -> bool:
    """Returns True if Python value is a rdflib.URIRef.

    Args:
        element (Python value)

    Returns:
        bool: True if Python value is a rdflib.URIRef
    """
    try:
        if type(element) is float:
            return False
        else:
            URIRef(element)
            return True
    except ValueError:
        return False

# #### RDF ####

def get_attribute_from_predicate(graph: Graph, predicate : Literal) -> str:
    """Returns the attribute witout the namespace.

    Args:
        graph (Graph): The Graph containing the namespaces
        predicate (Literal): The Literal to convert

    Returns:
        str: The attribute name
    """
    predStr = str(predicate)
    #Get all the namespaces in the graph
    for nameSpace in graph.namespaces():
        nameSpaceStr = str(nameSpace[1])
        if(predStr.__contains__(nameSpaceStr)):
            predStr = predStr.replace(nameSpaceStr, '')
            break
    return predStr

# TODO this should be generalised so it's easier to add new namespaces that are not hardcoded
# example:
# namespaces = [('exif','http://www.w3.org/2003/12/exif/ns#' ),...]
# for namespace in namespaces:
#     graph.bind(namespace[0],rdflib.Namespace(namespace[1]))
def bind_ontologies(graph : Graph) -> Graph:
    """Returns a graph that binds in its namespace the ontologies that GEMOMAPI uses and that are not in the rdflib.\n\n

     Features (ontologies):
        1. exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns#') \n
        2. geo=rdflib.Namespace('http://www.opengis.net/ont/geosparql#') \n
        3. gom=rdflib.Namespace('https://w3id.org/gom#') \n
        4.  omg=rdflib.Namespace('https://w3id.org/omg#') \n
        5. fog=rdflib.Namespace('https://w3id.org/fog#')\n
        6. v4d=rdflib.Namespace('https://w3id.org/v4d/core#')\n
        7. openlabel=rdflib.Namespace('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#')\n
        8. e57=rdflib.Namespace('http://libe57.org#')\n
        9. xcr=rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')\n
        10. ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')\n

    Returns:
        Graph with update namespace 
    """
    exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns#')
    graph.bind('exif', exif)
    geo=rdflib.Namespace('http://www.opengis.net/ont/geosparql#') #coordinate system information
    graph.bind('geo', geo)
    gom=rdflib.Namespace('https://w3id.org/gom#') # geometry representations => this is from mathias
    graph.bind('gom', gom)
    omg=rdflib.Namespace('https://w3id.org/omg#') # geometry relations
    graph.bind('omg', omg)
    fog=rdflib.Namespace('https://w3id.org/fog#')
    graph.bind('fog', fog)
    v4d=rdflib.Namespace('https://w3id.org/v4d/core#')
    graph.bind('v4d', v4d)
    v4d3D=rdflib.Namespace('https://w3id.org/v4d/3D#')
    graph.bind('v4d3D', v4d3D)
    openlabel=rdflib.Namespace('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#')
    graph.bind('openlabel', openlabel)
    e57=rdflib.Namespace('http://libe57.org#')
    graph.bind('e57', e57)
    xcr=rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
    graph.bind('xcr', xcr)
    ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')
    graph.bind('ifc', ifc)
    loa=rdflib.Namespace('https://docplayer.net/131921614-Usibd-level-of-accuracy-loa-specification-guide.html#')
    graph.bind('loa', loa)    
    return graph

# instead of cleaning the attributes, we should only serialise the rdf variables
def clean_attributes_list(list:list) -> list:
    """Returns a 'cleaned' list of class attributes that are not to be serialized in the graph. This includes data attributes such as resources (point clouds, images), linkedNodes, etc.\n

    **NOTE**: RDF Graphs are meant to store metadata, not actual GBs of data that should be contained in their native file formats.\n

    Args:
        list (list): list of class attributes

    Returns:
        list (list): 'cleaned' list of class attributes 
    """
    #NODE
    excludedList=['graph','resource','graphPath','subject','resource','fullResourcePath','kwargs', 'orientedBoundingBox','type']
    #BIMNODE    
    excludedList.extend(['ifcElement'])
    #MESHNODE
    excludedList.extend(['mesh'])
    #IMGNODE
    excludedList.extend(['exifData','xmlData','image','features2d','pinholeCamera'])
    #PCDNODE
    excludedList.extend(['pcd','e57Pointcloud','e57xmlNode','e57image','features3d'])
    #SESSIONNODE
    excludedList.extend(['linkedNodes'])

    cleanedList = [ elem for elem in list if elem not in excludedList]
    return cleanedList

def check_if_path_is_valid(path:str)-> bool:
    """Returns True if the given path is an existing folder.\n

    Args:
        path (str): path to folder or file

    Returns:
        bool: True if exsists.
    """
    folder=get_folder(path)
    if os.path.isdir(path):
        return True
    elif os.path.exists(folder):
        return True
    else:
        return False

def validate_timestamp(value) -> datetime:
    """Format value as datetime ("%Y-%m-%dT%H:%M:%S")

    Args:
        1.value ('Tue Dec  7 09:38:13 2021')\n
        2.value ("1648468136.033126")\n
        3.value ("2022:03:13 13:55:30")\n
        4.value (datetime)\n
        5.value ("2022-03-13 13:55:30")\n

    Raises:
        ValueError: 'timestamp formatting ("%Y-%m-%dT%H:%M:%S") failed for tuple, datetime and string formats.'

    Returns:
        datetime
    """
    string=str(value)
  
    try:
        return datetime.datetime.strptime(string, "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") 
    except:
        pass
    try:         
        test=datetime.datetime.strptime(string, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") 
        return test
    except:
        pass
    try:         
        test=datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") 
        return test
    except:
        pass
    try:         
        test=datetime.datetime.strptime(string, "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%dT%H:%M:%S") 
        return test
    except:
        pass
    try:
        return datetime.datetime.utcfromtimestamp(float(string)).strftime('%Y-%m-%dT%H:%M:%S')
    except:
        raise ValueError('no valid time formatting found e.g. 1.value (Tue Dec  7 09:38:13 2021) 2.value (1648468136.033126) 3.value (2022:03:13 13:55:30)')

def get_node_resource_extensions(objectType:str) -> list:
    """Returns the potential file formats for different types of Nodes' resources (images, pcd, ortho, etc.).\n

    Args:
        objectType (str): Type of node class

    Returns:
        list with possible extensions e.g. .obj, .ply for MeshNodes
    """    
    if 'MeshNode' in objectType:        
        return MESH_EXTENSION
    if 'SessionNode' in objectType:        
        return MESH_EXTENSION
    elif 'BIMNode' in objectType:        
        return MESH_EXTENSION
    elif 'PointCloudNode' in objectType:        
        return PCD_EXTENSION    
    elif 'ImageNode' in objectType:        
        return IMG_EXTENSION
    elif 'OrthoNode' in objectType:        
        return IMG_EXTENSION
    else:
        return ['.txt']+MESH_EXTENSION+PCD_EXTENSION+IMG_EXTENSION+RDF_EXTENSIONS

# This whole function can be replaced with: "return v4d[type(node).__name__]" where node is the node object
def get_node_type(objectType:str) -> URIRef:
    """Return the type of Node as an rdflib literal. By default, 'Node' is returned.

    Features:
        1. MeshNode\n
        2. BIMNode\n
        3. PointCloudNode\n
        4. GeometryNode\n
        5. ImageNode\n
        6. OrthoNode\n
        7. SessionNode\n
        8. linesetNode\n
        9. Node\n

    Args:
        objectType (str): Type of node class.\n

    Returns:
        URIRef (nodeType)
    """
    if 'MeshNode' in objectType:        
        return v4d['MeshNode']
    elif 'BIMNode' in objectType:        
        return v4d['BIMNode']
    elif 'PointCloudNode' in objectType:        
        return v4d['PointCloudNode']
    elif 'GeometryNode' in objectType:        
        return v4d['GeometryNode']
    elif 'ImageNode' in objectType:        
        return v4d['ImageNode']
    elif 'OrthoNode' in objectType:        
        return v4d['OrthoNode']
    elif 'SessionNode' in objectType:        
        return v4d['SessionNode']
    elif 'linesetNode' in objectType:        
        return v4d['linesetNode']
    else:
        return v4d['Node']

def get_data_type(value) -> XSD.ENTITY:
    """Return XSD dataType of Python value. By default, string is returned as the XSD.entity.\n

    Args:
        value (any): data of any Python value (boolean, int, float, dateTime, string)

    Returns:
        XSD.ENTITY 
    """
    if 'bool' in str(type(value)):        
        return XSD.boolean
    elif 'int' in str(type(value)):  
        return XSD.integer
    elif 'float' in str(type(value)):  
        return XSD.float
    elif 'date' in str(type(value)):  
        return XSD.dateTime   
    else:
        return XSD.string

# NOTE maybe we should use decorators to define the attribute in the node itself to prevent double writing
def match_uri(attribute :str) -> URIRef:
    """ Returns fitting predicate from class attribute given the following prefix ontologies.\n
    By default, the v4d ontology[attributeName] is returned.

    Features (ontologies):
        1. exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns#') \n
        2. geo=rdflib.Namespace('http://www.opengis.net/ont/geosparql#') \n
        3. gom=rdflib.Namespace('https://w3id.org/gom#') \n
        4.  omg=rdflib.Namespace('https://w3id.org/omg#') \n
        5. fog=rdflib.Namespace('https://w3id.org/fog#')\n
        6. v4d=rdflib.Namespace('https://w3id.org/v4d/core#')\n
        7. openlabel=rdflib.Namespace('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#')\n
        8. e57=rdflib.Namespace('http://libe57.org#')\n
        9. xcr=rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')\n
        10. ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')\n

    Returns:
        URIRef: predicate 
    """
    #OPENLABEL
    if attribute in ['timestamp','sensor']:
        return openlabel[attribute]
    #E57
    elif attribute in ['cartesianBounds','cartesianTransform','geospatialTransform','pointCount','e57XmlPath','e57Path','e57Index','e57Image']:
        return  e57[attribute]
    #GOM
    elif attribute in ['coordinateSystem']:
        return  gom[attribute]
    #IFC
    elif attribute in ['ifcPath','className','globalId','phase','ifcName']:
        return  ifc[attribute]
    #EXIF
    elif attribute in ['xResolution','yResolution','resolutionUnit','imageWidth','imageHeight']:
        return  exif[attribute]
    #XCR
    elif attribute in ['focalLength35mm','principalPointU','principalPointV','distortionCoeficients','gsd']:
        return  xcr[attribute]
    #XCR
    elif attribute in ['isDerivedFromGeometry']:
        return  omg[attribute]
    #V4D
    else:
        return v4d[attribute]
