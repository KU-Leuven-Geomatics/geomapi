"""
General Basic functions to support other modules.
"""
import datetime
import os
from pathlib import Path
import re
import time
import dateutil.parser
import math
from typing import Callable, List, Optional, Union
import random
import importlib
import numpy as np
from rdflib import RDF, XSD, Graph, Literal, URIRef,Namespace,RDFS

#### GLOBAL VARIABLES ####

RDF_EXTENSIONS = [".TTL"]
IMG_EXTENSIONS = [".JPG", ".PNG", ".JPEG",".TIF"]
MESH_EXTENSIONS = [".OBJ",".PLY",".FBX" ]
PCD_EXTENSIONS = [".PCD", ".E57",".PTS", ".PLY",'.LAS','.LAZ']
BIM_EXTENSIONS=[".IFC"]
CAD_EXTENSIONS=[".PLY",".DXF",".TFW"]
TIME_FORMAT = "%Y-%m-%d %H-%M-%S"

RDFMAPPINGS = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
geomapi_ontology_path=os.path.join(project_root, 'geomapi', 'ontology', 'geomapi_ontology.ttl')

GEOMAPI_GRAPH=Graph().parse(geomapi_ontology_path) if os.path.exists(geomapi_ontology_path) else Graph().parse('https://w3id.org/geomapi')
GEOMAPI_PREFIXES = {prefix: Namespace(namespace) for prefix, namespace in GEOMAPI_GRAPH.namespace_manager.namespaces()}
GEOMAPI_NAMESPACE = Namespace('https://w3id.org/geomapi#')

IFC_GRAPH=Graph().parse("https://standards.buildingsmart.org/IFC/DEV/IFC4/ADD2_TC1/OWL/ontology.ttl")
IFC_NAMESPACE = Namespace("https://standards.buildingsmart.org/IFC/DEV/IFC4/ADD2_TC1/OWL#")

#### BASIC OPERATIONS ####

def time_function(func, *args):
    """Measures how long the functions takes to run and returns the result 

    Args:
        func (function): The function to measure, write without ()
        *args (Tuple) : The arguments for the function, pass as a tuple with a * in front to pass the arguments separately

    Returns:
        object: The result of the function
    """

    start = time.time()
    result = func(*args)
    end = time.time()
    print("Completed function `" + func.__name__ + "()` in", np.round(end - start,3), "seconds")
    return result

def get_timestamp(path : str) -> str:
    """Returns the timestamp ('%Y-%m-%dT%H:%M:%S') from a filepath.

    Args:
        path (str): filepath

    Returns:
        dateTime (str): '%Y-%m-%dT%H:%M:%S' The creation date (Windows) or the last modified date (Linux)
    """
    if(os.path.exists(path)):
        ctime=os.path.getctime(path)
        dtime=datetime.datetime.fromtimestamp(ctime)
        return literal_to_datetime(dtime)
    raise ValueError("Path does not exist")

def get_random_color(range:int=1) -> np.ndarray:
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

def map_to_2d_array(input_data: list | np.ndarray) -> np.ndarray:
    """
    Converts the input data into a 2D NumPy array.

    Args:
        input_data (list or numpy.ndarray): The input data, which can be a list or a NumPy array.

    Returns:
        numpy.ndarray: A 2D NumPy array representation of the input data. If the input is a 1D array,
                    it is expanded to a 2D array with a single row.
    """
    if isinstance(input_data, list):
        input_data = np.array(input_data)

    # Ensure input_data is at least 2D
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)
    
    return input_data

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
    elif type(item) is list:
        return item
    else:
        return [item]

def split_list(list, n:int=None,l:int=None) -> list:
    """Split list into approximately equal chunks. Last list might have an unequal number of elements.

    Args:
        list (object): list to split.\n
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

def replace_str_index(text:str,index:int,replacement:str='_') -> str:
    """Replace a string character at the location of the index with the replacement. index must be in the range of the string \n

    Args:
        1. text (str)\n
        2. index (int): index to replace. -1 indicates the end of the string \n
        3. replacement (str, optional): replacement character. Defaults to '_'.\n

    Returns:
        text (str) with updated characters
    """
    # If the index is -1, replace the last char
    if(index == -1): index = len(text)-1
    # raise an error if index is outside of the string
    if index not in range(len(text)):
        raise ValueError("index outside given string")
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def get_list_of_files(folder: Path | str , ext: str = None) -> list:
    """
    Get a list of all filepaths in the folder and sub folders that match the given file extension.

    Args:
        folder: The path to the folder as a string or Path object
        ext: Optional. The file extension to filter by, e.g., ".txt". If None, all files are returned.

    Returns:
        A list of filepaths that match the given file extension.
    """
    folder = Path(folder)  # Ensure the folder is a Path object
    allFiles = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            # check for an optional extension
            if ext is None or filepath.lower().endswith(ext.lower()):
                allFiles.append(Path(filepath))
    return allFiles

##### Ontology functions #######

def get_geomapi_classes() -> List[URIRef]:
    """Get all classes from the GEOMAPI graph.

    Returns:
        - List[URIRef]: A list of ontology classes.
    """
    query = '''
    SELECT ?class
    WHERE {
        ?class a owl:Class.
    }
    '''
    result = GEOMAPI_GRAPH.query(query)
    return [row['class'] for row in result]

def get_method_for_datatype(datatype : str) -> str:
    """Get the method associated with a datatype in the GEOMAPI graph.

    Args:
        - datatype (str): The URI of the datatype. example: "https://w3id.org/geomapi#matrix"

    Returns:
        - str or None: The URI of the method if found, otherwise None.
    """
    query = '''
    PREFIX geomapi: <https://w3id.org/geomapi#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?method
    WHERE {
        ?datatype a rdfs:Datatype ;
                  geomapi:method ?method .
        FILTER (?datatype = <%s>)
    }
    ''' % datatype
    result = GEOMAPI_GRAPH.query(query)
    for row in result:
        return str(row.method)
    return None

def apply_method_to_object(datatype: str, obj: object) -> object:
    """
    Dynamically run a function from a GEOMAPI datatype.

    Args:
        - datatype (str): The URI of the datatype.
        - obj (object): The object to apply the method to.

    Returns:
        - object: The result of applying the method to the object.
    """
    method_name = get_method_for_datatype(datatype)
    if not method_name:
        method_name = f"geomapi.utils.literal_to_number"

    # Dynamically import the method
    components = method_name.split('.')
    module_path = '.'.join(components[:-1])
    method = components[-1]    
    mod = importlib.import_module(module_path)
    func = getattr(mod, method)
    
    # Apply the method to the object
    return func(obj)

def get_subject_name(subject:URIRef) -> str:
    """Get the main body of a URIRef graph subject

    Args:
        - subject (URIRef)

    Returns:
        - name (str): the name in the URIRef 
    """
    string=subject.toPython()
    return string.split('/')[-1].split('#')[-1]

def get_subject_graph(graph:Graph, subject:URIRef = None) -> Graph:
    """Returns a subselection of the full Graph that only contains the triples of one subject.

    Args:
        - graph (Graph) 
        - subject (URIRef, optional): If no subject is provided, the first one is picked. Defaults to first subject in the graph.
    
    Returns:
        - Graph with the triples of one subject
    """
    #input validation       
    if(subject is not None and subject not in graph.subjects()): 
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

def get_attribute_from_predicate(graph: Graph, predicate : Literal) -> str:
    """Returns the attribute without the namespace.

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

#### CONVERSIONS ####

def literal_to_number(literal:  "str | Literal") -> float | int:
    """Tries to convert the string to a number

    Args:
        literal (str | Literal): the input literal

    Returns:
        int, float, str: the converted value
    """
    try:
        if '.' in literal:
            return float(literal)
        else:
            return int(literal)
    except ValueError:
        pass

    # If all conversions fail, return the literal as string
    return literal

def literal_to_string(literal:  "str | Literal")->str:
    """Returns string from rdflib.literal.\n

    Args:
        literal (rdflib): literal containing value, if the value is None returns None

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
        raise ValueError('Conversion error')

def literal_to_int(literal: "str | Literal") -> int:
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
        raise ValueError('Conversion error')
    
def literal_to_float(literal:  "str | Literal") -> float:
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

def literal_to_list(literal:  "str | Literal")->list:
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
                    try: 
                        res.append(int(item))
                        continue
                    except: pass
                    try:
                        res.append(float(item))
                        continue
                    except: pass
                    try:
                        res.append(str(item))
                        continue
                    except: pass
                    try:
                        res.append(URIRef(item))
                        continue
                    except:
                        pass
                return res
        return None  
    except:
        raise ValueError

def literal_to_matrix(input: Union[str, "Literal"]) -> np.ndarray:
    """
    Parse a string representation of a matrix into a NumPy array of floats.

    Args:
        - input (str | Literal): A string representing a matrix.

    Returns:
        - np.ndarray: A NumPy array representation of the matrix.

    Raises:
        - ValueError: If the input does not represent a valid matrix.
    """
    try:
        input_string = str(input).replace(',', ' ')

        if input_string.lower() == "none":
            return None

        # Insert newlines between adjacent brackets if missing
        cleaned_string = input_string.replace('][', ']\n[')

        # Clean and normalize the string
        cleaned_string = ' '.join(cleaned_string.strip().split())
        cleaned_string = cleaned_string.replace('[', '').replace(']', '')

        # Convert to NumPy array
        float_array = np.fromstring(cleaned_string, sep=' ')

        # Validate row consistency
        rows = input_string.strip().split(']')
        row_lengths = [len(row.replace('[', '').strip().split()) for row in rows if row.strip()]
        if len(set(row_lengths)) != 1:
            raise ValueError("The rows of the input matrix do not have the same length.")
        
        n_cols = row_lengths[0]

        # Reshape and remove redundant dimensions
        return np.squeeze(float_array.reshape(-1, n_cols))

    except Exception as e:
        raise ValueError(f"Error parsing string to float array: {e}")

def literal_to_datetime(input: Union[str, "Literal"], asStr: bool = True, millies: bool = False) -> datetime.datetime | str:
    """
    Validate and convert various timestamp formats into a standardized datetime.

    Args:
        - input (str | Literal): The input timestamp.
        - asStr (bool, optional): If True, return as a formatted string. Defaults to True.
        - millies (bool, optional): If True, include milliseconds in the output. Defaults to False.

    Returns:
        - datetime.datetime | str: A datetime object or formatted string.

    Raises:
        - ValueError: If the input cannot be parsed into a valid timestamp.
    """

    def return_as(val: datetime.datetime, asStr: bool, millies: bool):
        if asStr:
            fmt = '%Y-%m-%dT%H:%M:%S.%f' if millies else '%Y-%m-%dT%H:%M:%S'
            return val.strftime(fmt)
        return val

    string = str(input)

    # Handle special format: "YYYY:MM:DD HH:MM:SS"
    try:
        dt = datetime.datetime.strptime(string, "%Y:%m:%d %H:%M:%S")
        return return_as(dt, asStr, millies)
    except ValueError:
        pass

    # Handle standard or ISO datetime formats
    try:
        dt = dateutil.parser.parse(string)
        return return_as(dt, asStr, millies)
    except (ValueError, TypeError):
        pass

    # Handle Unix timestamps (float or int)
    try:
        dt = datetime.datetime.fromtimestamp(float(string), tz=datetime.timezone.utc)
        return return_as(dt, asStr, millies)
    except (ValueError, TypeError, OSError):
        pass

    # No valid format found
    raise ValueError(
        "No valid time format found. Expected formats include:\n"
        "  - 'Tue Dec  7 09:38:13 2021'\n"
        "  - '1648468136.033126' (Unix timestamp)\n"
        "  - '2022:03:13 13:55:30'"
    )



def xml_to_float(xml: str) -> float:
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

#### VALIDATION ####

def validate_string(string:str|Path, replacement ='_') -> str:
    """Checks path validity. A string is considered invalid if it cannot be serialized by rdflib or is not Windows subscribable.\n
    If not valid, The function adjusts path naming to be Windows compatible.

    Features (invalid characters):
        "()^[^<>{}[] ~`],|*$ /\:"

    Args:
        path (str): string to check
        replacement (character): characters to replace invalid charters in the string 

    Returns:
        str: cleansed string
    """
    string=str(string)
    prefix=''
    if 'file:///' in string:
        string=string.replace('file:///','')
        prefix='file:///'
    elif 'http://' in string:
        string=string.replace('http://','')
        prefix='http://'
    for idx,element in enumerate(string):
        if element in r"()^[^<>{}[] ~`],|*$ /\:": #
            string=replace_str_index(string,index=idx,replacement=replacement)
    string=prefix+string
    return string

def validate_uri(list:List[URIRef], subject:URIRef) ->bool:
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

def validate_path(path:str)-> bool:
    """Returns True if the given path is an existing folder.

    Args:
        - path (str): path to folder or file

    Returns:
        bool: True if exsists.
    """
    folder=Path(path).parent
    if os.path.isdir(path):
        return True
    elif os.path.exists(folder):
        return True
    else:
        return False





# #### RDF ####

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

def bind_ontologies(graph : Graph=Graph()) -> Graph:
    """Returns a graph that binds in its namespace the ontologies that GEMOMAPI uses and that are not in the rdflib.

     Features (ontologies):
        - @prefix bot: <https://w3id.org/bot#> .
        - @prefix dbp: <http://dbpedia.org/ontology/> .
        - @prefix dcterms: <http://purl.org/dc/terms/> .
        - @prefix dggs: <https://w3id.org/dggs/as> .
        - @prefix fog: <https://w3id.org/fog#> .
        - @prefix geo: <http://www.opengis.net/ont/geosparql#> .
        - @prefix geomapi: <https://w3id.org/geomapi#> .
        - @prefix gom: <https://w3id.org/gom#> .
        - @prefix ifc: <http://standards.buildingsmart.org/IFC/DEV/IFC2x3/TC1/OWL#> .
        - @prefix omg: <https://w3id.org/omg#> .
        - @prefix owl: <http://www.w3.org/2002/07/owl#> .
        - @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        - @prefix vann: <http://purl.org/vocab/vann/> .
        - @prefix voaf: <http://purl.org/vocommons/voaf#> .
        - @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    Returns:
        Graph with updated namespaces 
    """
    # Iterate through all namespaces in the geomapi ontology 
    for prefix, namespace in GEOMAPI_GRAPH.namespaces():
        # Bind each namespace to the new graph
        graph.bind(prefix, Namespace(namespace))
    
    return graph

def get_node_resource_extensions(node: object) -> list:
    """
    Retrieves the resource extensions associated with the given node's class name 
    by executing a SPARQL query on the GEOMAPI_GRAPH.

    Args:
        node (object): The object whose class name is used in the SPARQL query.

    Returns:
        list: A list of extensions (as strings) associated with the node's class.
    """
    query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX dbp: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?extension
        WHERE {{
            ?class rdf:type owl:Class ;
               rdfs:label ?label .
        OPTIONAL {{ ?class dbp:extension ?extension }}  # In case not all have extensions
        FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{node.__class__.__name__}")))
        }}
    """
    # Execute the query on the GEOMAPI_GRAPH
    results = GEOMAPI_GRAPH.query(query)

    # Extract and return the extensions
    return [str(row['extension']) for row in results]

def get_node_type(cls) -> URIRef:
    """Return the type of Node as an rdflib literal. By default, URIRef(Node) is returned.

    Returns:
        URIRef (nodeType)
    """
    query = f"""
        SELECT ?class
        WHERE {{
            ?class a owl:Class .
            FILTER (CONTAINS(STR(?class), "{cls.__class__.__name__}"))
        }}
        """        
    # Perform the query
    result = GEOMAPI_GRAPH.query(query)

    # Extract and return the class if found
    return next((URIRef(row['class']) for row in result), None)

def get_data_type(value) -> "XSD.ENTITY":
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

def get_geomapi_data_types() -> List[URIRef]:
    """
    Get all datatype properties from the GEOMAPI graph.

    Returns:
        - List[URIRef]: A list of datatype properties.
    """
    query = f"""
    SELECT ?datatypeProperty
    WHERE {{
        ?datatypeProperty a owl:DatatypeProperty.
    }}
    """        
    # Perform the query
    result = GEOMAPI_GRAPH.query(query)

    # Extract and return the class if found
    return [row[0] for row in result]


def get_ifcowl_uri(value:str=None) -> URIRef:
    """
    Get the IFCOWL URI corresponding to a class name.

    Args:
        - value (str, optional): A string to match an IFCOWL class. Defaults to None.

    Returns:
        - URIRef: The matched IFCOWL class URI, or IfcBuildingElement by default.
    """
    ifc_owl_classes = list(IFC_GRAPH.subjects(RDFS.subClassOf, IFC_NAMESPACE.IfcBuildingElement))
    
    if value is None:
        return IFC_NAMESPACE.IfcBuildingElement
    
    lower_value = value.lower()
    return next(
        (URIRef(row) for row in ifc_owl_classes if lower_value in row.toPython().lower()),
        IFC_NAMESPACE.IfcBuildingElement
    )
def get_ifcopenshell_class_name(value:URIRef) -> str:
    """
    Get the class name from an IFCOWL URIRef.

    Args:
        - value (URIRef): The IFCOWL URIRef.

    Returns:
        - str: The extracted class name.
    """
    return value.split('#')[-1]

### ONTOLOGY ###

def get_predicate_and_datatype(attribute_name: str):
    """
    Retrieve the URIRef and datatype for a given attribute name from the GEOMAPI ontology.

    Args:
        attribute_name (str): The name of the attribute to search for.

    Returns:
        tuple: A tuple containing the URIRef of the predicate, the URIRef of the datatype, and the namespace prefix.
               If the attribute is not found, returns a default predicate URIRef, XSD.string as datatype, and None for the prefix.
    """
    # Construct a SPARQL query to find the predicate and its rdfs:range
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?predicate ?range
    WHERE {{
        ?predicate rdfs:range ?range .
        FILTER (CONTAINS(STR(?predicate), "{attribute_name}"))
    }}
    """    
    result = GEOMAPI_GRAPH.query(query)

    # Extract and return the predicate and datatype if found
    for row in result:
        predicate = row.predicate
        datatype = row.range
        return URIRef(predicate), URIRef(datatype)
    
    # Return default predicate and None for datatype if no match is found
    return GEOMAPI_NAMESPACE[attribute_name], None

def get_relative_path(self, value: Path):
    """
    Get the relative path of a value with respect to the node graph's folder path.

    Args:
        - value (Path): The absolute path to convert.

    Returns:
        - str: The relative path if possible, otherwise the original path as string.
    """
    if self.graphPath:
        folder_path = Path(self.graphPath).parent
        try:
            return Path(os.path.relpath(value, folder_path)).as_posix()
        except Exception:
            pass
    return value.as_posix()

def rdf_property(predicate: Optional[str] = None, serializer: Optional[Callable] = None, datatype: Optional[str] = None):
    """
    Decorator to mark a property for RDF serialization with an optional custom serializer and datatype.

    Args:
        predicate (Optional[str]): The URI or string representing the RDF predicate. If not provided,
                                    the predicate will be inferred from the property name.
        serializer (Optional[Callable]): A custom function to serialize the property value before adding it to the RDF graph.
                                         If not provided, the default serialization will be used.
        datatype (Optional[str]): The datatype URI to associate with the property. If not provided, the datatype will be inferred
                                  from the property name.

    Returns:
        Callable: The decorated function that marks the property for RDF serialization.
    """
    def decorator(func: Callable) -> Callable:
        # Get default predicate and datatype if not provided
        _pred, _dat = get_predicate_and_datatype(func.__name__)

        # If predicate or datatype is not provided, use the default one
        _predicate = _pred if predicate is None else predicate
        _datatype = _dat if datatype is None else datatype

        # Store the RDF mapping in the RDFMAPPINGS dictionary
        RDFMAPPINGS[func.__name__] = {"predicate": _predicate, "serializer": serializer, "datatype": _datatype}
        return func
    
    return decorator


