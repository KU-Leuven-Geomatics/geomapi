import json
import sys
import os
import pandas as pd
from rdflib import Graph
from pathlib import Path
import rdflib
import open3d as o3d
import numpy as np
import cv2
import laspy
import pye57 
import time
import ifcopenshell
from rdflib import Graph, URIRef,Namespace, Literal, OWL,RDFS, RDF, XSD


# import geomapi
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.tools as tl 
import geomapi.utils.geometryutils as gmu
from geomapi.utils import GEOMAPI_PREFIXES

# path= os.path.join(parent_dir, 'geomapi','ontology', 'geomapi_ontology.ttl')
# GEOMAPI=Graph().parse(path)
# GEOMAPI_PREFIXES = {prefix: Namespace(namespace) for prefix, namespace in GEOMAPI.namespace_manager.namespaces()}
# GEOMAPI_NAMESPACE = Namespace('https://w3id.org/geomapi#')
        
class DataLoaderIndoorSite:
    def __init__(self,path=None):
        st = time.time()
        print(f'Creating Indoor Site DataLoader:')     
          
        #PATH
        self.path= Path.cwd() / "tests" / "testfiles"  if not path else path
    
        #GRAPH
        self.panoGraphPath=self.path / 'graphs' /  'pano_graph.txt'
        self.panoGraph=Graph().parse(self.panoGraphPath)
        self.panoSubject=next(s for s in self.panoGraph.subjects(RDF.type) if '00000-info' in s.toPython() )
        print(f'    loaded {self.panoGraphPath}')  

        #POINTCLOUD
        self.pcdPath=self.path / 'pano'/ "pano.pcd"
        self.pcd=o3d.io.read_point_cloud(str(self.pcdPath))
        print(f'    loaded {self.pcd}')

        #IMG
        self.imgPath=self.path /  'pano' / '00000-pano.jpg'
        self.image=cv2.imread(str(self.imgPath))
        print(f'    loaded {self.imgPath}')

        #IMG
        self.depthImgPath=self.path /  'pano' / '00000-pano_depth.png'
        self.depthImage=cv2.imread(str(self.depthImgPath))
        print(f'    loaded {self.depthImgPath}')

        #CSV
        self.csvPath = self.path /  'pano' / "pano-poses.csv"
        self.csvFile = open(self.csvPath, mode = 'r')
        self.csvData = list(pd.read_csv(self.csvFile))

        #JSON
        self.jsonPath = self.path /  'pano' / "00000-info.json"
        self.jsonFile = open(self.jsonPath, mode = 'r')
        self.jsonData = json.load(self.jsonFile)

        #RESOURCES temporary folder
        self.resourcePath= self.path / "resources"
        if not os.path.exists(self.resourcePath):
            os.mkdir(self.resourcePath)
        
        #FILES
        self.files=ut.get_list_of_files(self.path)
        
        et = time.time()
        print(f'DataLoader succesfully loaded in {et-st} seconds!')

#LOAD DATA -> reference this instance to load data only once
DATALOADERINDOORSITEINSTANCE = DataLoaderIndoorSite()
try:
    DATALOADERINDOORSITEINSTANCE = DataLoaderIndoorSite()
except:
    print(f'    DATALOADER failed due to path inconsistency. Create your own DATALOADER(path)')
    pass
    
if __name__ == '__main__':
    pass
    
