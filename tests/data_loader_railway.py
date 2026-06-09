import sys
import os
from rdflib import Graph
from pathlib import Path
import rdflib
import open3d as o3d
import numpy as np
import cv2
import ezdxf as cad
import laspy
import time
from rdflib import Graph, URIRef,Namespace, Literal, OWL,RDFS, RDF, XSD
import ezdxf 
from PIL import Image


# import geomapi
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.geometryutils as gmu
import geomapi.utils.cadutils as cadu
import geomapi.utils as ut
import geomapi.tools as tl
from geomapi.utils import GEOMAPI_PREFIXES


class DataLoaderRailway:
    def __init__(self,path=None):
        st = time.time()
        print(f'Creating Railway DataLoader:')
        
           
        #PATH
        self.path= Path.cwd() / "tests" / "testfiles"  if not path else path
        
        #CAD
        self.dxfPath= self.path / 'cad' / "railway.dxf" 
        self.dxf=ezdxf.readfile(self.dxfPath) #this takes 7s to load
        self.entity = next(entity for entity in self.dxf.modelspace().query("LINE"))
        self.line=cadu.ezdxf_entity_to_o3d(self.entity)
        self.cadGraphPath=self.path /  'graphs' / 'cad_graph.ttl'
        self.cadGraph=Graph().parse(self.cadGraphPath)
        self.cadSubject=next((s for s in self.cadGraph.subjects(RDF.type) if 'B02F' in s.toPython()),None )
        print(f'loaded {self.dxf}')
        
        #POINTCLOUD
        self.pcdPath=self.path / 'pcd'/"railway.laz"
        self.laz=laspy.read(str(self.pcdPath))
        self.pcdGraphpath=self.path / 'graphs' /  'pcd_graph.ttl'
        self.pcdGraph=Graph().parse(str(self.pcdGraphpath))
        self.pcdSubject=next((s for s in self.pcdGraph.subjects(RDF.type) if 'railway' in s.toPython()),None )
        print(f'    loaded {self.laz}')
        
        #MESH
        self.meshPath=self.path / 'mesh'/"railway.obj"
        self.mesh=o3d.io.read_triangle_mesh(str(self.meshPath))
        self.meshGraphPath=self.path / 'graphs' /  'mesh_graph.ttl'
        self.meshGraph=Graph().parse(self.meshGraphPath)
        self.meshSubject= next((s for s in self.meshGraph.subjects(RDF.type) if 'railway' in s.toPython()),None )
        print(f'    loaded {self.mesh}')    
        
        # IMG
        self.imgGraphPath=self.path /  'graphs' / 'img_graph.ttl'
        self.imgGraph=Graph().parse(str(self.imgGraphPath))      
        self.imageXmlPath = self.path / 'img' / 'railway.xml'            
        self.imagePath1=self.path / 'img' / "P0024688.jpg" 
        self.imageSubject1= next(s for s in self.imgGraph.subjects(RDF.type) if 'P0024688' in s.toPython() )
        self.image1=Image.open(self.imagePath1)
        self.imageCartesianTransform1= np.array([[ 5.83812227e-02, -9.98234429e-01,  1.09387827e-02, 2.63374319e+05],
                                    [-9.98294230e-01, -5.83833322e-02,  1.26659890e-04,1.51069035e+05],
                                    [ 5.12206323e-04, -1.09275183e-02, -9.99940162e-01,2.82230717e+02],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.00000000e+00]])
        self.imageSubject1=next((s for s in self.imgGraph.subjects(RDF.type) if 'P0024688' in s.toPython() ),None)
        self.focalLength1=21963.0445544689
        self.imageWidth1=11664
        self.imageHeight1=8750
        self.imageCenter = np.array([[self.imageWidth1/2.0, self.imageHeight1/2.0]])
        self.worldCoordinate= np.array([[263377.98, 151064.413 , 256.92,1]])
        self.imgCoordinate= np.array([[1676, 10007]]) #these are (column,row) coordinates
        self.distance=25.98837076368911
        print(f'    loaded {self.imagePath1}')           
        
        # #maybe 1 image is enough
        # self.imagePath2=self.path / 'img' / "P0024691.JPG" 
        # self.image2=Image.open(self.imagePath2)
        # self.imageCartesianTransform2= np.array([[ 8.16701918e-01,  5.76783553e-01,  1.78524640e-02,  1.00585779e+05],
        #                                         [ 5.76947600e-01, -8.16762274e-01, -5.55470424e-03,  1.96265377e+05],
        #                                         [ 1.13773570e-02 , 1.48364739e-02, -9.99825202e-01,  3.19327009e+01],
        #                                         [ 0.00000000e+00, 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
        # # self.imageSubject2=next(s for s in self.imgGraph.subjects(RDF.type) if 'P0024691' in s.toPython() )
        # self.focalLength2=21963.0445544689
        # self.imageWidth2=11664
        # self.imageHeight2=8750
        # self.imageCoordinate1= np.array([[ 9951,7081] ])
        # print(f'    loaded {self.imagePath2}')    
        
        #ORTHO
        # self.orthoPath1=self.path / 'ortho' / "railway_0.01m.jpg"
        # self.orthoTfwPath1=self.path / 'ortho' / "railway_0.01m.tfw"
        # self.ortho1=np.array(Image.open(self.orthoPath1))
        # self.gsd1=0.01
        # self.orthoCenter1= np.array([263395.50100000005, 151080.793 ,256]) #how to we get height?
        # print(f'    loaded {self.orthoPath1}')    
        
        self.orthoPath2=self.path / 'ortho' / "railway-0-0.tif"
        self.orthoTfwPath2=self.path / 'ortho' / "railway-0-0.tfw"
        self.orthoDxfPath2=self.path / 'ortho' / "railway-scheme.dxf"        
        self.ortho2=np.array(Image.open(self.orthoPath2))
        self.gsd2=0.01560589037
        self.orthoCenter2= np.array([263379.5193, 151089.1667,256]) #how to we get height?
        self.orthoCartesianTransform=np.array([[ 1,  0, 0,263379.5193],
                                                [0, -1, 0,151089.1667],
                                                [0,  0, -1,256],
                                                [ 0,  0, 0,1]])
        self.orthoHeight=256
        self.orthoWidth2= 31.961
        self.orthoHeight2= 31.961
        self.orthoGraphPath=self.path /  'graphs' / 'ortho_graph.ttl'
        self.orthoGraph=Graph().parse(str(self.orthoGraphPath))
        self.orthoSubject=next((s for s in self.orthoGraph.subjects(RDF.type) if 'railway-0-0' in s.toPython()),None )
        print(f'    loaded {self.orthoPath2}')    

        #RESOURCES temporary folder
        self.resourcePath= self.path / "resources"
        if not os.path.exists(self.resourcePath):
            os.mkdir(self.resourcePath)
        
        #FILES
        self.files=ut.get_list_of_files(self.path)
        
        et = time.time()
        print(f'DataLoader succesfully loaded in {et-st} seconds!')


def create_line():
    """
    Creates line
    """
    points = np.array([
    [0, 0, 0],  # Start point
    [1, 0, 0],  # End point
    ])

    lines = np.array([
        [0, 1],
    ])

    colors = np.array([
        [1, 0, 0],  # Red color
    ])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)  
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
    
#LOAD DATA -> reference this instance to load data only once
try:
    DATALOADERRAILWAYINSTANCE = DataLoaderRailway()
except:
    print(f'    DATALOADER failed due to path inconsistency. Create your own DATALOADER(path)')
    pass
    

if __name__ == '__main__':
    pass
