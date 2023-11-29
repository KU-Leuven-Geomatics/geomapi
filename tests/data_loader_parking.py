import sys
import os
from rdflib import Graph
from pathlib import Path
import rdflib
import open3d as o3d
import numpy as np
import cv2
import laspy
import pye57 
import time

# import geomapi
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.tools as tl 

class DataLoaderParking:
    def __init__(self):
        # Load your data here
        self._dataLoaded = False
        self.timesLoaded = 0
        
        #ONTOLOGIES
        self.exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns#')
        self.geo=rdflib.Namespace('http://www.opengis.net/ont/geosparql#') 
        self.gom=rdflib.Namespace('https://w3id.org/gom#') 
        self.omg=rdflib.Namespace('https://w3id.org/omg#') 
        self.fog=rdflib.Namespace('https://w3id.org/fog#')
        self.v4d=rdflib.Namespace('https://w3id.org/v4d/core#')
        self.openlabel=rdflib.Namespace('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#')
        self.e57=rdflib.Namespace('http://libe57.org#')
        self.xcr=rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
        self.ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')
        
        #LOAD DATA (only load data once across all tests)
        if not self._dataLoaded:
            self.timesLoaded+=1
            st = time.time()
            print('Creating Parking DataLoader:')
            
            #PATH
            self.path= Path.cwd() / "tests" / "testfiles"  
      
            #RESOURCES
            self.resourceGraphPath=self.path /  'graphs' / 'parking_resource_graph.ttl'
            self.resourceGraph=Graph().parse(str(self.resourceGraphPath))
            print(f'loaded {self.resourceGraphPath}')
            
            #POINTCLOUD
            self.pcdPath=self.path / 'pcd'/"parking.pcd"
            self.pcd=o3d.io.read_point_cloud(str(self.pcdPath))
            print(f'loaded {self.pcd}')

            self.e57Path1=self.path / 'pcd'/"lidar.e57"
            self.xmlPath=self.path / 'pcd'/"lidar.xml"
            e57_1 = pye57.E57( str(self.e57Path1))
            self.e571=e57_1.read_scan_raw(1) 
            print(f'loaded {self.e57Path1}')

            self.e57Path2=self.path / 'pcd'/"parking.e57"
            e57_2 = pye57.E57( str(self.e57Path2))
            self.e572=e57_2.read_scan_raw(0) 
            print(f'loaded {self.e57Path2}')
            
            self.lasPath=self.path / 'pcd'/"parking.las"
            self.las=laspy.read(str(self.lasPath))
            print(f'loaded {self.lasPath}')
            
            self.pcdGraphPath=self.path / 'graphs' /  'pcd_graph.ttl'
            self.pcdGraph=Graph().parse(self.pcdGraphPath)
            print(f'loaded {self.pcdGraphPath}')           

            
            #MESH
            self.meshPath=self.path / 'mesh'/ 'parking.obj'
            self.mesh=o3d.io.read_triangle_mesh(str(self.meshPath))
            print(f'loaded {self.mesh}') 
              
            self.meshGraphPath=self.path / 'graphs' /  'mesh_graph.ttl'
            self.meshGraph=Graph().parse(str(self.meshGraphPath))
            print(f'loaded {self.meshGraphPath}')           


            # #IFC
            # self.ifcPath=self.path / 'ifc' / "parking.ifc"
            # self.bimNodes=tl.ifc_to_nodes_multiprocessing(str(self.ifcPath)) #! Note: this uses geomapi functionality
            # print(f'loaded {len(self.bimNodes)} bimNodes from ifc file')
            
            # self.ifcGraphPath=self.path /  'graphs' / 'parking_ifc_graph.ttl'
            # self.ifcGraph=Graph().parse(str(self.ifcGraphPath))
            # print(f'loaded {self.ifcGraphPath}')           
  
            #IMG
            self.csvPath=self.path / 'img' / 'parking.csv' #! we don't do anything with the csv
            self.imgGraphPath=self.path /  'graphs' / 'road_img_graph.ttl'
            self.imgGraph=Graph().parse(str(self.imgGraphPath))
            print(f'loaded {self.imgGraphPath}')    

            self.imageXmpPath1 = self.path / 'img' / 'DJI_0085.xmp'
            self.imagePath1=self.path / 'img' / "DJI_0085.JPG" 
            self.image1=cv2.imread(str(self.imagePath1))
            self.imageCartesianTransform1= np.array([[-8.13902571e-02,  6.83059476e-01 ,-7.25813597e-01,  5.18276221e+01],
                                                    [ 9.96648497e-01,  4.97790854e-02, -6.49139139e-02 , 6.10007435e+01],
                                                    [-8.20972697e-03, -7.28664391e-01, -6.84821733e-01,  1.50408221e+01],
                                                    [ 0.00000000e+00 , 0.00000000e+00, 0.00000000e+00 , 1.00000000e+00]])
            print(f'loaded {self.imagePath1}')           

            self.imageXmpPath2 = self.path / 'img' / 'IMG_8834.xmp'
            self.imagePath2=self.path / 'img' / "IMG_8834.JPG" 
            self.image2=cv2.imread(str(self.imagePath2))
            self.imageCartesianTransform2= np.array([[ 4.12555151e-01,  4.12058430e-02 ,-9.10000179e-01, 6.68850552e+01],
                                                    [ 9.10841440e-01, -4.52553581e-03,  4.12731621e-01 , 4.52551195e+01],
                                                    [ 1.28887160e-02 ,-9.99140430e-01 ,-3.93990225e-02 , 5.45377093e+00],
                                                    [ 0.00000000e+00 , 0.00000000e+00  ,0.00000000e+00 , 1.00000000e+00]])
            print(f'loaded {self.imagePath2}')    

            #RESOURCES temporary folder
            self.resourcePath= self.path / "resources"
            if not os.path.exists(self.resourcePath):
                os.mkdir(self.resourcePath)
            
            #FILES
            self.files=ut.get_list_of_files(self.path)
            
            DataLoaderParking._dataLoaded = True
            et = time.time()
            print(f'DataLoader succesfully loaded in {et-st} seconds!')
            
if __name__ == '__main__':
    
    DataLoaderParking()
