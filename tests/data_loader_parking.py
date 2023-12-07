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
import ifcopenshell


# import geomapi
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut
import geomapi.tools as tl 
import geomapi.utils.geometryutils as gmu

        
class DataLoaderParking:
    def __init__(self,path=None):
        st = time.time()
        print(f'Creating Parking DataLoader:')     
          
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
        
        #PATH
        self.path= Path.cwd() / "tests" / "testfiles"  if not path else path
    
        #RESOURCES
        self.resourceGraphPath=self.path /  'graphs' / 'parking_resource_graph.ttl'
        self.resourceGraph=Graph().parse(str(self.resourceGraphPath))
        print(f'    loaded {self.resourceGraphPath}')
        #SESSION
        self.sessionGraphPath=self.path /  'graphs' / 'parking_session_graph.ttl'
        self.sessionGraph=Graph().parse(str(self.sessionGraphPath))
        print(f'    loaded {self.sessionGraphPath}')
        
        #POINTCLOUD
        self.pcdPath=self.path / 'pcd'/"parking.pcd"
        self.pcd=o3d.io.read_point_cloud(str(self.pcdPath))
        print(f'    loaded {self.pcd}')

        self.e57Path1=self.path / 'pcd'/"lidar.e57"
        self.e57XmlPath=self.path / 'pcd'/"lidar.xml"
        self.e571 = pye57.E57( str(self.e57Path1))
        self.e571Data=self.e571.read_scan_raw(1) 
        print(f'    loaded {self.e57Path1}')

        self.e57Path2=self.path / 'pcd'/"parking.e57"
        self.e572 = pye57.E57( str(self.e57Path2))
        self.e572Data=self.e572.read_scan_raw(0) 
        print(f'    loaded {self.e57Path2}')
        
        self.lasPath=self.path / 'pcd'/"parking.las"
        self.las=laspy.read(str(self.lasPath))
        print(f'    loaded {self.lasPath}')
        
        self.pcdGraphPath=self.path / 'graphs' /  'pcd_graph.ttl'
        self.pcdGraph=Graph().parse(self.pcdGraphPath)
        self.pcdSubject=next(s for s in self.pcdGraph.subjects() if 'parking' in s.toPython() )
        print(f'    loaded {self.pcdGraphPath}')           

        #MESH
        self.meshPath=self.path / 'mesh'/ 'parking.obj'
        self.mesh=o3d.io.read_triangle_mesh(str(self.meshPath))
        print(f'    loaded {self.mesh}') 
            
        self.meshGraphPath=self.path / 'graphs' /  'mesh_graph.ttl'
        self.meshGraph=Graph().parse(str(self.meshGraphPath))
        self.meshSubject= next(s for s in self.meshGraph.subjects() if 'parking' in s.toPython() )
        print(f'    loaded {self.meshGraphPath}')           

        # #IFC
        self.ifcPath=self.path / 'ifc' / "parking.ifc"
        # self.bimNodes=tl.ifc_to_nodes_multiprocessing(str(self.ifcPath)) #! Note: this uses geomapi functionality
        # print(f'loaded {len(self.bimNodes)} bimNodes from ifc file')
        
        self.ifcGraphPath=self.path /  'graphs' / 'parking_ifc_graph.ttl'
        self.ifcGraph=Graph().parse(str(self.ifcGraphPath))
        print(f'    loaded {self.ifcGraphPath}')      
        
        self.ifc = ifcopenshell.open(str(self.ifcPath))   
        self.ifcSlab=self.ifc.by_guid('2qZtnImXH6Tgdb58DjNlmF')
        self.ifcWall=self.ifc.by_guid('06v1k9ENv8DhGMCvKUuLQV')
        self.ifcBeam=self.ifc.by_guid('05Is7PfoXBjhBcbRTnzewz' )
        self.ifcColumn=self.ifc.by_guid('23JN72MijBOfF91SkLzf3a')

        self.slabMesh=gmu.ifc_to_mesh(self.ifcSlab)
        self.wallMesh=gmu.ifc_to_mesh(self.ifcWall)
        self.beamMesh=gmu.ifc_to_mesh(self.ifcBeam)
        self.columnMesh=gmu.ifc_to_mesh(self.ifcColumn) 
        self.bimMeshes= [self.slabMesh,
                        self.wallMesh,
                        self.beamMesh,
                        self.columnMesh]
        self.bimBoxes=[mesh.get_oriented_bounding_box() for mesh in [self.slabMesh,
                                                                     self.wallMesh,
                                                                     self.beamMesh,
                                                                     self.columnMesh] if mesh]
        for box in self.bimBoxes:
            box.color = [1, 0, 0]


        #IMG
        self.csvPath=self.path / 'img' / 'parking.csv' #! we don't do anything with the csv
        self.imgGraphPath=self.path /  'graphs' / 'parking_img_graph.ttl'
        self.imgGraph=Graph().parse(str(self.imgGraphPath))
        print(f'    loaded {self.imgGraphPath}')    

        self.imageXmpPath1 = self.path / 'img' / 'DJI_0085.xmp'
        self.imagePath1=self.path / 'img' / "DJI_0085.JPG" 
        self.image1=cv2.imread(str(self.imagePath1))
        self.imageCartesianTransform1= np.array([[-8.13902571e-02,  6.83059476e-01 ,-7.25813597e-01,  5.18276221e+01],
                                                [ 9.96648497e-01,  4.97790854e-02, -6.49139139e-02 , 6.10007435e+01],
                                                [-8.20972697e-03, -7.28664391e-01, -6.84821733e-01,  1.50408221e+01],
                                                [ 0.00000000e+00 , 0.00000000e+00, 0.00000000e+00 , 1.00000000e+00]])
        self.imageSubject1=next(s for s in self.imgGraph.subjects() if 'DJI_0085' in s.toPython() )
        print(f'    loaded {self.imagePath1}')           

        self.imageXmpPath2 = self.path / 'img' / 'IMG_8834.xmp'
        self.imagePath2=self.path / 'img' / "IMG_8834.JPG" 
        self.image2=cv2.imread(str(self.imagePath2))
        self.imageCartesianTransform2= np.array([[ 4.12555151e-01,  4.12058430e-02 ,-9.10000179e-01, 6.68850552e+01],
                                                [ 9.10841440e-01, -4.52553581e-03,  4.12731621e-01 , 4.52551195e+01],
                                                [ 1.28887160e-02 ,-9.99140430e-01 ,-3.93990225e-02 , 5.45377093e+00],
                                                [ 0.00000000e+00 , 0.00000000e+00  ,0.00000000e+00 , 1.00000000e+00]])
        self.imageSubject2=next(s for s in self.imgGraph.subjects() if 'IMG_8834' in s.toPython() )
        print(f'    loaded {self.imagePath2}')    

        #RESOURCES temporary folder
        self.resourcePath= self.path / "resources"
        if not os.path.exists(self.resourcePath):
            os.mkdir(self.resourcePath)
        
        #FILES
        self.files=ut.get_list_of_files(self.path)
        
        et = time.time()
        print(f'DataLoader succesfully loaded in {et-st} seconds!')

#LOAD DATA -> reference this instance to load data only once
try:
    DATALOADERPARKINGINSTANCE = DataLoaderParking()
except:
    print(f'    DATALOADER failed due to path inconsistency. Create your own DATALOADER(path)')
    pass
    
if __name__ == '__main__':
    pass
    
