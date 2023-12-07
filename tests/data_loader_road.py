import sys
import os
from rdflib import Graph
from pathlib import Path
import rdflib
import open3d as o3d
import numpy as np
import cv2
import ezdxf as cad
import pye57 
import time
import ifcopenshell

# import geomapi
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils.geometryutils as gmu
import geomapi.utils as ut
import geomapi.tools as tl 

class DataLoaderRoad:
    def __init__(self,path=None):
        st = time.time()
        print(f'Creating Road DataLoader:')
        
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
        PATH=self.path
        
        # #SESSION
        # self.sessionGraphPath=self.path /  'graphs' / 'road_session_graph.ttl'
        # self.sessionGraph=Graph().parse(str(self.sessionGraphPath))
        # print(f'    loaded {self.sessionGraphPath}')
        
        #CAD
        # self.cadPath= self.path / 'cad' / "road.dxf"
        # self.cad=cad.readfile(self.cadPath)
        # self.cadGraphPath=self.path /  'graphs' / 'road_cad_graph.ttl'
        # self.cadGraph=Graph().parse(self.cadGraphPath)
        # print(f'loaded {self.cad}')

        #IFC
        self.ifcPath=self.path / 'ifc' / "road.ifc"
        self.bimNodes=tl.ifc_to_nodes_multiprocessing(str(self.ifcPath)) #! Note: this uses geomapi functionality
        print(f'    loaded {len(self.bimNodes)} bimNodes from ifc file')
        
        self.ifc = ifcopenshell.open(str(self.ifcPath))   
        self.ifcRoad=self.ifc.by_guid('2PpmXyFnz1_QUVtw0_xe4L')
        self.ifcPipe=self.ifc.by_guid('2sFWN4Spr5fegnupyER9kl')
        self.ifcCollector=self.ifc.by_guid('3H7DyiHwT7hwR69f2l72fp' )
        self.ifcFoundation=self.ifc.by_guid('0jLihvjbj3jO6QBgJ8M5et')

        self.roadMesh=gmu.ifc_to_mesh(self.ifcRoad)
        self.pipeMesh=gmu.ifc_to_mesh(self.ifcPipe)
        self.collectorMesh=gmu.ifc_to_mesh(self.ifcCollector)
        self.foundationMesh=gmu.ifc_to_mesh(self.ifcFoundation)  
        self.bimMeshes= [self.roadMesh,
                        self.pipeMesh,
                        self.collectorMesh,
                        self.foundationMesh]
        self.bimBoxes=[mesh.get_oriented_bounding_box() for mesh in [self.roadMesh,
                                                                     self.pipeMesh,
                                                                     self.collectorMesh,
                                                                     self.foundationMesh] if mesh]
        for box in self.bimBoxes:
            box.color = [1, 0, 0]
   
        self.ifcGraphPath=self.path /  'graphs' / 'road_ifc_graph.ttl'
        self.ifcGraph=Graph().parse(self.ifcGraphPath)
        print(f'    loaded {self.ifcGraphPath}')    

        #POINTCLOUD
        self.pcdPath=self.path / 'pcd'/"road.pcd"
        self.pcd=o3d.io.read_point_cloud(str(self.pcdPath))

        print(f'    loaded {self.pcd}')

        self.e57Path=self.path / 'pcd'/"lidar.e57"
        self.e57 = pye57.E57( str(self.e57Path))
        self.e57Data=self.e57.read_scan_raw(0) 
        print(f'    loaded {self.e57Path}')
        
        self.pcdGraphpath=self.path / 'graphs' /  'pcd_graph.ttl'
        self.pcdGraph=Graph().parse(str(self.pcdGraphpath))
        self.pcdSubject=next(s for s in self.pcdGraph.subjects() if 'road' in s.toPython() )
        print(f'    loaded {self.pcdGraphpath}')
        
        #MESH
        self.meshPath=self.path / 'mesh'/"road.ply"
        self.mesh=o3d.io.read_triangle_mesh(str(self.meshPath))
        print(f'    loaded {self.mesh}')    
            
        self.meshGraphPath=self.path / 'graphs' /  'mesh_graph.ttl'
        self.meshGraph=Graph().parse(self.meshGraphPath)
        self.meshSubject= next(s for s in self.meshGraph.subjects() if 'road' in s.toPython() )
        print(f'    loaded {self.meshGraphPath}')    
        
        #IMG
        self.imgGraphPath=self.path /  'graphs' / 'road_img_graph.ttl'
        self.imgGraph=Graph().parse(str(self.imgGraphPath))
        print(f'    loaded {self.imgGraphPath}')    
        

        self.imageXmlPath = self.path / 'img' / 'road.xml'            
        self.imagePath1=self.path / 'img' / "101_0367_0007.JPG" 
        self.image1=cv2.imread(str(self.imagePath1))
        self.imageCartesianTransform1= np.array([[-7.99965974e-01, -5.98493762e-01 ,-4.31237396e-02,  1.00592066e+05],
                                                [-5.99164887e-01,  8.00618459e-01,  3.39417250e-03 , 1.96282855e+05],
                                                [ 3.24942709e-02 , 2.85534531e-02, -9.99063973e-01,  3.19272496e+01],
                                                [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
        self.imageSubject1=next(s for s in self.imgGraph.subjects() if '101_0367_0007' in s.toPython() )
        self.focalLength1=3693.1569475809993
        self.imageWidth1=5472
        self.imageHeight1=3648
        print(f'    loaded {self.imagePath1}')           
        
        
        self.imagePath2=self.path / 'img' / "101_0367_0055.JPG" 
        self.image2=cv2.imread(str(self.imagePath2))
        self.imageCartesianTransform2= np.array([[ 8.16701918e-01,  5.76783553e-01,  1.78524640e-02,  1.00585779e+05],
                                                [ 5.76947600e-01, -8.16762274e-01, -5.55470424e-03,  1.96265377e+05],
                                                [ 1.13773570e-02 , 1.48364739e-02, -9.99825202e-01,  3.19327009e+01],
                                                [ 0.00000000e+00, 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
        self.imageSubject2=next(s for s in self.imgGraph.subjects() if '101_0367_0055' in s.toPython() )
        self.focalLength2=3693.1569475809993
        self.imageWidth2=5472
        self.imageHeight2=3648
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
    DATALOADERROADINSTANCE = DataLoaderRoad()
except:
    print(f'    DATALOADER failed due to path inconsistency. Create your own DATALOADER(path)')
    pass
    

if __name__ == '__main__':
    pass
