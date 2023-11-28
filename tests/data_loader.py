from rdflib import Graph
import sys
import os
import geomapi.utils as ut
from pathlib import Path
import rdflib

class DataLoader:
    def __init__(self):
        # Load your data here
        self._data_loaded = False
        self.times_loaded=0
        
        #ONTOLOGIES
        self.exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns#')
        self.geo=rdflib.Namespace('http://www.opengis.net/ont/geosparql#') #coordinate system information
        self.gom=rdflib.Namespace('https://w3id.org/gom#') # geometry representations => this is from mathias
        self.omg=rdflib.Namespace('https://w3id.org/omg#') # geometry relations
        self.fog=rdflib.Namespace('https://w3id.org/fog#')
        self.v4d=rdflib.Namespace('https://w3id.org/v4d/core#')
        self.openlabel=rdflib.Namespace('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#')
        self.e57=rdflib.Namespace('http://libe57.org#')
        self.xcr=rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
        self.ifc=rdflib.Namespace('http://ifcowl.openbimstandards.org/IFC2X3_Final#')
        
        #LOAD DATA (only load data once across all tests)
        if not self._data_loaded:
            self.times_loaded+=1
            
            #PATH
            self.path= Path.cwd() / "tests" / "testfiles"  
            
            #GRAPH 1
            self.graphPath1=self.path / 'graphs' / 'parking_ifc_graph.ttl'
            self.graph1=Graph().parse(self.graphPath1)

            #GRAPH 2
            self.graphPath2=self.path /  'graphs' / 'resource_graph.ttl'
            self.graph2=Graph().parse(self.graphPath2)

            #GRAPH 3
            self.graphPath3=self.path / 'graphs' /  'pcd_graph.ttl'
            self.graph3=Graph().parse(self.graphPath3)

            #GRAPH 4
            self.graphPath4=self.path / 'graphs' /  'mesh_graph.ttl'
            self.graph4=Graph().parse(self.graphPath4)
                    
            #POINTCLOUD
            self.pcdPath=self.path / 'pcd'/"parking.pcd"
            self.e57Path=self.path / 'pcd'/"parking.e57"
            
            #MESH
            self.meshPath=self.path / 'mesh'/"parking.obj"
        
            #IMG
            self.image1Path=self.path / "img" / "IMG_8834.JPG"  
            
            #RESOURCES temporary folder
            self.resourcePath= self.path / "resources"
            if not os.path.exists(self.resourcePath):
                os.mkdir(self.resourcePath)
            
            #FILES
            self.files=ut.get_list_of_files(self.path)
            
            DataLoader._data_loaded = True
            
if __name__ == '__main__':
    DataLoader()