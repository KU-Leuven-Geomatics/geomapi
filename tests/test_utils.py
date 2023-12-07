#LIBRARIES
from datetime import datetime
import sys
import os
from pathlib import Path
import time
import shutil
import unittest
from multiprocessing.sharedctypes import Value
import numpy as np
import open3d as o3d
from rdflib import RDF, RDFS, Graph, Literal, URIRef,XSD
import PIL


#GEOMAPI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import geomapi.utils as ut

#DATA
sys.path.append(current_dir)
from data_loader_parking import DATALOADERPARKINGINSTANCE 
from data_loader_road import DATALOADERROADINSTANCE 


class TestUtils(unittest.TestCase):
    
    
    
    
 ################################## SETUP/TEARDOWN CLASS ######################
    @classmethod
    def setUpClass(cls):
        #execute once before all tests
        print('-----------------Setup Class----------------------')
        st = time.time()
        
        cls.dataLoaderParking = DATALOADERPARKINGINSTANCE
        cls.dataLoaderRoad = DATALOADERROADINSTANCE

        #TIME TRACKING 
        et = time.time()
        print("startup time: "+str(et - st))
        print('{:50s} {:5s} '.format('tests','time'))
        print('------------------------------------------------------')

    @classmethod
    def tearDownClass(cls):
        #execute once after all tests
        print('-----------------TearDown Class----------------------')   
        
        
        
        
        
################################## SETUP/TEARDOWN ######################

    def setUp(self):
        #execute before every test
        self.startTime = time.time()   

    def tearDown(self):
        #execute after every test
        t = time.time() - self.startTime
        print('{:50s} {:5s} '.format(self._testMethodName,str(t)))





################################## TEST FUNCTIONS ######################
    def test_bind_ontologies(self):
        graph=Graph()
        graph=ut.bind_ontologies(graph) 
        test=False
        for n in graph.namespaces():
            if 'v4d' in n:
                test=True
        self.assertTrue(test)

    def test_cartesianTransform_to_literal(self):
        literal=ut.cartesianTransform_to_literal(self.dataLoaderParking.imageCartesianTransform1)
        self.assertTrue('-8.13902571e-02' in literal)
       
    def test_check_if_uri_exists(self):
        list=[URIRef('4499de21-f13f-11ec-a70d-c8f75043ce59'),URIRef('http://IMG_2173'),URIRef('http://000_GM_Opening_Rectangular_Opening_Rectangular_1101520'),URIRef('43be9b1c-f13f-11ec-8e65-c8f75043ce59')]
        subject=URIRef('43be9b1c-f13f-11ec-8e65-c8f75043ce59')
        test=ut.check_if_uri_exists(list, subject)
        self.assertTrue(test)

        #incorrect one
        subject=URIRef('blablabla')
        test=ut.check_if_uri_exists(list, subject)
        self.assertFalse(test)
       
    def test_validate_timestamp(self):

        #string
        self.assertEqual(str(ut.validate_timestamp("2022:03:13 13:55:26")),"2022-03-13T13:55:26")

        #string
        self.assertEqual(str(ut.validate_timestamp('Tue Dec  7 09:38:13 2021')),"2021-12-07T09:38:13")

         #string
        self.assertEqual(str(ut.validate_timestamp("1648468136.033126")),"2022-03-28T11:48:56")

        #tuple
        self.assertEqual(str(ut.validate_timestamp(datetime(2022,3,13,13,55,26))),"2022-03-13T13:55:26")

        # invalid
        self.assertRaises(ValueError,ut.validate_timestamp,'qsdfqsdf')

    def test_validate_string(self):
        
        test=ut.validate_string('http://pointcloud2_0')
        self.assertEqual(test,'http://pointcloud2_0')
        test=ut.validate_string('http://po/int/cloud2_ 0:')
        self.assertEqual(test,'http://po_int_cloud2__0_')
        test=ut.validate_string('pointc&:lo ud2_0_')
        self.assertEqual(test,'pointc&_lo_ud2_0_')
        test=ut.validate_string('file:///pointc&:lo ud2_0')
        self.assertEqual(test,'file:///pointc&_lo_ud2_0')
        test=ut.validate_string('file:///pointc&:lo /ud2_0')
        self.assertEqual(test,'file:///pointc&_lo__ud2_0')
        test=ut.validate_string('4499de21-f13f-11ec-a70d-c8f75043ce59')
        self.assertEqual(test,'4499de21-f13f-11ec-a70d-c8f75043ce59')
        test=ut.validate_string('[this<has$to^change]')
        self.assertEqual(test,'_this_has_to_change_')

    def test_clean_attributes_list(self):
        list=['blablabla',
            'graph','graphPath','subject','fullResourcePath','kwargs', 'orientedBoundingBox',
            'ifcElement',
            'mesh',
            'exifData','xmlData','image','features2d','pinholeCamera',
            'pcd','e57Pointcloud','e57xmlNode','e57image','features3d',
            'linkedNodes']
        newList=ut.clean_attributes_list(list)
        self.assertEqual(len(newList),1)
      
    def test_dd_to_dms(self):
        dms=ut.dd_to_dms(6.5)
        self.assertEqual(dms[1],30)

    def test_dms_to_dd(self):
        #west
        dd=ut.dms_to_dd(6, 30, 0, 'W')
        self.assertEqual(dd,-6.5)
        #south
        dd=ut.dms_to_dd(6, 30, 0, 'S')
        self.assertEqual(dd,-6.5)
        #north
        dd=ut.dms_to_dd(6, 30, 0, 'N')
        self.assertEqual(dd,6.5)
      
    def test_filter_exif_gps_data(self):
        #tuple
        dd=ut.parse_exif_gps_data((6, 30, 0), reference= 'N') 
        self.assertEqual(dd,6.5)

        #float
        dd=ut.parse_exif_gps_data(6.5, reference= 'S') 
        self.assertEqual(dd,-6.5)
        
    def test_get_attribute_from_predicate(self):
        graph=Graph()
        graph=ut.bind_ontologies(graph)
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('http://libe57.org#pointCount')) 
        self.assertEqual(string,'pointCount')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('https://w3id.org/v4d/core#faceCount')) 
        self.assertEqual(string,'faceCount')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#timestamp')) 
        self.assertEqual(string,'timestamp')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('http://ifcowl.openbimstandards.org/IFC2X3_Final#className')) 
        self.assertEqual(string,'className')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('http://www.w3.org/2003/12/exif/ns#xResolution')) 
        self.assertEqual(string,'xResolution')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('https://w3id.org/gom#coordinateSystem')) 
        self.assertEqual(string,'coordinateSystem')
      
    def test_get_exif_data(self):
        im = PIL.Image.open(self.dataLoaderRoad.imagePath2) 
        exifData=ut.get_exif_data(im)
        self.assertIsNotNone(exifData["GPSInfo"])
        im.close()
       
    def test_get_extension(self):
        string=ut.get_extension(self.dataLoaderRoad.imagePath1)
        self.assertEqual(string,'.JPG')

    def test_get_filename(self):
        string=ut.get_filename(self.dataLoaderRoad.imagePath1)
        self.assertEqual(string,'101_0367_0007')

    def test_get_folder(self):
        string=ut.get_folder(self.dataLoaderRoad.imagePath2)
        self.assertEqual(string,os.path.join(self.dataLoaderRoad.path,'img'))

    def test_get_folder_path(self):
        string=ut.get_folder_path(self.dataLoaderRoad.imagePath2)
        self.assertEqual(string,os.path.join(self.dataLoaderRoad.path,'img'))

    def test_get_list_of_files(self):
        files=ut.get_list_of_files(self.dataLoaderParking.path)
        self.assertGreater(len(files),1)

    def test_get_paths_in_class(self):
        class tinyClass:
            def __init__(self,**kwargs):
                self.__dict__.update(kwargs)  

        test=tinyClass(path=1,myPath=2,thisisnotapat=3,resolution=4)
        paths=ut.get_paths_in_class(test)
        self.assertEqual(len(paths),2)

    def test_get_subject_graph(self):
        
        subject=URIRef('file:///Basic_Wall_211_WA_Ff1_Glued_brickwork_sandlime_150mm_1095339')

        #subject
        subject=next(s for s in self.dataLoaderParking.resourceGraph.subjects())
        newGraph=ut.get_subject_graph(graph=self.dataLoaderParking.resourceGraph,subject=subject)
        self.assertEqual(len(newGraph),len([t for t in self.dataLoaderParking.resourceGraph.triples((subject,None,None))]))

        #no subject        
        newGraph=ut.get_subject_graph(graph=self.dataLoaderParking.resourceGraph)
        self.assertIsNotNone(next(newGraph.subjects(RDF.type)))

        #wrong subject
        self.assertRaises(ValueError,ut.get_subject_graph,graph=self.dataLoaderParking.imgGraph,subject=URIRef('blabla'))

    def test_get_timestamp(self):
        timeStamp=ut.get_timestamp(self.dataLoaderParking.pcdGraphPath) 
        self.assertEqual(type(timeStamp),str)

    def test_get_variables_in_class(self):
        class tinyClass:
            def __init__(self,**kwargs):
                self.__dict__.update(kwargs)  

        test=tinyClass(path=1,myPath=2,thisisnotapat=3,resolution=4)
        variables=ut.get_variables_in_class(test)
        self.assertEqual(len(variables),4)

    def test_is_float(self):
        test=ut.is_float(0.5) 
        self.assertTrue(test)

        test=ut.is_float('0.5') 
        self.assertTrue(test)

        test=ut.is_float(1) 
        self.assertTrue(test)

        test=ut.is_float('1') 
        self.assertTrue(test)

        test=ut.is_float('qsfdsdf') 
        self.assertFalse(test)

    def test_is_int(self):
        test=ut.is_int(1) 
        self.assertTrue(test)

        test=ut.is_int('1') 
        self.assertTrue(test)

        test=ut.is_int('qsfdsdf') 
        self.assertFalse(test)

        test=ut.is_int(0.5) 
        self.assertTrue(test)
        
    def test_is_string(self):
        test=ut.is_string(1) 
        self.assertTrue(test)

        test=ut.is_string('1') 
        self.assertTrue(test)

        test=ut.is_string('qsfdsdf') 
        self.assertTrue(test)

        test=ut.is_string(0.5) 
        self.assertTrue(test)
            
    def test_is_uriref(self):
        test=ut.is_uriref(Literal(1)) 
        self.assertTrue(test)

        test=ut.is_uriref('1') 
        self.assertTrue(test)

        test=ut.is_uriref('qsfdsdf') 
        self.assertTrue(test)

        test=ut.is_uriref(0.5) 
        self.assertFalse(test)

    def test_item_to_list(self):
        item='qsdf'
        test=ut.item_to_list(item)
        self.assertEqual(len(test),1)
        item=['qsdf']
        test=ut.item_to_list(item)
        self.assertEqual(len(test),1)
        item=['qsdf',1]
        test=ut.item_to_list(item)
        self.assertEqual(len(test),2)

    def test_literal_to_array(self):
        #cartesianBounds
        item=Literal("[-12.33742784 -10.91544131  73.8353109   73.96926636   8.642    9.462     ]")
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,6)

        item=Literal([-12.33742784, -10.91544131,  73.8353109 ,  73.96926636 ,  8.642 ,   9.462     ])
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,6)

        item=Literal(np.array([-12.33742784, -10.91544131,  73.8353109 ,  73.96926636 ,  8.642 ,   9.462     ]))
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,6)

        item=Literal("[-12.33742784, -10.91544131,  73.8353109  , 73.96926636  , 8.642   , 9.462     ]")
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,6)

        item=Literal("None")
        test=ut.literal_to_array(item)
        self.assertEqual(test,None)

        #geospatialTransform
        item=Literal("[6.30  5  0]" )
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,3)
        
        item=Literal("[6.30 , 5 , 0]" )
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,3)

        item=Literal([6.30 , 5 , 0])
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,3)

        item=Literal(np.array([6.30 , 5 , 0]) )
        test=ut.literal_to_array(item)
        self.assertEqual(test.size,3)
        
        item=Literal("[None, None, None]" )
        test=ut.literal_to_array(item)
        self.assertEqual(test,None)

    def test_literal_to_linked_subjects(self):
        string="['file:///Basic_Wall_162_WA_f2_Retaining_concrete_300mm_-_tegen_beschoeiing_904659_0_Z_Q8COz94wZzVDqlx5_s', 'file:///Basic_Wall_162_WA_f2_Retaining_concrete_300mm_-_tegen_beschoeiing_904099_0_Z_Q8COz94wZzVDqlx5c6', 'file:///Basic_Wall_162_WA_f2_Retaining_concrete_300mm_-_tegen_beschoeiing_903697_0_Z_Q8COz94wZzVDqlx5Wq', 'file:///DJI_0085', 'file:///IMG_8834', 'file:///parking', 'file:///parking']"

        list=ut.literal_to_linked_subjects(string)
        gtlist=['file:///Basic_Wall_162_WA_f2_Retaining_concrete_300mm_-_tegen_beschoeiing_904659_0_Z_Q8COz94wZzVDqlx5_s', 'file:///Basic_Wall_162_WA_f2_Retaining_concrete_300mm_-_tegen_beschoeiing_904099_0_Z_Q8COz94wZzVDqlx5c6', 'file:///Basic_Wall_162_WA_f2_Retaining_concrete_300mm_-_tegen_beschoeiing_903697_0_Z_Q8COz94wZzVDqlx5Wq', 'file:///DJI_0085', 'file:///IMG_8834', 'file:///parking', 'file:///parking']

        (self.assertTrue(list[i]==gtlist[i]) for i in range(len(list)) )

    def test_check_if_subject_is_in_graph(self):
        #http
        self.assertTrue(ut.check_if_subject_is_in_graph(self.dataLoaderRoad.imgGraph,next(s for s in self.dataLoaderRoad.imgGraph.subjects(RDF.type))))

        #random
        graph=Graph()
        graph.add((URIRef('mySubject'),RDFS.label,Literal('label')))
        self.assertTrue(ut.check_if_subject_is_in_graph(graph,URIRef('mySubject')))

        #not in graph
        self.assertFalse(ut.check_if_subject_is_in_graph(self.dataLoaderRoad.meshGraph,URIRef('ikjuhygfds')))

    def test_get_graph_subject(self):
        #http
        self.assertIsNotNone(ut.get_graph_subject(self.dataLoaderParking.meshGraph,next(s for s in self.dataLoaderParking.meshGraph.subjects(RDF.type))))
        
        #file
        self.assertIsNotNone(ut.get_graph_subject(self.dataLoaderParking.pcdGraph,next(s for s in self.dataLoaderParking.pcdGraph.subjects(RDF.type))))

        #random
        graph=Graph()
        graph.add((URIRef('mySubject'),RDFS.label,Literal('label')))
        self.assertIsNotNone(ut.get_graph_subject(graph,URIRef('mySubject')))

        #not in graph
        self.assertRaises(ValueError,ut.get_graph_subject,self.dataLoaderParking.imgGraph,URIRef('kjhgfd'))

    def test_get_data_type(self):
        value=1
        dataType=ut.get_data_type(value)
        self.assertAlmostEqual(dataType,XSD.integer)
        
        value=0.1
        dataType=ut.get_data_type(value)
        self.assertAlmostEqual(dataType,XSD.float)

        value=[1,2,3]
        dataType=ut.get_data_type(value)
        self.assertAlmostEqual(dataType,XSD.string)

        value=datetime(1991,5,12,10,10,10)
        dataType=ut.get_data_type(value)
        self.assertAlmostEqual(dataType,XSD.dateTime)
        
        value=np.array([1,2,3])
        dataType=ut.get_data_type(value)
        self.assertAlmostEqual(dataType,XSD.string)

        value=(1,2,3)
        dataType=ut.get_data_type(value)
        self.assertAlmostEqual(dataType,XSD.string)

    def test_literal_to_cartesianTransform(self):
        item=Literal("[[-0.05442451  0.08978218  0.99447329 -8.94782375] [-0.78368672 -0.62101649  0.01317728 11.25314019] [ 0.6187674  -0.77863835  0.10415962  6.54284524] [ 0.          0.          0.          1.        ]]")
        test=ut.literal_to_cartesianTransform(item)
        self.assertEqual(test.size,16)

        item=Literal('[[-0.05442451  0.08978218  0.99447329 -8.94782375]\r\n [-0.78368672 -0.62101649  0.01317728 11.25314019]\r\n [ 0.6187674  -0.77863835  0.10415962  6.54284524]\r\n [ 0.          0.          0.          1.        ]]')
        test=ut.literal_to_cartesianTransform(item)
        self.assertEqual(test.size,16)
        
        item=Literal([-0.05442451,  0.08978218 , 0.99447329 ,-8.94782375,-0.78368672, -0.62101649 , 0.01317728, 11.25314019, 0.6187674 , -0.77863835 , 0.10415962 , 6.54284524, 0.  ,        0.    ,      0.     ,     1.        ])
        test=ut.literal_to_cartesianTransform(item)
        self.assertEqual(test.size,16)

        item=Literal(np.array([[-0.05442451,  0.08978218 , 0.99447329 ,-8.94782375],
                                 [-0.78368672, -0.62101649 , 0.01317728, 11.25314019],
                                  [ 0.6187674 , -0.77863835 , 0.10415962 , 6.54284524] ,
                                  [ 0.  ,        0.    ,      0.     ,     1.        ]]))
        test=ut.literal_to_cartesianTransform(item)
        self.assertEqual(test.size,16)

        item=Literal("[[-0.05442451 , 0.08978218 , 0.99447329 ,-8.94782375] [-0.78368672, -0.62101649,  0.01317728, 11.25314019] [ 0.6187674  ,-0.77863835 , 0.10415962 , 6.54284524] [ 0.  ,        0.    ,      0.       ,   1.        ]]")
        test=ut.literal_to_cartesianTransform(item)
        self.assertEqual(test.size,16)

        item=Literal("None")
        test=ut.literal_to_cartesianTransform(item)
        self.assertIsNone(test)

    def test_literal_to_float(self):
        item=Literal(0.5)
        test=ut.literal_to_float(item)
        self.assertIsInstance(test,float)

        item=Literal(5)
        test=ut.literal_to_float(item)
        self.assertIsInstance(test,float)

        item=Literal('0.5')
        test=ut.literal_to_float(item)
        self.assertIsInstance(test,float)

        item=Literal('blabla')
        self.assertRaises(ValueError,ut.literal_to_float,item)

    def test_literal_to_int(self):
        item=Literal(0.5)
        test=ut.literal_to_int(item)
        self.assertIsInstance(test,int)

        item=Literal(5)
        test=ut.literal_to_int(item)
        self.assertIsInstance(test,int)

        item=Literal('5')
        test=ut.literal_to_int(item)
        self.assertIsInstance(test,int)

        item=Literal('blabla')
        self.assertRaises(ValueError,ut.literal_to_int,item)

    def test_literal_to_list(self):
        item=Literal(0.5)
        test=ut.literal_to_list(item)
        self.assertIsInstance(test,list)

        item=Literal('[-0.126115439984335, 0.0981832072267781, 0.0312044509604729]')
        test=ut.literal_to_list(item)
        self.assertEqual(len(test),3)

        item=Literal('[-0.126115439984335 0.0981832072267781 0.0312044509604729]')
        test=ut.literal_to_list(item)
        self.assertEqual(len(test),3)

        item=Literal('[None None ]')
        test=ut.literal_to_list(item)
        self.assertIsNone(test)

    def test_literal_to_orientedBounds(self):
        item=Literal("[[-0.05442451  0.08978218  0.99447329 ] [-0.78368672 -0.62101649  0.01317728 ] [ 0.6187674  -0.77863835    6.54284524] [ 0.          0.                    1.        ] [-0.05442451   0.99447329 -8.94782375] [-0.78368672  0.01317728 11.25314019] [   -0.77863835  0.10415962  6.54284524] [ 0.                    0.          1.        ]]")
        test=ut.literal_to_orientedBounds(item)
        self.assertEqual(test.size,24)

        item=Literal([-0.05442451,  0.08978218 , 0.99447329 ,-8.94782375,-0.78368672, -0.62101649 , 0.01317728, 11.25314019, 0.6187674 , -0.77863835 , 0.10415962 , 6.54284524, 0.  ,        0.    ,      0.  , -0.77863835 , 0.10415962 , 6.54284524, -0.77863835 , 0.10415962 , 6.54284524, -0.77863835 , 0.10415962 , 6.54284524        ])
        test=ut.literal_to_orientedBounds(item)
        self.assertEqual(test.size,24)

        item=Literal(np.array([[-0.05442451,  0.08978218 , 0.99447329 ],
                                 [-0.78368672, -0.62101649 , 0.01317728],
                                  [ 0.6187674 , -0.77863835 , 0.10415962 ] ,
                                  [ 0.  ,        0.    ,      0.     ,          ],
                                  [-0.05442451,  0.08978218 , 0.99447329 ],
                                 [-0.78368672, -0.62101649 , 0.01317728],
                                  [ 0.6187674 , -0.77863835 , 0.10415962 ] ,
                                  [ 0.  ,        0.    ,      0.     ,          ]]))
        test=ut.literal_to_orientedBounds(item)
        self.assertEqual(test.size,24)

        item=Literal("[[-0.05442451 , 0.08978218 , 0.99447329 ] [-0.78368672, -0.62101649 , 0.01317728 ] [ 0.6187674  ,-0.77863835  ,  6.54284524] [ 0.    ,      0.          ,          1.        ] [-0.05442451  , 0.99447329 ,-8.94782375] [-0.78368672 , 0.01317728, 11.25314019] [   -0.77863835,  0.10415962 , 6.54284524] [ 0.       ,             0.   ,       1.        ]]")
        test=ut.literal_to_orientedBounds(item)
        self.assertEqual(test.size,24)

        item=Literal("None")
        test=ut.literal_to_orientedBounds(item)
        self.assertIsNone(test)

    def test_literal_to_string(self):
        item=Literal(0.5)
        test=ut.literal_to_string(item)
        self.assertIsInstance(test,str)

        item=Literal(5)
        test=ut.literal_to_string(item)
        self.assertIsInstance(test,str)

        item=Literal('blabla')
        test=ut.literal_to_string(item)
        self.assertIsInstance(test,str)

        item=Literal('None')
        test=ut.literal_to_string(item)
        self.assertIsNone(test)

    def test_literal_to_uriref(self):
        item=Literal('blabla')
        test=ut.literal_to_uriref(item).toPython()
        self.assertEqual(test,'blabla')

        item=Literal('None')
        test=ut.literal_to_uriref(item)
        self.assertIsNone(test)

        self.assertRaises(ValueError,ut.literal_to_uriref,5)

        self.assertRaises(ValueError,ut.literal_to_uriref,0.5)

    def test_match_uri(self):
        test=ut.match_uri('timestamp').toPython()
        self.assertEqual(test,'https://www.asam.net/index.php?eID=dumpFile&t=f&f=3876&token=413e8c85031ae64cc35cf42d0768627514868b2f#timestamp')
        test=ut.match_uri('cartesianBounds').toPython()
        self.assertEqual(test,'http://libe57.org#cartesianBounds')
        test=ut.match_uri('coordinateSystem').toPython()
        self.assertEqual(test,'https://w3id.org/gom#coordinateSystem')
        test=ut.match_uri('ifcPath').toPython()
        self.assertEqual(test,'http://ifcowl.openbimstandards.org/IFC2X3_Final#ifcPath')
        test=ut.match_uri('xResolution').toPython()
        self.assertEqual(test,'http://www.w3.org/2003/12/exif/ns#xResolution')
        test=ut.match_uri('focalLength35mm').toPython()
        self.assertEqual(test,'http://www.w3.org/1999/02/22-rdf-syntax-ns#focalLength35mm')
        test=ut.match_uri('qfdsfqsf').toPython()
        self.assertEqual(test,'https://w3id.org/v4d/core#qfdsfqsf')

    def test_parse_dms(self):
        item="[[6 , 30 , 0, N ]]"
        test=ut.parse_dms(item)
        self.assertEqual(test,6.5)

        item="[6  30  0 N ]"
        test=ut.parse_dms(item)
        self.assertEqual(test,6.5)

        item=(6 , 30 , 0 , 'N' )
        test=ut.parse_dms(item)
        self.assertEqual(test,6.5)

        item=np.array([6 , 30 , 0 , 'N'])
        test=ut.parse_dms(item)
        self.assertEqual(test,6.5)
    
    def test_replace_str_index(self):
        item="rrrr"
        test=ut.replace_str_index(item,index=0,replacement='_')
        self.assertEqual(test[0],'_')

    def test_string_to_list(self):
        #cartesianBounds
        item="[-12.33742784 -10.91544131  73.8353109   73.96926636   8.642    9.462     ]"
        test=ut.string_to_list(item)
        self.assertEqual(len(test),6)

        item="[-12.33742784, -10.91544131,  73.8353109  , 73.96926636  , 8.642   , 9.462     ]"
        test=ut.string_to_list(item)
        self.assertEqual(len(test),6)

        item="None"
        test=ut.string_to_list(item)
        self.assertIsNone(test)
        
        item="[None, None, None]" 
        test=ut.string_to_list(item)
        self.assertIsNone(test)

    def test_string_to_rotation_matrix(self):
        #cartesianBounds
        item="[-12.33742784 -10.91544131  73.8353109   73.96926636   8.642    9.462 -12.33742784 -10.91544131  73.8353109    ]"
        test=ut.string_to_rotation_matrix(item)
        self.assertEqual(test.size,9)

        item="[-12.33742784, -10.91544131,  73.8353109  , 73.96926636  , 8.642   , 9.462  ,-12.33742784, -10.91544131,  73.8353109   ]"
        test=ut.string_to_rotation_matrix(item)
        self.assertEqual(test.size,9)

        item="[[-12.33742784, -10.91544131,  73.8353109 ] ,   [ 73.96926636  , 8.642   , 9.462  ] ,  [-12.33742784, -10.91544131,  73.8353109   ]]"
        test=ut.string_to_rotation_matrix(item)
        self.assertEqual(test.size,9)

        item="None"
        self.assertRaises(ValueError,ut.string_to_rotation_matrix,item)
        
        item="[[None, None, None],[None, None, None],[None, None, None]]"     
        self.assertRaises(ValueError,ut.string_to_rotation_matrix,item)

    def test_xcr_to_alt(self):
        item="642069440/10000"
        test=ut.xcr_to_alt(item)
        self.assertEqual(test,64206.9440)

        item="None"
        test=ut.xcr_to_alt(item)
        self.assertIsNone(test)
        
    def test_xcr_to_lat(self):
        item="179.992700159232641N"
        test=ut.xcr_to_lat(item)
        self.assertEqual(test,179.992700159232641)

        item="179.992700159232641S"
        test=ut.xcr_to_lat(item)
        self.assertEqual(test,-179.992700159232641)

        item="None"
        test=ut.xcr_to_lat(item)
        self.assertIsNone(test)

    def test_xcr_to_long(self):
        item="66.587349536158328E"
        test=ut.xcr_to_long(item)
        self.assertEqual(test,66.587349536158328)

        item="66.587349536158328W"
        test=ut.xcr_to_long(item)
        self.assertEqual(test,-66.587349536158328)

        item="None"
        test=ut.xcr_to_long(item)
        self.assertIsNone(test)

if __name__ == '__main__':
    unittest.main()
