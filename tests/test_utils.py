#LIBRARIES
from datetime import datetime
import sys
import os
import time
import unittest
import pytest
import numpy as np
from rdflib import RDF, RDFS, Graph, Literal, URIRef,XSD


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
        #if os.path.exists(cls.dataLoader.resourcePath):
        #    shutil.rmtree(cls.dataLoader.resourcePath) 
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
    def test_time_function(self):
        self.assertEqual(ut.time_function(max, 5,6), 6)
    
    def test_get_timestamp(self):
        timeStamp=ut.get_timestamp(self.dataLoaderParking.pcdGraphPath) 
        self.assertEqual(type(timeStamp),str)

    def test_random_color(self):
        self.assertLessEqual(np.max(ut.get_random_color()), 1)
        self.assertLessEqual(np.max(ut.get_random_color(255)), 255)

    def test_map_to_2d_array(self):
        # Regular 3d vector
        vector = [1,0,0]
        vector2d = [[1,0,0]]
        vector3d = [[[1,0,0]]]
        np.testing.assert_array_equal(ut.map_to_2d_array(vector),vector2d)
        np.testing.assert_array_equal(ut.map_to_2d_array(np.array(vector)),vector2d)
        np.testing.assert_array_equal(ut.map_to_2d_array(vector2d),vector2d)
        np.testing.assert_array_equal(ut.map_to_2d_array(vector3d),vector3d)

    def test_item_to_list(self):
        listTest = [1,2,3,4]
        array = np.array([1,2,3,4])
        self.assertTrue(isinstance(ut.item_to_list(listTest), list))
        self.assertTrue(isinstance(ut.item_to_list(array), list))
        self.assertTrue(isinstance(ut.item_to_list("test"), list))

    def test_get_geomapi_classes(self):
        classes = ut.get_geomapi_classes()
        self.assertIsNotNone(classes)

    def test_get_method_for_datatype(self):
        self.assertEqual(str(ut.get_method_for_datatype("https://w3id.org/geomapi#matrix")), "geomapi.utils.literal_to_matrix")

    def test_apply_method_to_object(self):
        matrix = np.array(([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]))
        matrixString = """[[1 2 3 4]
        [1 2 3 4]
        [1 2 3 4]
        [1 2 3 4]]"""
        np.testing.assert_array_equal(matrix, ut.apply_method_to_object("https://w3id.org/geomapi#matrix", matrixString))

    def test_split_list(self):
        list  = np.arange(100)

        splitList = ut.split_list(list, 10)
        self.assertEqual(len(splitList), 10)
        splitList = ut.split_list(list, 9) # not cleanly divisable
        self.assertEqual(len(splitList), 9)
        splitList = ut.split_list(list,l = 11)
        self.assertEqual(len(splitList), 10)
        self.assertEqual(len(splitList[0]), 11)
        self.assertEqual(len(splitList[-1]), 1) #check last element

    def test_replace_str_index(self):
        item="rrrr"
        test=ut.replace_str_index(item,index=0,replacement='_')
        self.assertEqual(test,'_rrr')
        test=ut.replace_str_index(item,index=-1,replacement='_')
        self.assertEqual(test,'rrr_')
        with pytest.raises(ValueError):
            ut.replace_str_index(item,index=10,replacement='_')

    def test_get_list_of_files(self):
        files=ut.get_list_of_files(self.dataLoaderParking.path)
        self.assertGreater(len(files),1)

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
        
    def test_literal_to_matrix_complex(self):
        test_cases = [
            {
                "input": """[[ 3.48110207e-01  9.37407536e-01  9.29487057e-03  2.67476305e+01]
                            [-9.37341584e-01  3.48204779e-01 -1.20077869e-02  6.17326932e+01]
                            [-1.44927083e-02 -4.53243552e-03  9.99884703e-01  4.84636987e+00]
                            [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]""",
                "expected": np.array([
                    [0.348110207, 0.937407536, 0.00929487057, 26.7476305],
                    [-0.937341584, 0.348204779, -0.0120077869, 61.7326932],
                    [-0.0144927083, -0.00453243552, 0.999884703, 4.84636987],
                    [0.0, 0.0, 0.0, 1.0]
                ]),
                "should_raise": False
            },
            {
                "input": """[[ 3.48110207e-01  9.37407536e-01  9.29487057e-03  2.67476305e+01]\n
                            [-9.37341584e-01  3.48204779e-01 -1.20077869e-02  6.17326932e+01]\n
                            [-1.44927083e-02 -4.53243552e-03  9.99884703e-01  4.84636987e+00]\n
                            [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]""",
                "expected": np.array([
                    [0.348110207, 0.937407536, 0.00929487057, 26.7476305],
                    [-0.937341584, 0.348204779, -0.0120077869, 61.7326932],
                    [-0.0144927083, -0.00453243552, 0.999884703, 4.84636987],
                    [0.0, 0.0, 0.0, 1.0]
                ]),
                "should_raise": False
            },
            {
                "input": """[[ 3.48110207e-01  9.37407536e-01  9.29487057e-03  2.67476305e+01]\r
                            [-9.37341584e-01  3.48204779e-01 -1.20077869e-02  6.17326932e+01]\r
                            [-1.44927083e-02 -4.53243552e-03  9.99884703e-01  4.84636987e+00]\r
                            [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]""",
                "expected": np.array([
                    [0.348110207, 0.937407536, 0.00929487057, 26.7476305],
                    [-0.937341584, 0.348204779, -0.0120077869, 61.7326932],
                    [-0.0144927083, -0.00453243552, 0.999884703, 4.84636987],
                    [0.0, 0.0, 0.0, 1.0]
                ]),
                "should_raise": False
            },
            {
                "input": """[[1 2 3]
                            [4 5 6]
                            [7 8 9]
                            [10 11 12]]""",
                "expected": np.array(  [[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9],
                                        [10, 11, 12]]),
                "should_raise": False
            },
            {
                "input": """[[1 2 3][4 5 6][7 8 9][10 11 12]]""",
                "expected": np.array(  [[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9],
                                        [10, 11, 12]]),
                "should_raise": False
            },
            {
                "input": """[1  2  3]""",
                "expected": np.array([1, 2, 3]),
                "should_raise": False
            }
        ]

        for i, case in enumerate(test_cases, start=1):
            if case["should_raise"]:
                with self.assertRaises(ValueError):
                    ut.literal_to_matrix(case["input"])
            else:
                result = ut.literal_to_matrix(case["input"])
                np.testing.assert_array_almost_equal(result, case["expected"], err_msg=f"Test case {i} failed")

    def test_literal_to_matrix(self):
        #cartesianBounds
        item=Literal("[-12.33742784 -10.91544131  73.8353109   73.96926636   8.642    9.462     ]")
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,6)

        item=Literal([-12.33742784, -10.91544131,  73.8353109 ,  73.96926636 ,  8.642 ,   9.462     ])
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,6)

        item=Literal(np.array([-12.33742784, -10.91544131,  73.8353109 ,  73.96926636 ,  8.642 ,   9.462     ]))
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,6)

        item=Literal("[-12.33742784, -10.91544131,  73.8353109  , 73.96926636  , 8.642   , 9.462     ]")
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,6)

        item=Literal("None")
        test=ut.literal_to_matrix(item)
        self.assertEqual(test,None)

        #geospatialTransform
        item=Literal("[6.30  5  0]" )
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,3)
        
        item=Literal("[6.30 , 5 , 0]" )
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,3)

        item=Literal([6.30 , 5 , 0])
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,3)

        item=Literal(np.array([6.30 , 5 , 0]) )
        test=ut.literal_to_matrix(item)
        self.assertEqual(test.size,3)
        
        #cartesianTransform
        literal=self.dataLoaderParking.imgGraph.value(subject=next(s for s in self.dataLoaderParking.imgGraph.subjects(RDF.type)),predicate=URIRef('https://w3id.org/geomapi#cartesianTransform'))
        test=ut.literal_to_matrix(literal)
        self.assertEqual(test.shape,(4,4))
        
        #orientedBoundingBox
        literal=self.dataLoaderParking.imgGraph.value(subject=next(s for s in self.dataLoaderParking.imgGraph.subjects(RDF.type)),predicate=URIRef('https://w3id.org/geomapi#orientedBoundingBox'))
        test=ut.literal_to_matrix(literal)
        self.assertEqual(test.size,9)
        
        #convexHull
        literal=self.dataLoaderParking.imgGraph.value(subject=next(s for s in self.dataLoaderParking.imgGraph.subjects(RDF.type)),predicate=URIRef('https://w3id.org/geomapi#convexHull'))
        test=ut.literal_to_matrix(literal)
        self.assertEqual(test.shape,(5,3))

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

        item=Literal("None")
        test=ut.literal_to_float(item)
        self.assertIsNone(test)

        item=Literal('blabla')
        self.assertRaises(ValueError,ut.literal_to_float,item)
    
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

    def test_literal_to_list(self):
        item=Literal(0.5)
        test=ut.literal_to_list(item)
        self.assertIsInstance(test,list)
        self.assertEqual(test[0], 0.5)

        item=Literal('[-0.126115439984335, 0.0981832072267781, 0.0312044509604729]')
        test=ut.literal_to_list(item)
        self.assertEqual(len(test),3)

        item=Literal('[-0.126115439984335,0.0981832072267781,0.0312044509604729]')
        test=ut.literal_to_list(item)
        self.assertEqual(len(test),3)

        item=Literal('[-0.126115439984335 0.0981832072267781 0.0312044509604729]')
        test=ut.literal_to_list(item)
        self.assertEqual(len(test),3)

        item=Literal('[None None ]')
        test=ut.literal_to_list(item)
        self.assertIsNone(test)

        item="[kip ei hond]"
        test=ut.literal_to_list(item)
        self.assertEqual(len(test),3)

        item="[kip, ei, hond]"
        test=ut.literal_to_list(item)
        self.assertEqual(len(test),3)

        item="None"
        test=ut.literal_to_list(item)
        self.assertIsNone(test)
        
        item="[None, kip, 0.5]" 
        test=ut.literal_to_list(item)
        self.assertIsNone(test)

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

    def test_literal_to_number(self):
        item=Literal(0.5)
        test=ut.literal_to_number(item)
        self.assertIsInstance(test,float)

        item=Literal(5)
        test=ut.literal_to_number(item)
        self.assertIsInstance(test,int)

        item=Literal('5')
        test=ut.literal_to_number(item)
        self.assertIsInstance(test,int)

        item=Literal('blabla')
        test=ut.literal_to_number(item)
        self.assertIsInstance(test,str)


    def test_xml_to_float(self):
        item="10"
        test=ut.xml_to_float(item)
        self.assertEqual(test,10.0)

        item="None"
        self.assertRaises(ValueError,ut.xml_to_float,item)

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

#### VALIDATION ####

    def test_check_if_uri_exists(self):
        list=[URIRef('4499de21-f13f-11ec-a70d-c8f75043ce59'),URIRef('http://IMG_2173'),URIRef('http://000_GM_Opening_Rectangular_Opening_Rectangular_1101520'),URIRef('43be9b1c-f13f-11ec-8e65-c8f75043ce59')]
        subject=URIRef('43be9b1c-f13f-11ec-8e65-c8f75043ce59')
        test=ut.validate_uri(list, subject)
        self.assertTrue(test)

        #incorrect one
        subject=URIRef('blablabla')
        test=ut.validate_uri(list, subject)
        self.assertFalse(test)

    def test_get_subject_name(self):
        self.assertEqual(ut.get_subject_name(URIRef("http://images#P0024688")),"P0024688")

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

    def test_literal_to_datetime(self):

        #string
        self.assertEqual(str(ut.literal_to_datetime("2022:03:13 13:55:26")),"2022-03-13T13:55:26")
        #string
        self.assertEqual(str(ut.literal_to_datetime('Tue Dec  7 09:38:13 2021')),"2021-12-07T09:38:13")
        #string
        self.assertEqual(str(ut.literal_to_datetime("1648468136.033126", millies=True)),"2022-03-28T11:48:56.033126")
        #datetime object
        self.assertEqual(str(ut.literal_to_datetime(datetime(2022,3,13,13,55,26))),"2022-03-13T13:55:26")
        # invalid
        self.assertRaises(ValueError,ut.literal_to_datetime,'qsdfqsdf')

    def test_check_if_subject_is_in_graph(self):
        #http
        self.assertTrue(ut.check_if_subject_is_in_graph(self.dataLoaderRoad.imgGraph,next(s for s in self.dataLoaderRoad.imgGraph.subjects(RDF.type))))
        self.assertTrue(ut.check_if_subject_is_in_graph(self.dataLoaderParking.imgGraph,URIRef("images#IMG_8834")))

        #random
        graph=Graph()
        graph.add((URIRef('mySubject'),RDFS.label,Literal('label')))
        self.assertTrue(ut.check_if_subject_is_in_graph(graph,URIRef('mySubject')))

        #not in graph
        self.assertFalse(ut.check_if_subject_is_in_graph(self.dataLoaderRoad.meshGraph,URIRef('ikjuhygfds')))

    def test_get_graph_intersection(self):
        intersectionGraph = ut.get_graph_intersection([self.dataLoaderRoad.imgGraph,self.dataLoaderRoad.imgGraph])
        self.assertEqual(len(list(intersectionGraph.subjects())),len(list(self.dataLoaderRoad.imgGraph.subjects())))

    def test_get_attribute_from_predicate(self):
        graph=Graph()
        graph=ut.bind_ontologies(graph)
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('https://w3id.org/geomapi#cartesianTransform')) 
        self.assertEqual(string,'cartesianTransform')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('http://purl.org/dc/terms/created')) 
        self.assertEqual(string,'created')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('http://standards.buildingsmart.org/IFC/DEV/IFC2x3/TC1/OWL#className')) 
        self.assertEqual(string,'className')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('http://www.w3.org/2003/12/exif/ns#xResolution')) 
        self.assertEqual(string,'xResolution')
        string=ut.get_attribute_from_predicate(graph, predicate =Literal('https://w3id.org/gom#coordinateSystem')) 
        self.assertEqual(string,'coordinateSystem')

    def test_bind_ontologies(self):
        graph=Graph()
        graph=ut.bind_ontologies(graph) 
        test=False
        for n in graph.namespaces():
            if 'geomapi' in n:
                test=True
        self.assertTrue(test)


      
    def test_get_node_resource_extensions(self):
        class ImageNode():
            def __init__(self):
                pass
        node = ImageNode()
        self.assertEqual(ut.get_node_resource_extensions(node), ut.IMG_EXTENSIONS)

    def test_get_node_type(self):
        class ImageNode():
            def __init__(self):
                pass
        node = ImageNode()
        self.assertEqual(ut.get_node_type(node), URIRef("https://w3id.org/geomapi#ImageNode"))

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

    def test_get_geomapi_data_types(self):
        self.assertGreater(len(ut.get_geomapi_data_types()),0)

    def test_get_ifcowl_uri(self):
        self.assertEqual(ut.get_ifcowl_uri(), ut.IFC_NAMESPACE.IfcBuildingElement)

    def test_get_ifcopenshell_class_name(self):
        self.assertEqual(ut.get_ifcopenshell_class_name("ifc#balk"), "balk")

    def test_get_predicate_and_datatype(self):
        self.assertEqual(ut.get_predicate_and_datatype("cartesianTransform"), 
                         (URIRef('https://w3id.org/geomapi#cartesianTransform'),URIRef("https://w3id.org/geomapi#matrix")))

if __name__ == '__main__':
    unittest.main()
