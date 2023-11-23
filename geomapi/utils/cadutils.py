
"""
cadutils - a Python library for validating CAD objects.

**NOTE**: these tools solely function properly on dxf files with meter units. Format your files appropriately to avoid errors.
"""

import os
import os.path
from typing import List, Tuple
import itertools


import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from geomapi.nodes import *

import ezdxf
from ezdxf.groupby import groupby


def ezdxf_to_o3d(dxf,
                 dtypes:List[str]=['LINE','ARC','CIRCLE','POINT','SPLINE','POLYLINE','LWPOLYLINE','MESH','ELLIPSE','SOLID','3DFACE'],
                 layers:List[str]=None,
                 join_geometries:bool=True,
                 explode_blocks:bool=True) -> Tuple[List[o3d.geometry.Geometry],List[str]]:
    """Convert a set of entities in a dxf file to a set of o3d.geometry.LineSet/TriangleMesh/PointCloud objects with their corresponding layers.

    Args:
        dxf (str or ezdxf): directly provide the dxf path or preread the dxf using ezdxf.readfile().
        dtypes (List[str], optional): list of entity names to query. Defaults to the main types in a dxf file i.e. ['LINE','ARC','CIRCLE','POINT','SPLINE','POLYLINE','LWPOLYLINE','MESH','ELLIPSE','SOLID','3DFACE'].
        layers (List[str], optional): list of layer names to query. If None, all layers are considered
        join_geometries (bool, optional): merge geometries of the same type. This optimizes their use. Defaults to True.
        group_per_layer (bool, optional): If true, create list of lists of geometries per layer. Defaults to True.
        explode_blocks (bool, optional): deconstruct blocks and add them to the geometries. Defaults to True.

    Returns:
        Tuple[List[o3d.geometry.Geometry],List[str]]: _description_
    """
    #open dxf file if not ezdxf entitiy
    if str(type(dxf)) ==str:
        print(f'Reading dxf file...')
        dxf = ezdxf.readfile(dxf)
    
    # check units
    if dxf.header['$INSUNITS'] !=6:
        units=ezdxf.units.decode(dxf.header['$INSUNITS'])
        print(f'Warning: dxf has {units} units while meters are expected!') 

    #select layout: Modelspace,Paperspace,Blocklayout
    msp = dxf.modelspace()    

    #gather layers
    layers=[key.casefold() for key in dxf.layers.entries.keys()]  if layers is None else [key.casefold() for key in ut.item_to_list(layers)]
    assert all(key.casefold() in dxf.layers.entries.keys() for key in layers ), f' some layers are not found, please check spelling.'        
    print(f'{len(layers)} layers found.')  

    #explode blocks into seperate geometries    
    if explode_blocks:
        print(f'Exploding blocks...')        
        [e.explode() for e in msp if e.dxftype()=='INSERT']

    #gather entities        
    entities=[]
    [entities.extend(list(msp.query(t))) for t in ut.item_to_list(dtypes)]
    print(f'{len(entities)} entities found.')  
    
    #convert entities  
    print(f'Converting entities to open3d geometries...')  
    #get groups
    group = groupby(entities=entities, dxfattrib="layer")        
    geometry_groups=[]    
    layer_groups=[]
    #convert per group
    for layer, groupentities in group.items():
        if layer.casefold() in layers:
            geometries,_=ezdxf_entities_to_o3d(groupentities)
            geometries=gmu.join_geometries(geometries)    if join_geometries else geometries
            geometries=ut.item_to_list(geometries)
            geometry_groups.append(geometries) if len(geometries)>0 else None
            layer_groups.append(layer) if len(geometries)>0 else None

    print(f'Produced {len(list(itertools.chain(*geometry_groups)))} open3d geometries in {len(layer_groups)} layers.')  
    return geometry_groups,layer_groups
    

def ezdxf_entities_sample_point_cloud(entities:List[ezdxf.entities.DXFEntity]) -> Tuple[o3d.geometry.PointCloud,np.ndarray]:
    
    
    # point_clouds=o3d.geometry.PointCloud()
    # ilist=[]
    # jlist=[]
    
    # #convert 
    # geometries,_=ezdxf_entities_to_o3d(entities)
    
    # for entity in entities:
        
        

    #     # Get line segments from the LineSet
    #     pointArray=np.asarray(lineset.points)
    #     points = []

    #     for j,line in enumerate(np.asarray(lineset.lines)):
    #         #get start and end
    #         start_point = pointArray[line[0]]
    #         end_point = pointArray[line[1]]
    #         #get direction and length
    #         direction = end_point - start_point
    #         length = np.linalg.norm(direction)
    #         #compute number of points
    #         num_points = int(length / step_size)
    #         if num_points > 0:
    #             step = direction / num_points
    #             p=[start_point + r * step for r in range(num_points + 1)]
    #             points.extend(p)
                
    #             #keep track of identity of the points
    #             ilist.extend(np.full((len(p), 1), i))
    #             jlist.extend(np.full((len(p), 1), j))
                
    #     # Convert the sampled points to an o3d PointCloud
    #     point_cloud = o3d.geometry.PointCloud()
    #     point_cloud.points = o3d.utility.Vector3dVector(points)
    #     color=lineset.colors[0]
    #     point_cloud.paint_uniform_color(color)
    #     point_clouds+=point_cloud
        
    # #compile identidyarray & point cloud
    # indentityArray=np.column_stack((np.array(ilist),np.array(jlist)))

    return point_clouds,indentityArray

def ezdxf_entities_to_o3d(entities:List[ezdxf.entities.DXFEntity]) -> Tuple[List[o3d.geometry.Geometry],List[str]]:
    """Convert ezdxf entities to a set of o3d.geometry.LineSet/TriangleMesh objects with their corresponding layers.

    Args:
        entities (List[ezdxf.entities.DXFEntity])

    Returns:
        Tuple[List[o3d.geometry.Geometry],List[str]]: geometries,layers
    """
    # match entity.dxftype():
    #     case 'LINE':
    #         o3d_geometries.append(line_to_o3d(entity))
    entities=ut.item_to_list(entities)     
    geometries=[]
    layers=[]        
    for entity in entities:
        if entity.dxftype() == 'LINE':
            g,l=lines_to_o3d(entity)            
        elif entity.dxftype() == 'ARC':
            g,l=arcs_to_o3d(entity)
        elif entity.dxftype() == 'CIRCLE':
            g,l=circles_to_o3d(entity)
        elif entity.dxftype() == 'POINT':
            g,l=points_to_o3d(entity)
        elif entity.dxftype() == 'SPLINE':
            g,l=splines_to_o3d(entity)
        elif entity.dxftype() == 'POLYLINE':
            g,l=polylines_to_o3d(entity)
        elif entity.dxftype() == 'LWPOLYLINE':
            g,l=lwpolylines_to_o3d(entity)
        elif entity.dxftype() == 'MESH':
            g,l=meshes_to_o3d(entity)
        elif entity.dxftype() == 'ELLIPSE':
            g,l=ellipses_to_o3d(entity)
        # elif entity.dxftype() == 'HATCH':
        #     o3d_geometries.append(hatch_to_o3d(entity))
        elif entity.dxftype() == 'SOLID' or entity.dxftype() =='3DFACE' or entity.dxftype() =='3DSOLID':
            g,l=solids_to_o3d(entity)
        else:
            continue
        geometries.append(g[0])
        layers.append(l[0])
    return geometries,layers

 
# def ezdxf_to_o3d_grouped_per_layer(dxf,layers:List[str]=None,explode_blocks:bool=True, join_geometries:bool=True) -> Tuple[List[o3d.geometry.Geometry],List[str]]:
#     """Convert ezdxf entity groups to a set of o3d.geometry.LineSet/TriangleMesh objects with their corresponding layers.

#     Args:
#         dxf (str or ): path to dxf file or the already read dxf
#         layers (List[str],optional): list of layer names to query. If None, all layers are considered
#         explode_blocks(bool,optional): deconstruct blocks and add them to the geometries. Defaults to True.

#     Returns:
#         Tuple[List[o3d.geometry.Geometry],List[str]]: geometries,layers
#     """
#     #open dxf file if not ezdxf entitiy
#     if str(type(dxf)) ==str:
#         print(f'Reading dxf file...')
#         dxf = ezdxf.readfile(dxf)
    
#     # check units
#     if dxf.header['$INSUNITS'] !=6:
#         units=ezdxf.units.decode(dxf.header['$INSUNITS'])
#         print(f'Warning: dxf has {units} units while meters are expected!') 

#     #select layout: Modelspace,Paperspace,Blocklayout
#     msp = dxf.modelspace()    

#     #explode blocks into seperate geometries    
#     if explode_blocks:
#         print(f'Exploding blocks...')        
#         [e.explode() for e in msp if e.dxftype()=='INSERT']
     
#     #gather layers
#     layers=[key.casefold() for key in dxf.layers.entries.keys()] if layers is None else layers
#     print(f'{len(layers)} layers found.')  
     
#     #gather entities   
#     group = groupby(entities=msp, dxfattrib="layer")

#     #convert entities
#     print(f'Converting entities to open3d geometries...')     
#     geometry_groups=[]    
#     counter=0
#     for layer, entities in group.items():
#         if layer.casefold() in layers:
#             counter+=len(entities)
#             geometries,_=ezdxf_entities_to_o3d(entities)
#             geometries=gmu.join_geometries(geometries)    if join_geometries else geometries
#             geometries=ut.item_to_list(geometries)
#             geometry_groups.append(geometries) if len(geometries)>0 else geometry_groups.append(None)
#     print(f'Converted {counter} entities in {len(layers)}.')  
#     return geometry_groups,layers

def lines_to_o3d(entities:List[ezdxf.entities.line.Line])-> Tuple[List[o3d.geometry.LineSet],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.

    Args:
        entities (List[ezdxf.entities.line.Line])

    Returns:
        Tuple[List[o3d.geometry.LineSet],List[str]]: line_sets, layers
    """   
    entities=ut.item_to_list(entities)
    
    geometries=[]
    layers=[]
    for entity in entities:
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector([np.array( entity.dxf.start),np.array( entity.dxf.end)]) 
        line_set.lines =o3d.utility.Vector2iVector(np.array([[0,1]]))  
        line_set.colors= o3d.utility.Vector3dVector(np.array([np.repeat(entity.dxf.color/256,3)]))
        geometries.append(line_set)
        layers.append(entity.dxf.layer)
    return geometries, layers


def circles_to_o3d(entities:List[ezdxf.entities.arc.Arc])-> Tuple[List[o3d.geometry.LineSet],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.

    Args:
        entities (List[ezdxf.entities.arc.Arc])

    Returns:
        Tuple[List[o3d.geometry.LineSet],List[str]]: line_sets, layers
    """
    entities=ut.item_to_list(entities)
    
    geometries=[]
    layers=[]
    for entity in entities:          
        #get points
        points=np.array(list(entity.vertices(angles=np.arange(0,360,10))))
        #get lines
        start=np.arange(start=0,stop=points.shape[0]-1 )[..., np.newaxis]    
        end=np.arange(start=1,stop=points.shape[0] )[..., np.newaxis] 
        lines = np.hstack((start, end))        
        #create lineset
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector(points)  
        line_set.lines = o3d.utility.Vector2iVector(lines)        
        line_set.paint_uniform_color(np.repeat(entity.dxf.color/256,3))
        geometries.append(line_set)
        layers.append(entity.dxf.layer)
        
    return geometries, layers

def arcs_to_o3d(entities:List[ezdxf.entities.arc.Arc])-> Tuple[List[o3d.geometry.LineSet],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.

    Args:
        entities (List[ezdxf.entities.arc.Arc])

    Returns:
        Tuple[List[o3d.geometry.LineSet],List[str]]: line_sets, layers
    """
    entities=ut.item_to_list(entities)
    
    geometries=[]
    layers=[]
    for entity in entities:          
        #get points
        points=np.array(list(entity.vertices(angles=entity.angles(10))))
        #get lines
        start=np.arange(start=0,stop=points.shape[0]-1 )[..., np.newaxis]    
        end=np.arange(start=1,stop=points.shape[0] )[..., np.newaxis] 
        lines = np.hstack((start, end))        
        #create lineset
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector(points)  
        line_set.lines = o3d.utility.Vector2iVector(lines)        
        line_set.paint_uniform_color(np.repeat(entity.dxf.color/256,3))
        geometries.append(line_set)
        layers.append(entity.dxf.layer)
        
    return geometries, layers

def ellipses_to_o3d(entities:List[ezdxf.entities.arc.Arc])-> Tuple[List[o3d.geometry.LineSet],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.

    Args:
        entities (List[ezdxf.entities.arc.Arc])

    Returns:
        Tuple[List[o3d.geometry.LineSet],List[str]]: line_sets, layers
    """
    entities=ut.item_to_list(entities)
    
    geometries=[]
    layers=[]
    for entity in entities:            
        #get points every 10°
        # params=list(entity.params(num=36))
        # points=np.array(list(entity.vertices(params)))        
        points=np.array(list(entity.vertices(params=entity.params(10))))

        
        #get lines
        start=np.arange(start=0,stop=points.shape[0] -1)[..., np.newaxis]    
        end=np.arange(start=1,stop=points.shape[0] )[..., np.newaxis] 
        # end=np.append(end,0)[..., np.newaxis] 
        lines = np.hstack((start, end))
        #create lineset
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector(points)  
        line_set.lines = o3d.utility.Vector2iVector(lines)        
        line_set.paint_uniform_color(np.repeat(entity.dxf.color/256,3))
        geometries.append(line_set)
        layers.append(entity.dxf.layer)
        
    return geometries, layers

def points_to_o3d(entities:List[ezdxf.entities.point.Point])->  Tuple[np.ndarray,List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.

    Args:
        entities (List[ezdxf.entities.point.Point])

    Returns:
       Tuple[np.ndarray,str]: point,entity.dxf.layer
    """   
    entities=ut.item_to_list(entities)
    
    geometries=[]
    layers=[]
    for entity in entities:
        geometries.append(np.array(entity.dxf.location))    
        layers.append(entity.dxf.layer)
    return np.array(geometries),layers


def splines_to_o3d(entities:List[ezdxf.entities.spline.Spline])-> Tuple[List[o3d.geometry.LineSet],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.
    
    **NOTE**: A spline is reconstructed as a lineset between the control points. It does NOT represent the actual curve.

    Args:
        entities (List[ezdxf.entities.spline.Spline])

    Returns:
        Tuple[List[o3d.geometry.LineSet],List[str]]: line_sets, layers
    """   
    entities=ut.item_to_list(entities)
    
    geometries=[]
    layers=[]
    for entity in entities:
        #get points
        points=np.array(entity.control_points)
        #get lines    
        start=np.arange(start=0,stop=points.shape[0]-1 )[..., np.newaxis]    
        end=np.arange(start=1,stop=points.shape[0] )[..., np.newaxis] 
        # if entity.closed:
        #     start=np.append(start,points.shape[0])[..., np.newaxis]         
        #     end=np.append(end,0)[..., np.newaxis]         
        lines = np.hstack((start, end))
        
        #create lineset
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector(points)  
        line_set.lines = o3d.utility.Vector2iVector(lines)        
        line_set.paint_uniform_color(np.repeat(entity.dxf.color/256,3))
        geometries.append(line_set)
        layers.append(entity.dxf.layer)
        
    return geometries, layers

def solids_to_o3d(entities:List[ezdxf.entities.solid.Solid])-> Tuple[List[o3d.geometry.TriangleMesh],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.
    
    **NOTE**: A spline is reconstructed as a lineset between the control points. It does NOT represent the actual curve.

    Args:
        entities (List[ezdxf.entities.spline.Spline])

    Returns:
        Tuple[List[o3d.geometry.LineSet],List[str]]: line_sets, layers
    """   
    entities=ut.item_to_list(entities)
    
    geometries=[]
    layers=[]
    for entity in entities:
        #get points & faces 
        p0=entity.dxf.vtx0
        p1=entity.dxf.vtx1
        p2=entity.dxf.vtx2
        p3=entity.dxf.vtx3
        if p2==p3:
            points=np.array([p0,p1,p2])
            faces=np.array([0,1,2])
        else:
            points=np.array([p0,p1,p2,p3])
            faces=np.array([[0,1,2],[0,2,3]])
            
        #construct mesh            
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(points)  
        mesh.triangles=o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.repeat(entity.dxf.color/256,3))
        geometries.append(mesh)
        layers.append(entity.dxf.layer)
        
    return geometries, layers

def lwpolylines_to_o3d(entities:List[ezdxf.entities.lwpolyline.LWPolyline])-> Tuple[List[o3d.geometry.LineSet],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.
    
    **NOTE**: Spline and ARC segments are abstracted as linesegments between the control points. It does NOT represent the actual curve.
    
    Args:
        entities (List[ezdxf.entities.lwpolyline.LWPolyline])

    Returns:
        Tuple[List[o3d.geometry.LineSet],List[str]]: 03d geometries, layers
    """   
    entities=ut.item_to_list(entities)

    geometries=[]
    layers=[]
    
    for entity in entities:
        points=[]
        #get points
        for p in entity.get_points():
            points.append(np.array([p[0],p[1],entity.dxf.elevation]))
        points=np.array(points)
        #get lines    
        start=np.arange(start=0,stop=points.shape[0]-1 )[..., np.newaxis]    
        end=np.arange(start=1,stop=points.shape[0] )[..., np.newaxis] 
        if entity.dxf.flags==1:
            start=np.append(start,points.shape[0])[..., np.newaxis]         
            end=np.append(end,0)[..., np.newaxis]         
        lines = np.hstack((start, end))            
        #create lineset
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector(points)  
        line_set.lines = o3d.utility.Vector2iVector(lines)        
        line_set.paint_uniform_color(np.repeat(entity.dxf.color/256,3))        
        geometries.append(line_set)
        layers.append(entity.dxf.layer)
        
    return geometries, layers
        
def polylines_to_o3d(entities:List[ezdxf.entities.polyline.Polyline])-> Tuple[List[o3d.geometry.Geometry],List[str]]:       
    """Convert ezdxf entities to o3d.geometry.LineSet and Trianglemesh objects.
    AcDbPolyFaceMesh objects will be returned as o3d.Geometry.TriangleMesh objects.
    
    **NOTE**: Spline and ARC segments are abstracted as linesegments between the control points. It does NOT represent the actual curve.
    
    Args:
        entities (List[ezdxf.entities.polyline.Polyline])

    Returns:
        Tuple[List[o3d.geometry.Geometry],List[str]]: 03d geometries, layers
    """   
    entities=ut.item_to_list(entities)

    geometries=[]
    layers=[]
    
    for entity in entities:
        #get layer
        layers.append(entity.dxf.layer)
        
        if entity.get_mode()== 'AcDbPolyFaceMesh': 
            #get color
            color=entity.dxf.color
            #convert to mesh
            entity=ezdxf.render.MeshBuilder.from_polyface (entity ) 
            #get points
            points=np.array(entity.vertices)
            #get faces
            faces=gmu.split_quad_faces(np.array(entity.faces))
            #construct mesh            
            mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector(points)  
            mesh.triangles=o3d.utility.Vector3iVector(faces)
            mesh.paint_uniform_color(np.repeat(color/256,3))
            geometries.append(mesh)
            
        elif entity.get_mode()=='AcDb2dPolyline' or entity.get_mode()=='AcDb3dPolyline':
            #get points
            points,layers=points_to_o3d(entity.vertices)

            #get lines    
            start=np.arange(start=0,stop=points.shape[0]-1 )[..., np.newaxis]    
            end=np.arange(start=1,stop=points.shape[0] )[..., np.newaxis] 
            # if entity.is_closed:
            #     start=np.append(start,points.shape[0])[..., np.newaxis]         
            #     end=np.append(end,0)[..., np.newaxis]         
            lines = np.hstack((start, end))            
            #create lineset
            line_set = o3d.geometry.LineSet() 
            line_set.points = o3d.utility.Vector3dVector(points)  
            line_set.lines = o3d.utility.Vector2iVector(lines)        
            line_set.paint_uniform_color(np.repeat(entity.dxf.color/256,3))        
            geometries.append(line_set)
            
        elif entity.get_mode()=='Polymesh':
            print('Polymesh transformer not implemented')
            
        else:
            continue
        
    return geometries, layers
    
def meshes_to_o3d(entities:ezdxf.entities.mesh.Mesh)-> Tuple[List[o3d.geometry.Geometry],List[str]]:       
    """NOT IMPLEMENTED
    
    Args:
        entities (List[ezdxf.entities.mesh.Mesh])

    Returns:
        Tuple[List[o3d.geometry.Geometry],List[str]]: 03d geometries, layers
    """   
    geometries=[]
    layers=[]
    return geometries, layers


def sample_pcd_from_linesets(linesets:List[o3d.geometry.LineSet],step_size:float=0.1)-> Tuple[o3d.geometry.PointCloud,np.ndarray]:
    """Sample a point cloud from a set of o3d.geometry.LineSet elements (color is inherited)

    Args:
        linesets (List[o3d.geometry.LineSet]): linesets to sample. 
        step_size(float,optional):spacing between points. Defaults to 0.1m.

    Returns:
        Tuple[List[o3d.geometry.PointCloud],np.ndarray]: point_clouds, identityarray with integers of the origin of the points
    """
    point_clouds=o3d.geometry.PointCloud()
    ilist=[]
    jlist=[]
    
    for i,lineset in enumerate(linesets):

        # Get line segments from the LineSet
        pointArray=np.asarray(lineset.points)
        points = []

        for j,line in enumerate(np.asarray(lineset.lines)):
            #get start and end
            start_point = pointArray[line[0]]
            end_point = pointArray[line[1]]
            #get direction and length
            direction = end_point - start_point
            length = np.linalg.norm(direction)
            #compute number of points
            num_points = int(length / step_size)
            if num_points > 0:
                step = direction / num_points
                p=[start_point + r * step for r in range(num_points + 1)]
                points.extend(p)
                
                #keep track of identity of the points
                ilist.extend(np.full((len(p), 1), i))
                jlist.extend(np.full((len(p), 1), j))
                
        # Convert the sampled points to an o3d PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        color=lineset.colors[0]
        point_cloud.paint_uniform_color(color)
        point_clouds+=point_cloud
        
    #compile identidyarray & point cloud
    indentityArray=np.column_stack((np.array(ilist),np.array(jlist)))

    return point_clouds,indentityArray

# def select_lineset_inliers(linesets,points)->o3d.geometry.LineSet:
#     linesetselection=np.empty((points.shape[0],2))

#     #iterate through linesets
#     for i,p in enumerate(points):
        
#         for linesetidx,lineset in enumerate(sublinesets):
            
#             #iterate through line
#             for lineidx,line in enumerate(lineset.lines):
#                 # # get p0 and p1 in the size of the input points
#                 # p0=np.tile(lineset.points[line[0]], (points.shape[0], 1)) 
#                 # p1=np.tile(lineset.points[line[1]], (points.shape[0], 1))
#                 p0=lineset.points[line[0]]
#                 p1=lineset.points[line[1]]
                
#                 #test if any point is on the line  -> # 0.0 <= dot(p1-p0,p-p0)/|p-p0| <= 1.0
#                 # print(np.sum((p-p0)**2))
#                 dot=np.dot(p1-p0,p-p0)/np.sum((p-p0)**2)
                
#                 if (dot>=0) & (dot <=1 ):
#                     linesetselection[i,0]=linesetidx
#                     linesetselection[i,1]=lineidx
#     return(linesetselection)               
#             # print('line')
#             # dot=np.sum(np.dot(p1-p0,(points-p0).T),axis=1) /np.sum((points-p0)**2,axis=1)
        
# #         # create tuple 
# #         np.where((dot>=0) & (dot <=1 ),
# #                  (linesetidx,lineidx,dot),
# #                  (np.nan,np.nan,dot))
            
        
# # return linesetselection.append((linesetidx,lineidx,distance))
        
# # return True if (dot>=0 or dot <=1 ) else False


# def create_unique_mapping(array:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
#     """Create a unique mapping of an array

#     Args:
#         array (np.ndarray): first column of the array will be used for the sorting.

#     Returns:
#         Tuple[np.ndarray,np.ndarray]: unique_values, mapping (shape of input array)
#     """
#     #get first array #! this is a bit flawed and supposes that every x-value is unique
#     a=array[:,0]
#     unique_values=np.unique(array,axis=0)
    
#     # build dictionary
#     fwd = np.argsort(a)
#     asorted = a[fwd]
#     keys = np.unique(asorted) 
#     lower = np.searchsorted(asorted, keys)
#     higher = np.append(lower[1:], len(asorted))

#     inv = {key: fwd[lower_i:higher_i]
#             for key, lower_i, higher_i
#             in zip(keys, lower, higher)}
    
#     # remap values to 0,1,2,....
#     mapping=np.zeros(array.shape[0])
#     i=0
#     for _,value in inv.items():
#             for v in value:
#                     mapping[v]=i
#             i+=1
    
#     return unique_values,mapping

# def ezdxf_lines_to_open3d_linesets(ezdxf_lines:List[Tuple[ezdxf.math.vector.Vector,ezdxf.math.vector.Vector,int,str,float]]) -> List[o3d.geometry.LineSet]: #
#     """Convert ezdxf lines to open3D linesets. Create 1 lineset per layer.

#     Args:
#         ezdxf_lines (List[Tuple[ezdxf.math.vector.Vector,ezdxf.math.vector.Vector,int,str,float]]): (start, end, color, layer, thickness)

#     Returns:
#         List[o3d.geometry.LineSet]
#     """  
#     # get layers (str)
#     layers=list(set([line[3] for line in ezdxf_lines]))

#     #Convert linesets
#     linesets = []  
#     for layer in layers:

#         # get lines in layer
#         lines=[line for line in ezdxf_lines if line[3]==layer]
        
#         # get start and endpoints       
#         array=np.empty((len(lines)*2,3))
#         for i,line in enumerate(lines): 
#             array[2*i,:]= np.array(line[0])
#             array[2*i+1,:]= np.array(line[1])
        
#         #get mapping    
#         # unique_values,mapping = create_unique_mapping(array) #! this is faulty
#         unique_rows, index_mapping = np.unique(array, axis=0, return_inverse=True)

#         #get points and lines and create lineset
#         line_set = o3d.geometry.LineSet() 
#         lineArray=np.reshape(index_mapping,(len(lines),2))
#         line_set.points = o3d.utility.Vector3dVector(unique_rows)  
#         line_set.lines = o3d.utility.Vector2iVector(lineArray)
        
#         #colorize per color 
#         color=next(line[2] for line in ezdxf_lines if line[3]==layer)/256
#         line_set.paint_uniform_color(np.repeat(color,3))
#         linesets.append(line_set)

#     return linesets

def get_linesets_inliers_in_box(linesets:List[o3d.geometry.LineSet],box:o3d.geometry.OrientedBoundingBox,point_cloud:o3d.geometry.PointCloud,identityArray:np.ndarray) -> List[o3d.geometry.LineSet]:
    """Returns the segments of the linesets that have sampled pointcloud points falling within a certain bounding box.
    This function should be used together with:\\
        1. vt.sample_pcd_from_linesets(linesets,step_size=0.1)\\
        2.vt.create_selection_box_from_image_boundary_points(n,roi,meshNode.resource,z=5) \\

    Args:
        linesets (List[o3d.geometry.LineSet]): linesets from which the segments will be selected
        box (o3d.geometry.OrientedBoundingBox): bounding box that is used to filter the point cloud points
        point_cloud (o3d.geometry.PointCloud): sampled points on the linesets
        identityArray (np.ndarray): array with integers that reflect which point cloud point belongs to which lineset

    Returns:
        List[o3d.geometry.LineSet]: _description_
    """
    assert len(np.asarray(point_cloud.points))==identityArray.shape[0], f'length of point cloud and identityarray are not equal'
        
    #compute point_cloud inliers in box
    idxList=box.get_point_indices_within_bounding_box(point_cloud.points)
    if len(idxList)==0:
        return []    
        
    #retrieve which linesets are visible in the box
    idx=identityArray[idxList]
    unique_rows, _ = np.unique(idx, axis=0, return_inverse=True)
    
    #split lists per lineset -> create split_rows dictionary of mapping
    split_rows = {}
    for row in unique_rows:
        key = row[0]
        if key not in split_rows:
            split_rows[key] = []
        split_rows[key].append(row)
    
    #get linesegments and build new linesets
    sublinesets=[]
    for key,value in split_rows.items():
        #get lineset
        lineset=linesets[key]
        #get linesegments
        linesegments=[linesets[key].lines[v[1]] for v in value]
        #create new linesets -> currently still has all redundant points
        line_set = o3d.geometry.LineSet() 
        line_set.points = o3d.utility.Vector3dVector(lineset.points)  
        line_set.lines = o3d.utility.Vector2iVector(linesegments)
        sublinesets.append(line_set)
    return sublinesets


# def cad_show_lines(entities:List[ezdxf.entities.DXFEntity]):

#     # Extract all line entities from the DXF file
#     lines=[e for e in entities if e.dxftype()== 'LINE']

#     # Plot all the lines using Matplotlib
#     for line in lines:
#         x1, y1, _ = line.dxf.start
#         x2, y2, _ = line.dxf.end
#         plt.plot([x1, x2], [y1, y2])

#     plt.show()


# def create_selection_box_from_image_boundary_points(n:ImageNode,roi:Tuple[int,int,int,int],mesh:o3d.geometry.TriangleMesh,z:float=5)->o3d.geometry.OrientedBoundingBox:
#     """Create a selection box from an ImageNode, a region of interest (roi) and a mesh to raycast.
#     A o3d.geometry.OrientedBoundingBox will be created on the location of the intersection of the rays with the mesh.
#     The height of the box is determined by the offset of z in both positive and negative Z-direction

#     Args:
#         n (ImageNode): Imagenode used for the raycasting (internal and external camera paramters)
#         roi (Tuple[int,int,int,int]): region of interest (rowMin,rowMax,columnMin,columnMax)
#         mesh (o3d.geometry.TriangleMesh): mesh used for the raycasting
#         z (float, optional): offset in height of the bounding box. Defaults to [-5m:5m].

#     Returns:
#         o3d.geometry.OrientedBoundingBox or None (if not all rays hit the mesh)
#     """
#     box=None
    
#     #create rays for boundaries
#     uvCoordinates=np.array([[roi[0],roi[2]], # top left
#                             [roi[0],roi[3]], # top right
#                             [roi[1],roi[2]], # bottom left
#                             [roi[1],roi[3]] # bottom right
#                             ])
#     # transform uvcoordinates  to world coordinates to rays   
#     rays=n.create_rays(uvCoordinates)
    
#     # cast rays to 3D mesh 
#     distances,_=gmu.compute_raycasting_collisions(mesh,rays)
    
#     if all(np.isnan(distances)==False): #if all rays hit
#         #compute endpoints 
#         _,endpoints=gmu.rays_to_points(rays,distances)
        
#         #create box of projected points
#         points=np.vstack((gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z],[0,0,0,1]])),
#                         gmu.transform_points(endpoints,transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-z],[0,0,0,1]]))))
#         box=o3d.geometry.OrientedBoundingBox.create_from_points(o3d.cpu.pybind.utility.Vector3dVector(points))
#         box.color=[1,0,0]     
#     return box 


# def dxf_extract_lines(dxf_path:str) -> List[Tuple[ezdxf.math.vector.Vector,ezdxf.math.vector.Vector,int,str,float]]:
#     """Import a DXF and extract all line assets in the modelspace.  
    
#     Args:
#         dxf_path (str): path to dxf

#     Returns:
#         List[Tuple[ezdxf.math.vector.Vector,ezdxf.math.vector.Vector,int,str,float]]: (start, end, color, layer, thickness)

#     """
#     #open dxf file
#     dwg = ezdxf.readfile(dxf_path)
    
#     #select layout: Modelspace,Paperspace,Blocklayout
#     modelspace = dwg.modelspace()
    
#     #parse lines
    
    
#     lines = []
#     for entity in modelspace:
#         if entity.dxftype() == 'LINE': #! make bigger linesets
      
#             start = entity.dxf.start
#             end = entity.dxf.end
#             color = entity.dxf.color
#             layer = entity.dxf.layer
#             thickness = entity.dxf.lineweight
#             lines.append((start, end, color, layer, thickness))
    
#     return lines