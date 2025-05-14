
"""
cadutils - a Python library for validating CAD objects.

**NOTE**: these tools solely function properly on dxf files with meter units. Format your files appropriately to avoid errors.
"""

from typing import List, Tuple
import itertools


import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import numpy as np
import open3d as o3d
import math

import ezdxf
from ezdxf.document import Drawing
from ezdxf.groupby import groupby


def ezdxf_to_o3d(dxf: str | Drawing,
                 dtypes:List[str]=['LINE','ARC','CIRCLE','POINT','SPLINE','POLYLINE','LWPOLYLINE','MESH','ELLIPSE','SOLID','3DFACE'],
                 layers:List[str]=None,
                 join_geometries:bool=True,
                 explode_blocks:bool=True) -> Tuple[List[o3d.geometry.Geometry],List[str]]:
    """Convert a set of entities in a dxf file to a set of o3d.geometry.LineSet/TriangleMesh/PointCloud objects with their corresponding layers.

    **NOTE**: be aware of the scale factor!
    
    Args:
        - dxf (str or Drawing): directly provide the dxf path or pre-read the dxf using ezdxf.readfile().
        - dtypes (List[str], optional): list of entity names to query. Defaults to the main types in a dxf file i.e. ['LINE','ARC','CIRCLE','POINT','SPLINE','POLYLINE','LWPOLYLINE','MESH','ELLIPSE','SOLID','3DFACE'].
        - layers (List[str], optional): list of layer names to query. If None, all layers are considered
        - join_geometries (bool, optional): merge geometries of the same type. This optimizes their use. Defaults to True.
        - group_per_layer (bool, optional): If true, create list of lists of geometries per layer. Defaults to True.
        - explode_blocks (bool, optional): deconstruct blocks and add them to the geometries. Defaults to True.

    Returns:
        Tuple[List[o3d.geometry.Geometry],List[str]]: _description_
    """
    #open dxf file if not ezdxf entitiy
    if isinstance(dxf, str):
        print(f'Reading dxf file...')
        dxf = ezdxf.readfile(dxf)
    
    # check units
    if dxf.header['$INSUNITS'] !=6:
        units=ezdxf.units.decode(dxf.header['$INSUNITS'])
        print(f'Warning: dxf has {units} units while meters are expected!') 

    #select layout: Modelspace,Paperspace,Blocklayout
    msp = dxf.modelspace()    

    #gather layers
    layers=[l for l in dxf.layers] if layers is None else [dxf.layers.get(l) for l in ut.item_to_list(layers)]
    allLayers=[l.dxf.name for l in dxf.layers]   
    # assert all(l in dxf.layers.entries for l in layers ), f' some layers are not found, please check spelling.'   
    # layerColors=[]
    # layerColors=[np.array(get_rgb_from_aci(l.dxf.color))/255 for l in dxf.layers] #if layers is None else [np.array(get_rgb_from_aci(dxf.layers.get(l).dxf.color))/255 for l in ut.item_to_list(layers)]
    # for entity in layers:
    #     color = (0, 0, 0)  # Default to black
    #     if entity.dxf.hasattr('color'):  # Check if ACI color is defined
    #         aci = entity.dxf.color # this is not the correct index
    #         color = get_rgb_from_aci(aci)
    #     elif entity.dxf.hasattr('true_color'):  # Check if True Color is defined
    #         true_color = entity.dxf.true_color
    #         color = ((true_color & 0xFF0000) >> 16, (true_color & 0x00FF00) >> 8, true_color & 0x0000FF)
    #     layerColors.append(np.array(color)/255)
    #     # layerColors.append(np.array([c / 255.0 for c in color]))
        
    
    # layerColors=[np.repeat(dxf.layers.entries.get(l).color/256,3) for l in layers]     
    # print(layerColors)
    print(f'{len(layers)} layers found.')  

    #explode blocks into seperate geometries    
    if explode_blocks:
        print(f'Exploding blocks...')        
        [e.explode() for e in msp if e.dxftype()=='INSERT'] # these inserts contain metadata that you can use

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
        if layer in allLayers:
            #watch out for points, as they are formatted as [x1,y1,z1,x2,y2,z2,...]
            geometries=[ezdxf_entity_to_o3d(entity) for entity in groupentities]
            geometries=gmu.join_geometries(geometries)    if join_geometries else geometries
            #find the color of the layer
            color=np.array(get_rgb_from_aci(dxf.layers.get(layer).dxf.color))/255
            if len(geometries)>0:
                for g in geometries:
                    try:
                        g.paint_uniform_color(color)
                    except: 
                        print("geometry type:",type(g) ,"cannot be given colors.")
                        pass
            # [g.paint_uniform_color(layerColors[layers.index(layer)]) for g in geometries if len(geometries)>0 and layer in layers]
            geometries=ut.item_to_list(geometries)
            geometry_groups.append(geometries) if len(geometries)>0 else None
            layer_groups.append(layer) if len(geometries)>0 else None

    print(f'Produced {len(list(itertools.chain(*geometry_groups)))} open3d geometries in {len(layer_groups)} layers.')  
    return geometry_groups,layer_groups

def ezdxf_entity_to_o3d(entity:ezdxf.entities.DXFEntity) -> o3d.geometry.Geometry:
    """
    Process a DXF entity and convert it to an Open3D object.

    Parameters:
        - entity (ezdxf.entities.dxfentity.DXFEntity): The DXF entity to process.

    Returns:
        - the Open3D object

    Supported DXF types and their corresponding functions:
        - LINE: line_to_o3d
        - ARC: arc_to_o3d
        - CIRCLE: circle_to_o3d
        - POINT: point_to_o3d
        - SPLINE: spline_to_o3d
        - POLYLINE: polyline_to_o3d
        - LWPOLYLINE: lwpolyline_to_o3d
        - MESH: mesh_to_o3d
        - ELLIPSE: ellips_to_o3d
        - 3DFACE: solid_to_o3d

    Unsupported DXF types:
        - SOLID
        - 3DSOLID
        - HATCH
    """
    # Mapping DXF types to corresponding functions
    dxf_to_o3d = {
        'LINE': line_to_o3d,
        'ARC': arc_to_o3d,
        'CIRCLE': circle_to_o3d,
        'POINT': point_to_o3d,
        'SPLINE': spline_to_o3d,
        'POLYLINE': polyline_to_o3d,
        'LWPOLYLINE': lwpolyline_to_o3d,
        'MESH': mesh_to_o3d,
        'ELLIPSE': ellips_to_o3d,
        '3DFACE': solid_to_o3d,
        'INSERT': insert_to_o3d
    }

    # Determine the type of the entity and process accordingly
    entity_type = entity.dxftype()
    if entity_type in dxf_to_o3d:
        return dxf_to_o3d[entity_type](entity)
    elif entity_type in ['SOLID', '3DSOLID','HATCH']:
        print(f'{entity_type} not supported')
        

def calculate_angle_between_lines(line1: tuple | list | np.ndarray, line2: tuple | list | np.ndarray) -> float:
    """Calculate the angle between two lines.

    This function takes two lines defined by their endpoints and calculates the angle between them.

    Args:
        line1 (tuple): A tuple containing the endpoints of the first line in the form ((x1, y1), (x2, y2)).
        line2 (tuple): A tuple containing the endpoints of the second line in the form ((x3, y3), (x4, y4)).

    Returns:
        float: The angle in degrees between the two lines.

    Example:
        line1 = ((1, 1), (3, 3))
        line2 = ((1, 1), (2, 0))
        angle = calculate_angle_between_lines(line1, line2)
        print(angle)  # Output: 45.0 degrees

    Note:
        - The function calculates the angle by finding the angles of the lines with respect to the x-axis and computing the absolute difference.
        - The result is the acute angle between the two lines.
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    angle1 = math.atan2(y2 - y1, x2 - x1)
    angle2 = math.atan2(y4 - y3, x4 - x3)
    
    angle_diff = abs(math.degrees(angle1 - angle2)) % 180
    
    return angle_diff

def calculate_perpendicular_distance(line1: tuple | list | np.ndarray, line2: tuple | list | np.ndarray) -> float:
    """Calculate the perpendicular distance between two lines.

    This function takes two lines defined by their endpoints and calculates the minimum perpendicular distance between them.

    Args:
        line1 (tuple): A tuple containing the endpoints of the first line in the form ((x1, y1), (x2, y2)).
        line2 (tuple): A tuple containing the endpoints of the second line in the form ((x3, y3), (x4, y4)).

    Returns:
        float: The minimum perpendicular distance between the two lines.

    Note:
        - The function calculates the perpendicular distance by finding the shortest distance between the endpoints of the lines.
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the slope of the first line
    if x2 - x1 != 0:
        slope1 = (y2 - y1) / (x2 - x1)
    else:
        slope1 = float('inf')

    # Calculate the slope of the second line
    if x4 - x3 != 0:
        slope2 = (y4 - y3) / (x4 - x3)
    else:
        slope2 = float('inf')

    # Check if lines are parallel
    if slope1 == slope2:
        return math.dist(line1[0], line2[0])  # Return the distance between endpoints

    # Calculate the intersection point of the two lines
    if slope1 == float('inf'):
        intersection_x = x1
        intersection_y = slope2 * (x1 - x3) + y3
    elif slope2 == float('inf'):
        intersection_x = x3
        intersection_y = slope1 * (x3 - x1) + y1
    else:
        intersection_x = (y1 - y3 + slope2 * x3 - slope1 * x1) / (slope2 - slope1)
        intersection_y = slope1 * (intersection_x - x1) + y1

    # Calculate the perpendicular distance between the intersection point and both lines
    distance1 = math.dist((intersection_x, intersection_y), line1[0])
    distance2 = math.dist((intersection_x, intersection_y), line2[0])

    return min(distance1, distance2)

def insert_to_o3d(entity:ezdxf.entities.insert.Insert)-> o3d.geometry.LineSet:       
    """Convert ezdxf entity to o3d.geometry.LineSet.

    Args:
        entity (ezdxf.entities.insert.Insert)

    Returns:
        o3d.geometry.LineSet
    """    
    #create a box
    g = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)

    # Create scaling matrix
    scale_matrix = np.diag([ 1/entity.dxf.xscale,  1/entity.dxf.yscale,  1/entity.dxf.zscale, 1.0])

    # Create rotation matrix (assuming rotation around z-axis)
    theta = np.radians( entity.dxf.rotation) #this is in degrees
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Create translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = np.array( entity.dxf.insert)#/entity.dxf.xscale #insertion point

    # Create extrusion matrix
    extrude_vector = np.array( entity.dxf.extrusion)#/entity.dxf.xscale #scale
    extrude_vector /= np.linalg.norm(extrude_vector)
    extrude_matrix = np.eye(4)
    extrude_matrix[:3, 2] = extrude_vector

    # Combine transformations
    transformation_matrix = translation_matrix @ extrude_matrix @ rotation_matrix @ scale_matrix

    # Apply transformation to mesh
    g.transform(transformation_matrix)
    
    return gmu.mesh_get_lineset(g)


def circle_to_o3d(entity:ezdxf.entities.arc.Arc)-> o3d.geometry.LineSet:       
    """Convert ezdxf entity to o3d.geometry.LineSet.

    Args:
        entity (ezdxf.entities.arc.Arc)

    Returns:
        o3d.geometry.LineSet
    """       
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
        
    return line_set

def line_to_o3d(entity:ezdxf.entities.line.Line)-> o3d.geometry.LineSet:       
    """Convert ezdxf entity to o3d.geometry.LineSet.

    Args:
        entity (ezdxf.entities.line.Line)

    Returns:
        o3d.geometry.LineSet
    """   
    line_set = o3d.geometry.LineSet() 
    line_set.points = o3d.utility.Vector3dVector([np.array( entity.dxf.start),np.array( entity.dxf.end)]) 
    line_set.lines =o3d.utility.Vector2iVector(np.array([[0,1]]))  
    line_set.colors= o3d.utility.Vector3dVector(np.array([np.repeat(entity.dxf.color/256,3)]))
    return line_set

def arc_to_o3d(entity:ezdxf.entities.arc.Arc)-> o3d.geometry.LineSet:          
    """Convert ezdxf entity to o3d.geometry.LineSet.

    Args:
        entity (ezdxf.entities.arc.Arc)

    Returns:
        o3d.geometry.LineSet
    """    
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
    return line_set

def ellips_to_o3d(entity:ezdxf.entities.arc.Arc)-> o3d.geometry.LineSet:            
    """Convert ezdxf entity to o3d.geometry.LineSet.

    Args:
        - entity (ezdxf.entities.arc.Arc)

    Returns:
        o3d.geometry.LineSet
    """       
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
    return line_set

def point_to_o3d(entity:ezdxf.entities.point.Point)->  np.ndarray:       
    """Convert ezdxf entity to ndarray.

    Args:
        - entity (ezdxf.entities.point.Point)

    Returns:
        ndarray: points
    """         
    return np.array(entity.dxf.location)

def spline_to_o3d(entity:ezdxf.entities.spline.Spline)-> o3d.geometry.LineSet:  
    """Convert ezdxf entity to o3d.geometry.LineSet.
    
    **NOTE**: A spline is reconstructed as a lineset between the control points. It does NOT represent the actual curve.

    Args:
        - entity (ezdxf.entities.spline.Spline)

    Returns:
        - o3d.geometry.LineSet
    """  
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
    return line_set

def solid_to_o3d(entity:ezdxf.entities.solid.Solid)-> o3d.geometry.LineSet:       
    """Convert ezdxf entities to o3d.geometry.LineSet objects.
    
    **NOTE**: A spline is reconstructed as a lineset between the control points. It does NOT represent the actual curve.
    
    **NOTE**: only 3DFACE entities are supported (3p or 4p meshfaces)

    Args:
        entities (ezdxf.entities.solid.Solid)

    Returns:
        o3d.geometry.LineSet
    """   
    try:
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
        line_set=gmu.mesh_get_lineset(mesh)
    except:
        print(f'Error: {entity.dxf.handle} is not a 3DFACE entity.')        
        
    return line_set

def lwpolyline_to_o3d(entity: ezdxf.entities.lwpolyline.LWPolyline) -> o3d.geometry.LineSet:
    """Convert ezdxf entity to o3d.geometry.LineSet.

    Args:
        - entity (ezdxf.entities.lwpolyline.LWPolyline)

    Returns:
        - o3d.geometry.LineSet
    """
    points = []
    # Get points
    for p in entity.get_points():
        points.append(np.array([p[0], p[1], entity.dxf.elevation]))
    points = np.array(points)

    # Create lines
    start = np.arange(start=0, stop=points.shape[0] - 1)[..., np.newaxis]
    end = np.arange(start=1, stop=points.shape[0])[..., np.newaxis]

    # Check if the LWPolyline is closed and add a line segment to close it
    if entity.is_closed:
        start = np.append(start, [points.shape[0] - 1, 0])[..., np.newaxis]
        end = np.append(end, [0, 1])[..., np.newaxis]

    lines = np.hstack((start, end))

    # Create lineset
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Create a color array
    color = np.array([entity.dxf.color / 256.0, entity.dxf.color / 256.0, entity.dxf.color / 256.0])

    # Set the uniform color
    line_set.paint_uniform_color(color)

    return line_set

        
def polyline_to_o3d(entity:ezdxf.entities.polyline.Polyline)-> o3d.geometry.LineSet:       
    """Convert ezdxf entity to o3d.geometry.LineSet 
    AcDbPolyFaceMesh objects will also be returned as o3d.geometry.LineSet (later we will expand this to o3d.Geometry.TriangleMesh).
    
    **NOTE**: Spline and ARC segments are abstracted as linesegments between the control points. It does NOT represent the actual curve.
    
    Args:
        entity (ezdxf.entities.polyline.Polyline)

    Returns:
        o3d.geometry.LineSet    
    """   
    if entity.get_mode()== 'AcDbPolyFaceMesh': 
        #get color
        color=entity.dxf.color
        #convert to mesh
        entity=ezdxf.render.MeshBuilder.from_polyface (entity ) 
        #get points
        points=np.array([point_to_o3d(p) for p in entity.vertices])
        #get faces
        faces=gmu.split_quad_faces(np.array(entity.faces))
        #construct mesh            
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(points)  
        mesh.triangles=o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.repeat(color/256,3))
        
    elif entity.get_mode()=='AcDb2dPolyline' or entity.get_mode()=='AcDb3dPolyline':
        #get points
        points=np.array([point_to_o3d(p) for p in entity.vertices])

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
            
    elif entity.get_mode()=='Polymesh':
        print('Polymesh transformer not implemented')
        return None

    return line_set
    
def mesh_to_o3d(entity:ezdxf.entities.mesh.Mesh)-> o3d.geometry.TriangleMesh:       
    """NOT IMPLEMENTED
    
    Args:
        entity (ezdxf.entities.mesh.Mesh)

    Returns:
        o3d.geometry.TriangleMesh
    """   
    return None

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
                print(p)
                #keep track of identity of the points
                ilist.extend(np.full((len(p), 1), i))
                jlist.extend(np.full((len(p), 1), j))
                
        # Convert the sampled points to an o3d PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if(lineset.has_colors()):
            color=lineset.colors[0]
            point_cloud.paint_uniform_color(color)
        point_clouds+=point_cloud
        
    #compile identidyarray & point cloud
    indentityArray=np.column_stack((np.array(ilist),np.array(jlist)))

    return point_clouds,indentityArray

def get_linesets_inliers_in_box(linesets:List[o3d.geometry.LineSet],box:o3d.geometry.OrientedBoundingBox,point_cloud:o3d.geometry.PointCloud,identityArray:np.ndarray) -> List[o3d.geometry.LineSet]:
    """Returns the segments of the linesets that have sampled pointcloud points falling within a certain bounding box.
    This function should be used together with:\\
        1. cu.sample_pcd_from_linesets(linesets,step_size=0.1)\\
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

def get_rgb_from_aci(aci:int=7):
    """Return RGB from AutoCAD Color Index (ACI) RGB equivalents https://gohtx.com/acadcolors.php

    Args:
        aci (int): AutoCAD Color Index (ACI)

    Returns:
        (int,int,int): RGB
    """
    aci_to_rgb_map = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (255, 255, 0),
        3: (0, 255, 0),
        4: (0, 255, 255),
        5: (0, 0, 255),
        6: (255, 0, 255),
        7: (255, 255, 255),
        8: (65, 65, 65),
        9: (128, 128, 128),
        10: (255, 0, 0),
        11: (255, 170, 170),
        12: (189, 0, 0),
        13: (189, 126, 126),
        14: (129, 0, 0),
        15: (129, 86, 86),
        16: (104, 0, 0),
        17: (104, 69, 69),
        18: (79, 0, 0),
        19: (79, 53, 53),
        20: (255, 63, 0),
        21: (255, 191, 170),
        22: (189, 46, 0),
        23: (189, 141, 126),
        24: (129, 31, 0),
        25: (129, 96, 86),
        26: (104, 25, 0),
        27: (104, 78, 69),
        28: (79, 19, 0),
        29: (79, 59, 53),
        30: (255, 127, 0),
        31: (255, 212, 170),
        32: (189, 94, 0),
        33: (189, 157, 126),
        34: (129, 64, 0),
        35: (129, 107, 86),
        36: (104, 52, 0),
        37: (104, 86, 69),
        38: (79, 39, 0),
        39: (79, 66, 53),
        40: (255, 191, 0),
        41: (255, 234, 170),
        42: (189, 141, 0),
        43: (189, 173, 126),
        44: (129, 96, 0),
        45: (129, 118, 86),
        46: (104, 78, 0),
        47: (104, 95, 69),
        48: (79, 59, 0),
        49: (79, 73, 53),
        50: (255, 255, 0),
        51: (255, 255, 170),
        52: (189, 189, 0),
        53: (189, 189, 126),
        54: (129, 129, 0),
        55: (129, 129, 86),
        56: (104, 104, 0),
        57: (104, 104, 69),
        58: (79, 79, 0),
        59: (79, 79, 53),
        60: (191, 255, 0),
        61: (234, 255, 170),
        62: (141, 189, 0),
        63: (173, 189, 126),
        64: (96, 129, 0),
        65: (118, 129, 86),
        66: (78, 104, 0),
        67: (95, 104, 69),
        68: (59, 79, 0),
        69: (73, 79, 53),
        70: (127, 255, 0),
        71: (212, 255, 170),
        72: (94, 189, 0),
        73: (157, 189, 126),
        74: (64, 129, 0),
        75: (107, 129, 86),
        76: (52, 104, 0),
        77: (86, 104, 69),
        78: (39, 79, 0),
        79: (66, 79, 53),
        80: (63, 255, 0),
        81: (191, 255, 170),
        82: (46, 189, 0),
        83: (141, 189, 126),
        84: (31, 129, 0),
        85: (96, 129, 86),
        86: (25, 104, 0),
        87: (78, 104, 69),
        88: (19, 79, 0),
        89: (59, 79, 53),
        90: (0, 255, 0),
        91: (170, 255, 170),
        92: (0, 189, 0),
        93: (126, 189, 126),
        94: (0, 129, 0),
        95: (86, 129, 86),
        96: (0, 104, 0),
        97: (69, 104, 69),
        98: (0, 79, 0),
        99: (53, 79, 53),
        100: (0, 255, 63),
        101: (170, 255, 191),
        102: (0, 189, 46),
        103: (126, 189, 141),
        104: (0, 129, 31),
        105: (86, 129, 96),
        106: (0, 104, 25),
        107: (69, 104, 78),
        108: (0, 79, 19),
        109: (53, 79, 59),
        110: (0, 255, 127),
        111: (170, 255, 212),
        112: (0, 189, 94),
        113: (126, 189, 157),
        114: (0, 129, 64),
        115: (86, 129, 107),
        116: (0, 104, 52),
        117: (69, 104, 86),
        118: (0, 79, 39),
        119: (53, 79, 66),
        120: (0, 255, 191),
        121: (170, 255, 234),
        122: (0, 189, 141),
        123: (126, 189, 173),
        124: (0, 129, 96),
        125: (86, 129, 118),
        126: (0, 104, 78),
        127: (69, 104, 95),
        128: (0, 79, 59),
        129: (53, 79, 73),
        130: (0, 255, 255),
        131: (170, 255, 255),
        132: (0, 189, 189),
        133: (126, 189, 189),
        134: (0, 129, 129),
        135: (86, 129, 129),
        136: (0, 104, 104),
        137: (69, 104, 104),
        138: (0, 79, 79),
        139: (53, 79, 79),
        140: (0, 191, 255),
        141: (170, 234, 255),
        142: (0, 141, 189),
        143: (126, 173, 189),
        144: (0, 96, 129),
        145: (86, 118, 129),
        146: (0, 78, 104),
        147: (69, 95, 104),
        148: (0, 59, 79),
        149: (53, 73, 79),
        150: (0, 127, 255),
        151: (170, 212, 255),
        152: (0, 94, 189),
        153: (126, 157, 189),
        154: (0, 64, 129),
        155: (86, 107, 129),
        156: (0, 52, 104),
        157: (69, 86, 104),
        158: (0, 39, 79),
        159: (53, 66, 79),
        160: (0, 63, 255),
        161: (170, 191, 255),
        162: (0, 46, 189),
        163: (126, 141, 189),
        164: (0, 31, 129),
        165: (86, 96, 129),
        166: (0, 25, 104),
        167: (69, 78, 104),
        168: (0, 19, 79),
        169: (53, 59, 79),
        170: (0, 0, 255),
        171: (170, 170, 255),
        172: (0, 0, 189),
        173: (126, 126, 189),
        174: (0, 0, 129),
        175: (86, 86, 129),
        176: (0, 0, 104),
        177: (69, 69, 104),
        178: (0, 0, 79),
        179: (53, 53, 79),
        180: (63, 0, 255),
        181: (191, 170, 255),
        182: (46, 0, 189),
        183: (141, 126, 189),
        184: (31, 0, 129),
        185: (96, 86, 129),
        186: (25, 0, 104),
        187: (78, 69, 104),
        188: (19, 0, 79),
        189: (59, 53, 79),
        190: (127, 0, 255),
        191: (212, 170, 255),
        192: (94, 0, 189),
        193: (157, 126, 189),
        194: (64, 0, 129),
        195: (107, 86, 129),
        196: (52, 0, 104),
        197: (86, 69, 104),
        198: (39, 0, 79),
        199: (66, 53, 79),
        200: (191, 0, 255),
        201: (234, 170, 255),
        202: (141, 0, 189),
        203: (173, 126, 189),
        204: (96, 0, 129),
        205: (118, 86, 129),
        206: (78, 0, 104),
        207: (95, 69, 104),
        208: (59, 0, 79),
        209: (73, 53, 79),
        210: (255, 0, 255),
        211: (255, 170, 255),
        212: (189, 0, 189),
        213: (189, 126, 189),
        214: (129, 0, 129),
        215: (129, 86, 129),
        216: (104, 0, 104),
        217: (104, 69, 104),
        218: (79, 0, 79),
        219: (79, 53, 79),
        220: (255, 0, 191),
        221: (255, 170, 234),
        222: (189, 0, 141),
        223: (189, 126, 173),
        224: (129, 0, 96),
        225: (129, 86, 118),
        226: (104, 0, 78),
        227: (104, 69, 95),
        228: (79, 0, 59),
        229: (79, 53, 73),
        230: (255, 0, 127),
        231: (255, 170, 212),
        232: (189, 0, 94),
        233: (189, 126, 157),
        234: (129, 0, 64),
        235: (129, 86, 107),
        236: (104, 0, 52),
        237: (104, 69, 86),
        238: (79, 0, 39),
        239: (79, 53, 66),
        240: (255, 0, 63),
        241: (255, 170, 191),
        242: (189, 0, 46),
        243: (189, 126, 141),
        244: (129, 0, 31),
        245: (129, 86, 96),
        246: (104, 0, 25),
        247: (104, 69, 78),
        248: (79, 0, 19),
        249: (79, 53, 59),
        250: (51, 51, 51),
        251: (80, 80, 80),
        252: (105, 105, 105),
        253: (130, 130, 130),
        254: (190, 190, 190),
        255: (255, 255, 255)
    }
    return aci_to_rgb_map.get(aci, (0, 0, 0))  # Default to black if ACI is not found


# def ezdxf_entities_to_o3d(entities:List[ezdxf.entities.DXFEntity]) -> Tuple[List[o3d.geometry.Geometry],List[str],List[str]]:
#     """Convert ezdxf entities to a set of o3d.geometry.LineSet/TriangleMesh objects with their corresponding layers.

#     Args:
#         - entities (List[ezdxf.entities.DXFEntity]): a list of ezdxf entities.

#     Returns:
#         Tuple[List[o3d.geometry.Geometry],List[str],List[str]]: geometries, uris, layers
#     """
#     entities=ut.item_to_list(entities)     
#     geometries=[]
#     layers=[]      
#     uris=[]  
#     g,l,u=None,None,None
#     for entity in entities:
#         if entity.dxftype() == 'LINE':
#             g,l,u=lines_to_o3d(entity)            
#         elif entity.dxftype() == 'ARC':
#             g,l,u=arcs_to_o3d(entity)
#         elif entity.dxftype() == 'CIRCLE':
#             g,l,u=circles_to_o3d(entity)
#         elif entity.dxftype() == 'POINT':
#             g,l,u=points_to_o3d(entity)
#         elif entity.dxftype() == 'SPLINE':
#             g,l,u=splines_to_o3d(entity)
#         elif entity.dxftype() == 'POLYLINE':
#             g,l,u=polylines_to_o3d(entity)
#         elif entity.dxftype() == 'LWPOLYLINE':
#             g,l,u=lwpolylines_to_o3d(entity)
#         elif entity.dxftype() == 'MESH':
#             g,l,u=meshes_to_o3d(entity)
#         elif entity.dxftype() == 'ELLIPSE':
#             g,l,u=ellipses_to_o3d(entity)
#         # elif entity.dxftype() == 'HATCH':
#         #     o3d_geometries.append(hatch_to_o3d(entity))
#         elif entity.dxftype() == 'SOLID':
#             print('not supported')
#         elif entity.dxftype() =='3DSOLID':
#             print('not supported')
#         elif entity.dxftype() =='3DFACE':
#             g,l,u=solids_to_o3d(entity)
#         elif isinstance(entity,ezdxf.entities.insert.Insert):
#             g,l,u=insert_to_o3d(entity)
#         else:
#             continue
#         if g and l and u:
#             geometries.append(g[0])
#             layers.append(l[0])
#             uris.append(u[0])
#     geometries=ut.item_to_list(geometries)
#     layers=ut.item_to_list(layers)
#     uris=ut.item_to_list(uris)
#     return geometries,uris,layers

# def ezdxf_entities_sample_point_cloud(entities:List[ezdxf.entities.DXFEntity]) -> Tuple[o3d.geometry.PointCloud,np.ndarray]:
    
    
#     # point_clouds=o3d.geometry.PointCloud()
#     # ilist=[]
#     # jlist=[]
    
#     # #convert 
#     # geometries,_=ezdxf_entities_to_o3d(entities)
    
#     # for entity in entities:
        
        

#     #     # Get line segments from the LineSet
#     #     pointArray=np.asarray(lineset.points)
#     #     points = []

#     #     for j,line in enumerate(np.asarray(lineset.lines)):
#     #         #get start and end
#     #         start_point = pointArray[line[0]]
#     #         end_point = pointArray[line[1]]
#     #         #get direction and length
#     #         direction = end_point - start_point
#     #         length = np.linalg.norm(direction)
#     #         #compute number of points
#     #         num_points = int(length / step_size)
#     #         if num_points > 0:
#     #             step = direction / num_points
#     #             p=[start_point + r * step for r in range(num_points + 1)]
#     #             points.extend(p)
                
#     #             #keep track of identity of the points
#     #             ilist.extend(np.full((len(p), 1), i))
#     #             jlist.extend(np.full((len(p), 1), j))
                
#     #     # Convert the sampled points to an o3d PointCloud
#     #     point_cloud = o3d.geometry.PointCloud()
#     #     point_cloud.points = o3d.utility.Vector3dVector(points)
#     #     color=lineset.colors[0]
#     #     point_cloud.paint_uniform_color(color)
#     #     point_clouds+=point_cloud
        
#     # #compile identidyarray & point cloud
#     # indentityArray=np.column_stack((np.array(ilist),np.array(jlist)))

#     return point_clouds,indentityArray

 
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