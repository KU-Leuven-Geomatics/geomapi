
"""Tools to asses the progress on (road) construction sites."""

import numpy as np
import open3d as o3d
import geomapi.utils as ut
import geomapi.utils.geometryutils as gmu
import geomapi.utils.imageutils as iu
from typing import List,Tuple
from colour import Color
from geomapi.nodes import ImageNode
import copy 
import pandas as pd

import pyvista as pv
from vtk import vtkCellArray, vtkPoints, vtkPolyData, vtkTriangle

def create_voxel_block_grid_and_raytrace(pcd,imageNode):
    """THIS CURRENTLY DOESN'T WORK BUT IS A PLACEHOLDER.

    Args:
        pcd (_type_): _description_
        imageNode (_type_): _description_
    """
    intrinsic=imageNode.get_intrinsic_camera_parameters().intrinsic_matrix
    extrinsic=imageNode.cartesianTransform
    w=imageNode.imageWidth
    h=imageNode.imageHeight
    depth_scale=1
    depth_min=0
    depth_max=30
    vbg = o3d.t.geometry.VoxelBlockGrid(
                    attr_names=('tsdf', 'weight', 'color'),
                    attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                    attr_channels=((1), (1), (3)),
                    voxel_size=3.0 / 512,
                    block_resolution=16,
                    block_count=50000)
    active_voxels=vbg.compute_unique_block_coordinates(pcd)
    print(active_voxels)
    result = vbg.ray_cast(block_coords=active_voxels,
                              intrinsic=o3d.core.Tensor(intrinsic.intrinsic_matrix),
                              extrinsic=extrinsic,
                              width=640,
                              height=480,
                              render_attributes=[
                                  'depth', 'normal', 'color', 'index',
                                  'interp_ratio'
                              ],
                              depth_scale=depth_scale,
                              depth_min=depth_min,
                              depth_max=depth_max,
                              weight_threshold=1,
                              range_map_down_factor=8)
    # fig, axs = plt.subplots(2, 2)

    # # Colorized depth
    # colorized_depth = o3d.t.geometry.Image(result['depth']).colorize_depth(depth_scale,depth_min,depth_max)
    # # Render color via indexing
    # vbg_color = vbg.attribute('color').reshape((-1, 3))
    # nb_indices = result['index'].reshape((-1))
    # nb_interp_ratio = result['interp_ratio'].reshape((-1, 1))
    # nb_colors = vbg_color[nb_indices] * nb_interp_ratio
    # sum_colors = nb_colors.reshape((480, 640, 8, 3)).sum(
    #     (2)) / 255.0
    # t=colorized_depth.as_tensor().cpu().numpy()
    # print(t)
    # axs[0, 0].imshow(colorized_depth.as_tensor().cpu().numpy())
    # axs[0, 0].set_title('depth')

    # axs[0, 1].imshow(result['normal'].cpu().numpy())
    # axs[0, 1].set_title('normal')

    # axs[1, 0].imshow(result['color'].cpu().numpy())
    # axs[1, 0].set_title('color via kernel')

    # axs[1, 1].imshow(sum_colors.cpu().numpy())
    # axs[1, 1].set_title('color via indexing')

    # plt.tight_layout()
    # plt.show()

def subdivide_pcd_per_box (pcd,size:List[float]=None, parts:List[int]=None)->Tuple[List[int],List[int]]:
    """Subdivide a point cloud according to a set of boxes (either by size of number of parts).
    
    .. image:: ../../../docs/pics/subselection1.PNG

    Args:
        pcd (las,dataframe or open3d): Point Cloud (only points are used)
        size (list[float], optional): X, Y and Z size of the subdivided boxes in meter e.g. [10,10,5].
        parts (list[int], optional): X, Y and Z number of parts to divide the box in e.g. [7,7,1].

    Returns:
        Tuple[List[int],List[int]]: pathLists with names formatted as 'pcd_{a}_{b}_{c}_{d}' that correspond to xyz order, idxLists with indices per box.
    """
    # conver to o3d.geometry.point cloud if necessary
    if 'laspy.lasdata' in str(type(pcd)):
        pointcloud=o3d.geometry.PointCloud()
        pointcloud.points=o3d.utility.Vector3dVector(pcd.xyz)
        pcd=pointcloud
    elif 'pandas.DataFrame' in str(type(pcd)):
        pointcloud=o3d.geometry.PointCloud()
        pointcloud.points=o3d.utility.Vector3dVector(pcd.iloc[0:2].to_numpy())
        pcd=pointcloud
        
    # create box
    box=pcd.get_axis_aligned_bounding_box()

    # subdivide box into boxes
    boxes,names=gmu.divide_box_in_boxes(box,size=size) if size is not None else gmu.divide_box_in_boxes(box,parts=parts)

    # select indices per boxes
    pathLists=[]
    idxLists=[]
    for box,name in zip(boxes,names):
        pathLists.append(f'pcd_{name[0]}_{name[1]}_{name[2]}')
        idxLists.append(box.get_point_indices_within_bounding_box(pcd.points))
    return idxLists,pathLists

def subdivide_pcd_per_octree(pcd,maxDepth:int=4,lowEnd=0,highEnd=2000000)-> Tuple[List[int],List[int]]:
    """Create an octree of various point clouds and subdivide it according to a voxel octree. For each depth level, the data is divided 8-fold.
    This function returns the indices per voxel if that node has a number of points between lowEnd and highEnd.

    Args:
        pcd (las,dataframe or open3d): Point Cloud (only points are used)
        maxDepth (int, optional): depth of the octree. Defaults to 4 (this is also the maxDepth).
        lowEnd (int, optional): minimum number of points in a node for it to be added to the export. Defaults to 0.
        highEnd (int, optional): maximum number of points in a node for it to be added to the export. If this is surpassed, its children will be assessed. Defaults to 2000000.

    Returns:
        Tuple[List[int],List[int]]: pathLists with names formatted as 'pcd_{a}_{b}_{c}_{d}' that correspond to the depth ,idxLists with indices per valid node
    """
    #validate inputs
    assert maxDepth >0 and maxDepth <=4, f'maxDepth  should be >0 and <=4 (deeper branchin does not make sense and is not implemented)'
    
    # conver to o3d.geometry.point cloud if necessary
    if 'laspy.lasdata' in str(type(pcd)):
        pointcloud=o3d.geometry.PointCloud()
        pointcloud.points=o3d.utility.Vector3dVector(pcd.xyz)
        pcd=pointcloud
    elif 'pandas.DataFrame' in str(type(pcd)):
        pointcloud=o3d.geometry.PointCloud()
        pointcloud.points=o3d.utility.Vector3dVector(pcd.iloc[0:2].to_numpy())
        pcd=pointcloud
    
    #create octree
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    
    #iterate over point cloud node children (maxDepth 4) and return the indices if len(node.indices) are between lowEnd and highEnd
    a=-1
    pathLists=[]
    idxLists=[]
    # first level
    for node in octree.root_node.children[0:1]: # depth1
        a+=1
        if (getattr(node,'indices',None) is not None and 
            len(node.indices)> lowEnd and 
            len(node.indices)< highEnd):
            pathLists.append(f'pcd_{a}')
            idxLists.append(node.indices)
        else:
            # second level
            b=-1
            for childnode in getattr(node,'children',[]): #depth 2
                b+=1
                if (getattr(childnode,'indices',None) is not None and 
                    len(childnode.indices)> lowEnd and 
                    len(childnode.indices)< highEnd):
                    pathLists.append(f'pcd_{a}_{b}')
                    idxLists.append(childnode.indices)
                else:
                    #third level
                    c=-1
                    for grandchildnode in getattr(childnode,'children',[]): #depth 3
                        c+=1
                        if (getattr(grandchildnode,'indices',None) is not None and 
                            len(grandchildnode.indices)> lowEnd and 
                            len(grandchildnode.indices)< highEnd):
                            pathLists.append(f'pcd_{a}_{b}_{c}')
                            idxLists.append(grandchildnode.indices)
                        else:
                            #fourth level
                            d=-1
                            for grandgrandchildnode in getattr(grandchildnode,'children',[]): #depth 4
                                d+=1
                                if (getattr(grandgrandchildnode,'indices',None) is not None and 
                                    len(grandgrandchildnode.indices)> lowEnd and 
                                    len(grandgrandchildnode.indices)< highEnd):
                                    pathLists.append(f'pcd_{a}_{b}_{c}_{d}')
                                    idxLists.append(grandgrandchildnode.indices)
    return idxLists,pathLists

def pcd_to_octree(pcd:o3d.geometry.PointCloud, maDepth:int=7,colorUse:int=0)->o3d.geometry.Octree:
    """Create octree of point cloud and optionally color it consistently.

    Args:
        pcd (o3d.geometry.PointCloud): 
        colorUse (int, optional): If 0, the colors per leafNode will be averaged. If 1, the dominant color per LeafNode will be retained (this is quite slow). Defaults to 0.

    Returns:
        o3d.geometry.Octree: 
    """
    octree=o3d.geometry.Octree(max_depth=maDepth)
    octree.convert_from_point_cloud(pcd)
    def f_traverse_set_majority_color(node, node_info):
        """Callback function that changes the color to the dominant color in each leafnode.
        """
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):               
                myPcd=pcd.select_by_index(node.indices)
                colors=np.asarray(myPcd.colors)
                c=np.dot(colors[...,:3], [0.2989, 0.5870, 0.1140])
                _, counts = np.unique(c, return_counts=True)
                color=colors[np.argmax(counts)]
                node.color=color
        return None
    octree.traverse(f_traverse_set_majority_color) if colorUse==1 else None
    
    return octree

def capture_image_and_depth_viewer(pcdNode,imgNodes,imagePath,depthPath):
    """Visualizer currently is bugged in Open3D and doesn't allow custom sizes. 
    """
    pcd=copy.deepcopy(pcdNode.resource)
    # voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.4)
    # n=imgNodes[0]
    # intrinsic=n.get_intrinsic_camera_parameters().intrinsic_matrix
    # extrinsic=n.cartesianTransform
    # w=int(n.imageWidth/8)
    # h=int(n.imageHeight/8)
    # depth_scale=1
    # depth_min=0
    # depth_max=30
    # camera_parameters=n.get_pinhole_camera_parameters(downsampling=8)
    # voxelGrid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,0.4)
    # viewer = o3d.visualization.Visualizer() #! this currently is bugged in Open3D and doesn't allow custom sizes
    # viewer.create_window()
    # viewer.add_geometry(voxelGrid)
    # # control = viewer.get_view_control() #! this currently is bugged in Open3D
    # # print(control.convert_to_pinhole_camera_parameters().intrinsic)
    # # control.convert_from_pinhole_camera_parameters(camera_parameters)
    # viewer.capture_depth_image(os.path.join(imagePath, "{:04d}-depth.png".format(1)), do_render=True, depth_scale=1000.0)
    # viewer.capture_screen_image(os.path.join(depthPath, "{:04d}-color.png".format(1)), do_render=True)
    # viewer.destroy_window()

def assign_point_cloud_information(source_cloud:o3d.geometry.PointCloud,ref_clouds:List[o3d.geometry.PointCloud],class_names:List[int]=None)->np.array:
    ref_cloud,ref_arr=gmu.create_identity_point_cloud(ref_clouds)
    indices,distances=gmu.compute_nearest_neighbors(np.asarray(source_cloud.points),np.asarray(ref_cloud.points))
    index=ref_arr[indices]
    distances=distances[:,0]
    class_names=np.argmax(ref_arr)      if class_names is None  else class_names
    arr=np.zeros(len(np.asarray(source_cloud.points)))
    for ind in np.unique(index):
        locations=np.where(index ==ind)
        np.put(arr,locations,class_names[ind])
    return arr

def filter_img_classifcation_by_neighbors (predictions:np.array,shape: Tuple[int,int]=None,weight:float=3)->np.array:
    """Filters an initial raster prediction based on the classification of surrounding values.\n
    Every value is replaced by the most occuring value in the 9 surrounding raster values weighted by the initial value

    Args:
        predictions (np.array): _description_
        shape (Tuple[int,int]): shape of the raster
        weight (float, optional): influence of the initial value compared to neighboring values. Defaults to 3.

    Returns:
        np.array: _description_
    """
    #validate inputs
    shape=predictions.shape    if not shape     else shape
    predictions=np.reshape(predictions,shape)
    newPredictions=predictions
    #select most frequently occuring value in vincinity of each region (multiplied by weight for initial pixel)
    for i in range(1,predictions.shape[0]-1):
        for j in range(1,predictions.shape[1]-1):
            arr=np.hstack((predictions[i-1:i+1,j-1:j+1].flatten(),np.full((weight-1),predictions[i,j])))         
            newPredictions[i,j]=np.argmax(np.bincount(arr)) 
    return np.reshape(newPredictions,(-1,1)).flatten()

def remap_color_images_to_masks(images:List[np.array],colorList:np.array=None)->np.array:
    """Remap the values of an image (RGB or grayscale) to indices given a colorList. 

    **NOTE**: this is slow and error prone
    
    Args:
        1.images (List[np.array]): RGB or grayscale imagery.\n
        2.colorList (np.array, optional): list with RGB or grayscale colors used for the remapping. If no colors are provided, the unique colors of the first image will be taken. Defaults to None.

    Returns:
        np.array: image [m,n,1] with indices as mask
    """
    #validate inputs
    images=images if type(images)==list else [images]    
    images=[img if img.shape[2]==1 else np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) for img in images]
    
    #create colors equal to number of unique values if no colorList is provided
    colorList=colorList if colorList is not None else np.reshape(np.unique(images[0]),(len(np.unique(images[0])),1))
    colorList=colorList if colorList.shape[1]==1 else np.dot(colorList[...,:3], [0.2989, 0.5870, 0.1140])

    newImages=[]
    for img in images:
        #remap values
        for i,c in enumerate(colorList):
            # test1=
            # img=np.put(img,np.where(np.isclose(img,c,atol=0.01)),i)
            img=np.where(np.isclose(img,c,atol=0.01),i,img)
            
        #replace other values by 0
        img=np.where(img<1,0,img)
        newImages.append(img)
    return newImages
    
def project_meshes_to_rgbd_images (meshes:List[o3d.geometry.TriangleMesh], imgNodes:List[ImageNode], scale:float=1.0, fill_black:int=0)->Tuple[List[np.array],List[np.array]]:
    """Project a set of meshes given camera parameters.

    .. image:: ../../../docs/pics/Raycasting_6.PNG

    Args:
        1.meshes (List[o3d.geometry.TriangleMesh]): set of TriangleMeshes.\n
        2.imgNodes (List[ImageNode]): should contain imageWidth,imageHeight,cartesianTransform and focalLength35mm\n
        3.scale (float, optional): scale to apply to imagery (typically for downscaling). Defaults to 1.\n
        4.fill_black (int, optional): Region to fill in black pixels. 5 is a good value.\n
        
    Returns:
        Tuple[List[np.array],List[np.array]]: colorImages,depthImages
    """
    #validate meshes
    mesh=gmu.join_geometries(ut.item_to_list(meshes))
    imgNodes=ut.item_to_list(imgNodes)
    
    #create lists
    colorImages=[]
    depthImages=[]    
    
    #create raytracing scene
    scene = o3d.t.geometry.RaycastingScene()
    reference=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(reference)
    
    #create a colorArray from the mesh triangles (color of first vertex is taken)
    colors=np.asarray(mesh.vertex_colors)
    indices=np.asarray(mesh.triangles)[:,0]
    triangle_colors=colors[indices]
    #append black color at the end of the array for the invalid hits
    triangle_colors=np.vstack((triangle_colors,np.array([0,0,0])))
    
    #create rays  
    for n in imgNodes:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                                        intrinsic_matrix =n.get_intrinsic_camera_parameters().intrinsic_matrix,
                                        extrinsic_matrix =np.linalg.inv(n.get_cartesian_transform()),
                                        width_px=int(n.imageWidth),
                                        height_px=int(n.imageHeight))
        
        #apply scale
        rays=rays.numpy()
        rays=o3d.core.Tensor(rays[::int(1/scale),::int(1/scale)])
        #cast rays
        ans = scene.cast_rays(rays) 
        
        #get triangle_ids that are hit per ray
        triangle_ids=ans['primitive_ids'].numpy() # triangles
        rows,columns=triangle_ids.shape        
        triangle_ids=triangle_ids.flatten()
        # replace invalid id's by last (which is the above added black color)
        np.put(triangle_ids,np.where(triangle_ids==scene.INVALID_ID),triangle_colors.shape[0]-1) 
        
        #select colors 
        colors=triangle_colors[triangle_ids]

        #reshape array back to normal
        colorImage=np.reshape(colors,(rows,columns,3))
                
        #fill black if necessary
        colorImage=iu.fill_black_pixels(colorImage,fill_black)         if fill_black !=0       else colorImage
        depthImage=iu.fill_black_pixels(ans['t_hit'].numpy(),fill_black)         if fill_black !=0       else ans['t_hit'].numpy()

        #add to list
        colorImages.append(colorImage)    
        depthImages.append(depthImage)

    return colorImages,depthImages

def project_pcd_to_rgbd_images (pointClouds:List[o3d.geometry.Geometry], imgNodes:List[ImageNode], depth_max:float=15, fill_black:int=0)->Tuple[List[np.array],List[np.array]]:
    """Project a set of point cloud geometries given camera parameters. The given

    .. image:: ../../../docs/pics/rgbd.png

    Args:
        1.pointClouds (List[o3d.geometry.PointCloud]): set of o3d point clouds.\n
        2.imgNodes (List[ImageNode]): should contain imageWidth,imageHeight,cartesianTransform and focalLength35mm\n
        3.depth_max (float, optional): cut off distance. Defaults to 15m.\n
        4.fill_black (int, optional): Region to fill in black pixels. 5 is a good value.\n
        

    Returns:
        Tuple[List[np.array],List[np.array]]: colorImages,depthImages
    """
    #validate point clouds
    pointClouds=ut.item_to_list(pointClouds)
    pointClouds=gmu.join_geometries(pointClouds)

    #convert point cloud to tensor
    pcd = o3d.t.geometry.PointCloud.from_legacy(pointClouds)
    
    colorImages=[]
    depthImages=[]
    #project color and depth from point cloud to images
    for n in ut.item_to_list(imgNodes):
        # intrinsic=n.get_intrinsic_camera_parameters().intrinsic_matrix
        # extrinsic= np.linalg.inv(n.get_cartesian_transform()) 
        rgbd_reproj = pcd.project_to_rgbd_image(intrinsics =n.get_intrinsic_camera_parameters().intrinsic_matrix,
                                        extrinsics =np.linalg.inv(n.get_cartesian_transform()),
                                        width=int(n.imageWidth),
                                        height=int(n.imageHeight),
                                        depth_scale=1.0,
                                        depth_max=depth_max)
        colorImage=np.asarray(rgbd_reproj.color.to_legacy())
        colorImage=iu.fill_black_pixels(colorImage,fill_black)         if fill_black !=0       else colorImage
        depthImage=np.asarray(rgbd_reproj.depth.to_legacy())
        depthImage=iu.fill_black_pixels(depthImage,fill_black)         if fill_black !=0       else depthImage
                
        # reverse order so orientation matches image
        # colorImage=colorImage[::-1,::-1]
        # depthImage=depthImage[::-1,::-1]
    
        colorImages.append(colorImage)
        depthImages.append(depthImage)
    return colorImages,depthImages
        
def get_average_cartesian_transform_ortho(list):
    i=0
    sum1=0
    sum2=0
    sum3=0
    average1=0
    average2=0
    average3=0
    length=0
    matrix=[]

    while i<len(list):
        sum1+=list[i][0][3]
        sum2+=list[i][1][3]
        sum3+=list[i][2][3]
        i+=1

    length=len(list)

    average1=sum1/length
    average2=sum2/length
    average3=sum3/length

    matrix=np.array([[1,0,0,average1],[0,1,0,average2],[0,0,1,average3],[0,0,0,1]])
    return matrix

def create_xy_grids (geometries:List[o3d.geometry.TriangleMesh], resolution:float=0.1, direction:str='Down')-> np.array:
    """Generates a grid of rays (x,y,z,nx,ny,nz) with a spatial resolution from a set of input meshes.\n

    Args:
        1.geometries (List[o3d.geometry.TriangleMesh]): geometries to generate the grid from. grid will be placed at the highest of the lowest point.\n
        2.resolution (float, optional): XY resolution of the grid. Default stepsize is 0.1m.\n
        3.direction (str, optional): 'Up' or 'Down'. Position and direction of the grid. If 'Down', the grid is placed at the highest point with the orientation looking downards (0,0,-1). Defaults to 'Down'.

    Returns:
        np.array[x*y,6]: grid of arrays (x,y,z,nx,ny,nz)
    """
    rays = []

    for g in geometries:
        # create values
        minBound=g.get_min_bound()
        maxBound=g.get_max_bound()
        x = np.arange(minBound[0], maxBound[0],resolution )
        y = np.arange(minBound[1], maxBound[1], resolution )

        if direction == 'Down':
            z=maxBound[2]
            xx, yy = np.meshgrid(x, y)
            zz=np.full((x.size,y.size),z)
            array = np.zeros((np.size(xx), 6))
            array[:, 0] = np.reshape(xx, -1)
            array[:, 1] = np.reshape(yy, -1)
            array[:, 2] = np.reshape(zz, -1)
            array[:, 3] = np.zeros((xx.size,1))[0]
            array[:, 4] = np.zeros((xx.size,1))[0]
            array[:, 5] = -np.ones((xx.size,1))[0]
            ray = o3d.core.Tensor(array,dtype=o3d.core.Dtype.Float32)
            rays.append(ray)
        else:
            z=minBound[2]
            xx, yy = np.meshgrid(x, y)
            zz=np.full((x.size,y.size),z)
            array = np.zeros((np.size(xx), 6))
            array[:, 0] = np.reshape(xx, -1)
            array[:, 1] = np.reshape(yy, -1)
            array[:, 2] = np.reshape(zz, -1)
            array[:, 3] = np.zeros((xx.size,1))[0]
            array[:, 4] = np.zeros((xx.size,1))[0]
            array[:, 5] = np.ones((xx.size,1))[0]
            ray = o3d.core.Tensor(array,dtype=o3d.core.Dtype.Float32)
            rays.append(ray)
    return rays

def volume_mesh_BIM(depthmapFBIM:np.array, depthmapBimMin:np.array,depthmapBimMax:np.asarray,resolution:float=0.1)-> np.array:
    """Calculate the volume per element , three different options where:
            1) mesh is beneath the bim\n
            2) mesh is above the bim\n
            3) mesh is between the top and bottom of the bim\n

    **NOTE**: heinder, move this to tools

    Args:
        1. depthmapFBIM (np.array[:,1]): The distances between the grid per object and the top of the mesh.\n
        2. depthmapBimMin (np.array[:,1]): The distances between the grid per object and the bottom of the bim.\n
        3. depthmapBimMax (np.array[:,1]): The distances between the grid per object and the top of the bim.\n
        4. resolution (np.array[:,1], optional): Resolution of the grid.Defaults to 0.1m.\n

    Returns:
        array of volumes per bim object 
    """    
    m=0
    volume=[]
    while m<len(depthmapFBIM):
        n=0
        v=0
        while n<len(depthmapFBIM[m]):          
            if abs(depthmapFBIM[m][n])<100 and abs((depthmapBimMin[m][n]).numpy())<100 and abs(depthmapBimMax[m][n])<100:
                if depthmapFBIM[m][n] >= depthmapBimMin[m][n]:
                    d=(0)
                elif depthmapFBIM[m][n] < depthmapBimMax[m][n]:
                    d=((depthmapBimMin[m][n] -depthmapBimMax[m][n]).numpy())
                else: 
                    d=((depthmapBimMin[m][n]-depthmapFBIM[m][n]).numpy())
                v+=(d*resolution*resolution)
            n=n+1
        volume.append(v)
        m=m+1            
    return volume

def volume_theoretical_BIM( depthmapBimMin:np.array,depthmapBimMax:np.asarray,resolution:float=0.1)-> np.array:
    """Calculate the theoretical volume per element (m³).\n

    **NOTE**: heinder, move this to tools
    
    Args:
        1. depthmapFBIM (np.array[:,1]): The distances between the grid per object and the top of the mesh.\n
        2. depthmapBimMin (np.array[:,1]): The distances between the grid per object and the bottom of the bim.\n
        3. resolution (np.array[:,1], optional): Resolution of the grid.Defaults to 0.1m.\n

    Returns:
        array of theoretcial volumes per bim object 
    """  
    m=0
    volume=[]
    while m<len(depthmapBimMin):
        n=0
        v=0
        while n<len(depthmapBimMin[m]):
            if abs((depthmapBimMin[m][n]).numpy())<1000 and abs(depthmapBimMax[m][n])<1000:
                d=abs((depthmapBimMin[m][n] -depthmapBimMax[m][n]).numpy())
                v+=(d*resolution*resolution)
            n=n+1
        volume.append(v)
        m=m+1
    return volume

def calculate_completion(volumeMeshBIM:np.array,volumeBIM:np.array)->np.array:
    """Calculate the percentual completion (%) using the theoretical and practical volumes of the bim objects.

    **NOTE**: heinder, move this to tools

    Args:
        1. volumeMeshBIM (np.array[:,1]): The volume between mesh and BIM.\n
        2. volumeBIM (np.array[:,1]): The theoretical BIM.\n

    Returns:
        array of completness [0-1]"""

    completion=[]
    for i,element in enumerate(volumeMeshBIM): 
        if not volumeBIM[i] == 0:
            completion.append(element/volumeBIM[i])
        else:
            completion.append(None)
    return completion

def color_BIMNode(completion, BIMNodes):
    """Colors the BIM mesh geometries in the computed LOA color    
    
    **NOTE**: heinder, move this to tools
    
    
    Args:
        1. LOAs (_type_): results of the LOA analysis
        2. BIMNodes (List[BIMNode]): List of the BIMNodes in the project
        
    Returns:
        None
    """
    for BIMNode in BIMNodes:
        if BIMNode.resource:
                BIMNode.resource.paint_uniform_color([0.5,0.5,0.5])
    for i,BIMNode in enumerate(BIMNodes):
        if not completion[i] == None:
                if not BIMNode.resource:
                    BIMNode.get_resource()
                if completion[i]>=0.95:
                    BIMNode.resource.paint_uniform_color([0,1,0])
                if completion[i]<0.95:
                    BIMNode.resource.paint_uniform_color([1,1,0])
                if completion[i]<=0.50:
                    BIMNode.resource.paint_uniform_color([1,0.76,0])
                if completion[i]<=0.25:
                    BIMNode.resource.paint_uniform_color([1,0,0])

def remove_edges_volume_calculation(depthmapDifference,pcdFlightMax,distance:int=1):
    """    **NOTE**: heinder, move this to tools

    """
    pcd = pcdFlightMax
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    i=0 #0
    list=[]
    while i< len(depthmapDifference):
        if pd.isna(depthmapDifference[i]) == True or np.isinf(depthmapDifference[i]):
            list.append(i)
        i+=1

    for item in list:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[item], distance)
        for x in idx:
            if  not (np.isinf(depthmapDifference[x]) or  pd.isna(depthmapDifference[x])):
                depthmapDifference[x]=np.inf
    return depthmapDifference

def color_pointcloud_by_height(pointcloud: o3d.geometry.PointCloud, heights, buckets: int = 5, hmax:float = 10, buffer:float = 0.03):
    """Colors the resulting point cloud of the LOA analysis in a gradient by distance between the matched points from the reference and the source (very slow)

    **NOTE**: heinder, move this to tools. this is a really crappy function

    Args:
        pointcloud (o3d.geometry.PointCloud): Point cloud from the LOA determination or pointcloud matching its the returned indeces
        heights (nx1 array): Array containing the distances between two matched points
        buckets (int, optional): Number of intervals to be colored in. Defaults to 5.
        dmax (float, optional): Distances higher then this distance will be ignored. Defaults to 10.
        byElement (bool, optional): If the LOA must be computed per element of for the enitre cloud. Defaults to False.
        

    Returns:
        o3d.geometry.PointCloud()
    """
    print(pointcloud)

    pointcloud.paint_uniform_color([1,1,1])

    heights[heights == np.inf] = np.min(heights)
    max = np.nanmax(np.asarray(heights))
    # print(max)
    if max > hmax:
        max = hmax

    heights[heights == -np.inf] = np.min(heights)
    min = np.nanmin(np.asarray(heights))
    # print(min)
    if min < -hmax:
        min = -hmax
    
    interval = max / buckets
    lb = 0
    ub = lb+interval
    green = Color("lightgreen")
    colors = list(green.range_to(Color("darkgreen"),buckets))
    colors = [c.rgb for c in colors]
    bucket=0
    while ub <= max:
        places2 = np.where(np.asarray(heights) <= ub)[0]
        # print(places2)
        places3 = np.where(np.asarray(heights) > lb)[0]
        # print(places3)
        for place2 in places2:
            if place2 in places3:
                np.asarray(pointcloud.colors)[place2] = colors[bucket]
        lb = ub
        ub += interval
        bucket +=1


    interval = np.abs(min / buckets)
    ub = 0
    lb = ub-interval 
    

    red = Color("red")
    colors = list(red.range_to(Color("darkred"),buckets))
    colors = [c.rgb for c in colors]
    bucket=0
    
    while lb > min :
        places2 = np.where(np.asarray(heights) <= ub)[0]
        places3 = np.where(np.asarray(heights) > lb)[0]
        for place2 in places2:
            if place2 in places3:
                np.asarray(pointcloud.colors)[place2] = colors[bucket]
        ub = lb
        lb -= interval
        bucket +=1

    places2 = np.where(np.asarray(heights) <= buffer)[0]
    places3 = np.where(np.asarray(heights) > -buffer)[0]
    for place2 in places2:
        if place2 in places3:
            np.asarray(pointcloud.colors)[place2] = [0.5,0.5,0.5]
    
    return pointcloud

def create_xy_grid (geometry, resolution:float=0.1, direction:str='Down', offset:int=10)-> np.array:
    """Generates a grid of rays (x,y,z,nx,ny,nz) with a spatial resolution from a set of input meshes.\n

    **NOTE**: MB, this is ugly code

    Args:
        1.geometries (List[o3d.geometry.TriangleMesh]): geometries to generate the grid from. grid will be placed at the highest of the lowest point.\n
        2.resolution (float, optional): XY resolution of the grid. Default stepsize is 0.1m. \n
        3.direction (str, optional): 'Up' or 'Down'. Position and direction of the grid. If 'Down', the grid is placed at the highest point with the orientation looking downards (0,0,-1). Defaults to 'Down'.

    Returns:
        np.array[x*y,6]: grid of arrays (x,y,z,nx,ny,nz)
    """       
    # create values
    minBound=geometry.get_min_bound()
    maxBound=geometry.get_max_bound()
    x = np.arange(minBound[0], maxBound[0],resolution )
    y = np.arange(minBound[1], maxBound[1], resolution )

    if direction == 'Down':
        z=maxBound[2]+offset
        xx, yy = np.meshgrid(x, y)
        zz=np.full((x.size,y.size),z)
        array = np.zeros((np.size(xx), 6))
        array[:, 0] = np.reshape(xx, -1)
        array[:, 1] = np.reshape(yy, -1)
        array[:, 2] = np.reshape(zz, -1)
        array[:, 3] = np.zeros((xx.size,1))[0]
        array[:, 4] = np.zeros((xx.size,1))[0]
        array[:, 5] = -np.ones((xx.size,1))[0]
        ray = o3d.core.Tensor(array,dtype=o3d.core.Dtype.Float32)
    else:
        z=minBound[2]-offset
        xx, yy = np.meshgrid(x, y)
        zz=np.full((x.size,y.size),z)
        array = np.zeros((np.size(xx), 6))
        array[:, 0] = np.reshape(xx, -1)
        array[:, 1] = np.reshape(yy, -1)
        array[:, 2] = np.reshape(zz, -1)
        array[:, 3] = np.zeros((xx.size,1))[0]
        array[:, 4] = np.zeros((xx.size,1))[0]
        array[:, 5] = np.ones((xx.size,1))[0]
        ray = o3d.core.Tensor(array,dtype=o3d.core.Dtype.Float32)
    return ray

def get_mesh_intersections(geometry:o3d.geometry.Geometry, grid:np.array)-> np.array:
    """Returns [N , X * Y] matrix of distances between a grid and a set of input geometries. \n

    Args:
        1. geometries (List[o3d.geometry.TriangleMesh]): N geometries to compute the distance to.\n
        2. grid(o3d.core.Tensor): Tensor 
       
    Returns:
        np.array: 2D distance array [N , X * Y]
    """
    # create grid
    rays=grid
    # construct raycastscenes
    scene = o3d.t.geometry.RaycastingScene()
    gl = o3d.t.geometry.TriangleMesh.from_legacy(geometry)
    scene = o3d.t.geometry.RaycastingScene()
    id = scene.add_triangles(gl)
    ans = scene.cast_rays(rays)
    return ans['t_hit'].numpy()

def get_bim_intersections (geometries:List[o3d.geometry.TriangleMesh], rays:np.array)-> np.array:
    """Returns [N , X * Y] matrix of distances between a grid and a set of input geometries. \n

    **NOTE**: don't call this BIM when its just meshes. not enough tensor information. this function appears twice

    Args:
        1.geometries (List[o3d.geometry.TriangleMesh]): N geometries to compute the distance to.\n
        2.grid(o3d.core.Tensor): Tensor 
       
    Returns:
        np.array: 2D distance array [N , X * Y]
    """
    intersections=[]
    scene = o3d.t.geometry.RaycastingScene()
    for i,g in enumerate(geometries):
        gl = o3d.t.geometry.TriangleMesh.from_legacy(g)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(gl)
        ans = scene.cast_rays(rays[i])
        intersections.append(ans['t_hit'].numpy())
    return intersections

def get_mesh_intersectionsBIM (geometry, grid:np.array)-> np.array:
    """Returns [N , X * Y] matrix of distances between a grid and a set of input geometries. \n

    **NOTE**: don't call this BIM when its just meshes. not enough tensor information. this function appears twice

    Args:
        1.geometries (List[o3d.geometry.TriangleMesh]): N geometries to compute the distance to.\n
        2.grid(o3d.core.Tensor): Tensor 
       
    Returns:
        np.array: 2D distance array [N , X * Y]
    """
    
    # create grid
    rays=grid
       
    # construct raycastscenes
    scene = o3d.t.geometry.RaycastingScene()

    b=[]

    gl = o3d.t.geometry.TriangleMesh.from_legacy(geometry)
    scene = o3d.t.geometry.RaycastingScene()
    id = scene.add_triangles(gl)


    b=[]
    n=0
    while n<len(grid):
        ans = scene.cast_rays(rays[n])
        b.append(ans['t_hit'].numpy())
        n=n+1

    return b

def get_scene_intersections (geometries:List[o3d.geometry.TriangleMesh],mesh1:o3d.geometry.TriangleMesh , mesh2:o3d.geometry.TriangleMesh,resolution:float=0.1, direction:str='Down')-> np.array:
    """Returns [N , d] matrix of distances between a grid and a set of input geometries. \n

    **NOTE**: don't call this BIM when its just meshes. not enough tensor information. this function appears twice

    Args:
        1.geometries (List[o3d.geometry.TriangleMesh]): N geometries to compute the distance to.\n
        2.grid(o3d.core.Tensor): Tensor 
        
    Returns:
        np.array: 2D distance array [N , d]
    """

    # create grid
    rays=create_xy_grid(geometries,resolution=resolution,direction=direction)
        
    # construct raycastscenes
    scene = o3d.t.geometry.RaycastingScene()

    b=[]
    for g in geometries:
        gl = o3d.t.geometry.TriangleMesh.from_legacy(g)
        scene = o3d.t.geometry.RaycastingScene()
        id = scene.add_triangles(gl)
        ans = scene.cast_rays(rays)
        b.append(ans['t_hit'].numpy())
    
    b=np.asarray(b).T

    distance1=0

    mesh1l = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    scene = o3d.t.geometry.RaycastingScene()
    id = scene.add_triangles(mesh1l)
    ans = scene.cast_rays(rays)
    distance1=(np.asarray([ans['t_hit'].numpy()]))

    distance1=distance1.T

    distance2=0
    
    mesh2l = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)
    scene = o3d.t.geometry.RaycastingScene()
    id = scene.add_triangles(mesh2l)
    ans = scene.cast_rays(rays)
    distance2=(np.asarray([ans['t_hit'].numpy()]))

    distance2=distance2.T

    print(b.shape)
    print(distance1.shape)
    print(distance2.shape)

    array=np.block([b,distance1,distance2])
    return array

def get_rays_raycast (geometries, direction:str='Down'):
    """Generates a grid of rays (x,y,z,nx,ny,nz) with a spatial resolution from a set of input meshes.\n

    **NOTE**: move this to tools

    Args:
        1.geometries (List[o3d.geometry.TriangleMesh]): geometries to generate the grid from. grid will be placed at the highest of the lowest point.\n
        2.resolution (float, optional): XY resolution of the grid. Default stepsize is 0.1m.
        3.direction (str, optional): 'Up' or 'Down'. Position and direction of the grid. If 'Down', the grid is placed at the highest point with the orientation looking downards (0,0,-1). Defaults to 'Down'.

    Returns:
        np.array[x*y,6]: grid of arrays (x,y,z,nx,ny,nz)
    """
    rays = []
    for g in geometries:
        # create values
        if direction == 'Down':
            points=np.asarray(g.croppedPcdMax.points)
            # print(len(points))
            zero=np.array([(np.zeros(len(g.croppedPcdMin.points)))]).T
            # print((zero))
            minusOne=np.array([-np.ones(len(g.croppedPcdMin.points))]).T
            ray=np.float32(np.column_stack((points,zero,zero,minusOne)))
            rays.append(ray)
        else:
            points=np.asarray(g.croppedPcdMin.points)
            zero=np.array([np.zeros(len(g.croppedPcdMin.points))]).T
            # print(len(zero))
            plusOne=np.array([np.ones(len(g.croppedPcdMin.points))]).T
            ray=np.float32(np.column_stack((points,zero,zero,plusOne)))
            rays.append(ray)
    return rays 



def get_mesh_intersection_with_grid(geometry: o3d.geometry.Geometry, grid: np.array) -> np.array:
    """ Finds the intersection of a mesh and a grid of rays.

    Args:
        1. geometry (o3d.geometry.Geometry): The mesh to intersect with the rays.
        2. grid (np.array): A 2D numpy array representing the rays to cast. Each row should contain the origin and direction of a ray.

    .. image:: ../../../docs/pics/2dGrid.JPG


    Returns:
        np.array: A 1D numpy array containing the intersection point along the ray for each ray in the grid.

    Example:
        mesh = o3d.io.read_triangle_mesh("mesh.ply")
        grid = np.array([[0,0,0, 0,0,1], [1,0,0, 0,0,1], [2,0,0, 0,0,1]])
        intersections = get_mesh_intersection(mesh, grid)
        print(intersections)
    """
    rays = grid
    scene = o3d.t.geometry.RaycastingScene()
    gl = o3d.t.geometry.TriangleMesh.from_legacy(geometry)
    scene = o3d.t.geometry.RaycastingScene()
    id = scene.add_triangles(gl)
    ans = scene.cast_rays(rays)
    hit_primitives = ans['primitive_ids'].numpy()
    array = []
    for pri in hit_primitives:
        if pri != 4294967295:
            array.append(pri)
    unique_arr = list(set(array))
    
    return unique_arr, ans['t_hit'].numpy()


def get_top_lineset_from_meshes(bimNodes):
    """Computes the visible boundary edges from the top of a set of BIMNodes.

    This function creates grids above the meshes and computes the intersections with the 3D meshes. The computed edges are returned as Open3D.LineSet objects.

    .. image:: ../../../docs/pics/get_top_lineset.PNG

    
    Args:
        bimNodes (list): A list of BIMNodes.

    Returns:
        list [Open3D.Lineset]: A list of LineSet objects representing the computed visible edges.
    """
    resolution=0.1
    target_meshes=[]
    target_triangles=[]
    for i,BIMNode in enumerate(bimNodes):
        BIMNode.topgrid=create_xy_grid(BIMNode.resource,resolution=resolution,direction='Down',offset=10) #offset in m
        BIMNode.bottomgrid=create_xy_grid(BIMNode.resource,resolution=resolution,direction='Up',offset=10) #offset in m
        
        xyzFlightMax=BIMNode.topgrid[:,0:3]  
        BIMNode.pcdFlightMax = o3d.geometry.PointCloud()
        BIMNode.pcdFlightMax.points = o3d.utility.Vector3dVector(xyzFlightMax.numpy())
        xyzFlightMin=BIMNode.bottomgrid[:,0:3]  
        pcdFlightMin = o3d.geometry.PointCloud()
        pcdFlightMin.points = o3d.utility.Vector3dVector(xyzFlightMin.numpy())
        BIMNode.toppcdgrid=BIMNode.pcdFlightMax
        BIMNode.intersectionMesh,BIMNode.depthmapplus=get_mesh_intersection_with_grid(BIMNode.resource,BIMNode.topgrid)
        BIMNode.bottompcdgrid=pcdFlightMin
        mesh=BIMNode.resource
        target_ids=BIMNode.intersectionMesh
        triangles = np.array(mesh.triangles)
        primitive_ids = np.arange(len(triangles))
        target_triangle = np.array(mesh.triangles)[np.in1d(primitive_ids, target_ids)]
        target_triangles.append(target_triangle)
        target_mesh = o3d.geometry.TriangleMesh()
        target_mesh.vertices = mesh.vertices
        target_mesh.triangles = o3d.utility.Vector3iVector(target_triangles[i])
        color = np.random.rand(3)  # red
        target_mesh.paint_uniform_color(color)
        target_meshes.append(target_mesh)

    meshes=[]
    edgestotal=[]
    multi_mesh = pv.MultiBlock()
    multi_edge = pv.MultiBlock()

    for mesh_o3d in target_meshes:
        points = vtkPoints()
        vertices = vtkCellArray()
        triangles = vtkCellArray()

        for i in range(len(mesh_o3d.vertices)):
            points.InsertNextPoint(mesh_o3d.vertices[i])
        for i in range(len(mesh_o3d.triangles)):
            triangle = vtkTriangle()
            triangle.GetPointIds().SetId(0, mesh_o3d.triangles[i][0])
            triangle.GetPointIds().SetId(1, mesh_o3d.triangles[i][1])
            triangle.GetPointIds().SetId(2, mesh_o3d.triangles[i][2])
            triangles.InsertNextCell(triangle)

        vtk_mesh = vtkPolyData()
        vtk_mesh.SetPoints(points)
        vtk_mesh.SetPolys(triangles)
        mesh_pv = pv.wrap(vtk_mesh)
        edges = mesh_pv.extract_feature_edges(45)
        multi_mesh.append(mesh_pv)
        multi_edge.append(edges)
        meshes.append(mesh_pv)
        edgestotal.append(edges)
    for i in range(len(meshes), len(multi_edge)):
        multi_edge[i].color = [1,0,0]

    points=[]
    lines=[]
    for edge in edgestotal:
        point=edge.points
        line=edge.lines
        lines.append(line)
        points.append(point)

    linesets=[]
    for count, point in enumerate(points):
        pointsi=point
        linesi=lines[count]
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(pointsi)
        line_indices = linesi.reshape(-1, 3)
        lineset.lines = o3d.utility.Vector2iVector(line_indices[:, 1:3])
        linesets.append(lineset)

    return linesets


def determine_percentage_of_coverage(sources: List[o3d.geometry.TriangleMesh], reference:o3d.geometry.PointCloud,threshold:float=0.1)-> np.array:
    """Returns the Percentage-of-Coverage (PoC) of every source geometry when compared to a reference geometry. The PoC is defined as the ratio of points on the boundary surface of the source that lie within a Euclidean distance threshold hold of the reference geometry. sampled point cloud on the boundary surface of the sources with a resolution of e.g. 0.1m. \n

    .. math::
    
        p_{i'} = \{ p \mid ∀ p \in p_i : p_i \cap n_i \}
        
        c_i = \\frac{{|P_{i'}|}}{{|P_i|}}

    
    E.g. a mesh of a beam of which half the surface lies within 0.1m of a point cloud will have a PoC of 0.5.
    
    Args:
        1. sources (o3d.geometry.TriangleMesh/PointCloud): geometries to determine the PoC for. \n
        2. reference (o3d.geometry.PointCloud): reference geometry for the Euclidean distance calculations.\n
        3. threshold (float, optional): sampling resolution of the boundary surface of the source geometries. Defaults to 0.1m.\n

    Raises:
        ValueError: Sources must be o3d.geometry (PointCloud or TriangleMesh)

    Returns:
        List[percentages[0-1.0]] per source
    """
    #if no list, list
    sources=ut.item_to_list(sources)
    sourcePCDs=[]
    indentityArray=None
    percentages=[0.0]*len(sources)

    # check whether source and reference are close together to minize calculations
    ind=gmu.get_box_inliers(reference.get_oriented_bounding_box(), [geometry.get_oriented_bounding_box() for geometry in sources])
    if not ind:
        return percentages

    # sample o3d.geometry and create identitylist so to track the indices.
    for i,source in enumerate(sources):  
        if i in ind:      
            if 'PointCloud' in str(type(source)) :
                sourcePCD=source.voxel_down_sample(threshold)
                indentityArray=np.vstack((indentityArray,np.full((len(sourcePCD.points), 1), i)))
            elif 'TriangleMesh' in str(type(source)):
                area=source.get_surface_area()
                count=int(area/(threshold*threshold))
                sourcePCD=source.sample_points_uniformly(number_of_points=count)
                indentityArray=np.vstack((indentityArray,np.full((len(sourcePCD.points), 1), i)))
            sourcePCDs.append(sourcePCD)
        else:
            sourcePCDs.append(None)

    indentityArray=indentityArray.flatten()
    indentityArray=np.delete(indentityArray,0)

    #compute distances
    joinedPCD=gmu.join_geometries(sourcePCDs)
    distances=joinedPCD.compute_point_cloud_distance(reference)

    #remove distances > threshold
    ind=np.where(np.asarray(distances) <= threshold)[0]
    if ind.size ==0:
        return percentages
    indexArray=[indentityArray[i] for i in ind.tolist()]

    #count occurences
    unique_elements, counts_elements = np.unique(indexArray, return_counts=True)
    for i,n in enumerate(unique_elements):
        percentages[n]=counts_elements[i]/len(sourcePCDs[n].points)
    return percentages
