import torch
import numpy as np
# import MVRegC  #  uncommment this if you want to run the N-ICP baseline.
import open3d as o3d
from scipy.spatial import KDTree


 
def rigid_fit( X, Y, w, eps=0.0001):
    '''
    @param X: source frame [B, N,3]
    @param Y: target frame [B, N,3]
    @param w: weights [B, N,1]
    @param eps:
    @return:
    '''
    # https://ieeexplore.ieee.org/document/88573

    bsize = X.shape[0]
    device = X.device
    W1 = torch.abs(w).sum(dim=1, keepdim=True)
    w_norm = w / (W1 + eps)
    mean_X = (w_norm * X).sum(dim=1, keepdim=True)
    mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
    Sxy = torch.matmul( (Y - mean_Y).transpose(1,2), w_norm * (X - mean_X) )
    Sxy = Sxy.cpu().double()
    U, D, V = Sxy.svd() # small SVD runs faster on cpu
    S = torch.eye(3)[None].repeat(bsize,1,1).double()
    UV_det = U.det() * V.det()
    S[:, 2:3, 2:3] = UV_det.view(-1, 1,1)
    svT = torch.matmul( S, V.transpose(1,2) )
    R = torch.matmul( U, svT).float().to(device)
    t = mean_Y.transpose(1,2) - torch.matmul( R, mean_X.transpose(1,2) )
    return R, t



def ED_warp(x, g, R, t, w):
    """ Warp a point cloud using the embeded deformation
    https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
    :param x: point location
    :param g: anchor location
    :param R: rotation
    :param t: translation
    :param w: weights
    :return:
    """
    y = ( (R @ (x[:,None] - g)[..., None] ).squeeze() + g + t ) * w[...,None]
    y = y.sum(dim=1)
    return y



def map_pixel_to_pcd(valid_pix_mask):
    ''' establish pixel to point cloud mapping, with -1 filling for invalid pixels
    :param valid_pix_mask:
    :return:
    '''
    image_size = valid_pix_mask.shape
    pix_2_pcd_map = torch.cumsum(valid_pix_mask.view(-1), dim=0).view(image_size).long() - 1
    pix_2_pcd_map [~valid_pix_mask] = -1
    return pix_2_pcd_map

def pc_2_uv_np(pcd, intrin):
    '''
    :param pcd: nx3
    :param intrin: 3x3 mat
    :return:
    '''

    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    fx, cx, fy, cy = intrin[0, 0], intrin[0, 2], intrin[1, 1], intrin[1, 2]
    u = (fx * X / Z + cx).astype(int)
    v = (fy * Y / Z + cy).astype(int)
    return np.stack([u,v], -1 )

def pc_2_uv(pcd, intrin):
    '''
    :param pcd: nx3
    :param intrin: 3x3 mat
    :return:
    '''

    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    fx, cx, fy, cy = intrin[0, 0], intrin[0, 2], intrin[1, 1], intrin[1, 2]
    u = (fx * X / Z + cx).to(torch.long)
    v = (fy * Y / Z + cy).to(torch.long)
    return torch.stack([u,v], -1 )

def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def meshgrid2d(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def depth2pointcloud(z, intrin, downsampling=True, sampling_points = 3000):
    B, H, W = list(z.shape)
    y, x = meshgrid2d(B, H, W)
    z = torch.reshape(z, [B, H, W])
    # fx, x0, fy, y0 = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    fx, fy, x0, y0 = split_intrinsics(intrin)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    # print("shape of xyz", xyz.shape)
    temp_ = []

    for i in range(B):
        temp_pcd = xyz[i][~torch.all(xyz[i] == 0, dim=-1)]
        # xyz[i] = xyz[i][[not torch.all(xyz[i][idx] == 0) for idx in range(xyz[i].shape[0])], :]
        if downsampling:
            if temp_pcd.shape[0] > sampling_points:
                idx = torch.randperm(temp_pcd.shape[0])[:sampling_points]
                temp_pcd = temp_pcd[idx,:]
            elif (temp_pcd.shape[0] > 0):
                # print('intra_liver_pcd_tgt shape: ', intra_liver_pcd_tgt.shape)
                temp_pcd = torch.pad(temp_pcd, ((0, sampling_points - temp_pcd.shape[0]),(0,0)), 'wrap')
                
            else:
                temp_pcd = torch.zeros([sampling_points, 3])
        temp_.append(temp_pcd)
    temp_ = torch.stack(temp_, dim=0)
    # print("shape of after xyz", xyz.shape)
    
    return temp_.contiguous()

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    B, H, W = list(z.shape)

    fx = torch.reshape(fx, [B,1,1])
    fy = torch.reshape(fy, [B,1,1])
    x0 = torch.reshape(x0, [B,1,1])
    y0 = torch.reshape(y0, [B,1,1])
    
    # unproject py3d
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)

    ## test py3d 2 blender ##
    # z = -1 * z
    # x = (z/fx)*(x-x0) * (-1)
    # y = (z/fy)*(y-y0)
    ## test ##

    ## test py3d 2 opencv ##
    # z = z
    # x = (z/fx)*(x-x0) * (-1)
    # y = (z/fy)*(y-y0) * (-1)
    ## test ##
    
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    xyz = torch.stack([x,y,z], dim=2)
    return xyz


def depth_2_pc(depth, intrin):
    '''
    :param depth:
    :param intrin: 3x3 mat
    :return:
    '''
    # if len(depth.shape) > 2:
    #     depth = depth[:, :, 0]
    fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    height, width = depth[0].shape
    u = torch.arange(width) * torch.ones([height, width])
    v = torch.arange(height) * torch.ones([width, height])
    v = torch.transpose(v)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return torch.stack([X, Y, Z])

def dpt_2_cld(self, dpt, cam_scale, K):
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd / cam_scale
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)
    return cld, choose

def depth_to_mesh(depth_image,
                 mask_image,
                 intrin,
                 depth_scale=1000.,
                 max_triangle_distance=0.04):
    """
    :param depth_image:
    :param mask_image:
    :param intrin:
    :param depth_scale:
    :param max_triangle_distance:
    :return:
    """
    width = depth_image.shape[1]
    height = depth_image.shape[0]


    mask_image[mask_image > 0] = 1
    depth_image = depth_image * mask_image

    point_image = depth_2_pc(depth_image / depth_scale, intrin)
    point_image = point_image.astype(np.float32)

    vertices, faces, vertex_pixels = MVRegC.depth_to_mesh(point_image, max_triangle_distance)

    return vertices, faces, vertex_pixels, point_image


def compute_graph_edges(vertices, valid_vertices, faces, node_indices, num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS ) :

    num_nodes = node_indices.shape[0]
    num_vertices = vertices.shape[0]

    graph_edges              = -np.ones((num_nodes, num_neighbors), dtype=np.int32)
    graph_edges_weights      =  np.zeros((num_nodes, num_neighbors), dtype=np.float32)
    graph_edges_distances    =  np.zeros((num_nodes, num_neighbors), dtype=np.float32)
    node_to_vertex_distances = -np.ones((num_nodes, num_vertices), dtype=np.float32)

    visible_vertices = np.ones_like(valid_vertices)
    MVRegC.compute_edges_geodesic( vertices, visible_vertices, faces, node_indices,
                                   graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances,
                                   num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS )

    return graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances


def get_deformation_graph_from_depthmap (depth_image, intrin, config, debug_mode=False):
    '''
    :param depth_image:
    :param intrin:
    :return:
    '''

    #########################################################################
    # Options
    #########################################################################
    # Depth-to-mesh conversion
    max_triangle_distance = config.max_triangle_distance
    # Node sampling and edges computation
    node_coverage = config.node_coverage  # in meters
    USE_ONLY_VALID_VERTICES = config.USE_ONLY_VALID_VERTICES
    num_neighbors = config.num_neighbors
    ENFORCE_TOTAL_NUM_NEIGHBORS = config.ENFORCE_TOTAL_NUM_NEIGHBORS
    SAMPLE_RANDOM_SHUFFLE = config.SAMPLE_RANDOM_SHUFFLE
    REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = config.REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS



    #########################################################################
    """convert depth to mesh"""
    #########################################################################
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    mask_image=depth_image>0
    # fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    vertices, faces, vertex_pixels, point_image = depth_to_mesh(depth_image, mask_image, intrin, max_triangle_distance=max_triangle_distance, depth_scale=1000.)
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    assert num_vertices > 0 and num_faces > 0


    #########################################################################
    """Erode mesh, to not sample unstable nodes on the mesh boundary."""
    #########################################################################
    non_eroded_vertices = MVRegC.erode_mesh(vertices, faces, 0, 0)



    #########################################################################
    """Sample graph nodes"""
    #########################################################################
    valid_vertices = non_eroded_vertices
    node_coords, node_indices = MVRegC.sample_nodes ( vertices, valid_vertices, node_coverage, USE_ONLY_VALID_VERTICES, SAMPLE_RANDOM_SHUFFLE)
    num_nodes = node_coords.shape[0]



    #########################################################################
    """visualize surface and non-eroded points"""
    #########################################################################
    if debug_mode:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[non_eroded_vertices.reshape(-1), :]))
        pcd_nodes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_coords))
        o3d.visualization.draw_geometries([mesh,  pcd_nodes], mesh_show_back_face=True)


    #########################################################################
    """Compute graph edges"""
    #########################################################################
    graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances = \
        compute_graph_edges(vertices, valid_vertices, faces, node_indices, num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS )
    # graph_edges = MVRegC.compute_edges_euclidean(node_coords,num_neighbors, 0.05)


    #########################################################################
    "Remove nodes"
    #########################################################################
    valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)
    node_id_black_list = []
    if REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS:
        MVRegC.node_and_edge_clean_up(graph_edges, valid_nodes_mask)
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()
    # else:
    #     print("You're allowing nodes with not enough neighbors!")
    # print("Node filtering: initial num nodes", num_nodes, "| invalid nodes", len(node_id_black_list),
    #       "({})".format(node_id_black_list))




    #########################################################################
    """Compute pixel anchors"""
    #########################################################################
    pixel_anchors = np.zeros((0), dtype=np.int32)
    pixel_weights = np.zeros((0), dtype=np.float32)
    MVRegC.compute_pixel_anchors_geodesic( node_to_vertex_distances, valid_nodes_mask, vertices, vertex_pixels, pixel_anchors, pixel_weights, width, height, node_coverage)
    # print("Valid pixels:", np.sum(np.all(pixel_anchors != -1, axis=2)))



    #########################################################################
    """filter invalid nodes"""
    #########################################################################
    node_coords = node_coords[valid_nodes_mask.squeeze()]
    node_indices = node_indices[valid_nodes_mask.squeeze()]
    graph_edges = graph_edges[valid_nodes_mask.squeeze()]
    graph_edges_weights = graph_edges_weights[valid_nodes_mask.squeeze()]
    graph_edges_distances = graph_edges_distances[valid_nodes_mask.squeeze()]


    #########################################################################
    """Check that we have enough nodes"""
    #########################################################################
    num_nodes = node_coords.shape[0]
    if (num_nodes == 0):
        print("No nodes! Exiting ...")
        exit()


    #########################################################################
    """Update node ids"""
    #########################################################################
    if len(node_id_black_list) > 0:
        # 1. Mapping old indices to new indices
        count = 0
        node_id_mapping = {}
        for i, is_node_valid in enumerate(valid_nodes_mask):
            if not is_node_valid:
                node_id_mapping[i] = -1
            else:
                node_id_mapping[i] = count
                count += 1

        # 2. Update graph_edges using the id mapping
        for node_id, graph_edge in enumerate(graph_edges):
            # compute mask of valid neighbors
            valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

            # make a copy of the current neighbors' ids
            graph_edge_copy = np.copy(graph_edge)
            graph_edge_weights_copy = np.copy(graph_edges_weights[node_id])
            graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

            # set the neighbors' ids to -1
            graph_edges[node_id] = -np.ones_like(graph_edge_copy)
            graph_edges_weights[node_id] = np.zeros_like(graph_edge_weights_copy)
            graph_edges_distances[node_id] = np.zeros_like(graph_edge_distances_copy)

            count_valid_neighbors = 0
            for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                if is_valid_neighbor:
                    # current neighbor id
                    current_neighbor_id = graph_edge_copy[neighbor_idx]

                    # get mapped neighbor id
                    if current_neighbor_id == -1:
                        mapped_neighbor_id = -1
                    else:
                        mapped_neighbor_id = node_id_mapping[current_neighbor_id]

                    graph_edges[node_id, count_valid_neighbors] = mapped_neighbor_id
                    graph_edges_weights[node_id, count_valid_neighbors] = graph_edge_weights_copy[neighbor_idx]
                    graph_edges_distances[node_id, count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                    count_valid_neighbors += 1

            # normalize edges' weights
            sum_weights = np.sum(graph_edges_weights[node_id])
            if sum_weights > 0:
                graph_edges_weights[node_id] /= sum_weights
            else:
                print("Hmmmmm", graph_edges_weights[node_id])
                raise Exception("Not good")

        # 3. Update pixel anchors using the id mapping (note that, at this point, pixel_anchors is already free of "bad" nodes, since
        # 'compute_pixel_anchors_geodesic_c' was given 'valid_nodes_mask')
        MVRegC.update_pixel_anchors(node_id_mapping, pixel_anchors)



    #########################################################################
    """Compute clusters."""
    #########################################################################
    graph_clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32)
    clusters_size_list = MVRegC.compute_clusters(graph_edges, graph_clusters)
    # print("clusters_size_list", clusters_size_list)


    #########################################################################
    """visualize valid pixels"""
    #########################################################################
    if debug_mode:
        from utils.vis import save_grayscale_image
        pixel_anchors_image = np.sum(pixel_anchors, axis=2)
        pixel_anchors_mask = np.copy(pixel_anchors_image).astype(np.uint8)
        raw_pixel_mask = np.copy(pixel_anchors_image).astype(np.uint8) * 0
        pixel_anchors_mask[pixel_anchors_image == -4] = 0
        raw_pixel_mask[depth_image > 0] = 1
        pixel_anchors_mask[pixel_anchors_image > -4] = 1
        save_grayscale_image("../output/pixel_anchors_mask.jpeg", pixel_anchors_mask)
        save_grayscale_image("../output/depth_mask.jpeg", raw_pixel_mask)


    #########################################################################
    """visualize graph"""
    #########################################################################
    if debug_mode:
        from utils.vis import node_o3d_spheres, merge_meshes

        node_mesh = node_o3d_spheres(node_coords, node_coverage * 0.1, color=[1, 0, 0])
        edges_pairs = []
        for node_id, edges in enumerate(graph_edges):
            for neighbor_id in edges:
                if neighbor_id == -1:
                    break
                edges_pairs.append([node_id, neighbor_id])
        from utils.line_mesh import LineMesh

        line_mesh = LineMesh(node_coords, edges_pairs, radius=0.002)
        edge_mesh = line_mesh.cylinder_segments
        edge_mesh = merge_meshes(edge_mesh)
        edge_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh, node_mesh, edge_mesh], mesh_show_back_face=True)


    model_data = {
        "graph_nodes": torch.from_numpy( node_coords),
        "graph_edges": torch.from_numpy( graph_edges).long(),
        "graph_edges_weights": torch.from_numpy( graph_edges_weights),
        "graph_clusters": graph_clusters,
        "pixel_anchors": torch.from_numpy( pixel_anchors),
        "pixel_weights": torch.from_numpy( pixel_weights),
        "point_image": torch.from_numpy( point_image).permute(1,2,0)
    }


    return model_data




def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]



def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx


def multual_nn_correspondence(src_pcd_deformed, tgt_pcd, search_radius=0.3, knn=1):

    src_idx = np.arange(src_pcd_deformed.shape[0])

    s2t_dists, ref_tgt_idx = knn_point_np (knn, tgt_pcd, src_pcd_deformed)
    s2t_dists, ref_tgt_idx = s2t_dists[:,0], ref_tgt_idx [:, 0]
    valid_distance = s2t_dists < search_radius

    _, ref_src_idx = knn_point_np (knn, src_pcd_deformed, tgt_pcd)
    _, ref_src_idx = _, ref_src_idx [:, 0]

    cycle_src_idx = ref_src_idx [ ref_tgt_idx ]

    is_mutual_nn = cycle_src_idx == src_idx

    mutual_nn = np.logical_and( is_mutual_nn, valid_distance)
    correspondences = np.stack([src_idx [ mutual_nn ], ref_tgt_idx[mutual_nn] ] , axis=0)

    return correspondences

def xyz_2_uv(pcd, intrin):
    ''' np function
    :param pcd: nx3
    :param intrin: 3x3 mat
    :return:
    '''

    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    fx, cx, fy, cy = intrin[0, 0], intrin[0, 2], intrin[1, 1], intrin[1, 2]
    u = (fx * X / Z + cx)
    v = (fy * Y / Z + cy)

    if type(u) == np.ndarray:
        return np.stack([u, v], -1).astype(int)
    else :
        return torch.stack( [u,v], -1).long()


def add_gaussian_noise(point_cloud, noise_percent=10):
    min_val = np.min(point_cloud)
    max_val = np.max(point_cloud)
    data_range = max_val - min_val
    
    if data_range == 0:
        return point_cloud  # 避免除零错误
    
    sigma = data_range * noise_percent / 100
    noise = np.random.normal(0, sigma, point_cloud.shape)
    noisy_pointcloud = point_cloud + noise
    
    # 裁剪到合理范围（例如±3σ）
    lower_bound = min_val - 3 * sigma
    upper_bound = max_val + 3 * sigma
    noisy_pointcloud = np.clip(noisy_pointcloud, lower_bound, upper_bound)
    
    return noisy_pointcloud


def calculate_sparse_overlap(source, target, radius=0.01):
    """
    计算目标点云在源点云中的点包含重叠比例。

    参数：
    - source (np.ndarray): Nx3 源点云（已对齐）
    - target (np.ndarray): Mx3 目标点云（已对齐）
    - radius (float): 邻域半径阈值（单位：与坐标系一致，如米）

    返回：
    - ratio (float): 重叠比例（0~1）
    """
    # 构建源点云的KD-Tree
    kdtree = KDTree(source)
    
    # 统计符合条件的目标点数量
    count = 0
    for point in target:
        # 查询最近邻距离
        dist, _ = kdtree.query(point, k=1)  # k=1表示只找最近点
        if dist <= radius:
            count += 1
    
    return count / len(target) if len(target) > 0 else 0.0