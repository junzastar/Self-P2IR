import logging
import math
import numpy as np
from lib.transformations import quaternion_from_matrix

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


def sample_rotations_12():
    """ tetrahedral_group: 12 rotations

    """
    group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],

                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                      [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

                      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]])
    # return group.astype(float)
    quaternion_group = np.zeros((12, 4))
    for i in range(12):
        quaternion_group[i] = quaternion_from_matrix(group[i])
    return quaternion_group.astype(float)


def sample_rotations_24():
    """ octahedral_group: 24 rotations

    """
    group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],

                      [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                      [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
                      [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
                      [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],

                      [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                      [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],
                      [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
                      [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],

                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                      [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

                      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],

                      [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                      [[0, 0, 1], [0, -1, 0], [-1, 0, 0]],
                      [[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
                      [[0, 0, -1], [0, -1, 0], [1, 0, 0]]])
    # return group.astype(float)
    quaternion_group = np.zeros((24, 4))
    for i in range(24):
        quaternion_group[i] = quaternion_from_matrix(group[i])
    return quaternion_group.astype(float)


def sample_rotations_60():
    """ icosahedral_group: 60 rotations

    """
    phi = (1 + math.sqrt(5)) / 2
    R1 = np.array([[-phi/2, 1/(2*phi), -0.5], [-1/(2*phi), 0.5, phi/2], [0.5, phi/2, -1/(2*phi)]])
    R2 = np.array([[phi/2, 1/(2*phi), -0.5], [1/(2*phi), 0.5, phi/2], [0.5, -phi/2, 1/(2*phi)]])
    group = [np.eye(3, dtype=float)]
    n = 0
    while len(group) > n:
        n = len(group)
        set_so_far = group
        for rot in set_so_far:
            for R in [R1, R2]:
                new_R = np.matmul(rot, R)
                new = True
                for item in set_so_far:
                    if np.sum(np.absolute(item - new_R)) < 1e-6:
                        new = False
                        break
                if new:
                    group.append(new_R)
                    break
            if new:
                break
    # return np.array(group)
    group = np.array(group)
    quaternion_group = np.zeros((60, 4))
    for i in range(60):
        quaternion_group[i] = quaternion_from_matrix(group[i])
    return quaternion_group.astype(float)

def load_obj(path_to_file):
    """ Load obj file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices

    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces

def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * face_vertices[0, :] + \
        sqrt_r1 * (1 - r2) * face_vertices[1, :] + \
        sqrt_r1 * r2 * face_vertices[2, :]

    return point


def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C


def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :],
                         faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points


def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts

def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


######### lepard utils ###
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


def blend_scene_flow (query_loc, reference_loc, reference_flow , knn=3) :
    '''approximate flow on query points
    this function assume query points are sub-/un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    dists, idx = knn_point_np (knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    return blended_flow


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