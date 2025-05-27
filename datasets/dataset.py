import torch.utils.data as data
from PIL import Image
import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import _pickle as cPickle
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import open3d as o3d
from lib.PLY import read_ply, write_ply
from scipy.spatial.transform import Rotation
import yaml
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences
import re
from lib.geometry import add_gaussian_noise

# import imgaug.augmenters as iaa
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    
    img_width = 1024
    img_length = 1280

    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 640)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

class PoseDataset(data.Dataset):
    def __init__(self, mode, config, data_augmentation=False):

        self.patients = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

        self.mode = mode
        self.root = config.data_root
        self.real_syn = config.dataset

        self.data_augmentation = data_augmentation

        self.list_label = []
        self.list_img = []
        self.list_liverPcd = []
        self.list_patient = []
        self.list_rank = []
        self.list_scale = []
        self.list_posemeta = {}
        self.pre_model = {}
        self.pre_model_normal = {}
        self.list_camK = {}

        self.rot_factor = 1.
        self.angle_sigma = 0.06
        self.angle_clip = 0.18
        self.augment_noise = 0.002
        self.scale_low = 0.8
        self.scale_high = 1.25

        self.max_points = config.max_points #30720
        self.intra_points = 30000
        self.overlap_radius = 0.0065 #0.005
        self.rng = np.random


        item_count = 0
        for patient in self.patients:
            if self.mode == 'train':
                if self.real_syn == 'real':
                    input_file = open('{0}/{1}/train_real.txt'.format(self.root, '%02d' % patient))
                else:
                    input_file = open('{0}/{1}/train_syn.txt'.format(self.root, '%02d' % patient))
                
            else:
                if self.real_syn == 'real':
                    input_file = open('{0}/{1}/test_real.txt'.format(self.root, '%02d' % patient))
                else:
                    input_file = open('{0}/{1}/test_syn.txt'.format(self.root, '%02d' % patient))
            while 1:
                item_count = item_count + 1
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                if self.real_syn == 'syn':
                    self.list_label.append('{0}/{1}/syn/labels/lbl{2}.png'.format(self.root, '%02d' % patient, input_line[-5:]))
                    self.list_liverPcd.append('{0}/{1}/syn/liverPcds/{2}.ply'.format(self.root, '%02d' % patient, input_line))
                    self.list_patient.append(patient)
                    self.list_rank.append(input_line[-5:])
                else:
                    self.list_label.append('{0}/{1}/real/labels/{2}/label.png'.format(self.root, '%02d' % patient, input_line))
                    self.list_img.append('{0}/{1}/real/labels/{2}/img.png'.format(self.root, '%02d' % patient, input_line))
                    self.list_liverPcd.append('{0}/{1}/real/liverPcds/{2}.ply'.format(self.root, '%02d' % patient, input_line))
                    self.list_patient.append(patient)
                    match = re.search(r"frame_(\d+)_json", input_line)
                    if match:
                        cur_rank = int(match.group(1))
                    self.list_rank.append(cur_rank)
                    self.list_scale.append('{0}/{1}/real/scale/{2}.txt'.format(self.root, '%02d' % patient, input_line))

            if self.real_syn == 'syn':
                pose_file = open('{0}/{1}/syn/camPose.yml'.format(self.root, '%02d' % patient), 'r')
                self.list_posemeta[patient] = yaml.safe_load(pose_file)

                camK_file = open('{0}/{1}/syn/CAM_K.yml'.format(self.root, '%02d' % patient), 'r')
                # intrinsic = yaml.safe_load(camK_file)
                intrinsic = np.array(yaml.safe_load(camK_file)['00000'][0]['cam_K']).reshape(3,3)
                self.list_camK[patient] = intrinsic
            else:
                camK_file = open('{0}/{1}/real/CAM_K.yml'.format(self.root, '%02d' % patient), 'r')
                # intrinsic = yaml.safe_load(camK_file)
                intrinsic = np.array(yaml.safe_load(camK_file)['00000'][0]['cam_K']).reshape(3,3)
                self.list_camK[patient] = intrinsic

            # load pre-operative model
            self.preope_mesh_path = '{0}/{1}/model/reconstructed_mesh_world_m.obj'.format(self.root, '%02d' % patient)

            preope_mesh = o3d.io.read_triangle_mesh(self.preope_mesh_path)
            preope_pcd = preope_mesh.sample_points_uniformly(200000)
            preope_pcd = preope_pcd.voxel_down_sample(voxel_size=0.0006)
            self.pre_model[patient] = np.array(preope_pcd.points) # m

            ## here we need to know the normal of the liver point cloud ###
            preope_mesh.compute_vertex_normals()
            original_normals = np.array(preope_mesh.vertex_normals)
            from scipy.spatial import cKDTree
            tree = cKDTree(np.array(preope_mesh.vertices))
            _, indices = tree.query(self.pre_model[patient], k=1)
            propagated_normals = original_normals[indices].squeeze()
            self.pre_model_normal[patient] = propagated_normals
            ## here we need to know the normal of the liver point cloud ###

            print("Patient {0} buffer loaded".format(patient))

        self.length = len(self.list_label)
        print("{0} data is {1}".format(self.mode, self.length))

    def random_idx(self):
        n = self.length
        idx = self.rng.randint(0, n)
        # item = self.list[idx]
        return idx
    
    def get_item(self, index):

        ### read the gt pose ###
        which_patient = self.list_patient[index]
        if self.real_syn == 'syn':
            gt_pose = self.list_posemeta[which_patient]
        rank = self.list_rank[index]
        cam_K = self.list_camK[which_patient]

        ## get real image size (H, W)
        if self.real_syn == 'real':
            if which_patient == 5:
                img_size = np.array([2160,3840])
            else:
                img_size = np.array([1080,1920])
        else:
            img_size = np.array([540,960])
        
        if self.real_syn == 'syn':
            pose_R = np.array(gt_pose[rank][0]["cam_R_c2w"]).reshape(3,3).astype(np.float32)
            pose_t = np.array(gt_pose[rank][0]["cam_t_c2w"]).reshape(3,1).astype(np.float32) / 1000.0 # m

            tsfm = to_tsfm(pose_R, pose_t) # C2W so, we need to inverse it
            tsfm = np.linalg.inv(tsfm) ##[4,4]
            pose_R = tsfm[:3,:3].astype(np.float32)
            pose_t = tsfm[:3,3].reshape(3,1).astype(np.float32) # W2C (S2T)
        else:
            pose_R = np.eye(3).astype(np.float32)
            pose_t = np.zeros([3,1]).astype(np.float32)
            tsfm = np.eye(4).astype(np.float32)
        # try:
        # print("whcih patient: {0}, path:{1}".format(which_patient, self.list_label[index]))
        if self.real_syn == 'syn':
            labels = np.array(Image.open(self.list_label[index]).convert('L'))
            imgs = np.ones_like(labels) ## original images
        else:
            labels = np.array(Image.open(self.list_label[index]))
            imgs = np.array(Image.open(self.list_img[index])) ## original images
         
        if self.real_syn == 'syn':
            liver_labels = labels.copy()
            liver_labels[liver_labels==0]=1
            liver_labels[liver_labels!=1]=0
        else:
            liver_labels = labels.copy()

        # except:
        #     print("whcih patient: {0}, path:{1}",format(which_patient, self.list_label[index]))
        # print("liver pixel number: ", np.sum(liver_labels == 1))
        if np.sum(liver_labels == 1) < 100:
            print("liver pixel number: ", np.sum(liver_labels == 1))
            return None
        
        ### here we need load the scale (to rescale the src for rendering) ###
        if self.real_syn == 'real':
            scale = np.loadtxt(self.list_scale[index])
        else:
            scale = np.array(1.0)
        ### here we need load the scale (to rescale the src for rendering) ###

        preope_pcd_src = self.pre_model[which_patient]
        preope_pcd_src_normal = self.pre_model_normal[which_patient]

        intra_liver_pcd = o3d.io.read_point_cloud(self.list_liverPcd[index])
        ### here we need to compute the center of get_axis_aligned_bounding_box ###
        if self.real_syn == 'syn':
            bounding_box_center = np.array([0., 0., 0.]).astype(np.float32)
        else:
            axis_aligned_bounding_box = intra_liver_pcd.get_axis_aligned_bounding_box()
            [center_x, center_y, center_z] = axis_aligned_bounding_box.get_center()
            bounding_box_center = np.array([center_x, center_y, center_z]).astype(np.float32) / 1000.0 # m
        ### here we need to compute the center of get_axis_aligned_bounding_box ###
        intra_liver_pcd = intra_liver_pcd.uniform_down_sample(every_k_points = 8)
        intra_liver_pcd = intra_liver_pcd.remove_duplicated_points()

        intra_liver_pcd, ind = intra_liver_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    

        # if we get too many points, we do some downsampling
        intra_liver_pcd_tgt = np.array(intra_liver_pcd.points) / 1000.0 # m
        intra_liver_pcd_tgt = intra_liver_pcd_tgt[~np.all(intra_liver_pcd_tgt == 0., axis=-1)]

        

        
        ### here we need to transform tgt from opencv to blender
        if self.real_syn == 'real':
            ocv2blender = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            intra_liver_pcd_tgt = np.dot(ocv2blender, intra_liver_pcd_tgt.T).T
       
        else:
            ocv2blender = np.array([[1,0,0],[0,1,0],[0,0,1]])

        assert not np.any(np.isnan(intra_liver_pcd_tgt))
      
        if (preope_pcd_src.shape[0] > self.max_points):
            idx = self.rng.permutation(preope_pcd_src.shape[0])[:self.max_points]
            preope_pcd_src = preope_pcd_src[idx]
            preope_pcd_src_normal = preope_pcd_src_normal[idx]

        if (intra_liver_pcd_tgt.shape[0] > self.max_points):
            idx = self.rng.permutation(intra_liver_pcd_tgt.shape[0])[:self.max_points]
            intra_liver_pcd_tgt = intra_liver_pcd_tgt[idx]
        elif (intra_liver_pcd_tgt.shape[0] > 0):
            # print('intra_liver_pcd_tgt shape: ', intra_liver_pcd_tgt.shape)
            intra_liver_pcd_tgt = np.pad(intra_liver_pcd_tgt, ((0, self.max_points - intra_liver_pcd_tgt.shape[0]),(0,0)), 'wrap')
            
        else:
            intra_liver_pcd_tgt = np.zeros([self.max_points, 3])

        # print('intra_liver_pcd_tgt shape: ', intra_liver_pcd_tgt.shape)
        # print('preope_pcd_src shape: ', preope_pcd_src.shape)

        ### here we need to reconstruct the pre-operative model for saving the face information ####
        # 创建一个 Open3D 点云对象
        pcd_poisson = o3d.geometry.PointCloud()
        pcd_poisson.points = o3d.utility.Vector3dVector(preope_pcd_src)
        pcd_poisson.normals = o3d.utility.Vector3dVector(preope_pcd_src_normal)
        # 使用泊松重建算法重建网格
        # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_poisson, depth=10)[0]
        # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_poisson, 0.01)
        radii = [0.005, 0.01, 0.02, 0.04]
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_poisson, o3d.utility.DoubleVector(radii))
        # reconstructed_vertices = np.asarray(poisson_mesh.vertices)
        reconstructed_faces = np.asarray(poisson_mesh.triangles)
        # o3d.io.write_triangle_mesh('/home/jiking/users/jun/SelfTraining/test_data/recons_model.ply', poisson_mesh)
        ### here we need to reconstruct the pre-operative model for saving the face information ####

        # add gaussian noise
        if self.data_augmentation:
            if self.real_syn == 'syn':
                # rotate the point cloud
                # euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
                euler_ab = np.clip(self.angle_sigma*self.rng.randn(3), -self.angle_clip, self.angle_clip) # anglez, angley, anglex [-10.3, 10.3]
                rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
                
                ## add random scale
                scales = self.rng.uniform(self.scale_low, self.scale_high)    # 0.8~1.25间的随机缩放
                scale_matrix = np.diag([scales,scales,scales]) # [x,y,z] same scale
                scale_matrix_inv = np.linalg.inv(scale_matrix)

                if (self.rng.rand(1)[0] > 0.5):
                    preope_pcd_src = np.matmul(rot_ab, preope_pcd_src.T).T
                    pose_R = np.matmul(pose_R, rot_ab.T)

                    preope_pcd_src = np.matmul(scale_matrix, preope_pcd_src.T).T
                    pose_R = np.matmul(pose_R, scale_matrix_inv)
                else:
                    intra_liver_pcd_tgt = np.matmul(rot_ab, intra_liver_pcd_tgt.T).T
                    pose_R = np.matmul(rot_ab, pose_R)
                    pose_t = np.matmul(rot_ab, pose_t)

                    intra_liver_pcd_tgt = np.matmul(scale_matrix, intra_liver_pcd_tgt.T).T
                    pose_R = np.matmul(scale_matrix, pose_R)
                    pose_t = np.matmul(scale_matrix, pose_t)

                preope_pcd_src = preope_pcd_src + (self.rng.rand(preope_pcd_src.shape[0], 3) - 0.5) * self.augment_noise
                intra_liver_pcd_tgt = intra_liver_pcd_tgt + (self.rng.rand(intra_liver_pcd_tgt.shape[0], 3) - 0.5) * self.augment_noise
            else:
                preope_pcd_src = preope_pcd_src + (self.rng.rand(preope_pcd_src.shape[0], 3) - 0.5) * self.augment_noise
                intra_liver_pcd_tgt = intra_liver_pcd_tgt + (self.rng.rand(intra_liver_pcd_tgt.shape[0], 3) - 0.5) * self.augment_noise
       
        if self.real_syn == 'syn':
            correspondences = get_correspondences(to_o3d_pcd(preope_pcd_src), to_o3d_pcd(intra_liver_pcd_tgt), tsfm, self.overlap_radius)
            if correspondences.shape[0] < 1000:
                print("correspondences shape: ", correspondences.shape)
                return None
        else:
            correspondences = np.ones((100,3))
       

        src_feats = np.ones_like(preope_pcd_src[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(intra_liver_pcd_tgt[:, :1]).astype(np.float32)
        rgbs = np.zeros_like(intra_liver_pcd_tgt).astype(np.float32)

        ##### ablation study 1. varying noise on the intraoperative pcd #####
        # intra_liver_pcd_tgt = add_gaussian_noise(intra_liver_pcd_tgt, 2.5) # 0.2%，0.5%，1%，2%, 2.5%
        ##### ablation study 1. varying noise on the intraoperative pcd #####

        assert not np.any(np.isnan(preope_pcd_src))
        assert not np.any(np.isnan(intra_liver_pcd_tgt))
        assert not np.any(np.isnan(src_feats))
        assert not np.any(np.isnan(tgt_feats))

        assert not np.any(np.isnan(pose_R))
        assert not np.any(np.isnan(pose_t))
        assert not np.any(np.isnan(liver_labels))
        # assert not np.any(np.isnan(correspondences))
        # assert not np.any(np.isnan(self.preope_mesh_path))
        # assert not np.any(np.isnan(liver_depth))

        # print("data loaded successfully!")
      
        item_dict = dict(
            preope_pcd_src=preope_pcd_src.astype(np.float32), \
            preope_pcd_src_normal=preope_pcd_src_normal.astype(np.float32), \
            preope_reconstructed_faces = reconstructed_faces.astype(np.int32), \
            intra_liver_pcd_tgt=intra_liver_pcd_tgt.astype(np.float32), \
            src_feats=src_feats.astype(np.float32), \
            tgt_feats=tgt_feats.astype(np.float32), \
            pose_R=pose_R.astype(np.float32),\
            pose_t=pose_t.astype(np.float32),\
            liver_labels=liver_labels.astype(np.int32),\
            liver_imgs = imgs.astype(np.uint8),\
            correspondences=correspondences,\
            o3d_pre_mesh=self.preope_mesh_path,
            cam_K = cam_K.astype(np.float32),
            rgbs = rgbs.astype(np.float32), ## texture
            img_size = img_size.astype(np.int32),
            ocv2blender = ocv2blender.astype(np.float32),
            scale = scale.astype(np.float32),
            bbx_center = bounding_box_center.astype(np.float32),
            which_patient = np.array(which_patient).astype(np.int32),
        )

        return item_dict
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.mode == 'train':
    
            data = self.get_item(idx)
            while data is None:
                item_name = self.random_idx()
               
                data = self.get_item(item_name)
            return data
        else:
           
            data = self.get_item(idx)
            while data is None:
                item_name = self.random_idx()
               
                data = self.get_item(item_name)
            return data
            # return self.get_item(idx)



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    from datasets.dataloader import get_dataloader, get_datasets
    from configs.models import architectures
    import open3d as o3d
    import numpy as np
    config_pth = '/home/jiking/users/jun/SelfTraining/configs/train/main_config.yaml'
    with open(config_pth,'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    dataset = PoseDataset(mode='train', config=config, data_augmentation=True)

    for idx in range(1):
        print("processiong: {0}".format(idx),end='r')
        data = dataset.__getitem__(idx)
        print("preope_pcd_src shape: ", data['preope_pcd_src'].shape)
        print("intra_liver_pcd_tgt shape: ", data['intra_liver_pcd_tgt'].shape)
        print("src_feats shape: ", data['src_feats'].shape)
        print("tgt_feats shape: ", data['tgt_feats'].shape)
        print("pose_R shape: ", data['pose_R'].shape)
        print("pose_t shape: ", data['pose_t'].shape)
        print("liver_labels shape: ", data['liver_labels'].shape)
        # print("liver_depth shape: ", data['liver_depth'].shape)
        print("correspondences shape: ", data['correspondences'].shape)
       
