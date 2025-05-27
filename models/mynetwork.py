import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import numpy as np
from tqdm.notebook import tqdm
import imageio
import cv2

import pytorch3d
# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

## external imports
from lib.geometry import depth_2_pc,depth2pointcloud
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences

from models.mesh_render import MeshRender
from models.NDP_deform import NDP_Deformation
# from PoseNet import PoseNet
from models.lepard.lepard import Pipeline

from models.pointnet.pointnet import Pointnet2MSG
from models.structure_adaptor import PriorAdaptor


class DeformRegis(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # self.camera_K = config.camera_K
        # if len(self.camera_K.shape) < 3:
        #     self.camera_K_batch = self.camera_K.unsqueeze(0).repeat(config.batch_size,1,1).to(config.device)

        self.posenet = Pipeline(config)
        self.deformNet = NDP_Deformation(config)
        self.renderer = MeshRender(config)
        self.device = config.device
        self.num_points = config.max_points
        self.real_syn = config.dataset

        ### structure similarity ###
        self.geo_local = Pointnet2MSG(0)

        self.sim_mat = PriorAdaptor(emb_dims=64, n_heads=4)

        self.intra_model_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.pre_model_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 3, 1),
            nn.Sigmoid(),
        )

        conv1d_stpts_prob_modules = []
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.ReLU())
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        self.lowrank_projection = nn.Sequential(*conv1d_stpts_prob_modules)
        
        
        
        
       
    
    def np_depth_to_colormap(self, depth):
        """ depth: [H, W] """
        depth_normalized = np.zeros(depth.shape)

        valid_mask = depth > -0.9 # valid
        if valid_mask.sum() > 0:
            d_valid = depth[valid_mask]
            depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())

            depth_np = (depth_normalized * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
            depth_normalized = depth_normalized
        else:
            print('!!!! No depth projected !!!')
            depth_color = depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
        return depth_color, depth_normalized

    def forward(self, input, timers=None):
        '''
        preMeshes: Pre-operative liver model, pcd [N,3] by default
        renderer: Pytorch3d renderer
        intraRef: Intra-operative reference liver data [pointcloud + image]
        pose: Estimated pose from object coordinate system to camera coordinate system
        '''
        self.timers = timers

        
        preMeshes = input['o3d_pre_mesh_list'] # str
        
        
        if self.timers: self.timers.tic('rigid registration')
        pred_data = self.posenet(input, timers)
        if self.timers: self.timers.toc('rigid registration')

        
        
        
        preope_pcd_src = input['batched_src_pcd'] # B, N, C
        
       
        camera_K_batch = input['cam_k'] # B, N, C
        
        intraOpe_pcd = input['batched_tgt_pcd'] # B, N, C
        
        assert preope_pcd_src.shape == intraOpe_pcd.shape
        bs, n_pts = intraOpe_pcd.size()[:2]
        

        pred_R = pred_data['R_s2t_pred'] # s2t // w2c
        pred_t = pred_data['t_s2t_pred'] 

        
        tsfm = {}
        tsfm['rot'] = pred_R
        tsfm['trans'] = pred_t

        

        repositioned_mesh = torch.bmm(preope_pcd_src, pred_R.permute(0,2,1)) + pred_t.permute(0,2,1)
        
        
        
        
        if self.real_syn == 'real':
            

            intra_model_local = self.geo_local(intraOpe_pcd) # Bs 64, Npts
            intra_model_global = self.intra_model_global(intra_model_local) # Bs 1024, 1

            lowrank_proj = self.lowrank_projection(intra_model_local) # Bs 3, Npts
            
            weighted_points_features = torch.sum(lowrank_proj[:, None, :, :] * intra_model_local[:, :, None, :], dim=3)

            

            pre_model_local = self.geo_local(preope_pcd_src) ## Bs 64, Npts
            
            similarity_matrix = self.sim_mat(pre_model_local, weighted_points_features, weighted_points_features)
            pre_model_global = self.pre_model_global(similarity_matrix) ## Bs 1024, 1


            assign_feat = torch.cat((intra_model_local, intra_model_global.repeat(1, 1, n_pts), pre_model_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
            assign_mat = self.assignment(assign_feat)
            assign_mat = assign_mat.view(-1, n_pts, 3).contiguous()
            
            assign_mat = F.softmax(assign_mat, dim=2)
            


            ##### here we need to deform the source pcd #####
            self.deformNet.load_pcds(repositioned_mesh, intraOpe_pcd, assign_mat)
            
            if self.timers: self.timers.tic('deformation')
            warped_pcd, _, _= self.deformNet.register() ## note: the output is the warped pcd, which is in camera coordinate system
            if self.timers: self.timers.toc('deformation')
            ori_preope_pcd_src_warped = torch.bmm(pred_R.permute(0,2,1), (warped_pcd - pred_t.permute(0,2,1)).permute(0,2,1)).permute(0,2,1)
        
            verts_flow = ori_preope_pcd_src_warped - preope_pcd_src # Bs, N, 3
            ##### here we need to deform the source pcd #####

            weighted_verts_flow = assign_mat * verts_flow
            
            

            updata_verts_ = preope_pcd_src + weighted_verts_flow # deformed source pcd
            

            deformed_src_verta = updata_verts_.clone()


            # print("tsfm: ", tsfm)
            if self.real_syn == 'real':
                updata_verts_ = torch.bmm(tsfm['rot'], updata_verts_.permute(0,2,1)).permute(0,2,1) + tsfm['trans'].permute(0,2,1) 

                updata_verts_ = torch.bmm(input['ocv2blender'], updata_verts_.permute(0,2,1)).permute(0,2,1) ## opencv camera coordinate system
                  
                center_p_batch = input['bbx_center'].unsqueeze(1)
                scale = 1. / input['scale'].unsqueeze(1)
                scale = scale.unsqueeze(-1).repeat(1, n_pts, 1)
                
                updata_verts_ = updata_verts_ - center_p_batch
                updata_verts_ = updata_verts_ * scale
                updata_verts_ = updata_verts_ + center_p_batch


            if self.timers: self.timers.tic('rendering')
            mask, depthmap = self.renderer(updata_verts_, tsfm, input['rgbs'], input['img_size'], camera_K_batch, input)
            if self.timers: self.timers.toc('rendering')
            

            rendered_pcd = depth2pointcloud(depthmap, camera_K_batch, downsampling=True, sampling_points = self.num_points)

            rendered_pcd_ocv = rendered_pcd - input['bbx_center'].unsqueeze(1)
            rendered_pcd_ocv = rendered_pcd_ocv * input['scale'].unsqueeze(1).unsqueeze(-1).repeat(1, n_pts, 1)
            rendered_pcd_ocv = rendered_pcd_ocv + input['bbx_center'].unsqueeze(1)
            rendered_pcd_bld = torch.bmm(input['ocv2blender'], rendered_pcd_ocv.permute(0,2,1)).permute(0,2,1) ## blender camera coordinate system

            
            
            
            pred_data.update({ 'mask': mask})
            pred_data.update({ 'rendered_pcd': rendered_pcd})
            pred_data.update({ 'rendered_dpt': depthmap})
            pred_data.update({ 'deformed_src': deformed_src_verta})
            pred_data.update({ 'deformed_src_vis': updata_verts_})
            pred_data.update({ 'rendered_pcd_bld': rendered_pcd_bld})

        
        return pred_data
    


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    from datasets.dataloader import get_dataloader, get_datasets
    from configs.models import architectures
    config_pth = '/configs/train/main_config.yaml'
    with open(config_pth,'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23456', world_size=1, rank=0)
    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     init_method='env://',
    # )

    if config.gpu_mode:
        config.device = torch.device("cuda:2")
    else:
        config.device = torch.device('cpu')

    camera_K = torch.from_numpy(np.array([[360.0,0.0,480.0],[0.0,360.0,270.0],[0.0,0.0,1.0]]))
    print('camera_K shape: ', camera_K.shape)
    config.camera_K = camera_K
    config.kpfcn_config.architecture = architectures[config.dataset]
    config.img_size = (540,960)
    model = DeformRegis(config).to(config.device)

    train_set, val_set, test_set = get_datasets(config)
    train_loader, neighborhood_limits = get_dataloader(train_set,config,shuffle=False)

    for inputs in train_loader:
        for k, v in inputs.items():
            if type(v) == list:
                if type(v[0]) in [str, np.ndarray]:
                    pass
                else:
                    inputs [k] = [item.to(config.device) for item in v]
            elif type(v) in [ dict, float, type(None), np.ndarray, str]:
                pass
            else:
                inputs [k] = v.to(config.device)

        pred_data = model(inputs)
        
        print("mask shape: ", pred_data['mask'].shape)
        print("rendered_pcd shape: ", pred_data['rendered_dpt'].shape)
        del pred_data
        del inputs
