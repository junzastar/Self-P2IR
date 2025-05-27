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
        # self.geo_local = Pointnet2MSG(0)
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
        
        
        
        
        # # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        # self.register_buffer('image_ref', image_ref)
        
        # # Create an optimizable parameter for the x, y, z position of the camera. 
        # self.camera_position = nn.Parameter(
        #     torch.from_numpy(np.array([3.0,  6.9, +2.5], dtype=np.float32)).to(meshes.device))
    
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

        # device = preMeshes.device
        
        # preope_pcd_src = input['src_pcd_list'] # B, N, C
        # intraOpe_pcd = input['tgt_pcd_list'] # B, N, C
        preMeshes = input['o3d_pre_mesh_list'] # str
        
        # intraOpe_img = input['liver_label_list'] # B, C, H, W
        # print("mymodel 1:{}".format(torch.cuda.memory_allocated(0)))

        pred_data = self.posenet(input, timers)

        # print("mymodel 2:{}".format(torch.cuda.memory_allocated(0)))
        
        # preope_pcd_src = input['s_pcd'] # B, N, C
        preope_pcd_src = input['batched_src_pcd'] # B, N, C
        # preope_pcd_src = torch.stack(input['src_pcd_list'], dim=0)
       
        camera_K_batch = input['cam_k'] # B, N, C
        # intraOpe_pcd = input['t_pcd'] # B, N, C
        intraOpe_pcd = input['batched_tgt_pcd'] # B, N, C
        # intraOpe_pcd = torch.stack(input['tgt_pcd_list'], dim=0)
        assert preope_pcd_src.shape == intraOpe_pcd.shape
        bs, n_pts = intraOpe_pcd.size()[:2]
        # print('preope_pcd_src shape: ', preope_pcd_src.shape)
        # print('intraOpe_pcd shape: ', intraOpe_pcd.shape)

        pred_R = pred_data['R_s2t_pred'] # s2t // w2c
        pred_t = pred_data['t_s2t_pred'] 
        # print('R shape: ', pred_R.shape)
        # print('t shape: ', pred_t.shape)
        # tsfm = torch.eye(4).to(self.device)
        # tsfm[:3,:3]=pred_R
        # tsfm[:3,3]=pred_t.flatten()
        # tsfm = torch.cat([pred_R, pred_t],2)
        # tsfm = torch.cat((tsfm, torch.Tensor([[0,0,0,1]]).to(self.device)), 1)
        # print('tsfm shape: ', tsfm.shape)
        # tsfm = to_tsfm(pred_R, pred_t)
        tsfm = {}
        tsfm['rot'] = pred_R
        tsfm['trans'] = pred_t

        # print("pred_R and pred_t: ", pred_R.requires_grad, pred_t.requires_grad)

        repositioned_mesh = torch.bmm(preope_pcd_src, pred_R.permute(0,2,1)) + pred_t.permute(0,2,1)
        # repositioned_mesh = torch.add(repositioned_mesh, pred_t.permute(0,2,1))
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/repositioned_mesh.txt', repositioned_mesh[0].cpu().numpy())


        ### here we want to project 3D points onto 2D plane ###
        # K = camera_K_batch[0].detach().cpu().numpy()
        # p3d = repositioned_mesh[0].detach().cpu().numpy()
        # p3d = np.dot(input['ocv2blender'][0].cpu().numpy(), p3d.T).T
        # center_p = input['bbx_center'][0].detach().cpu().numpy()
        # # center_p = np.array([9.92050e-5,3.44450e-6,8.48053e-4])
        # scale_factor = 1. / input['scale'][0].detach().cpu().numpy()
        # # scale_factor = 1. / 122.117216
        # centered_cloud = p3d - center_p
        # scaled_cloud = centered_cloud * scale_factor
        # p3d = scaled_cloud + center_p
        
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/repositioned_mesh_scaled.txt', p3d)
        # p2d = np.dot(p3d, K.T)
        # p2d_3 = p2d[:, 2] # 深度数据 [N,1]
        # p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
        # p2d[:, 2] = p2d_3
        # # print("p2ds: {0}".format(p2d))
        # p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
        # white_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        # h, w = white_image.shape[0], white_image.shape[1]
        # color=[(255, 0, 0)]
        # if type(color) == tuple:
        #     color = [color]
        # if len(color) != p2d.shape[0]:
        #     color = [color[0] for i in range(p2d.shape[0])]
        # for pt_2d, c in zip(p2d, color):
        #     pt_2d[0] = np.clip(pt_2d[0], 0, w)
        #     pt_2d[1] = np.clip(pt_2d[1], 0, h)
        #     # print("color: {0}".format(c))
            
        #     white_image = cv2.circle(
        #         white_image, (pt_2d[0], pt_2d[1]), 1, c, -1
        #     )
        # cv2.imwrite('/home/jiking/users/jun/SelfTraining/test_data/projected_p2d_noDef.jpg', white_image)
        # ### here we want to project 3D points onto 2D plane ###
        
        
        

        ## structure similarity ##
        # if preope_pcd_src.shape[1] > 5000:
        #     idx = torch.randperm(preope_pcd_src.shape[1])[:2048]
        #     preope_pcd_src = preope_pcd_src[:,idx,:]
        #     intraOpe_pcd = intraOpe_pcd[:,idx,:]

        intra_model_local = self.geo_local(intraOpe_pcd) # Bs 64, Npts
        intra_model_global = self.intra_model_global(intra_model_local) # Bs 1024, 1

        lowrank_proj = self.lowrank_projection(intra_model_local) # Bs 3, Npts
        # weighted_xyz = torch.sum(lowrank_proj[:, :, :, None] * intraOpe_pcd[:, None, :, :], dim=2)
        # print("shape of these: ", intra_model_local.shape, intra_model_global.shape, lowrank_proj[:, None, :, :].shape, intra_model_local[:, :, None, :].shape)
        weighted_points_features = torch.sum(lowrank_proj[:, None, :, :] * intra_model_local[:, :, None, :], dim=3)

        # print("shape of weighted_points_features: ", weighted_points_features.shape)

        pre_model_local = self.geo_local(preope_pcd_src) ## Bs 64, Npts
        # print("shape of pre_model_local: ", pre_model_local.shape)
        similarity_matrix = self.sim_mat(pre_model_local, weighted_points_features, weighted_points_features)
        pre_model_global = self.pre_model_global(similarity_matrix) ## Bs 1024, 1


        assign_feat = torch.cat((intra_model_local, intra_model_global.repeat(1, 1, n_pts), pre_model_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, n_pts, 3).contiguous()
        # print('before assign_mat: ', assign_mat.shape, assign_mat)
        # # assign_mat = F.softmax(assign_mat, dim=2)
        # print('after assign_mat: ', assign_mat.shape, assign_mat)


        ##### here we need to deform the source pcd #####
        self.deformNet.load_pcds(repositioned_mesh, intraOpe_pcd, assign_mat)
        # print("shape of these: ", repositioned_mesh.shape, intraOpe_pcd.shape)
        warped_pcd, _, _= self.deformNet.register() ## note: the output is the warped pcd, which is in camera coordinate system
        ori_preope_pcd_src_warped = torch.bmm(pred_R.permute(0,2,1), (warped_pcd - pred_t.permute(0,2,1)).permute(0,2,1)).permute(0,2,1)
        # verts_flow = warped_pcd - preope_pcd_src # Bs, N, 3
        verts_flow = ori_preope_pcd_src_warped - preope_pcd_src # Bs, N, 3
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/preope_pcd_src.txt', preope_pcd_src[0].cpu().numpy())
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/ori_preope_pcd_src_warped.txt', ori_preope_pcd_src_warped[0].cpu().numpy())
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/intraOpe_pcd.txt', intraOpe_pcd[0].cpu().numpy())

        # print("shape of these: ", ori_preope_pcd_src_warped.shape, preope_pcd_src.shape, verts_flow.shape)
        ##### here we need to deform the source pcd #####

        weighted_verts_flow = assign_mat * verts_flow
        
        # print("shape of weighted_verts_flow: ", weighted_verts_flow.shape, weighted_verts_flow)
        # print("shape of verts_flow: ", verts_flow.shape, verts_flow)
        # updata_verts_ = preope_pcd_src + weighted_verts_flow # deformed source pcd
        updata_verts_ = preope_pcd_src  # deformed source pcd

        deformed_src_verta = updata_verts_.clone()

        #### comparison results ####
        # updata_verts_ = ori_preope_pcd_src_warped
        #### comparison results ####
        
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/weighted_verts_flow.txt', weighted_verts_flow[0].detach().cpu().numpy())
        # print("updata_verts_ and tsfm: ", updata_verts_.requires_grad, tsfm['rot'].requires_grad, tsfm['trans'].requires_grad)

        # print("tsfm: ", tsfm)
        if self.real_syn == 'real':
            updata_verts_ = torch.bmm(tsfm['rot'], updata_verts_.permute(0,2,1)).permute(0,2,1) + tsfm['trans'].permute(0,2,1) 
            # tsfm['rot'] = torch.bmm(input['ocv2blender'], tsfm['rot']) # opencv
            # print("shape of tsfm['rot'],updata_verts_, tsfm['trans']: ", tsfm['rot'].shape, updata_verts_.shape,tsfm['trans'].shape)
            updata_verts_ = torch.bmm(input['ocv2blender'], updata_verts_.permute(0,2,1)).permute(0,2,1) ## opencv camera coordinate system
            # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/updata_verts_opencv.txt', updata_verts_[0].detach().cpu().numpy())

            # center_p_batch = torch.tensor([9.92050e-5,3.44450e-6,8.48053e-4]).to(self.device).unsqueeze(0) # B, 3
            # center_p_batch = center_p_batch.unsqueeze(1).repeat(bs, n_pts, 1)
            center_p_batch = input['bbx_center'].unsqueeze(1)
            scale = 1. / input['scale'].unsqueeze(1)
            scale = scale.unsqueeze(-1).repeat(1, n_pts, 1)
            # print("updata_verts_ ,center_p_batch: shape: ", updata_verts_.shape, center_p_batch.shape)
            # print("input['scale'] ,scale: shape: ", input['scale'].shape, scale.shape)
            updata_verts_ = updata_verts_ - center_p_batch
            updata_verts_ = updata_verts_ * scale
            updata_verts_ = updata_verts_ + center_p_batch

            # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/updata_verts_oripcd.txt', updata_verts_[0].detach().cpu().numpy())


        mask, depthmap = self.renderer(updata_verts_, tsfm, input['rgbs'], input['img_size'], camera_K_batch, input)
        # print("shape of mask and depthmap: ", mask.shape, depthmap.shape, input['liver_label'].shape)

        # print("what is the mask: ", np.unique(mask[0].detach().cpu().numpy()))
        # print("what is the input['liver_label']: ", np.unique(input['liver_label'][0].detach().cpu().numpy()))
        
        # depth_color, depth_normalized = self.np_depth_to_colormap(depthmap[0].detach().cpu().numpy())
        # import os
        # cv2.imwrite(os.path.join('/home/jiking/users/jun/SelfTraining/test_data', 'rendered_depth.jpg'), (depth_normalized*255).astype(np.uint8))

        rendered_pcd = depth2pointcloud(depthmap, camera_K_batch, downsampling=True, sampling_points = self.num_points)

        rendered_pcd_ocv = rendered_pcd - input['bbx_center'].unsqueeze(1)
        rendered_pcd_ocv = rendered_pcd_ocv * input['scale'].unsqueeze(1).unsqueeze(-1).repeat(1, n_pts, 1)
        rendered_pcd_ocv = rendered_pcd_ocv + input['bbx_center'].unsqueeze(1)
        rendered_pcd_bld = torch.bmm(input['ocv2blender'], rendered_pcd_ocv.permute(0,2,1)).permute(0,2,1) ## blender camera coordinate system

        # print("mask and rendered_pcd_bld: ", rendered_pcd_bld.shape, rendered_pcd_bld.requires_grad)
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/tgt_pcd[0].txt', intraOpe_pcd[0].detach().cpu().numpy())
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/src_pcd[0].txt', preope_pcd_src[0].detach().cpu().numpy())
        # # np.savetxt('/home/jun/Desktop/project/Liver_registration/SelfTraining/test_data/tgt_pcd[1].txt', intraOpe_pcd[1].detach().cpu().numpy())
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/rendered_pcd[0].txt', rendered_pcd[0].detach().cpu().numpy())
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/rendered_pcd_ocv[0].txt', rendered_pcd_ocv[0].detach().cpu().numpy())
        # np.savetxt('/home/jiking/users/jun/SelfTraining/test_data/rendered_pcd_bld[0].txt', rendered_pcd_bld[0].detach().cpu().numpy())
        # # np.savetxt('/home/jun/Desktop/project/Liver_registration/SelfTraining/test_data/rendered_pcd[1].txt', rendered_pcd[1].detach().cpu().numpy())
        # cv2.imwrite('/home/jiking/users/jun/SelfTraining/test_data/rendered_mask[0].png', (mask[0].detach().cpu().numpy() * 255).astype(np.uint8))
        # # cv2.imwrite('/home/jun/Desktop/project/Liver_registration/SelfTraining/test_data/rendered_mask[1].png', (mask[1].detach().cpu().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('/home/jiking/users/jun/SelfTraining/test_data/gt_mask[0].png', (input['liver_label'][0].detach().cpu().numpy() * 255).astype(np.uint8))

        # import sys 
        # sys.exit()

        # # # Render the image using the updated camera position. Based on the new position of the 
        # # camera we calculate the rotation and translation matrices
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        
        # image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        # # Calculate the silhouette loss
        # loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        # print("mymodel 5:{}".format(torch.cuda.memory_allocated(0)))
        pred_data.update({ 'mask': mask})
        # pred_data.update({ 'rendered_dpt': depthmap})
        pred_data.update({ 'rendered_pcd': rendered_pcd})
        pred_data.update({ 'rendered_dpt': depthmap})
        pred_data.update({ 'deformed_src': deformed_src_verta})
        pred_data.update({ 'deformed_src_vis': updata_verts_})
        pred_data.update({ 'rendered_pcd_bld': rendered_pcd_bld})
        

        # print("mymodel 6:{}".format(torch.cuda.memory_allocated(0)))
        
        # assert not torch.any(torch.isnan(mask))
        # assert not torch.any(torch.isnan(depthmap))
        # assert not torch.any(torch.isnan(rendered_pcd))
        
        return pred_data
    


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    from datasets.dataloader import get_dataloader, get_datasets
    from configs.models import architectures
    config_pth = '/home/jiking/users/jun/SelfTraining/configs/train/main_config.yaml'
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
