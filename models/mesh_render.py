import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import sys
import cv2
import numpy as np
from PIL import Image
import yaml
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from lib.vis import project_mesh_to_image

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesAtlas,
    BlendParams,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
)
from pytorch3d.transforms import Transform3d, Rotate, Translate

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from skimage import io
import os
import open3d as o3d
# from .geometry import  depth_2_pc



def opencv_to_pytorch3d(T):
    ''' ajust axis
    :param T: 4x4 mat
    :return:
    '''
    origin = np.array(((-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    origin = torch.from_numpy(origin).float().to(T)
    return T @ origin

def center_normalize_verts(verts):
    print('verts before', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    """ normalize """
    verts -= verts.mean(0, keepdim=True)
    verts /= verts.max()
    print('normalized', verts.max(0)[0], verts.min(0)[0], verts.mean(0))
    return verts

class MeshRender(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.camera = None
        self.device = config.device
        self.real_syn = config.dataset


        # self.img_size = config.img_size # (height, width) of the image

        # self.camera_K = config.camera_K 

        # self.renderer = self.init_camera(config)



    def load_mesh_single_texture(self, obj_filename, device, normalize_verts=False):
        mesh = load_objs_as_meshes(obj_filename, device=device, load_textures = False)

        # print("what is the mesh: ", mesh._verts_list)

        # verts = mesh.verts_packed()
        # faces = mesh.faces_packed()
        # textures = mesh.textures

        # if normalize_verts:
        #     verts = center_normalize_verts(verts)
        # mesh = Meshes(verts=[verts], faces=[faces], textures=mesh.textures)
        return mesh
    
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

    # def init_camera(self,config):
        # elev = 0  # Constant
        # azim = torch.tensor(101.0, requires_grad=True)  # Optimizable parameter
        # azim_goal = 130

        # R_goal, T_goal = look_at_view_transform(
        #     dist=1, elev=elev, azim=azim_goal, up=torch.Tensor((0, -1, 0)).unsqueeze(0)
        # )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        # config.img_size = (540,960)
        H, W = config.img_size[0], config.img_size[1]
        intrinsics = config.camera_K
        image_size = ((H, W),)  # (h, w)
        fcl_screen = ((-intrinsics[0][0], -intrinsics[1][1]),)  # fcl_ndc * min(image_size) / 2
        prp_screen = ((intrinsics[0][2], intrinsics[1][2]), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2

        cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size, device=config.device)

        raster_settings = RasterizationSettings(
            image_size=image_size[0],
            blur_radius= np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100, 
            bin_size=0,
        )

        # lights = AmbientLights(device=device)
        #lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            )

        # Create a Phong renderer by composing a rasterizer and a shader.
        # if args.shader == 'mask':
        shader = pytorch3d.renderer.SoftSilhouetteShader()
        # print('Use mask SoftSilhouetteShader shader')

        renderer = MeshRendererWithFragments(
            rasterizer = rasterizer,
            shader=shader
        )
        return renderer
    
    def exponential_alpha(self, dist_xy, radius=0.03):
        dists2 = dist_xy.permute(0, 3, 1, 2)
        dists_norm = dists2 / (radius * radius)
        dists_norm = dists_norm.clamp(min=0.0)
        alpha = torch.exp(-1 * dists_norm)
        return alpha

    # def topy3d(self, Pose):
    #     w2cR, w2cT = Pose['rot'], Pose['trans'] ## w2c (need to transform it to c2w)
    #     c2wR = w2cR.permute(0, 2, 1).contiguous()
    #     c2wT = torch.bmm((-1) * c2wR, w2cT)
    #     # R = torch.stack([-R[:, 0], R[:, 1], -R[:, 2]], 1) # from RDF to LUF for Rotation
    #     py3d_c2wR = torch.stack([-c2wR[:, :, 0], c2wR[:, :, 1], -c2wR[:, :, 2]], 2) # from RDF to LUF for Rotation
    #     # print("R,t after B2P: ", R.shape, T.shape)
    #     new_w2cR = py3d_c2wR
    #     new_w2cT = torch.bmm((-1)*new_w2cR, c2wT)

    #     # new_c2w = torch.cat([R, T], 1)
    #     # w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))

    #     # R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix
    #     # R = R[None] # batch 1 for rendering
    #     # T = T[None] # batch 1 for rendering
    #     R = new_w2cR
    #     T = new_w2cT.squeeze(-1).contiguous()
    #     return R.contiguous(),T

    def setup_renderer(self, camera):
        # Initialize a camera.
        # print(camera)
        if camera is not None:
            """
            The camera coordinate sysmte in COLMAP is right-down-forward
            Blender is right-up-back
            Pytorch3D is left-up-forward
            """
            ###### test #####
            w2c = camera['w2c']
            c2w = torch.linalg.inv(w2c) 
            R, T = c2w[:3, :3], c2w[:3, 3:]
            # R = torch.stack([-R[:, 0], R[:, 1], -R[:, 2]], 1) # from RDF to LUF for Rotation

            R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from opencv to py3d for Rotation
            
            new_c2w = torch.cat([R, T], 1)
            w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]]).to(device=self.device)), 0))
            R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix

            
            R = R[None] # batch 1 for rendering
            T = T[None] # batch 1 for rendering
            ###### test #####


            # new_c2w = torch.cat([R, T], 1)
            # w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))

            # R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix
            # R = R[None] # batch 1 for rendering
            # T = T[None] # batch 1 for rendering
            # R = new_w2cR
            # T = new_w2cT.squeeze(-1).contiguous()

            # print("shape of R, T: ", R.shape, T.shape)

            """ Downsample images size for faster rendering """
            H, W = int(camera['H']), int(camera['W'])
            # H, W = int(H / args.down), int(W / args.down)

            intrinsics = camera['intrinsics']
            # print("intrinsics: ", intrinsics)

            image_size = ((H, W),)  # (h, w)
            fcl_screen = ((intrinsics[0][0], intrinsics[1][1]),)  # fcl_ndc * min(image_size) / 2
            prp_screen = ((intrinsics[0][2], intrinsics[1][2]), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
            # prp_screen = ((W / 2., H / 2.), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
            # fcl_screen = ((360., 360.),)
            # prp_screen = ((480., 270.), )
    
            cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size, R=R, T=T, device=self.device)

        # print('Camera R T', R, T)

        # Define the settings for rasterization and shading.
        ### try point cloud rendering ###
        raster_settings = PointsRasterizationSettings(
            image_size=image_size[0], 
            radius = 0.02, #0.03
            points_per_pixel = 1,
            # bin_size = 100
        )
        ### try point cloud rendering ###
        # print(image_size)
        # raster_settings = RasterizationSettings(
        #     image_size=image_size[0],
        #     blur_radius=0.0, 
        #     faces_per_pixel=1, 
        # )

        lights = AmbientLights(device=self.device)
        #lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # rasterizer = MeshRasterizer(
        #         cameras=cameras,
        #         raster_settings=raster_settings
        #     )
        rasterizer = PointsRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        # Create a Phong renderer by composing a rasterizer and a shader.
        # if args.shader == 'mask':
        shader = pytorch3d.renderer.SoftSilhouetteShader()
        # print('Use mask SoftSilhouetteShader shader')

        # renderer = MeshRendererWithFragments(
        #     rasterizer = rasterizer,
        #     shader=shader
        # )
        renderer = PointsRenderer(
                rasterizer = rasterizer,
                compositor=AlphaCompositor()
            )

        render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'lights': lights,
                'rasterizer': rasterizer, 'renderer': renderer}

        return render_setup



    def forward(self, deformed_src, Pose, rgbs, img_size, cam_k, input, shader = 'mask'):
        '''
        MESH: The pre-operative model
        Pose: transformation matrix between the object and camera coordinate system
              By default, Pose means [c2w], there 'w' also means the object coordinate system, it is a dict.
        shader: we want to render the mask of the liver model
        '''
        n_batch = deformed_src.shape[0]
        batch_mask = []
        batch_depth = []
        for batch_idx in range(n_batch):
            camera = {}
            camera['R'] = Pose['rot'][batch_idx] # [3x3]
            # camera['R'][:, 0] = -camera['R'][:, 0] # flip x-axis
            # camera['R'][:, 1] = -camera['R'][:, 1] # flip y-axis
            # camera['R'][:, 2] = -camera['R'][:, 2] # flip z-axis

            camera['t'] = Pose['trans'][batch_idx] # [3x1]

            if self.real_syn == 'real':
                camera['R'] = torch.eye(3).to(device=self.device)
                camera['t'] = torch.zeros(3).unsqueeze(-1).to(device=self.device)

            all_pose = torch.cat([camera['R'],camera['t']],1)
            camera['w2c'] = torch.cat((all_pose, torch.Tensor([[0,0,0,1]]).to(device=self.device)), 0)
            # camera['R'] = camera_pose_R
            # camera['t'] = camera_pose_t
            # Rt_transforms = torch.eye(4)
            camera['H'] = img_size[batch_idx][0]
            camera['W'] = img_size[batch_idx][1]
            camera['intrinsics'] = cam_k[batch_idx] # [3x3]
            # print("shape of R, t, H, W, intrinsics: ", camera['R'].shape, camera['t'].shape, camera['H'], camera['W'], camera['intrinsics'].shape)

            render_setup = self.setup_renderer(camera)
            deformed_src_pcd = Pointclouds(points=[deformed_src[batch_idx]], features=[rgbs[batch_idx]])
            renderer = render_setup['rasterizer']
            idx, zbuf, dist_xy = renderer(deformed_src_pcd)

            # mask_renderer = render_setup['renderer']
            # mask = mask_renderer(deformed_src_pcd)[0, ...,:3]

            # Calculate PC coverage
            valid_pts = (idx >= 0).float()
            valid_ray = valid_pts[:, :, :, 0]

            # Calculate composite weights -- dist_xy is squared distance!!
            # Clamp weights to avoid 0 gradients or errors
            weights = self.exponential_alpha(dist_xy, 0.01)
            weights = weights.clamp(min=0.0, max=0.99)
            # Composite the raster for feats and depth
            idx = idx.long().permute(0, 3, 1, 2).contiguous()
            # == Rasterize depth ==
            # zero out weights -- currently applies norm_weighted sum
            w_normed = weights * (idx >= 0).float()
            w_normed = w_normed / w_normed.sum(dim=1, keepdim=True).clamp(min=1e-9)
            z_weighted = zbuf.permute(0, 3, 1, 2).contiguous() * w_normed.contiguous()
            z_weighted = z_weighted.sum(dim=1, keepdim=True)
            # print("idx, zbuf dist_xy and weights, w_normed, z_weighted: ", idx.requires_grad, zbuf.requires_grad, dist_xy.requires_grad, weights.requires_grad,w_normed.requires_grad,z_weighted.requires_grad)
            ## output mask and depth for each batch
            # mask = (valid_ray[0, ...] > 0).contiguous() # [H, W]
            # mask = torch.where(valid_ray[0, ...] > 0, True, False)
            mask =torch.where(valid_ray[0, ...] > 0, torch.ones_like(valid_ray[0, ...],requires_grad=True),torch.zeros_like(valid_ray[0, ...],requires_grad=True))
            depth = z_weighted[0, 0, :, :].contiguous()# [H, W]

            batch_mask.append(mask)
            batch_depth.append(depth)

        batch_depth = torch.stack(batch_depth, dim=0).contiguous()
        batch_mask = torch.stack(batch_mask, dim=0).contiguous()

        


        return batch_mask, batch_depth



if __name__ == '__main__':


    """data path"""
    seq_name = "moose6OK9_AttackTrotRM"
    seq_dir = "/home/liyang/workspace/NeuralTracking/GlobalReg/example/" + seq_name
    depth_name = "cam1_0015.png"
    intr_name = "cam1intr.txt"
    tgt_depth_image_path = os.path.join( seq_dir,"depth", "cam1_0009.png")
    intrinsics_path = os.path.join(seq_dir, intr_name)
    K = np.loadtxt(intrinsics_path)
    tgt_depth = io.imread( tgt_depth_image_path )/1000.
    tgt_pcd = depth_2_pc(tgt_depth,K).transpose(1,2,0)
    tgt_pcd = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float()
    # tgt_pcd = tgt_pcd - tgt_pcd.mean(dim=0, keepdim=True)



    renderer = PCDRender(K)

    img = renderer.render_pcd(tgt_pcd)

    plt.imshow(img[0, ..., :3].cpu().numpy())
    plt.show()
