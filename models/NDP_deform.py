import torch
import torch.nn as nn
import  sys
import numpy as np
from skimage import io

import pytorch3d
import torch.optim as optim
import gc

Runbaselines=True
if Runbaselines:
    from lib.geometry import *
    from geomloss import SamplesLoss

from .NDP_lossF import  arap_cost, landmark_cost, compute_truncated_chamfer_distance, nerfies_regularization, aiap_loss
from .NDP_base import *
from pytorch3d.loss.chamfer import chamfer_distance

sys.path.append("../")
from lib.vis import visualize_pcds


BCE = nn.BCELoss()


class NDP_Deformation():


    def __init__(self, config):


        self.tgt_pcd = None
        self.src_pcd = None
        self.device = config.device
        self.config = config
        self.deformation_model = config.deformation_model




    def load_raw_pcds_from_depth(self, source_depth_path, tgt_depth_path, K, landmarks=None):
        ''' creat deformation graph for N-ICP based registration
        '''

        assert self.deformation_model == "ED"

        self.intrinsics = K

        """initialize deformation graph"""
        depth_image = io.imread(source_depth_path)
        image_size = (depth_image.shape[0], depth_image.shape[1])
        data = get_deformation_graph_from_depthmap( depth_image, K, self.config)
        self.graph_nodes = data['graph_nodes'].to(self.device)
        self.graph_edges = data['graph_edges'].to(self.device)
        self.graph_edges_weights = data['graph_edges_weights'].to(self.device)
        # self.graph_clusters = data['graph_clusters']


        """initialize point clouds"""
        valid_pixels = torch.sum(data['pixel_anchors'], dim=-1) > -4
        self.src_pcd_raw = data["point_image"][valid_pixels].to(self.device)
        self.point_anchors = data["pixel_anchors"][valid_pixels].long().to(self.device)
        self.anchor_weight = data["pixel_weights"][valid_pixels].to(self.device)
        self.anchor_loc = data["graph_nodes"][self.point_anchors].to(self.device)
        self.frame_point_len = [len(self.src_pcd_raw)]


        """pixel to pcd map"""
        self.src_pix_2_pcd_map = [map_pixel_to_pcd(valid_pixels)]


        """define pcd renderer"""
        # self.renderer = PCDRender(K, img_size=image_size)


        """load target frame"""
        tgt_depth = io.imread( tgt_depth_path )/1000.
        depth_mask = torch.from_numpy(tgt_depth > 0)
        tgt_pcd = depth_2_pc(tgt_depth, self.intrinsics).transpose(1,2,0)
        self.tgt_pcd_raw = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float().to(self.device)
        self.tgt_pix_2_pcd_map = map_pixel_to_pcd(depth_mask)

        if landmarks is not None:
            s_uv , t_uv = landmarks
            s_id = self.src_pix_2_pcd_map[-1][s_uv[:, 1], s_uv[:, 0]]
            t_id = self.tgt_pix_2_pcd_map [ t_uv[:,1], t_uv[:,0]]
            valid_id = (s_id>-1) * (t_id>-1)
            s_ldmk = s_id[valid_id]
            t_ldmk = t_id[valid_id]
            self.landmarks = (s_ldmk, t_ldmk)
        else:
            self.landmarks = None


    def load_pcds(self, src, tgt, assign_mat, landmarks=None ):

        if  type(src) in [  np.ndarray] :
            src = torch.from_numpy(src)
            tgt = torch.from_numpy(tgt)

        self.src_pcd = src.to(self.device)
        self.tgt_pcd = tgt.to(self.device)

        
        self.landmarks = landmarks

        self.assign_mat = assign_mat.to(self.device)


    def register(self, **kwargs):
        if self.deformation_model == "Sinkhorn":
            return self.run_optimal_transport( **kwargs)
        
        if self.deformation_model == "ED": # Embeded_deformation, c.f. https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
            return self.optimize_Embeded_deformation( **kwargs)
        
        if self.deformation_model == "NSFP": # Neural scene flow prior, https://arxiv.org/abs/2111.01253
            return self.optimize_neural_SFlow(**kwargs)
        
        if self.deformation_model == "Nerfies": # Nerfies, https://arxiv.org/abs/2011.12948
            return self.optimize_Nerfies(**kwargs)
        
        if self.deformation_model == "NDP": # deformation pyramid
            return self.optimize_deformation_pyramid( **kwargs)
            
        
        if self.deformation_model == "DPF": # Dynamic Point Fields iccv23
            return self.optimize_dynamic_point_fields( **kwargs)

        raise KeyError()
    

    def optimize_dynamic_point_fields(self, visualize=False, timer = None):
        model = Siren(in_features=3,
                hidden_features=128,
                hidden_layers=3,
                out_features=3, outermost_linear=True,
                first_omega_0=30, hidden_omega_0=30.).to(self.device).train()
        l1_loss = torch.nn.L1Loss()
        optm = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        schedm = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, verbose=True, patience=1)

        use_aiap=True
        use_aiap_inter=True
        use_chamfer=True
        use_guidance=False

        if use_chamfer and self.src_pcd is None:
            print("no target cloud provided, ignoring...")
            use_chamfer = False

        if use_guidance and self.tgt_pcd is None:
            print("no keypoints provided, ignoring...")
            use_guidance = False

        guided_loss_total = 0
        chamfer_loss_total = 0
        aiap_loss_total = 0
        total_loss = 0
        n_r = 0
        
        n_chamfer_samples=1024
        aiap_inter_max=1.0
        aiap_weight=1.0e2
        guided_weight=1.0e4
        chamfer_weight=1.0e4
        iso_n_neighbors=5
        vsrc = vtrg = None

        xsrc = self.src_pcd
        xtrg = self.tgt_pcd

        for i in range(0, 2000):

            if use_aiap or use_chamfer or use_aiap_inter:
                # print("xsrc xtrg shape: ", xsrc.shape, xtrg.shape)

                src = torch.randperm(xsrc.shape[0])
                tgt = torch.randperm(xtrg.shape[0])
                xbatch_src = xsrc[src[: n_chamfer_samples]]
                xbatch_trg = xtrg[tgt[: n_chamfer_samples]]

                # xbatch_src = torch.Tensor(xsrc[np.random.choice(xsrc.shape[1], n_chamfer_samples, replace=False)]).to(self.device)
                xbatch_deformed = xbatch_src + model(xbatch_src)

            if use_aiap_inter:
            # additionally enforce isometry between frames
                t = aiap_inter_max * np.random.uniform()
                xbatch_deformed_halfway = xbatch_src + aiap_inter_max*model(xbatch_src)

            loss = 0

            if use_aiap:
                iso_loss = aiap_weight*aiap_loss(xbatch_src, xbatch_deformed, n_neighbors=iso_n_neighbors)
                loss += iso_loss
                aiap_loss_total += float(iso_loss)

            if use_aiap_inter:
                iso_loss += aiap_weight*aiap_loss(xbatch_src, xbatch_deformed_halfway, n_neighbors=iso_n_neighbors)
                loss += iso_loss
                aiap_loss_total += float(iso_loss)

            if use_guidance:
                vsrc_deformed = vsrc + model(vsrc)
                guidance_loss = guided_weight*l1_loss(vsrc_deformed, vtrg)
                loss += guidance_loss
                guided_loss_total += float(guidance_loss)

            if use_chamfer:
                # xbatch_trg = torch.Tensor(xtrg[np.random.choice(len(xtrg), n_chamfer_samples, replace=False)]).to(self.device)
                # print("xbatch_deformed xbatch_trg shape: ", xbatch_deformed.shape, xbatch_trg.shape)
                chamfer_loss = chamfer_weight*chamfer_distance(xbatch_deformed,
                                                                xbatch_trg)[0]
                loss += chamfer_loss
                chamfer_loss_total += float(chamfer_loss)

            total_loss += float(loss)
            n_r += 1

            optm.zero_grad()
            loss.backward()
            optm.step()

        torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        
        warped_pcd = model(xsrc)

        return warped_pcd, None, timer

    def optimize_deformation_pyramid(self, visualize=False, timer = None):

        config = self.config
        max_break_count=config.max_break_count
        break_threshold_ratio=config.break_threshold_ratio


        NDP = Deformation_Pyramid ( depth=config.depth,
                                    width=config.width,
                                    device=self.device,
                                    k0=config.k0,
                                    m=config.m,
                                    nonrigidity_est=config.w_reg > 0,
                                    rotation_format=config.rotation_format,
                                    motion=config.motion_type)


        self.src_pcd = self.src_pcd.to(self.device)

        if visualize:
            visualize_pcds(src_pcd = self.src_pcd, tgt_pcd= self.tgt_pcd)


        # cancel global translation
        
        src_mean = self.src_pcd.mean(dim=1, keepdims=True)
        tgt_mean = self.tgt_pcd.mean(dim=1, keepdims=True)
        src_pcd = self.src_pcd - src_mean
        tgt_pcd = self.tgt_pcd - tgt_mean

        

        src = torch.randperm(src_pcd.shape[0])
        tgt = torch.randperm(tgt_pcd.shape[0])
        s_sample = src_pcd[src[: config.samples]]
        t_sample = tgt_pcd[tgt[: config.samples]]
        
        assign_mat_sample = self.assign_mat[src[: config.samples]]


        if self.landmarks is not None:
            src_ldmk = self.landmarks[0] - src_mean
            tgt_ldmk = self.landmarks[1] - tgt_mean



        iter_cnt={}

        # print("number of levels: ", NDP.n_hierarchy)

        for level in range ( NDP.n_hierarchy):

            """freeze non-optimized level"""
            NDP.gradient_setup(optimized_level=level)


            optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr= self.config.lr )


            break_counter = 0
            loss_prev = 1e+6


            """optimize current level"""
            for iter in range(self.config.iters):

                # use  ldmk
                if self.landmarks is not None:

                    if config.w_cd > 0 :
                        src_pts = torch.cat( [ src_ldmk, s_sample ])
                        warped_pts, data = NDP.warp(src_pts, max_level=level, min_level=level)
                        warped_ldmk = warped_pts [: len(src_ldmk) ]
                        s_sample_warped = warped_pts [ len(src_ldmk):]
                        loss_ldmk =  torch.mean( torch.sum( (warped_ldmk - tgt_ldmk)**2, dim=-1))
                        loss_CD = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=config.trunc_cd)

                        loss = loss_ldmk + config.w_cd * loss_CD


                    else :
                        warped_ldmk, data = NDP.warp(src_ldmk, max_level=level, min_level=level)

                        loss = torch.mean( torch.sum( (warped_ldmk - tgt_ldmk)**2, dim=-1))

                else:

                    if timer: timer.tic("lvl_warp")
                    
                    s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
                    if timer: timer.toc("lvl_warp")

                    

                    
                    
                   
                    if timer: timer.tic("Chamfer")
                    
                    if s_sample_warped.dim() != 3:
                        s_sample_warped = s_sample_warped.unsqueeze(0)
                    
                    loss = compute_truncated_chamfer_distance(s_sample_warped, t_sample, trunc=1e+9)
                    if timer: timer.toc("Chamfer")


                if level > 0 and config.w_reg>0:
                    nonrigidity = data [level][1]
                    target = torch.zeros_like(nonrigidity)
                    reg_loss = BCE( nonrigidity, target )
                    loss = loss + config.w_reg* reg_loss




                # early stop
                if loss.item() < 1e-4:
                    break
                if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
                    break_counter += 1
                if break_counter >= max_break_count:
                    break
                loss_prev = loss.item()

                if timer: timer.tic("backprop")
                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()
                if timer: timer.toc("backprop")
            
            

            # use warped points for next level
            if self.landmarks is not None:
                src_ldmk = warped_ldmk.detach()

                if config.w_cd > 0 :
                    s_sample = s_sample_warped.detach()

            else:
                s_sample = s_sample_warped.detach()


        

        """freeze all level for inference"""
        NDP.gradient_setup(optimized_level=-1)

        torch.cuda.empty_cache()
        gc.collect()
        
        warped_pcd, data = NDP.warp(src_pcd)
        if visualize:
             visualize_pcds(tgt_pcd=tgt_pcd, warped_pcd=warped_pcd, rigidity=data[level][1])

        warped_pcd = warped_pcd + tgt_mean


        return warped_pcd,  iter_cnt, timer

    def optimize_Embeded_deformation(self, visualize=False):


        config = self.config


        if visualize:
            visualize_pcds(src_pcd = self.src_pcd_raw, tgt_pcd= self.tgt_pcd_raw)


        """translations"""
        node_translations = torch.zeros_like(self.graph_nodes)
        t = torch.nn.Parameter(node_translations)
        t.requires_grad = True

        """rotations"""
        phi = torch.zeros_like(self.graph_nodes)
        phi = torch.nn.Parameter (phi)
        phi.requires_grad = True


        """optimizer setup"""
        optimizer = optim.Adam([phi, t], lr= self.config.lr )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)


        max_break_count=config.max_break_count
        break_threshold_ratio=config.break_threshold_ratio
        break_counter = 0
        loss_prev = 1e+6

        samples = config.samples

        for i in range(self.config.iters):


            R = pytorch3d.transforms.axis_angle_to_matrix(phi)


            #sample points for NICP
            src = torch.randperm(self.src_pcd_raw.shape[0])
            tgt = torch.randperm(self.tgt_pcd_raw.shape[0])
            src_smpl_ind = src[:samples]
            tgt_smpl_ind = tgt[:samples]
            s_sample = self.src_pcd_raw[ src_smpl_ind ]
            t_sample = self.tgt_pcd_raw[ tgt_smpl_ind ]
            point_anchors = self.point_anchors[src_smpl_ind]
            anchor_loc = self.anchor_loc[src_smpl_ind]
            anchor_weight = self.anchor_weight[src_smpl_ind]

            anchor_trn = t [point_anchors]
            anchor_rot = R [point_anchors]

            if self.landmarks:
                s_ldmk_ind, t_ldmk_ind = self.landmarks
                s_ldmk = self.src_pcd_raw[s_ldmk_ind]
                t_ldmk = self.tgt_pcd_raw[t_ldmk_ind]

                ldmk_anchors = self.point_anchors[s_ldmk_ind]
                ldmk_anchor_loc = self.anchor_loc[s_ldmk_ind]
                ldmk_anchor_weight = self.anchor_weight[s_ldmk_ind]

                ldmk_anchor_trn = t[ldmk_anchors]
                ldmk_anchor_rot = R[ldmk_anchors]

                s_sample = torch.cat( [s_ldmk, s_sample])
                anchor_loc = torch.cat( [ ldmk_anchor_loc, anchor_loc])
                anchor_rot = torch.cat( [ ldmk_anchor_rot, anchor_rot])
                anchor_trn = torch.cat( [ldmk_anchor_trn, anchor_trn])
                anchor_weight = torch.cat([ldmk_anchor_weight, anchor_weight ])



            warped_pcd = ED_warp(s_sample, anchor_loc, anchor_rot, anchor_trn, anchor_weight)
            
            if warped_pcd.dim() != 3:
                warped_pcd = warped_pcd.unsqueeze(0)

            cd = compute_truncated_chamfer_distance(warped_pcd, t_sample, trunc=1e+10)


            reg = arap_cost(R, t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)


            if self.landmarks:
                warped_ldmk = warped_pcd[ : len(s_ldmk_ind) ]
                ldmk_loss = landmark_cost( warped_ldmk, t_ldmk)
            else :
                ldmk_loss = 0


            loss = \
                cd * config.w_cd + \
                reg * config.w_arap + \
                ldmk_loss * config.w_ldmk



            if loss.item() < 1e-5:
                break
            if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
                break_counter += 1
            if break_counter >= max_break_count:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


        "warp the full point cloud"
        anchor_trn = t [self.point_anchors]
        anchor_rot = R [self.point_anchors]

        warped_pcd = ED_warp(self.src_pcd_raw, self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)


        if visualize: visualize_pcds(src_pcd = self.src_pcd_raw, tgt_pcd= self.tgt_pcd_raw, warped_pcd=warped_pcd)


        # propagate motion to sampled points
        s_uv = pc_2_uv(self.src_pcd, self.intrinsics)
        s_id = self.src_pix_2_pcd_map[-1][s_uv[:, 1], s_uv[:, 0]]
        valid_id = s_id > -1
        warped_pcd = warped_pcd[s_id[valid_id]]

        return warped_pcd, valid_id


    def optimize_neural_SFlow(self, visualize=False):

        config = self.config
        max_break_count=config.max_break_count_NSFP
        break_threshold_ratio=config.break_threshold_ratio_NSFP


        model = Neural_Prior ( ).to(self.device)


        self.src_pcd = self.src_pcd.to(self.device)

        # cancel global translation
        src_mean = self.src_pcd.mean(dim=0, keepdims=True)
        tgt_mean = self.tgt_pcd.mean(dim=0, keepdims=True)
        src_pcd = self.src_pcd - src_mean
        tgt_pcd = self.tgt_pcd - tgt_mean


        if visualize:
            visualize_pcds(src_pcd = src_pcd, tgt_pcd= tgt_pcd)


        src = torch.randperm(src_pcd.shape[0])
        tgt = torch.randperm(tgt_pcd.shape[0])
        s_sample = src_pcd[src[: config.samples_NSFP]]
        t_sample = tgt_pcd[tgt[: config.samples_NSFP]]


        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr= self.config.lr_NSFP )

        break_counter = 0
        loss_prev = 1e+6


        for i in range(self.config.iters_NSFP):

            flow_pred = model(s_sample)
            s_sample_warped = s_sample + flow_pred
            if s_sample_warped.dim() != 3:
                s_sample_warped = s_sample_warped.unsqueeze(0)
            loss = compute_truncated_chamfer_distance(s_sample_warped, t_sample, trunc=1e+9)

            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
                break_counter += 1
            if break_counter >= max_break_count:
                break
            loss_prev = loss.item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if visualize:
            # warped_pcd, data = NDP.warp(src_pcd, max_level=level)
            flow_pred = model(src_pcd)
            warped_pcd = src_pcd + flow_pred
            visualize_pcds(  tgt_pcd= tgt_pcd, warped_pcd = warped_pcd)
            tmp = 0

        flow_pred = model(src_pcd)
        warped_pcd = src_pcd + flow_pred
        warped_pcd = warped_pcd + tgt_mean

        return warped_pcd, None, None
 
    def run_optimal_transport(self, visualize=False):
        '''
        refer to https://www.kernel-operations.io/geomloss/index.html#
        :return:
        '''
        if visualize:
            visualize_pcds(src_pcd =  self.src_pcd, tgt_pcd= self.tgt_pcd)
        config = self.config
        loss = SamplesLoss("sinkhorn", p=2, blur=config.blur_Sinkhorn, reach=config.reach_Sinkhorn )
        src_pcd = self.src_pcd #- src_mean
        tgt_pcd = self.tgt_pcd #- tgt_mean
        src = torch.randperm(src_pcd.shape[0])
        tgt = torch.randperm(tgt_pcd.shape[0])
        s_sample = src_pcd[src[: config.samples_Sinkhorn]]
        t_sample = tgt_pcd[tgt[: config.samples_Sinkhorn]]
        select_ind = torch.arange(src_pcd.shape[0]).long()
        select_ind = select_ind[src[: config.samples_Sinkhorn]]
        X_i = s_sample
        Y_j = t_sample
        x_i, y_j = X_i.clone(), Y_j.clone()
        x_i.requires_grad = True
        for i in range(config.Nsteps_Sinkhorn):  # Euler scheme ===============
            # Compute cost and gradient
            L_αβ = loss(x_i, y_j)
            [g] = torch.autograd.grad(L_αβ, [x_i])
            # in-place modification of the tensor's values
            x_i.data -= config.lr_Sinkhorn * len(x_i) * g
        if visualize:
            visualize_pcds(warped_pcd =  x_i, tgt_pcd= self.tgt_pcd)
        return  x_i, select_ind, None
    
    def optimize_Nerfies(self, visualize=False):
        '''
        :param landmarks:
        :return:
        '''



        config = self.config

        net = Nerfies_Deformation( max_iter=config.iters_Nerfies).to(self.device)

        for param in net.parameters():
            param.requires_grad = True


        """optimizer setup"""
        optimizer = optim.Adam(net.parameters(), lr= config.lr_Nerfies )

        # cancel global translation
        src_mean = self.src_pcd.mean(dim=0, keepdims=True)
        tgt_mean = self.tgt_pcd.mean(dim=0, keepdims=True)
        src_pcd = self.src_pcd - src_mean
        tgt_pcd = self.tgt_pcd - tgt_mean

        if visualize: visualize_pcds(src_pcd = src_pcd, tgt_pcd= tgt_pcd)


        src = torch.randperm(src_pcd.shape[0])
        tgt = torch.randperm(tgt_pcd.shape[0])
        s_sample = src_pcd[src[: config.samples_Nerfies]]
        t_sample = tgt_pcd[tgt[: config.samples_Nerfies]]



        max_break_count=config.max_break_count_Nerfies
        break_threshold_ratio=config.break_threshold_ratio_Nerfies
        break_counter = 0
        loss_prev = 1e+6

        for i in range(self.config.iters_Nerfies):

            warped_src, Jacobian = net(s_sample, iter =i)


            reg = nerfies_regularization(Jacobian)
            # ldmk_loss = torch.mean(  torch.sum( (warped_ldmk - tgt_ldmk)**2, dim=-1 ))
            if warped_src.dim() != 3:
                warped_src = warped_src.unsqueeze(0)

            cd = compute_truncated_chamfer_distance(warped_src, t_sample, trunc=1e+9)

            loss = cd + 0.001 * reg

            if visualize: print(i, loss)
            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
                break_counter += 1
            if break_counter >= max_break_count:
                break
            loss_prev = loss.item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        warped_pcd, _ = net(src_pcd, iter = i)


        if visualize: visualize_pcds(warped_pcd = warped_pcd, tgt_pcd= tgt_pcd)


        return warped_pcd + tgt_mean, None, None
