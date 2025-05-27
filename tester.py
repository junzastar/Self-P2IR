from trainer import Trainer
import torch
from tqdm import tqdm
from lib.loss import MatchMotionLoss as MML
import numpy as np
from models.lepard.matching import Matching as CM
from lib.loss import  ChamferLoss_l2, ChamferLoss_l1, BinaryDiceLoss
from extensions.chamfer_distance.chamfer_distance import ChamferDistance
import math
import os
import cv2
import open3d as o3d


class _synTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self, args)
        self.args = args


    def test(self):

        n = 1

        afmr = 0.
        arr_0_04 = 0
        arr_0_02 = 0
        arr_0_01 = 0
        arr_0_005 = 0
        arr_0_002 = 0
        air = 0
        arre = 0
        arte = 0

        for i in range(n): # combat ransac nondeterministic

            thr =0.15
            rr, ir, fmr, rre, rte = self.test_thr(thr)
            
            print("##### Results ###### \n")
            print( "conf_threshold: {}. \n".format(thr))
            print("registration recall < 0.04: {0}, < 0.02: {1}, < 0.01: {2}, < 0.005: {3}, < 0.002: {4}. \n"
                  .format(rr['recall_0_04'], rr['recall_0_02'],rr['recall_0_01'], rr['recall_0_005'],rr['recall_0_002']))
            print(" Inlier rate: {}. \n".format(ir))
            print("FMR: {}. \n".format(fmr)) 
            print('rotational error: {}. \n'.format(rre)) 
            print('translational error: {}. \n'.format(rte))
            
        

    def test_thr(self, conf_threshold=None):

        

        num_iter = math.ceil(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()


        self.model.eval()




        success_0_04 = 0.
        success_0_02 = 0.
        success_0_01 = 0.
        success_0_005 = 0.
        success_0_002 = 0.
        
        IR_0_01=0.
        IR_0_015=0.
        IR_0_005=0.
        IR_0_002=0.
        FMR_0_05=0.
        FMR_0_1=0.
        FMR_0_15=0.
        FMR_0_2=0.

        FMR_0_05_IR_0_015=0.
        FMR_0_05_IR_0_01=0.
        FMR_0_05_IR_0_005=0.
        FMR_0_05_IR_0_002=0.

        RRE = 0.
        RTE = 0.

        ## ablation study: overlap & registration recall ###
        overlap_list = []
        regis_rrlist_002 = []
        regis_rrlist_005 = []
        regis_rrlist_01 = []
        regis_rrlist_02 = []
        regis_rrlist_04 = []

        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch

                ##################################
                if self.timers: self.timers.tic('load batch')
                inputs = c_loader_iter.next()
                for k, v in inputs.items():
                    if type(v) == list:
                        if type(v[0]) in [str, np.ndarray]:
                            pass
                        else:
                            inputs [k] = [item.to(self.device) for item in v]
                    elif type(v) in [ dict, float, type(None), np.ndarray]:
                        pass
                    else:
                        inputs [k] = v.to(self.device)
                if self.timers: self.timers.toc('load batch')
                ##################################


                if self.timers: self.timers.tic('forward pass')
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                if self.timers: self.timers.toc('forward pass')



                match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=False)
                
                rot, trn, corrs_array, RR = MML.ransac_regist_coarse(data['s_pcd'], data['t_pcd'], data['src_mask'], data['tgt_mask'], match_pred, data)
                
                

                ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=0.1, idx=idx)
                
                # direction, which is more representative of the actual error)
                from lib import se3
                from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences
                
                assert rot.shape[0] == 1
                tsfm_est = torch.eye(4)
                tsfm_est[:3,:3]=rot[0]
                tsfm_est[:3,3]=trn[0].flatten()
                batched_rot = inputs['batched_rot']  # B,3,3
                batched_trn = inputs['batched_trn']
                gt_transforms = torch.eye(4)
                gt_transforms[:3,:3]=batched_rot[0]
                gt_transforms[:3,3]=batched_trn[0].flatten()


                residual_rotdeg, residual_transmag = compute_rre_rte(gt_transforms.unsqueeze(0).cpu().numpy(), tsfm_est.unsqueeze(0).cpu().numpy())
                
                

                vis = False
                if vis:
                    pcd = data['points'][0].cpu().numpy()
                    lenth = data['stack_lengths'][0][0]
                    spcd, tpcd = pcd[:lenth] , pcd[lenth:]

                    import mayavi.mlab as mlab
                    c_red = (224. / 255., 0 / 255., 125 / 255.)
                    c_pink = (224. / 255., 75. / 255., 232. / 255.)
                    c_blue = (0. / 255., 0. / 255., 255. / 255.)
                    scale_factor = 0.02
                    # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
                    mlab.points3d(spcd[:, 0], spcd[:, 1], spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.show()

                    spcd = ( np.matmul(rot, spcd.T) + trn ).T
                    mlab.points3d(spcd[:, 0], spcd[:, 1], spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.show()

                
                bs = len(rot)
                assert  bs==1
                success_0_04 += bs * RR['rr_0_04']
                success_0_02 += bs * RR['rr_0_02']
                success_0_01 += bs * RR['rr_0_01']
                success_0_005 += bs * RR['rr_0_005']
                success_0_002 += bs * RR['rr_0_002']

                IR_0_01 += bs*ir['ir_0_01']
                IR_0_015 += bs*ir['ir_0_015']
                IR_0_005 += bs*ir['ir_0_005']
                IR_0_002 += bs*ir['ir_0_002']
                
                FMR_0_05 += (ir['ir_0_01']>0.05).float()
                FMR_0_1 += (ir['ir_0_01']>0.1).float()
                FMR_0_15 += (ir['ir_0_01']>0.15).float()
                FMR_0_2 += (ir['ir_0_01']>0.2).float()

                ## 
                FMR_0_05_IR_0_015 += (ir['ir_0_015']>0.05).float()
                FMR_0_05_IR_0_01 += (ir['ir_0_01']>0.05).float()
                FMR_0_05_IR_0_005 += (ir['ir_0_005']>0.05).float()
                FMR_0_05_IR_0_002 += (ir['ir_0_002']>0.05).float()

                RRE += residual_rotdeg
                RTE += residual_transmag



            recall = {}
            recall['recall_0_04'] = success_0_04/len(self.loader['test'].dataset)
            recall['recall_0_02'] = success_0_02/len(self.loader['test'].dataset)
            recall['recall_0_01'] = success_0_01/len(self.loader['test'].dataset)
            recall['recall_0_005'] = success_0_005/len(self.loader['test'].dataset)
            recall['recall_0_002'] = success_0_002/len(self.loader['test'].dataset)
            IRate = {}
            IRate['ir_0_01'] = IR_0_01/len(self.loader['test'].dataset)
            IRate['ir_0_015'] = IR_0_015/len(self.loader['test'].dataset)
            IRate['ir_0_005'] = IR_0_005/len(self.loader['test'].dataset)
            IRate['ir_0_002'] = IR_0_002/len(self.loader['test'].dataset)
            FMR = {}
            FMR['fmr_0_05'] = FMR_0_05/len(self.loader['test'].dataset)
            FMR['fmr_0_1'] = FMR_0_1/len(self.loader['test'].dataset)
            FMR['fmr_0_15'] = FMR_0_15/len(self.loader['test'].dataset)
            FMR['fmr_0_2'] = FMR_0_2/len(self.loader['test'].dataset)
            
            FMR['fmr_0_05_ir_0_015'] = FMR_0_05_IR_0_015/len(self.loader['test'].dataset)
            FMR['fmr_0_05_ir_0_01'] = FMR_0_05_IR_0_01/len(self.loader['test'].dataset)
            FMR['fmr_0_05_ir_0_005'] = FMR_0_05_IR_0_005/len(self.loader['test'].dataset)
            FMR['fmr_0_05_ir_0_002'] = FMR_0_05_IR_0_002/len(self.loader['test'].dataset)
            # FMR = FMR/len(self.loader['test'].dataset)
            RRE = RRE/len(self.loader['test'].dataset)
            RTE = RTE/len(self.loader['test'].dataset)

            return recall, IRate, FMR, RRE, RTE

def compute_rre_rte(gt_transforms: np.ndarray, pred_transforms: np.ndarray):
    """Compute the Relative Rotation Error (RRE) and Relative Translation Error (RTE) between ground truth and predicted transforms.

    Args:
        gt_transforms: Ground truth transforms of size (N, 4, 4)
        pred_transforms: Predicted transforms of size (N, 4, 4)

    Returns:
        rre: Relative Rotation Error
        rte: Relative Translation Error
    """
    assert gt_transforms.shape == pred_transforms.shape, "Input shapes do not match"

    num_transforms = gt_transforms.shape[0]
    rre_sum = 0.0
    rte_sum = 0.0

    for i in range(num_transforms):
        gt_rot = gt_transforms[i, :3, :3]
        pred_rot = pred_transforms[i, :3, :3]
        gt_trans = gt_transforms[i, :3, 3]
        pred_trans = pred_transforms[i, :3, 3]

        # Compute rotation error using the trace of the rotation matrix
        trace = np.trace(np.dot(gt_rot.T, pred_rot))
        trace = np.clip(trace, -1.0, 3.0)  # Clip to ensure valid input for arccos
        angle_diff = np.arccos((trace - 1.0) / 2.0)
        rre_sum += angle_diff

        # Compute translation error as Euclidean distance
        trans_diff = np.linalg.norm(gt_trans - pred_trans)
        rte_sum += trans_diff

    rre = rre_sum / num_transforms
    rte = rte_sum / num_transforms

    return rre, rte

def blend_anchor_motion (query_loc, reference_loc, reference_flow , knn=3, search_radius=0.1) :
    '''approximate flow on query points
    this function assume query points are sub- or un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    from datasets.dataset_utils import knn_point_np
    dists, idx = knn_point_np (knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    mask = dists>search_radius
    dists[mask] = 1e+10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    mask = mask.sum(axis=1)<3

    return blended_flow, mask

def compute_nrfmr( match_pred, data, recall_thr=0.04):


    s_pcd, t_pcd = data['s_pcd'], data['t_pcd']

    s_pcd_raw = data ['src_pcd_list']
    sflow_list = data['sflow_list']
    metric_index_list = data['metric_index_list']

    batched_rot = data['batched_rot']  # B,3,3
    batched_trn = data['batched_trn']


    nrfmr = 0.

    for i in range ( len(s_pcd_raw)):

        # get the metric points' transformed position
        metric_index = metric_index_list[i]
        sflow = sflow_list[i]
        s_pcd_raw_i = s_pcd_raw[i]
        metric_pcd = s_pcd_raw_i [ metric_index ]
        metric_sflow = sflow [ metric_index ]
        metric_pcd_deformed = metric_pcd + metric_sflow
        metric_pcd_wrapped_gt = ( torch.matmul( batched_rot[i], metric_pcd_deformed.T) + batched_trn[i] ).T


        # use the match prediction as the motion anchor
        match_pred_i = match_pred[ match_pred[:, 0] == i ]
        s_id , t_id = match_pred_i[:,1], match_pred_i[:,2]
        s_pcd_matched= s_pcd[i][s_id]
        t_pcd_matched= t_pcd[i][t_id]
        motion_pred = t_pcd_matched - s_pcd_matched
        metric_motion_pred, valid_mask = blend_anchor_motion(
            metric_pcd.cpu().numpy(), s_pcd_matched.cpu().numpy(), motion_pred.cpu().numpy(), knn=3, search_radius=0.1)
        metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)

        debug = False
        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            metric_pcd_wrapped_gt = metric_pcd_wrapped_gt.cpu()
            metric_pcd_wrapped_pred = metric_pcd_wrapped_pred.cpu()
            err = metric_pcd_wrapped_pred - metric_pcd_wrapped_gt
            mlab.points3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(metric_pcd_wrapped_pred[ :, 0] , metric_pcd_wrapped_pred[ :, 1], metric_pcd_wrapped_pred[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.quiver3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], err[:, 0], err[:, 1], err[:, 2],
                          scale_factor=1, mode='2ddash', line_width=1.)
            mlab.show()

        dist = torch.sqrt( torch.sum( (metric_pcd_wrapped_pred - metric_pcd_wrapped_gt)**2, dim=1 ) )

        r = (dist < recall_thr).float().sum() / len(dist)
        nrfmr = nrfmr + r

    nrfmr = nrfmr /len(s_pcd_raw)

    return  nrfmr

class _realTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self, args)

    def test(self):

        
        import time
        start = time.time()
        aver_cd, aver_dice, std_cd, std_dice = self.test_thr()
        print("Mean dice: {:.8f}: +- {:.8f}\n".format(aver_dice, std_dice))
        print("Mean CD: {:.8f}: +- {:.8f}\n".format(aver_cd, std_cd))
        print( "time costs:", time.time() - start)

    def test_thr(self, conf_threshold=None):

        num_iter = math.ceil(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()


        self.model.eval()


        assert self.loader['test'].batch_size == 1

        DICE=[]
        CD= []
        OverLAP =[]

       
        DSC = BinaryDiceLoss()
        CD_ = ChamferDistance()

        n_sample = 0.

        # with torch.no_grad():
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if 'deformNet' in name:
                param.requires_grad = True
        for idx in tqdm(range(num_iter)): # loop through this epoch



            ##################################
            if self.timers: self.timers.tic('load batch')
            inputs = c_loader_iter.next()
            for k, v in inputs.items():
                if type(v) == list:
                    if type(v[0]) in [str, np.ndarray, None]:
                        pass
                    else:
                        inputs [k] = [item.to(self.device) for item in v]
                elif type(v) in [ dict, float, type(None), np.ndarray]:
                    pass
                else:
                    inputs [k] = v.to(self.device)
            if self.timers: self.timers.toc('load batch')
            ##################################


            if self.timers: self.timers.tic('forward pass')
            data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
            if self.timers: self.timers.toc('forward pass')

            with torch.no_grad():

                pred_R = data['R_s2t_pred'] # s2t // w2c
                pred_t = data['t_s2t_pred']
                

                deformed_src = data['deformed_src']
                tsfm_deformed_src = torch.bmm(pred_R, deformed_src.permute(0,2,1)).permute(0,2,1) + pred_t.permute(0,2,1)
                
                dist1, dist2 = CD_(data['batched_tgt_pcd'], tsfm_deformed_src)
                dist = (dist1 + dist2) * 1000.0 # mm
                
                dice = 1. - DSC(data['mask'], data['liver_label']) # 0-1
                dice = dice * 100.0 # percentage
                

                which_patient = inputs['which_patient'][0]
                print("Patient {0}, the {1} item: dice: {2}, CD: {3}".format(which_patient, idx, dice, torch.mean(dist).item()))
                
                CD.append(torch.mean(dist).item())
                DICE.append(dice.item())
        
        aver_cd = np.mean(CD)
        aver_dice = np.mean(DICE)
        std_cd = np.std(CD)
        std_dice = np.std(DICE)

        if self.timers: self.timers.print()

        return aver_cd, aver_dice, std_cd, std_dice




def get_trainer(config):
    if config.dataset == 'syn':
        return _synTester(config)
    elif config.dataset == 'real':
        return _realTester(config)
    else:
        raise NotImplementedError