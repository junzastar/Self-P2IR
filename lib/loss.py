import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd
# from lib.visualization import *
import nibabel.quaternions as nq
from sklearn.metrics import precision_recall_fscore_support
from lib.utils import blend_scene_flow, multual_nn_correspondence, knn_point_np
from models.lepard.matching import Matching as CM


from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance
from torch.nn.modules.loss import _Loss
CD = ChamferDistance()
EMD = EarthMoverDistance()

class ChamferLoss_l1(_Loss):
    def __init__(self):
        super(ChamferLoss_l1, self).__init__(True)

    def cd_loss_L1(self, pcs1, pcs2):
        """
        L1 Chamfer Distance.
        Args:
            pcs1 (torch.tensor): (B, N, 3)
            pcs2 (torch.tensor): (B, M, 3)
        """
        dist1, dist2 = CD(pcs1, pcs2)
        assert not torch.any(torch.isnan(dist1))
        assert not torch.any(torch.isnan(dist2))
        # print("loss this point >>>>>>>>>>>>>>: dist1 and dist2: ", dist1, dist2)
        dist1 = torch.sqrt(dist1 + 1e-9)
        dist2 = torch.sqrt(dist2 + 1e-9)
        return (torch.mean(dist1) + torch.mean(dist2)) / 2.0

    def forward(self, pcs1, pcs2):
        loss = self.cd_loss_L1(pcs1, pcs2)
        return loss

class ChamferLoss_l2(_Loss):
    def __init__(self):
        super(ChamferLoss_l2, self).__init__(True)
    def cd_loss_L2(self, pcs1, pcs2):
        """
        L2 Chamfer Distance.
        Args:
            pcs1 (torch.tensor): (B, N, 3)
            pcs2 (torch.tensor): (B, M, 3)
        """
        dist1, dist2 = CD(pcs1, pcs2)
        return torch.mean(dist1) + torch.mean(dist2)
    def forward(self,pcs1, pcs2):
        loss = self.cd_loss_L2(pcs1, pcs2)
        return loss
    
class silhouetteLoss(_Loss):
    def __init__(self):
        super(silhouetteLoss, self).__init__(True)
    def forward(self, pred, gt_ref):
                
        binary_mask = (pred > 1e-8).float()
        # print("shape of this :", pred.shape, gt_ref.shape, binary_mask)
        loss = F.binary_cross_entropy(pred, gt_ref, weight=binary_mask)
        # loss = torch.mean((pred - gt_ref) ** 2)
        # assert not torch.any(torch.isnan(loss))
        return loss

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        dice = (2. * intersection + self.smooth) / den

        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DepthL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        # assert not torch.any(torch.isnan(pred))
        # assert not torch.any(torch.isnan(gt))
        bs = pred.size()[0]
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        mask = gt > self.eps
        img1[~mask] = 0.
        img2[~mask] = 0.
        # return nn.L1Loss(reduction="sum")(img1, img2), pred.numel()
        loss = nn.L1Loss(reduction="sum")(img1, img2)
        loss = loss / mask.float().sum() * bs
        return loss


class DepthL2Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL2Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        mask = gt > self.eps
        img1[~mask] = 0.
        img2[~mask] = 0.
        return nn.MSELoss(reduction="sum")(img1, img2), pred.numel()


def ransac_pose_estimation(src_pcd, tgt_pcd, corrs, distance_threshold=0.05, ransac_n=3):
    src_pcd = to_o3d_pcd(src_pcd)
    tgt_pcd = to_o3d_pcd(tgt_pcd)
    corrs = o3d.utility.Vector2iVector(np.array(corrs).T)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=src_pcd, target=tgt_pcd, corres=corrs,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
    return result_ransac.transformation


def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html
    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,4,4]
    Returns:
    p (float): transformation error
    """

    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]

    return p.item()


class MatchMotionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.focal_alpha = config['focal_alpha']
        self.focal_gamma = config['focal_gamma']
        self.pos_w = config['pos_weight']
        self.neg_w = config['neg_weight']
        self.mot_w = config['motion_weight']
        self.mat_w = config['match_weight']
        self.motion_loss_type = config['motion_loss_type']

        self.match_type = config['match_type']
        self.positioning_type = config['positioning_type']


        self.registration_threshold = config['registration_threshold']

        self.confidence_threshold_metric = config['confidence_threshold_metric']
        self.inlier_thr = config['inlier_thr']
        self.fmr_thr = config['fmr_thr']
        self.mutual_nearest = config['mutual_nearest']
        self.dataset = config['dataset']


    def forward(self, data):
        loss_info = {}
        loss = self.ge_coarse_loss(data, loss_info)
        loss_info.update({ 'rigid_loss': loss})
        return loss_info


    def ge_coarse_loss(self, data, loss_info, eval_metric=False):


        if self.dataset == "4dmatch":
            s2t_flow = torch.zeros_like(data['s_pcd'])
            for i, cflow in enumerate(data['coarse_flow']):
                s2t_flow[i][: len(cflow)] = cflow

        loss = 0.

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        conf_matrix_pred = data['conf_matrix_pred']
        match_gt = data['coarse_matches']
        R_s2t_gt = data['batched_rot']
        t_s2t_gt = data['batched_trn']
        
        #get the overlap mask, for dense motion loss
        s_overlap_mask = torch.zeros_like(src_mask).bool()
        for bi, corr in enumerate (match_gt):
            s_overlap_mask[bi][ corr[0] ] = True
        # compute focal loss
        c_weight = (src_mask[:, :, None] * tgt_mask[:, None, :]).float()
        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        focal_coarse = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=c_weight)
        recall, precision = self.compute_match_recall( conf_matrix_gt, data['coarse_match_pred'])
        loss_info.update( { "focal_coarse": focal_coarse, "recall_coarse": recall, "precision_coarse": precision } )
        
        loss = loss + self.mat_w * focal_coarse

        

        if recall > 0.01 and self.mot_w > 0:
        # if self.mot_w > 0:
            R_s2t_pred = data["R_s2t_pred"]
            t_s2t_pred = data["t_s2t_pred"]

            #compute predicted flow. Note, if 4dmatch, the R_pred,t_pred try to find the best rigid fit of deformation
            src_pcd_wrapped_pred = (torch.matmul(R_s2t_pred, data['s_pcd'].transpose(1, 2)) + t_s2t_pred).transpose(1, 2)
            sflow_pred = src_pcd_wrapped_pred - data['s_pcd']


            if self.dataset == '4dmatch':
                spcd_deformed = data['s_pcd'] + s2t_flow
                src_pcd_wrapped_gt = (torch.matmul(R_s2t_gt, spcd_deformed.transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
            else : # 3dmatch
                src_pcd_wrapped_gt = (torch.matmul(R_s2t_gt, data['s_pcd'].transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
            sflow_gt = src_pcd_wrapped_gt - data['s_pcd']

            e1 = torch.sum(torch.abs(sflow_pred - sflow_gt), 2)
            e1 = e1[s_overlap_mask] # [data['src_mask']]
            l1_loss = torch.mean(e1)
            loss = loss + self.mot_w * l1_loss

        if self.positioning_type == "procrustes":

            for layer_ind in data["position_layers"]:
                # compute focal loss
                rpe_conf_matrix = data["position_layers"][layer_ind]["conf_matrix"]
                focal_rpe = self.compute_correspondence_loss(rpe_conf_matrix, conf_matrix_gt, weight=c_weight)
                recall, precision = self.compute_match_recall(conf_matrix_gt,
                                                              data["position_layers"][layer_ind]['match_pred'])
                
                loss = loss + self.mat_w * focal_rpe
               

                if recall >0.01 and self.mot_w > 0:
                # if self.mot_w > 0:
                    R_s2t_pred = data["position_layers"][layer_ind]["R_s2t_pred"]
                    t_s2t_pred = data["position_layers"][layer_ind]["t_s2t_pred"]

                    src_pcd_wrapped_pred = (torch.matmul(R_s2t_pred, data['s_pcd'].transpose(1, 2)) + t_s2t_pred).transpose(1, 2)
                    sflow_pred = src_pcd_wrapped_pred - data['s_pcd']


                    if self.dataset == '4dmatch':
                        spcd_deformed = data['s_pcd'] + s2t_flow
                        src_pcd_wrapped_gt = ( torch.matmul(R_s2t_gt, spcd_deformed.transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
                    else:  # 3dmatch
                        src_pcd_wrapped_gt = ( torch.matmul(R_s2t_gt, data['s_pcd'].transpose(1, 2)) + t_s2t_gt).transpose(1, 2)
                    sflow_gt = src_pcd_wrapped_gt - data['s_pcd']

                    e1 = torch.sum(torch.abs(sflow_pred - sflow_gt), 2) #[data['src_mask']]
                    e1 = e1[s_overlap_mask]  # [data['src_mask']]
                    l1_loss = torch.mean(e1)
                    loss = loss + self.mot_w * l1_loss
                    # print('step3 loss : ', loss)

        return loss


    @staticmethod
    def compute_nrfmr(match_pred, data, recall_thr=0.04):

        s_pcd, t_pcd = data['s_pcd'], data['t_pcd']

        s_pcd_raw = data['src_pcd_list']
        sflow_list = data['sflow_list']
        metric_index_list = data['metric_index_list']

        batched_rot = data['batched_rot']  # B,3,3
        batched_trn = data['batched_trn']

        nrfmr = 0.

        for i in range(len(s_pcd_raw)):

            # use the match prediction as the motion anchor
            match_pred_i = match_pred[match_pred[:, 0] == i]
            s_id, t_id = match_pred_i[:, 1], match_pred_i[:, 2]
            s_pcd_matched = s_pcd[i][s_id]
            t_pcd_matched = t_pcd[i][t_id]
            motion_pred = t_pcd_matched - s_pcd_matched

            if len(s_pcd_matched) >= 3 :

                # get the wrapped metric points
                metric_index = metric_index_list[i]
                sflow = sflow_list[i]
                s_pcd_raw_i = s_pcd_raw[i]
                metric_pcd = s_pcd_raw_i[metric_index]
                metric_sflow = sflow[metric_index]
                metric_pcd_deformed = metric_pcd + metric_sflow
                metric_pcd_wrapped_gt = (torch.matmul(batched_rot[i], metric_pcd_deformed.T) + batched_trn[i]).T

                # blend the motion for metric points
                try:
                    metric_motion_pred, valid_mask = MatchMotionLoss.blend_anchor_motion(
                        metric_pcd.cpu().numpy(), s_pcd_matched.cpu().numpy(), motion_pred.cpu().numpy(), knn=3,
                        search_radius=0.1)
                    metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)
                    dist = torch.sqrt(torch.sum((metric_pcd_wrapped_pred - metric_pcd_wrapped_gt) ** 2, dim=1))
                    r = (dist < recall_thr).float().sum() / len(dist)
                except :
                    r = 0

                nrfmr = nrfmr + r


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
                    mlab.points3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2],
                                  scale_factor=scale_factor, color=c_pink)
                    mlab.points3d(metric_pcd_wrapped_pred[:, 0], metric_pcd_wrapped_pred[:, 1],
                                  metric_pcd_wrapped_pred[:, 2], scale_factor=scale_factor, color=c_blue)
                    mlab.quiver3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2],
                                  err[:, 0], err[:, 1], err[:, 2],
                                  scale_factor=1, mode='2ddash', line_width=1.)
                    mlab.show()


        nrfmr = nrfmr / len(s_pcd_raw)

        return nrfmr

    @staticmethod
    def blend_anchor_motion(query_loc, reference_loc, reference_flow, knn=3, search_radius=0.1):
        '''approximate flow on query points
        this function assume query points are sub- or un-sampled from reference locations
        @param query_loc:[m,3]
        @param reference_loc:[n,3]
        @param reference_flow:[n,3]
        @param knn:
        @return:
            blended_flow:[m,3]
        '''
        dists, idx = knn_point_np(knn, reference_loc, query_loc)
        dists[dists < 1e-10] = 1e-10
        mask = dists > search_radius
        dists[mask] = 1e+10
        weight = 1.0 / dists
        weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
        blended_flow = np.sum(reference_flow[idx] * weight.reshape([-1, knn, 1]), axis=1, keepdims=False)

        mask = mask.sum(axis=1) < 3

        return blended_flow, mask

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        #corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        if self.match_type == "dual_softmax":
            pos_conf = conf[pos_mask]
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
            loss =  pos_w * loss_pos.mean()
            return loss

        elif self.match_type == "sinkhorn":
            # no supervision on dustbin row & column.
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
            return loss

    def match_2_conf_matrix(self, matches_gt, matrix_pred):
        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate (matches_gt) :
            matrix_gt [ b][ match[0],  match[1] ] = 1
        return matrix_gt


    @staticmethod
    def compute_match_recall(conf_matrix_gt, match_pred) : #, s_pcd, t_pcd, search_radius=0.3):
        '''
        @param conf_matrix_gt:
        @param match_pred:
        @return:
        '''

        pred_matrix = torch.zeros_like(conf_matrix_gt)

        b_ind, src_ind, tgt_ind = match_pred[:, 0], match_pred[:, 1], match_pred[:, 2]
        pred_matrix[b_ind, src_ind, tgt_ind] = 1.

        true_positive = (pred_matrix == conf_matrix_gt) * conf_matrix_gt

        recall = true_positive.sum() / conf_matrix_gt.sum()

        precision = true_positive.sum() / max(len(match_pred), 1)

        return recall, precision
    
    @staticmethod
    def weighted_svd(
        src_points,
        ref_points,
        weights=None,
        weight_thresh=0.0,
        eps=1e-5,
        return_transform=False,
    ):
        r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

        Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

        Args:
            src_points: torch.Tensor (B, N, 3) or (N, 3)
            ref_points: torch.Tensor (B, N, 3) or (N, 3)
            weights: torch.Tensor (B, N) or (N,) (default: None)
            weight_thresh: float (default: 0.)
            eps: float (default: 1e-5)
            return_transform: bool (default: False)

        Returns:
            R: torch.Tensor (B, 3, 3) or (3, 3)
            t: torch.Tensor (B, 3) or (3,)
            transform: torch.Tensor (B, 4, 4) or (4, 4)
        """
        if src_points.ndim == 2:
            src_points = src_points.unsqueeze(0)
            ref_points = ref_points.unsqueeze(0)
            if weights is not None:
                weights = weights.unsqueeze(0)
            squeeze_first = True
        else:
            squeeze_first = False

        batch_size = src_points.shape[0]
        if weights is None:
            weights = torch.ones_like(src_points[:, :, 0])
        weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
        weights = weights.unsqueeze(2)  # (B, N, 1)

        src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
        ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
        src_points_centered = src_points - src_centroid  # (B, N, 3)
        ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

        H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
        U, _, V = torch.svd(H.cpu())  # H = USV^T
        Ut, V = U.transpose(1, 2).cuda(), V.cuda()
        eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
        R = V @ eye @ Ut

        t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
        t = t.squeeze(2)

        if return_transform:
            transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
            transform[:, :3, :3] = R
            transform[:, :3, 3] = t
            if squeeze_first:
                transform = transform.squeeze(0)
            return transform
        else:
            if squeeze_first:
                R = R.squeeze(0)
                t = t.squeeze(0)
            return R, t

    @staticmethod
    def ransac_regist_coarse(batched_src_pcd, batched_tgt_pcd, src_mask, tgt_mask, match_pred, data ):
        s_len = src_mask.sum(dim=1).int()
        t_len = tgt_mask.sum(dim=1).int()
        bsize = len(batched_src_pcd)


        batched_src_pcd = MatchMotionLoss.tensor2numpy( batched_src_pcd)
        batched_tgt_pcd = MatchMotionLoss.tensor2numpy( batched_tgt_pcd)
        match_pred = MatchMotionLoss.tensor2numpy(match_pred)
        # gt_corres = np.array(data['correspondences_list'])

        rot = []
        trn = []
        corrs = []
        RR = {}

        for i in range(bsize):
            s_pcd = batched_src_pcd[i][:s_len[i]]
            t_pcd = batched_tgt_pcd[i][:t_len[i]]

            pair_i = match_pred[:, 0] == i
            n_pts = pair_i.sum()
            

            if n_pts < 3 :
                rot.append(torch.eye(3))
                trn.append(torch.zeros((3,1)))
                RR['rr_0_04'] = 0.0
                RR['rr_0_02'] = 0.0
                RR['rr_0_01'] = 0.0
                RR['rr_0_005'] = 0.0
                RR['rr_0_002'] = 0.0
                continue

            ind = match_pred[pair_i]
            s_ind, t_ind = ind[:, 1], ind[:, 2]

            
            ######## here we test the weighted_svd function #####
            # import time
            # time1 = time.time()
            R, t = MatchMotionLoss.weighted_svd(torch.from_numpy(s_pcd[s_ind]).cuda(), torch.from_numpy(t_pcd[t_ind]).cuda())
            rot.append(R)
            trn.append(t)
            corrs.append([s_ind, t_ind])
            # time2 = time.time()
            # print("time for weighted_svd: ", time2 - time1)
            
            # time1 = time.time()
            # pose = ransac_pose_estimation(s_pcd, t_pcd, [s_ind, t_ind], distance_threshold=0.001)
            # # # time2 = time.time()
            # # # print("time for ransac: ", time2 - time1)
            # pose = pose.copy()
            # rot.append(torch.from_numpy(pose[:3,:3]))
            # trn.append(torch.from_numpy(pose[:3,3:]))
            # print("pose from ransac: ", pose)
            # print("pose from weighted_svd: ", R, t)



            ##### compute RR #####
            # print("shape of gt_corres: ", gt_corres.shape)
            # s_ind_gt, t_ind_gt = gt_corres[i][:,0], gt_corres[i][:,1]
            # R = torch.from_numpy(pose[:3,:3].astype(np.float32)).cuda()
            # t = torch.from_numpy(pose[:3,3:].astype(np.float32)).flatten().cuda()
            s_pcd_oripts = data['batched_src_pcd'][i]
            gt_rot = data['batched_rot'][i]
            gt_trn = data['batched_trn'][i].flatten()
            # print("shape of gt_rot: ", gt_rot.shape)
            # print("shape of gt_trn: ", gt_trn.shape)
            # print("shape of R: ", R.shape)
            # print("shape of t: ", t.shape)
            # print("original dist: ", torch.norm(s_pcd_oripts[s_ind] - t_pcd[t_ind], dim=1).mean())
            s_tf_pred = torch.matmul(R, s_pcd_oripts.T).T + t
            s_tf = torch.matmul(gt_rot, s_pcd_oripts.T).T + gt_trn
            dist = torch.norm(s_tf_pred - s_tf, dim=1).mean()

            # dist = torch.norm(torch.from_numpy(t_pcd[t_ind]).cuda() - s_tf, dim=1).mean()
            # print("dist: ", dist)
            # dist = torch.norm(torch.from_numpy(s_pcd[s_ind]).cuda() - torch.from_numpy(t_pcd[t_ind]).cuda(), dim=1)
            rr_0_04 = (dist < 0.04).sum().float()
            rr_0_02 = (dist < 0.02).sum().float() 
            rr_0_01 = (dist < 0.01).sum().float() 
            rr_0_005 = (dist < 0.005).sum().float() 
            rr_0_002 = (dist < 0.002).sum().float()
            RR['rr_0_04'] = rr_0_04
            RR['rr_0_02'] = rr_0_02
            RR['rr_0_01'] = rr_0_01
            RR['rr_0_005'] = rr_0_005
            RR['rr_0_002'] = rr_0_002
            # print("RR: ", RR)
            


        return  torch.stack(rot, dim=0 ), torch.stack(trn , dim=0), np.array(corrs), RR #ndarray


    @staticmethod
    def compute_inlier_ratio(match_pred, data, inlier_thr, s2t_flow=None, idx=None):
        s_pcd, t_pcd = data['s_pcd'], data['t_pcd'] #B,N,3
        batched_rot = data['batched_rot'] #B,3,3
        batched_trn = data['batched_trn']

        if s2t_flow is not None: # 4dmatch
            s_pcd_deformed = s_pcd + s2t_flow
            s_pcd_wrapped = (torch.matmul(batched_rot, s_pcd_deformed.transpose(1, 2)) + batched_trn).transpose(1,2)
        else:  # 3dmatch
            s_pcd_wrapped = (torch.matmul(batched_rot, s_pcd.transpose(1, 2)) + batched_trn).transpose(1,2)
        
        

        s_pcd_matched = s_pcd_wrapped [match_pred[:,0], match_pred[:,1]]
        t_pcd_matched = t_pcd [match_pred[:,0], match_pred[:,2]]

        # print("shape of s_pcd_matched: ", s_pcd_matched.shape)
        # print("shape of t_pcd_matched: ", t_pcd_matched.shape)
        dist =  torch.sum( (s_pcd_matched - t_pcd_matched)**2 , dim= 1)

        inlier_0_01 = dist <  inlier_thr**2
        inlier_0_015 = dist <  0.015
        inlier_0_005 = dist <  0.005
        inlier_0_002 = dist <  0.002
        inlier_0_001 = dist <  0.001
        inlier_0_0005 = dist <  0.0005
        inlier_0_00001 = dist <  0.00001
        # dist = torch.norm(s_pcd_matched - t_pcd_matched, dim=1)
        # print("shape of this: ", s_pcd_matched.shape, t_pcd_matched.shape)
        # print("dist: ", dist)

        bsize = len(s_pcd)
        
        IR={}
        # RR = {}
        for i in range(bsize):
            pair_i = match_pred[:, 0] == i
            n_match = pair_i.sum()
            # print("n_match: ", n_match)
            inlier_i_0_01 = inlier_0_01[pair_i]
            inlier_i_0_015 = inlier_0_015[pair_i]
            inlier_i_0_005 = inlier_0_005[pair_i]
            inlier_i_0_002 = inlier_0_002[pair_i]
            inlier_i_0_001 = inlier_0_001[pair_i]
            inlier_i_0_0005 = inlier_0_0005[pair_i]
            inlier_i_0_00001 = inlier_0_00001[pair_i]

            n_inlier_0_01 = inlier_i_0_01.sum().float()
            n_inlier_0_015 = inlier_i_0_015.sum().float()
            n_inlier_0_005 = inlier_i_0_005.sum().float()
            n_inlier_0_002 = inlier_i_0_002.sum().float()
            n_inlier_0_001 = inlier_i_0_001.sum().float()
            n_inlier_0_0005 = inlier_i_0_0005.sum().float()
            n_inlier_0_00001 = inlier_i_0_00001.sum().float()
            if n_match <3:
                # IR.append( n_match.float()*0)
                IR['ir_0_01'] = n_match.float()*0
                IR['ir_0_015'] = n_match.float()*0
                IR['ir_0_005'] = n_match.float()*0
                IR['ir_0_002'] = n_match.float()*0
                IR['ir_0_001'] = n_match.float()*0
                IR['ir_0_0005'] = n_match.float()*0
                IR['ir_0_00001'] = n_match.float()*0
                
            else :
                IR['ir_0_01'] = n_inlier_0_01/n_match
                IR['ir_0_015'] = n_inlier_0_015/n_match
                IR['ir_0_005'] = n_inlier_0_005/n_match
                IR['ir_0_002'] = n_inlier_0_002/n_match
                IR['ir_0_001'] = n_inlier_0_001/n_match
                IR['ir_0_0005'] = n_inlier_0_0005/n_match
                IR['ir_0_00001'] = n_inlier_0_00001/n_match
            

        return IR #, RR



    @staticmethod
    def compute_registration_recall(R_est, t_est, data, thr=0.2):

        bs = len(R_est)
        success = 0.


        # if data['gt_cov'] is not None:

        err2 = thr ** 2

        gt = np.zeros( (bs, 4, 4))
        gt[:, -1,-1] = 1
        gt[:, :3, :3] = data['batched_rot'].cpu().numpy()
        gt[:, :3, 3:] = data['batched_trn'].cpu().numpy()

        pred = np.zeros((bs, 4, 4))
        pred[:, -1, -1] = 1
        pred[:, :3, :3] = R_est.detach().cpu().numpy()
        pred[:, :3, 3:] = t_est.detach().cpu().numpy()
        

        RR = {}
        for i in range(bs):

            p = computeTransformationErr( np.linalg.inv(gt[i]) @ pred[i], data['gt_cov'][i])

            # if p <= err2:
            #     success += 1
            

        rr = success / bs
        return rr


    @staticmethod
    def tensor2numpy(tensor):
        if tensor.requires_grad:
            tensor=tensor.detach()
        return tensor.cpu().numpy()