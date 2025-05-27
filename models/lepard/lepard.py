from models.lepard.blocks import *
from models.lepard.backbone import KPFCN
from models.lepard.transformer import RepositioningTransformer
from models.lepard.matching import Matching
from models.lepard.procrustes import SoftProcrustesLayer

class Pipeline(nn.Module):

    def __init__(self, config):
        super(Pipeline, self).__init__()
        self.config = config
        self.backbone = KPFCN(config['kpfcn_config'])
        self.pe_type = config['coarse_transformer']['pe_type']
        self.positioning_type = config['coarse_transformer']['positioning_type']
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])



    def forward(self, data,  timers=None):

        self.timers = timers

        if self.timers: self.timers.tic('kpfcn backbone encode')
        # print("lepard 1:{}".format(torch.cuda.memory_allocated(0)))
        coarse_feats = self.backbone(data, phase="coarse")
        # print("lepard 2:{}".format(torch.cuda.memory_allocated(0)))
        if self.timers: self.timers.toc('kpfcn backbone encode')
        # assert not torch.any(torch.isnan(coarse_feats))

        if self.timers: self.timers.tic('coarse_preprocess')
        # print("lepard 3:{}".format(torch.cuda.memory_allocated(0)))
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats (coarse_feats, data)
        # print("lepard 4:{}".format(torch.cuda.memory_allocated(0)))
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })
        if self.timers: self.timers.toc('coarse_preprocess')
        # print("shape of s_pcd and t_pcd: ", s_pcd.shape, t_pcd.shape)
        # print("shape of these tensor: ", src_feats.shape, tgt_feats.shape, s_pcd.shape, t_pcd.shape)
        # assert not torch.any(torch.isnan(src_feats))
        # assert not torch.any(torch.isnan(tgt_feats))
        

        if self.timers: self.timers.tic('coarse feature transformer')
        # print("lepard 5:{}".format(torch.cuda.memory_allocated(0)))
        src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers)
        # print("lepard 6:{}".format(torch.cuda.memory_allocated(0)))
        if self.timers: self.timers.toc('coarse feature transformer')
        # print("shape of these tensor: ", src_feats.shape, tgt_feats.shape, src_pe.shape, tgt_pe.shape)

        if self.timers: self.timers.tic('match feature coarse')
        # print("lepard 7:{}".format(torch.cuda.memory_allocated(0)))
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type = self.pe_type)
        # print("lepard 8:{}".format(torch.cuda.memory_allocated(0)))
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')
        # assert not torch.any(torch.isnan(conf_matrix_pred))
        # assert not torch.any(torch.isnan(coarse_match_pred))
        # assert not torch.any(torch.isnan(s_pcd))
        # assert not torch.any(torch.isnan(t_pcd))
        # assert not torch.any(torch.isnan(src_mask))
        # assert not torch.any(torch.isnan(tgt_mask))

        if self.timers: self.timers.tic('procrustes_layer')
        # print("lepard 9:{}".format(torch.cuda.memory_allocated(0)))
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        # print("lepard 10:{}".format(torch.cuda.memory_allocated(0)))
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')
        # assert not torch.any(torch.isnan(R))
        # assert not torch.any(torch.isnan(t))
        # data.update({"scale": scale})

        return data




    def split_feats(self, geo_feats, data):

        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        src_ind_coarse_split = data[ 'src_ind_coarse_split']
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']
        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]

        

        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

        return src_feats.view( b_size , src_pts_max , -1), \
               tgt_feats.view( b_size , tgt_pts_max , -1), \
               src_pcd.view( b_size , src_pts_max , -1), \
               tgt_pcd.view( b_size , tgt_pts_max , -1), \
               src_mask, \
               tgt_mask