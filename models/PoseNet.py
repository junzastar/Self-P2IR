import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self, ):
        super(PoseNet, self).__init__()
        

        

    def forward(self, src_pcd, tar_pcd):
        """ Only support batch size of 1

        Args:
            img: bs x 3 x H x W
            x: bs x n_p x 3
            choose: bs x n_p
            obj: bs x 1

        Returns:
            out_tx: 1 x n_p x 3
            out_rx: 1 x num_rot x 4
            out_cx: 1 x num_rot

        """
        
        

        return None