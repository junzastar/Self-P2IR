#!/usr/bin/env python3
import os
import yaml
import numpy as np


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

class CommonConfig:
    def __init__(self, ds_name='syn', cls_type=''):
        self.dataset_name = ds_name
        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = os.path.basename(self.exp_dir)
        self.resnet_ptr_mdl_p = os.path.abspath(
            os.path.join(
                self.exp_dir,
                'models/cnn/ResNet_pretrained_mdl'
            )
        )
        ensure_fd(self.resnet_ptr_mdl_p)

        # log folder
        self.cls_type = cls_type
        self.log_dir = os.path.abspath(
            os.path.join(self.exp_dir, 'train_log', self.dataset_name)
        )
        ensure_fd(self.log_dir)
        self.log_model_dir = os.path.join(self.log_dir, 'checkpoints', self.cls_type)
        ensure_fd(self.log_model_dir)
        self.log_eval_dir = os.path.join(self.log_dir, 'eval_results', self.cls_type)
        ensure_fd(self.log_eval_dir)
        self.log_traininfo_dir = os.path.join(self.log_dir, 'train_info', self.cls_type)
        ensure_fd(self.log_traininfo_dir)

        self.n_total_epoch = 25
        self.mini_batch_size = 2
        self.val_mini_batch_size = 2
        self.test_mini_batch_size = 1

        self.noise_trans = 0.05  # range of the random noise of translation added to the training data

        if self.dataset_name == 'syn':
            pass
        elif self.dataset_name == 'real':
            pass
        else:
            print("Unkonw dataset name!")

        self.intrinsic_matrix = {
            'syn_3Dircadb1': np.array([[360.0, 0.        , 480.0],
                                [0.      , 360.0  , 270.0],
                                [0.      , 0.        , 1.0]], np.float32),  
            'syn_3Dircadb2': np.array([[599.9999389648438, 0.        , 480.0],
                                [0.      , 599.9999389648438  , 270.0],
                                [0.      , 0.        , 1.0]], np.float32),      
        }

    def read_lines(self, p):
        with open(p, 'r') as f:
            return [
                line.strip() for line in f.readlines()
            ]


config = CommonConfig()
# vim: ts=4 sw=4 sts=4 expandtab
