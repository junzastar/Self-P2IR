from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os, torch, json, argparse, shutil
from easydict import EasyDict as edict
import yaml
from datasets.dataloader import get_dataloader, get_datasets
from models.mynetwork import DeformRegis
from lib.logger import setup_seed
from tester import get_trainer
from lib.loss import MatchMotionLoss, ChamferLoss_l2, silhouetteLoss,ChamferLoss_l1, BinaryDiceLoss
from lib.tictok import Timers
from configs.models import architectures
import numpy as np
import torch.backends.cudnn as cudnn

from torch import optim
import warnings
warnings.filterwarnings("ignore")
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

setup_seed(0)

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

yaml.add_constructor('!join', join)


if __name__ == '__main__':
    # load configs
    
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_id', type=list, default=[0, 1, 2, 3])
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--keep_batchnorm_fp32', default=True)
    parser.add_argument('--opt_level', default="O0", type=str,
                        help='opt level of apex mix presision trainig.')
    parser.add_argument('--syncBN', type=bool, default=True)
    args = parser.parse_args()
    

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    local_rank = torch.distributed.get_rank()
    args.local_rank = local_rank
    
    torch.cuda.set_device(args.local_rank)

    #### try paraller ##
    print("local_rank:", args.local_rank)
    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
    torch.manual_seed(0)

    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['dataset']+config['folder'], config['exp_dir'])
    config['tboard_dir'] = 'snapshot/%s/%s/tensorboard' % (config['dataset']+config['folder'], config['exp_dir'])
    config['save_dir'] = 'snapshot/%s/%s/checkpoints' % (config['dataset']+config['folder'], config['exp_dir'])
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
      
    
    config.device = torch.device('cuda:{}'.format(args.local_rank))
    config.local_rank = args.local_rank
    config.gpus = args.gpus
    
    # backup the
    if config.mode == 'train':
        os.system(f'cp -r models {config.snapshot_dir}')
        os.system(f'cp -r configs {config.snapshot_dir}')
        os.system(f'cp -r cpp_wrappers {config.snapshot_dir}')
        os.system(f'cp -r datasets {config.snapshot_dir}')
        os.system(f'cp -r kernels {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        shutil.copy2('main.py',config.snapshot_dir)

    # model initialization
    config.kpfcn_config.architecture = architectures[config.dataset]
    config.model = DeformRegis(config)
    config.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(config.model)

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
        
    #create learning rate scheduler
    if  'overfit' in config.exp_dir :
        config.scheduler = optim.lr_scheduler.MultiStepLR(
            config.optimizer,
            milestones=[config.max_epoch-1], # fix lr during overfitting
            gamma=0.1,
            last_epoch=-1)

    else:
        config.scheduler = optim.lr_scheduler.ExponentialLR(
            config.optimizer,
            gamma=config.scheduler_gamma,
        )


    config.timers = Timers()

    # create dataset and dataloader
    train_set, val_set, test_set = get_datasets(config)
    config.train_loader, neighborhood_limits, config.train_sampler = get_dataloader(train_set,config, mode = 'train',shuffle=False)
    config.val_loader, _ = get_dataloader(val_set, config, mode='val', shuffle=False, neighborhood_limits=neighborhood_limits)
    config.test_loader, _ = get_dataloader(test_set, config, mode='test', shuffle=False, neighborhood_limits=neighborhood_limits)
    
    
    config.syn_loss = MatchMotionLoss (config['train_loss'])
    config.self_training_CDloss = ChamferLoss_l2()
    config.self_training_maskloss = silhouetteLoss()
    config.self_training_diceloss = BinaryDiceLoss()
     
    torch.cuda.synchronize()
    trainer = get_trainer(config)
    if(config.mode=='train' and config.dataset == 'real'):
        trainer._load_pretrain("path to your own model trained on synthetic data/checkpoints/model_best_loss.pth")
        for name, param in config.model.named_parameters():
            if 'posenet' in name:
                param.requires_grad = False
    # import sys
    # sys.exit()
    if(config.mode=='train'):
        ### 
        args.world_size = args.gpus * args.nodes
        config.model = torch.nn.parallel.DistributedDataParallel(config.model, device_ids=[args.gpu_id], output_device=args.local_rank,find_unused_parameters=True)
        trainer.train()
    else:
        trainer.test()