import gc
import os

import torch
import torch.nn as nn
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.timer import AverageMeter
from lib.logger import Logger, validate_gradient
from lib.tictok import Timers

class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.start_iter = 0
        self.start_val_iter = 0
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.local_rank = args.local_rank
        self.gpus = args.gpus

        self.model = args.model
        self.model = self.model.to(self.device)
        self.real_syn = args.dataset


        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_dir = args.snapshot_dir

        self.iter_size = args.iter_size
        self.verbose_freq = args.verbose_freq // args.batch_size + 1
        if 'overfit' in self.config.exp_dir:
            self.verbose_freq = 500
        self.loss = args.syn_loss
        self.self_cdloss = args.self_training_CDloss
        self.self_maskloss = args.self_training_maskloss
        self.self_diceloss = args.self_training_diceloss

        self.best_loss = 1e5
        self.best_recall = -1e5
        if args.local_rank == 0:
            self.summary_writer = SummaryWriter(log_dir=args.tboard_dir)
            self.logger = Logger(args.snapshot_dir)
            self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')

        if (args.pretrain != ''):
            self._load_pretrain(args.pretrain)

        self.loader = dict()
        self.loader['train'] = args.train_loader
        self.loader['val'] = args.val_loader
        self.loader['test'] = args.test_loader
        self.train_sampler = args.train_sampler

        self.timers = args.timers

        with open(f'{args.snapshot_dir}/model', 'w') as f:
            f.write(str(self.model))
        f.close()

    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
                    
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename, _use_new_zipfile_serialization=False)

    def _load_pretrain(self, resume):
        print("loading pretrained", resume)
        if os.path.isfile(resume):
            state = torch.load(resume, map_location='cpu')
        
            self.model.load_state_dict(state['state_dict'], strict=False)

            self.best_recall = state['best_recall']
            if self.local_rank == 0:
                self.logger.write(f'Successfully load pretrained model from {resume}!\n')
                self.logger.write(f'Current best loss {self.best_loss}\n')
                self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']


    def inference_one_batch(self, inputs, phase):
        assert phase in ['train', 'val', 'test']
        inputs ['phase'] = phase


        if (phase == 'train'):
            self.model.train()
            self.optimizer.zero_grad()
            if self.timers: self.timers.tic('forward pass')
            
            data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
            
            if self.timers: self.timers.toc('forward pass')
            

            if self.timers: self.timers.tic('compute loss')
           
            if self.real_syn=='syn':
                loss_info = self.loss(data)
           
            if self.real_syn=='real':
                self_cdloss = self.self_cdloss(data['rendered_pcd_bld'], data['batched_tgt_pcd']).sum()
                
                self_maskloss = self.self_maskloss(data['mask'], data['liver_label']).sum()

                self_diceloss = self.self_diceloss(data['mask'], data['liver_label']).sum()

                loss_lst = [
                    (self_diceloss, 2.0), (self_maskloss, 0.5), (self_cdloss, 1.0),
                ]
                

                loss = sum([ls * w for ls, w in loss_lst])

                loss_info = {}
        
                loss_info['non_rigid_loss'] = loss
                
                loss_info['cdloss'] =  self_cdloss.item()
                loss_info['diceloss'] =  self_diceloss.item()
                loss_info.update({ 'silhouetteLoss': self_maskloss.item() })
                
                
            if self.timers: self.timers.toc('compute loss')


            if self.timers: self.timers.tic('backprop')
            if self.real_syn=='syn':
                loss_info['rigid_loss'].backward()
            else:
                loss_info['non_rigid_loss'].backward()
            if self.timers: self.timers.toc('backprop')


        else:
            self.model.eval()
            if self.real_syn=='syn':
                with torch.no_grad():
                    data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                    
                    
                    loss_info = self.loss(data)
                    
            else:
                

                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                    
                loss_info = {}
                
                self_cdloss = self.self_cdloss(data['rendered_pcd_bld'], data['batched_tgt_pcd'])
                
                self_maskloss = self.self_maskloss(data['mask'], data['liver_label'])

                self_diceloss = self.self_diceloss(data['mask'], data['liver_label'])

                loss_lst = [
                    (self_diceloss, 2.0), (self_maskloss, 0.5), (self_cdloss, 1.0),
                ]

                loss = sum([ls * w for ls, w in loss_lst])

            
                loss_info['loss'] = loss.item()
                loss_info['cdloss'] =  self_cdloss.item()
                loss_info['diceloss'] =  self_diceloss.item()
                loss_info.update({ 'silhouetteLoss': self_maskloss.item() })

        del data
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

        return loss_info


    def inference_one_epoch(self, epoch, phase):
        gc.collect()
        assert phase in ['train', 'val', 'test']

        # init stats meter
        stats_meter = None #  self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size) # drop last incomplete batch
        c_loader_iter = self.loader[phase].__iter__()
        
        self.optimizer.zero_grad()
        for c_iter in tqdm(range(num_iter // self.gpus)):  # loop through this epoch
        # with tqdm(total=num_iter // self.gpus, desc= phase
        # ) as pbar:

            if self.timers: self.timers.tic('one_iteration')

            ##################################
            
            if self.timers: self.timers.tic('load batch')
            inputs = c_loader_iter.next()
            # for c_iter, inputs in enumerate(self.loader[phase]):
                # for gpu_div_i, _ in enumerate(inputs):
            if not inputs:
                continue
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


            if self.timers: self.timers.tic('inference_one_batch')
            
            loss_info = self.inference_one_batch(inputs, phase)
            
            if self.timers: self.timers.toc('inference_one_batch')


            ###################################################
            # run optimisation
            if self.timers: self.timers.tic('run optimisation')
            # if ((c_iter + 1) % self.iter_size == 0 and phase == 'train'):
            if (phase == 'train'):
                gradient_valid = validate_gradient(self.model)
                if (gradient_valid):
                    self.optimizer.step()
                else:
                    self.logger.write('gradient not valid\n')
                    raise ValueError("LOSS IS NAN!")
                # self.optimizer.step()
                self.optimizer.zero_grad()
            if self.timers: self.timers.toc('run optimisation')
            ################################
            
            

            torch.cuda.empty_cache()
            gc.collect()

            if stats_meter is None:
                stats_meter = dict()
                for key, _ in loss_info.items():
                    stats_meter[key] = AverageMeter()
            for key, value in loss_info.items():
                stats_meter[key].update(value)

            if phase == 'train' :
                if (c_iter + 1) % self.verbose_freq == 0 and self.verbose  :
                    curr_iter = num_iter // self.gpus * (epoch - 1) + c_iter
                    lr = self._get_lr()
                    if self.local_rank == 0:
                        self.summary_writer.add_scalar('lr/lr', lr, curr_iter)
                    for key, value in stats_meter.items():
                        if self.local_rank == 0:
                            self.summary_writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)

                    dump_mess=True
                    if dump_mess:
                        message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter // self.gpus}]'
                        for key, value in stats_meter.items():
                            message += f'{key}: {value.avg:.6f}\t'
                        if self.local_rank == 0:
                            self.logger.write(message + '\n')


            if self.timers: self.timers.toc('one_iteration')


        # report evaluation score at end of each epoch
        if phase in ['val', 'test']:
            for key, value in stats_meter.items():
                if self.local_rank == 0:
                    self.summary_writer.add_scalar(f'{phase}/{key}', value.avg, epoch)

        message = f'{phase} Epoch: {epoch}'
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        if self.local_rank == 0:
            self.logger.write(message + '\n')

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        return stats_meter




    def train(self):
        print('start training...')
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        self.start_iter = 0
        for epoch in range(self.start_epoch, self.max_epoch):
            self.train_sampler.set_epoch(epoch)
            with torch.autograd.set_detect_anomaly(False):
                if self.timers: self.timers.tic('run one epoch')
                
                stats_meter = self.inference_one_epoch(epoch, 'train')
                
                if self.timers: self.timers.toc('run one epoch')

            self.scheduler.step()


            if  'overfit' in self.config.exp_dir :
                if stats_meter['loss'].avg < self.best_loss:
                    self.best_loss = stats_meter['loss'].avg
                    if self.local_rank == 0:
                        self._snapshot(epoch, 'best_loss')

                if self.local_rank == 0:
                    if self.timers: self.timers.print()

            else : # no validation step for overfitting
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
                    if 'deformNet' in name:
                        param.requires_grad = True

                if self.config.do_valid:
                    stats_meter = self.inference_one_epoch(epoch, 'val')
                    if stats_meter['loss'].avg < self.best_loss:
                        self.best_loss = stats_meter['loss'].avg
                        if self.local_rank == 0:
                            self._snapshot(epoch, 'best_loss')

                if self.local_rank == 0:
                    if self.timers: self.timers.print()
            
            for name, param in self.model.named_parameters():
                    param.requires_grad = True
                    if 'posenet' in name:
                        param.requires_grad = False

            del stats_meter
            torch.cuda.empty_cache()
            gc.collect()   

        # finish all epoch
        print("Training finish!")