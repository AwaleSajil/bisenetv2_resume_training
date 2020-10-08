#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

# apex
has_apex = True
try:
    from apex import amp
except ImportError:
    has_apex = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('-s', '--saveCheckpointDir', type=str, required=True, help = "folder to which intermidiate checkpoint are to be saved", default=None)
    parse.add_argument('-se', '--saveOnEveryEpoch', type=int, help = "Save a checkpoint after this many epoch", default=5000)
    parse.add_argument('-l', '--loadCheckpointLocation', type=str, help="location to the checkpoint you want to resume training" ,default=None)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.model]


def load_ckp(checkpoint_fpath, model, optimizer, lr_schdr):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_schdr.load_state_dict(checkpoint['lr_schdr'])
    return model, optimizer, lr_schdr, checkpoint['epoch']

def save_ckp(state, save_pth):
    torch.save(state, save_pth)


def set_model():
    net = model_factory[cfg.model_type](19)
    if not args.finetune_from is None:
        checkpoint = torch.load(args.finetune_from, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])

    net.to(device)
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim

def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train():
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(
            cfg.im_root, cfg.train_im_anns,
            cfg.ims_per_gpu, cfg.scales, cfg.cropsize,
            cfg.max_iter, mode='train', distributed=False)

    #finding max epoch to train
    dataset_length = len(dl.dataset)
    print("Dataset length: ", dataset_length)
    batch_size = cfg.ims_per_gpu
    iteration_per_epoch = int(dataset_length/batch_size)
    max_epoch = int(cfg.max_iter/iteration_per_epoch)
    print("Max_epoch: ", max_epoch)

    ## model
    net, criteria_pre, criteria_aux = set_model()

    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)


    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

     ##load checkpoin if exits for resuming training
    if  args.loadCheckpointLocation != None:
        net, optim, lr_schdr, start_epoch = load_ckp(args.loadCheckpointLocation, net, optim, lr_schdr)
    else:
        start_epoch = 0

    ## train loop
    for current_epoch in range(max_epoch):
        #on resumed training 'epoch' will be incremented from what was left else the sum is 0 anyways
        epoch = start_epoch + current_epoch

        for it, (im, lb) in enumerate(dl):
            
            im = im.to(device)
            lb = lb.to(device)

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            logits, *logits_aux = net(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
            if has_apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            lr_schdr.step()

            time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

            ## print training log message
            total_it = it + epoch*iteration_per_epoch
            if (it + 1) % 100 == 0:
                lr = lr_schdr.get_lr()
                lr = sum(lr) / len(lr)
                print_log_msg(
                    total_it, cfg.max_iter, lr, time_meter, loss_meter,
                    loss_pre_meter, loss_aux_meters)

        #save the checkpoint on every some epoch
        if (epoch + 1) % args.saveOnEveryEpoch == 0:
            if args.saveCheckpointDir != None:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optim.state_dict(),
                    'lr_schdr': lr_schdr.state_dict(),
                }
                epoch_no_str = (str(epoch+1)).zfill(len(str(cfg.max_iter)))
                ckt_name = 'checkpoint_epoch_' + epoch_no_str + '.pt'
                save_pth = osp.join(args.saveCheckpointDir, ckt_name)
                logger.info('\nsaving intermidiate checkpoint to {}'.format(save_pth))
                save_ckp(checkpoint, save_pth)



    ## dump the final model and evaluate the result
    checkpoint = {
                'epoch': max_epoch,
                'state_dict': net.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_schdr': lr_schdr.state_dict(),
                }
    save_pth = osp.join(args.saveCheckpointDir, 'model_final.pt')
    logger.info('\nsave Final models to {}'.format(save_pth))
    save_ckp(checkpoint, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    heads, mious = eval_model(net, 2, cfg.im_root, cfg.val_im_anns)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
    return


def main():
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    print("Args sent: ", args.saveCheckpointDir, args.loadCheckpointLocation, args.saveOnEveryEpoch)
    train()
    


if __name__ == "__main__":
    main()
