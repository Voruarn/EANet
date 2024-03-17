# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import util6.misc as misc
from util6.misc import NativeScalerWithGradNormCount as NativeScaler

from network.MAE import mae_vit_base, mae_vit_small, mae_vit_tiny

from engine_pretrain import train_one_epoch
import matplotlib.pyplot as plt
from PIL import Image

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def show_image(image, title='', cur_col=0, cols=6):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    # if cur_col<=cols:
    #     plt.title(title, fontsize=16)
    plt.axis('off')
    return

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base', type=str, metavar='MODEL',
                        help='Name of model to train:[mae_vit_base, mae_vit_small, mae_vit_tiny]')

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument("--ckpt", type=str,
            default=None,
              help="restore from checkpoint")

    parser.add_argument('--data_path', default=None, type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pred_dir', 
            default=None,
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', default=False, type=bool,
                        help='distributed')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # 
    return parser


def main(args):
    # misc.init_distributed_mode(args)

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(args)
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),  # 3 is bicubic
        transforms.RandomResizedCrop(args.input_size, scale=(1.0, 1.0)),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    # print(dataset_train)

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if  args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = eval(args.model)(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size)

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256


    if args.ckpt is not None and os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        print('epoch:', checkpoint['epoch'])
        try:
            model.load_state_dict(checkpoint['model'])
            print('try: load pth from:', args.ckpt)
        except:
            model_dict      = model.state_dict()
            pretrained_dict = checkpoint['model']
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)

            print('except: load pth from:', args.ckpt)

        model=model.to(device)  
    else:
        print("[!] Retrain")
        model.to(device)
    

    model.eval()
    cnt=0
    cols=12
    rows=4
    cur_col=1
    for data_iter_step, (samples, _) in enumerate(data_loader_train):

        samples = samples.to(device, non_blocking=True)
        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [36, 12]
        #H:4, W:12
        
        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, mask_ratio=args.mask_ratio)


            x=samples.detach().cpu()
            # run MAE
            # loss, y, mask = model(x.float(), mask_ratio=0.75)
            y = model.unpatchify(y)
            y = torch.einsum('nchw->nhwc', y).detach().cpu()

            # visualize the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
            
            x = torch.einsum('nchw->nhwc', x)

            # masked image
            im_masked = x * (1 - mask)

            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask

            
            plt.subplot(rows, cols, cur_col)
            show_image(x[0], "original", cur_col, cols)
            cur_col+=1

            plt.subplot(rows, cols, cur_col)
            show_image(im_masked[0], "masked", cur_col, cols)
            cur_col+=1

            # plt.subplot(rows, cols, (cnt%4)*4+3)
            # show_image(y[0], "reconstruction", cnt, rows)

            plt.subplot(rows, cols, cur_col)
            show_image(im_paste[0], "reconstruction", cur_col, cols)
            cur_col+=1

            # plt.savefig(args.pred_dir+'pred{}.png'.format(cnt), bbox_inches='tight')
            
        cnt+=1
        if cnt%(rows*(cols//3))==0:
            cur_col=1
            print('cnt:',cnt)
            plt.tight_layout()
            # plt.show()
            plt.savefig(args.pred_dir+'{}_pred{}.png'.format(args.model, cnt), bbox_inches='tight')
        # break



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
