from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data
from datasets.EORSSD_Dataset import EORSSDDataset
from metrics.SOD_metrics import SODMetrics
from network.EANet import EANet
from network.MAE import mae_vit_tiny, mae_vit_small, mae_vit_base
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import pytorch_iou

def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainset_path", type=str, 
        default='/opt/data/private/FYX/Dataset/EORSSD/Train',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='/opt/data/private/FYX/Dataset/EORSSD/Test',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='EORSSD', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
  
    parser.add_argument("--mae", type=str, default='mae_vit_base',
        help='model name:[mae_vit_tiny, mae_vit_small, mae_vit_base]')
    parser.add_argument("--model", type=str, default='EANet',
        help='model name:[EANet]')

    parser.add_argument("--epochs", type=int, default=100,
                        help="epoch number (default: 60)")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="total_itrs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=256)

    parser.add_argument("--n_cpu", type=int, default=8,
                        help="download datasets")
    parser.add_argument("--pretrained", type=str,
            default=None, 
            help="restore from checkpoint")
    parser.add_argument("--ckpt", type=str,
            default=None, help="restore from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--output_dir", type=str, default='./DHM_Ptr',
                        help="epoch interval for eval (default: 100)")
    return parser


def get_dataset(opts):

    train_dst = EORSSDDataset(is_train=True,voc_dir=opts.trainset_path, trainsize=opts.trainsize)
    val_dst = EORSSDDataset(is_train=False,voc_dir=opts.testset_path, trainsize=opts.trainsize)
    return train_dst, val_dst


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
IOU = pytorch_iou.IOU(size_average = True)


def mrs_loop(mrs):
    mrs_next=[mrs[i] for i in range(1, len(mrs))]
    mrs_next.append(mrs[0])
    return mrs_next

def main():
    opts = get_argparser().parse_args()
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

    tb_writer = SummaryWriter()
    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    opts.total_itrs=opts.epochs * (len(train_dst) // opts.batch_size)
    print('opts:',opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
        drop_last=True)  
   
    print("Dataset: %s, Train set: %d" %
          (opts.dataset, len(train_dst)))

    mae = eval(opts.mae)(img_size=opts.trainsize)
    model = eval(opts.model)(img_size=opts.trainsize)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    def save_ckpt(path):
        torch.save({
            "epoch": epoch+1,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)  
    
    if opts.pretrained is not None and os.path.isfile(opts.pretrained):
        checkpoint = torch.load(opts.pretrained, map_location=torch.device('cpu'))
        pretrained_epochs=checkpoint['epoch']
        print('pretrained_epochs:',pretrained_epochs)
        try:
            mae.load_state_dict(checkpoint['model'])
            print('try: mae load pth from:', opts.pretrained)
        except:
            model_dict      = mae.state_dict()
            pretrained_dict = checkpoint['model']
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                # if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v) and k.split('.')[0]=='layers':
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    print(k)
                    temp_dict[k] = v
                    load_key.append(k)
                # else:
                #     no_load_key.append(k)
            model_dict.update(temp_dict)
            mae.load_state_dict(model_dict)

            print('except: mae load pth from:', opts.pretrained)
        mae=mae.to(device)


    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model=model.to(device)
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]   
        
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model=model.to(device)

    mae.eval()
    mask_ratios=[0.3, 0.4, 0.5, 0.6, 0.7]
    for epoch in range(cur_epoch,opts.epochs):
        model.train()
        cur_itrs=0
        data_loader = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        
        for (images, gts) in data_loader:
            cur_mr=mask_ratios[cur_itrs%len(mask_ratios)]
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            gts = gts.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            # hybrid mask ratio
            loss, y, mask= mae(imgs=images, mask_ratio=cur_mr)
            
            x=images
            y = mae.unpatchify(y)
            # visualize the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = mae.unpatchify(mask)  # 1 is removing, 0 is keeping
            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask
            
            s1,s2,s3,s4, s1_sig,s2_sig,s3_sig,s4_sig= model(im_paste)
            loss1 = CE(s1, gts) + IOU(s1_sig, gts)
            loss2 = CE(s2, gts) + IOU(s2_sig, gts)
            loss3 = CE(s3, gts) + IOU(s3_sig, gts)
            loss4 = CE(s4, gts) + IOU(s4_sig, gts)
            total_loss = loss1 + loss2/2 + loss3/4 +loss4/8 

            running_loss += total_loss.data.item()

            total_loss.backward()
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, loss={:.4f}".format(epoch, opts.epochs, running_loss/cur_itrs)
            
            scheduler.step()

        mask_ratios=mrs_loop(mask_ratios)
        tags = ["train_loss", "learning_rate"]

        tb_writer.add_scalar(tags[0], (running_loss/cur_itrs), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        
        if (epoch+1) % opts.val_interval == 0:
            save_ckpt(opts.output_dir+'/latest_{}_{}_dhmptr.pth'.format(opts.model, opts.dataset))
            if (epoch+1)%50==0:
                save_ckpt(opts.output_dir+'/{}_{}_ep{}_dhmptr.pth'.format(opts.model, 
                                        opts.dataset, epoch+1))


if __name__ == '__main__':
    main()
