import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from reprod_log import ReprodLogger, ReprodDiffHelper
from paddle.io import DataLoader,SequenceSampler,BatchSampler

sys.path.append('..')
from dataloaders import pose_estimator,transforms
from dataloaders.kitti_loader import KittiDepth
from metrics import Result
import criteria


parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w',
                    '--workers',
                    default=2,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=21,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                         ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=8,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-5,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='E:/data/',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=['d', 'rgb', 'rgbd', 'g', 'gd'],
                    help='input: | '.join(['d', 'rgb', 'rgbd', 'g', 'gd']))
parser.add_argument('-l',
                    '--layers',
                    type=int,
                    default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained',
                    action="store_true",
                    help='use ImageNet pre-trained weights')
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument(
    '--rank-metric',
    type=str,
    default='rmse',
    choices=[m for m in dir(Result()) if not m.startswith('_')],
    help='metrics for which best result is sbatch_datacted')
parser.add_argument(
    '-m',
    '--train-mode',
    type=str,
    default="dense",
    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
    help='dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('--cpu', action="store_true", help='run on cpu')

args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
print(args)



if __name__ == '__main__':
    train_datasets = KittiDepth('train',args)
    val_datasets = KittiDepth('val',args)


    test_trainsampler = SequenceSampler(train_datasets)
    test_valsampler = SequenceSampler(val_datasets)

    test_batch_trainsampler = BatchSampler(sampler=test_trainsampler,batch_size=8)
    test_batch_valsampler = BatchSampler(sampler=test_valsampler,batch_size=8)

    train_loader = DataLoader(train_datasets,
                              num_workers=2,
                              batch_sampler=test_batch_trainsampler)
    val_loader = DataLoader(val_datasets,
                            num_workers=2,
                            batch_sampler=test_batch_valsampler)

    # for batch in train_loader:
    #     print(batch['rgb'].shape)
    #     print('--')
    # print('===='*20)
    #
    # for batch in val_loader:
    #     print(batch['rgb'].shape)
    #     print('--')



    logger_train_data = ReprodLogger()
    logger_train_data.add("length", np.array(len(train_datasets)))
    logger_val_data = ReprodLogger()
    logger_val_data.add('length',np.array(len(val_datasets)))

    category = ['rgb','d','gt','g']
    for idx, torch_batch in enumerate(train_loader):
        if idx >= 5:
            break
        for cate in category:
            logger_train_data.add(f"dataloader_{idx}", torch_batch[cate].detach().cpu().numpy())

    for idx, (torch_batch) in enumerate(val_loader):
        if idx >= 5:
            break
        for cate in category:
            logger_val_data.add(f"dataloader_{idx}", torch_batch[cate].detach().cpu().numpy())

    logger_train_data.save("../result/data_train_paddle.npy")
    logger_val_data.save("../result/data_val_paddle.npy")

    print('finished')