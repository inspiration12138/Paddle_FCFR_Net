import argparse
import sys
import os

import paddle
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

sys.path.append('..')
import criteria
from model import DepthCompletionNet
from metrics import Result
import helper

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
                    default=1,
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
                    default=True,
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
parser.add_argument('--cpu', default=True, action="store_true", help='run on cpu')
###########################################################################################cpu和pretrained的default
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


def train_one_epoch_paddle(inputs, model, criterion, optimizer_paddle, max_iter, reprod_logger):
    print('start test training...')

    for idx in range(max_iter):
        print('==> start idx {} ...'.format(idx))
        lr = helper.adjust_learning_rate(args.lr, optimizer_paddle, 21)

        inputs['d'][inputs['d'] > 85] = 85
        inputs['d'] = inputs['d'] / 85.0
        inputs['rgb'] = inputs['rgb'] / 255.0
        pred = model(inputs)
        print('==> model output completed')

        inputs['d'] = inputs['d'] * 85.0
        inputs['rgb'] = inputs['rgb'] * 255.0

        depth_loss = criterion(pred, inputs['gt'])  # +depth_criterion(global_features, res_gt)
        print('==> loss output completed')

        reprod_logger.add("loss_{}".format(idx), depth_loss.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx), np.array(lr))

        depth_loss.backward()
        optimizer_paddle.step()
        optimizer_paddle.clear_grad()

    reprod_logger.save("../result/losses_paddle.npy")


if __name__ == '__main__':
    max_iter = 3

    paddle_model = DepthCompletionNet(args)
    paddle_model.eval()
    model_named_params = [
        p for _, p in paddle_model.named_parameters() if not p.stop_gradient
    ]
    optimizer_paddle = paddle.optimizer.Adam(learning_rate=args.lr,parameters=model_named_params,
                                             weight_decay=float(args.weight_decay))
    print("==> model and optimizer create completed.")

    d = np.load("../data/d.npy")
    rgb = np.load("../data/rgb.npy")
    g = np.load("../data/g.npy")
    gt = np.load("../data/gt.npy")
    inputs = {"rgb": paddle.to_tensor(rgb), "d": paddle.to_tensor(d), "g": paddle.to_tensor(g),
              "gt": paddle.to_tensor(gt)}

    print('==> data load completed')
    # load paddle model
    # load data
    reprod_logger = ReprodLogger()
    print('==> logger create completed')

    depth_criterion = criteria.MaskedMSELoss() if (
            args.criterion == 'l2') else criteria.MaskedL1Loss()
    photometric_criterion = criteria.PhotometricLoss()
    smoothness_criterion = criteria.SmoothnessLoss()

    print('==> criteria set completed')

    paddle.set_device('cpu')
    train_one_epoch_paddle(inputs, paddle_model, depth_criterion, optimizer_paddle, max_iter, reprod_logger)
    # save the paddle output
