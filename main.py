import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import argparse
import time

import sys 
sys.path.append('/home/aistudio/external-libraries')

import paddle
import paddle.optimizer
import paddle.distributed as dist

from dataloaders.kitti_loader import load_calib,oheight,owidth,input_options,KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w',
                    '--workers',
                    default=4,
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
                    default=2,
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
                    default='/home/aistudio/dataset/paddle_part',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd', # 'gd
                    choices=input_options,
                    help='input: | '.join(input_options))
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

cuda = paddle.is_compiled_with_cuda() and not args.cpu
if cuda:
    paddle.set_flags({'FLAGS_benchmark':True})
    # import paddle.backends.cudnn as cudnn     ###############################
    # cudnn.benchmark = True                    ###############################
    device = paddle.device.set_device("gpu")
    print("=> using gpu for computation.")
else:
    device = paddle.device.set_device('cpu')
    print("=> using cpu for computation.")

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
    if cuda:
        kitti_intrinsics = kitti_intrinsics.cuda()


def iterate(mode, args, loader, model, optim, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optim, epoch)
    else:
        model.eval()
        lr = 0

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {
            key: val
            for key, val in batch_data.items() if val is not None
        }

        batch_data['d'] = paddle.transpose(batch_data['d'],perm=[0,3,1,2])
        batch_data['rgb'] = paddle.transpose(batch_data['rgb'],perm=[0,3,1,2])
        batch_data['gt'] = paddle.transpose(batch_data['gt'],perm=[0,3,1,2])

        batch_data['d'][batch_data['d'] > 85] = 85
        batch_data['d'] /= 85.0
        batch_data['rgb'] /= 255.0
        # print(batch_data['d'].max(), batch_data['rgb'].max())
        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None

        batch_data['d'].stop_gradient = False
        batch_data['rgb'].stop_gradient = False
        batch_data['gt'].stop_gradient = False


        data_time = time.time() - start

        start = time.time()
        pred = model(batch_data)
        # print(pred.max(),pred.min())
        # print(pred.max(), depth_pred.max(), lidar_pred.max(), global_features.max())
        batch_data['d'] *= 85.0
        batch_data['rgb'] *= 255.0
        # print(batch_data['d'].max(), batch_data['rgb'].max(), gt.max())
        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                # res_gt = gt-batch_data['d']
                # res_gt[gt==0]=0
                # if i==0:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+10*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # elif i==1:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+2*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # else:


                depth_loss = depth_criterion(pred,gt)  # +depth_criterion(global_features, res_gt)
                # depth_loss = depth_criterion(pred, gt)+0.1*depth_criterion(depth_pred, gt)+0.1*depth_criterion(lidar_pred, gt)+0.1*depth_criterion(global_features, res_gt)
                mask = paddle.to_tensor(gt < 1e-3,dtype = paddle.float32)

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_,
                                              batch_data['r_mat'],
                                              batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(
                        rgb_curr_, warped_, mask_) * (
                                                    2 ** (scale - num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0

            # backprop
            loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss

            optim.clear_grad()
            loss.backward()
            optim.step()

        gpu_time = time.time() - start
        if mode != 'train':
            pred[pred < 0.9] = 0.9
            pred[pred > 85] = 85
            # pred[depth_pred<0.9] = 0.9
            # pred[depth_pred>85] = 85
            # pred[lidar_pred<0.9] = 0.9
            # pred[lidar_pred>85] = 85
        # measure accuracy and record loss
        with paddle.no_grad():

            mini_batch_size = next(iter(batch_data.values())).shape[0]
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':


###############################################################################################


                result.evaluate(pred.detach(), gt.detach(), pred.detach(), pred.detach(), photometric_loss)
            [
                m.update(result, gpu_time, data_time, mini_batch_size)
                for m in meters
            ]
            logger.conditional_print(mode, i, epoch, lr, len(loader),block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred,epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)
        # if mode=='train':
        #    break

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best


@paddle.no_grad()
def iterate_val(mode, args, loader, model, optim, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optim, epoch)
    else:
        model.eval()
        lr = 0

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {
            key: val
            for key, val in batch_data.items() if val is not None
        }

        batch_data['d'] = paddle.transpose(batch_data['d'],perm=[0,3,1,2])
        batch_data['rgb'] = paddle.transpose(batch_data['rgb'],perm=[0,3,1,2])
        batch_data['gt'] = paddle.transpose(batch_data['gt'],perm=[0,3,1,2])

        batch_data['d'][batch_data['d'] > 85] = 85
        batch_data['d'] /= 85.0
        batch_data['rgb'] /= 255.0
        # print(batch_data['d'].max(), batch_data['rgb'].max())
        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - start

        start = time.time()
        pred = model(batch_data)
        # print(pred.max(),pred.min())
        # print(pred.max(), depth_pred.max(), lidar_pred.max(), global_features.max())
        batch_data['d'] *= 85.0
        batch_data['rgb'] *= 255.0
        # print(batch_data['d'].max(), batch_data['rgb'].max(), gt.max())
        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).astype(paddle.float32)
            elif 'dense' in args.train_mode:
                # res_gt = gt-batch_data['d']
                # res_gt[gt==0]=0
                # if i==0:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+10*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # elif i==1:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+2*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # else:
                depth_loss = depth_criterion(pred, gt)  # +depth_criterion(global_features, res_gt)
                # depth_loss = depth_criterion(pred, gt)+0.1*depth_criterion(depth_pred, gt)+0.1*depth_criterion(lidar_pred, gt)+0.1*depth_criterion(global_features, res_gt)
                mask = (gt < 1e-3).astype(paddle.float32)

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                # num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_,
                                              batch_data['r_mat'],
                                              batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(
                        rgb_curr_, warped_, mask_) * (
                                                    2 ** (scale - num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0

            # backprop
            loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss
            optim.clear_grad()
            loss.backward()
            optim.step()

        gpu_time = time.time() - start
        if mode != 'train':
            pred[pred < 0.9] = 0.9
            pred[pred > 85] = 85
            # depth_pred[depth_pred<0.9] = 0.9
            # depth_pred[depth_pred>85] = 85
            # lidar_pred[lidar_pred<0.9] = 0.9
            # lidar_pred[lidar_pred>85] = 85
        # measure accuracy and record loss
        with paddle.no_grad():
            mini_batch_size = next(iter(batch_data.values())).shape[0]
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.detach(), gt.detach(), pred.detach(), pred.detach(), photometric_loss)
            [
                m.update(result, gpu_time, data_time, mini_batch_size)
                for m in meters
            ]
            logger.conditional_print(mode, i, epoch, lr, len(loader),
                                     block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                   epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)
        # if mode=='train' and i==100:
        #    break

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best


def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = paddle.load(args.evaluate, map_location=device)
            args = checkpoint['args']
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    elif args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = paddle.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')

    paddle.device.set_device('gpu')

    dist.init_parallel_env()

    model = DepthCompletionNet(args)
    model_named_params = [ p for _, p in model.named_parameters() if not p.stop_gradient]
    optim = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model_named_params, weight_decay=args.weight_decay)

    optim.regularization = None

    print("completed.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = paddle.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_loader = paddle.io.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            use_buffer_reader=True)
        print("\t==> train_loader size:{}".format(len(train_loader)))
    val_dataset = KittiDepth('val', args)
    val_loader = paddle.io.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        use_buffer_reader=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate("val", args, val_loader, model, None, logger,
                                  checkpoint['epoch'])
        return

    # main loop
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optim, logger,epoch)  # train for one epoch
        result, is_best = iterate_val("val", args, val_loader, model, None, logger,epoch)  # evaluate on validation set
        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optim.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)
        paddle.device.cuda.empty_cache()

if __name__ == '__main__':
    main()









