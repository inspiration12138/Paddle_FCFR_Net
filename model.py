import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet

import numpy as np
import os
import cv2

def init_weights(m):
    if isinstance(m, nn.Conv2D) or isinstance(m, nn.Linear):
        nn.initializer.Normal(0, 1e-3)(m.weight)
        if m.bias is not None:
            nn.initializer.Constant(0)(m.bias)
    elif isinstance(m, nn.Conv2DTranspose):
        nn.initializer.Normal(0, 1e-3)(m.weight)
        if m.bias is not None:
            nn.initializer.Constant(0)(m.bias)
    elif isinstance(m, nn.BatchNorm2D):
        nn.initializer.Constant(1)(m.weight)
        nn.initializer.Constant(0)(m.bias)


# def init_weights(m):
#     if isinstance(m, nn.Conv2D) or isinstance(m, nn.Linear):
#         nn.initializer.Constant(value=1e-3)(m.weight)
#         if m.bias is not None:
#             nn.initializer.Constant(value=0.)(m.bias)
#     elif isinstance(m, nn.Conv2DTranspose):
#         nn.initializer.Constant(value=1e-3)(m.weight)
#         if m.bias is not None:
#             nn.initializer.Constant(value=0.)(m.bias)
#     elif isinstance(m, nn.BatchNorm2D):
#         nn.initializer.Constant(value=1)(m.weight)
#         nn.initializer.Constant(value=0.)(m.bias)

def my_conv(tensor_image):
    kernel = [[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]]
    kernel = paddle.to_tensor(kernel, stop_gradient=True).astype(paddle.float32).expand(shape=(1, 1, 5, 5))
    attr_p = paddle.ParamAttr(initializer=nn.initializer.Assign(kernel))
    weight = paddle.create_parameter(shape=[1, 1, 5, 5], dtype='float32', attr=attr_p)

    return F.conv2d(tensor_image, weight, padding=2)

def feature_fuse(d1, d2):

    # set M=N=3
    # print(d1.shape)
    b,c,h,w = d1.shape
    d1_2 = paddle.multiply(d1,d1)
    d2_2 = paddle.multiply(d2,d2)
    E1 = my_conv(d1_2.reshape((-1,1,h,w)))
    E2 = my_conv(d2_2.reshape((-1,1,h,w)))
    E1 = E1.reshape((b,c,h,w))
    E2 = E2.reshape((b,c,h,w))


    mask1 = paddle.ones_like(d1)
    place1 = np.where(E1<E2)
    mask1[place1] = 0
    F11 = d1*mask1

    mask2 = paddle.ones_like(d1)
    place2 = np.where(E1>=E2)
    mask2[place2] = 0
    F12 = d2 * mask2
    feature1 = F11+F12

    return feature1 * 2

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2D(in_channels=in_channels,out_channels=out_channels,
                  kernel_size=kernel_size,stride=stride,padding=padding,
                  bias_attr=bias))
    if bn:
        layers.append(nn.BatchNorm2D(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers:
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2DTranspose(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           output_padding=output_padding,
                           bias_attr=bias))
    if bn:
        layers.append(nn.BatchNorm2D(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers:
        init_weights(m)

    return layers


class DepthCompletionNet(nn.Layer):
    def __init__(self, args):
        assert (
                args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            args.layers)
        super(DepthCompletionNet, self).__init__()
        self.modality = args.input

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            #   self.conv1_d = conv_bn_relu(1,channels,kernel_size=3,stride=1,padding=1)
            self.conv1_d_2 = conv_bn_relu(1, 64, kernel_size=3, stride=1, padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3, 64, kernel_size=3, stride=1, padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        # self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model.layer1
        self.conv3 = pretrained_model.layer2
        self.conv4 = pretrained_model.layer3
        self.conv5 = pretrained_model.layer4
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=(512), out_channels=256, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=(768 + 512), out_channels=128, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128 + 256), out_channels=64, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64 + 128), out_channels=64, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=(128 + 64), out_channels=64, kernel_size=kernel_size, stride=1,
                                    padding=1)
        self.convtf = conv_bn_relu(in_channels=(128 + 64), out_channels=1, kernel_size=1, stride=1, bn=False,
                                   relu=False)

        ##############
        ## second path
        pretrained_model2 = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model2.apply(init_weights)
        # self.maxpool = pretrained_model._modules['maxpool']
        self.conv2_2 = pretrained_model2.layer1
        self.conv3_2 = pretrained_model2.layer2
        self.conv4_2 = pretrained_model2.layer3
        self.conv5_2 = pretrained_model2.layer4
        del pretrained_model2  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6_2 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # first layer
        # x['d'] /= 85.0
        # x['rgb'] /= 255.0
        if 'd' in self.modality:
            # x['d'] = paddle.transpose(x['d'],perm=[0,3,1,2])
            # conv1_d = self.conv1_d(x['d'])
            conv1_2 = self.conv1_d_2(x['d'])
        if 'rgb' in self.modality:
            # x['rgb'] = paddle.transpose(x['rgb'], perm=[0, 3, 1, 2])
            conv1 = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            # x['g'] = paddle.transpose(x['g'], perm=[0, 3, 1, 2])
            conv1 = self.conv1_img(x['g'])

        # if self.modality == 'rgbd' or self.modality == 'gd':
        #    conv1 = paddle.cat((conv1_d, conv1_img), 1)
        # else:
        #    conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv2_2 = self.conv2_2(conv1_2)
        b, c, h, w = conv2.shape
        cat_all = paddle.concat((conv2, conv2_2), 1)
        cat_reshape = paddle.reshape(cat_all, (b, 2, c, h, w))
        cat_transpose = cat_reshape.transpose([0, 2, 1, 3, 4])
        cat_view = cat_transpose.reshape(shape=[b, -1, h, w])
        conv2 = cat_view[:, :c, :, :]
        conv2_2 = cat_view[:, c:, :, :]

        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv3_2 = self.conv3_2(conv2_2)  # batchsize * ? * 176 * 608
        b, c, h, w = conv3.shape
        cat_all = paddle.concat((conv3, conv3_2), 1)
        cat_reshape = paddle.reshape(cat_all, (b, 2, c, h, w))
        cat_transpose = cat_reshape.transpose([0, 2, 1, 3, 4])
        # cat_transpose = paddle.transpose(cat_reshape, (1, 2))
        cat_view = cat_transpose.reshape(shape=[b, -1, h, w])
        conv3 = cat_view[:, :c, :, :]
        conv3_2 = cat_view[:, c:, :, :]

        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv4_2 = self.conv4_2(conv3_2)  # batchsize * ? * 88 * 304
        b, c, h, w = conv4.shape
        cat_all = paddle.concat((conv4, conv4_2), 1)
        cat_reshape = paddle.reshape(cat_all, (b, 2, c, h, w))
        cat_transpose = cat_reshape.transpose([0, 2, 1, 3, 4])
        # cat_transpose = paddle.transpose(cat_reshape, (1, 2))
        cat_view = cat_transpose.reshape(shape=[b, -1, h, w])
        conv4 = cat_view[:, :c, :, :]
        conv4_2 = cat_view[:, c:, :, :]

        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv5_2 = self.conv5_2(conv4_2)  # batchsize * ? * 44 * 152
        b, c, h, w = conv5.shape
        cat_all = paddle.concat((conv5, conv5_2), 1)
        cat_reshape = paddle.reshape(cat_all, (b, 2, c, h, w))
        cat_transpose = cat_reshape.transpose([0, 2, 1, 3, 4])
        # cat_transpose = paddle.transpose(cat_reshape, (1, 2))
        cat_view = cat_transpose.reshape(shape=[b, -1, h, w])
        conv5 = cat_view[:, :c, :, :]
        conv5_2 = cat_view[:, c:, :, :]

        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76
        conv6_2 = self.conv6_2(conv5_2)  # batchsize * ? * 22 * 76
        b, c, h, w = conv6.shape
        cat_all = paddle.concat((conv6, conv6_2), 1)
        cat_reshape = paddle.reshape(cat_all, (b, 2, c, h, w))
        cat_transpose = cat_reshape.transpose([0, 2, 1, 3, 4])
        # cat_transpose = paddle.transpose(cat_reshape, (1, 2))
        cat_view = cat_transpose.reshape(shape=[b, -1, h, w])
        conv6 = cat_view[:, :c, :, :]
        conv6_2 = cat_view[:, c:, :, :]

        conv6_all = feature_fuse(conv6, conv6_2)
        # decoder

        convt5 = self.convt5(conv6_all)
        y = paddle.concat((convt5, conv5, conv5_2), 1)  # add boosted module?

        convt4 = self.convt4(y)
        y = paddle.concat((convt4, conv4, conv4_2), 1)

        convt3 = self.convt3(y)
        y = paddle.concat((convt3, conv3, conv3_2), 1)

        convt2 = self.convt2(y)
        y = paddle.concat((convt2, conv2, conv2_2), 1)

        convt1 = self.convt1(y)
        y = paddle.concat((convt1, conv1, conv1_2), 1)

        y = self.convtf(y)

        depth_pred = y + x['d']

        if self.training:
            depth_pred = 85 * depth_pred
        else:
            min_distance = 0.9
            depth_pred = F.relu(85 * depth_pred - min_distance) + min_distance

        return depth_pred













