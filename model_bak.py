import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet


# def init_weights(m):
#     if isinstance(m, nn.Conv2D) or isinstance(m, nn.Linear):
#         nn.initializer.Normal(0,1e-3)(m.weight)
#         if m.bias is not None:
#             nn.initializer.Constant(0)(m.bias)
#     elif isinstance(m, nn.Conv2DTranspose):
#         nn.initializer.Normal(0, 1e-3)(m.weight)
#         if m.bias is not None:
#             nn.initializer.Constant(0)(m.bias)
#     elif isinstance(m, nn.BatchNorm2D):
#         nn.initializer.Constant(1)(m.weight)
#         nn.initializer.Constant(0)(m.bias)

def init_weights(m):
    if isinstance(m, nn.Conv2D) or isinstance(m, nn.Linear):
        nn.initializer.Constant(value=1e-3)(m.weight)
        if m.bias is not None:
            nn.initializer.Constant(value=0.)(m.bias)
    elif isinstance(m, nn.Conv2DTranspose):
        nn.initializer.Constant(value=1e-3)(m.weight)
        if m.bias is not None:
            nn.initializer.Constant(value=0.)(m.bias)
    elif isinstance(m, nn.BatchNorm2D):
        nn.initializer.Constant(value=1)(m.weight)
        nn.initializer.Constant(value=0.)(m.bias)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding)
    )
    if bn:
        layers.append(nn.BatchNorm2D(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2))  # 不确定是不是inplace的
    layers = nn.Sequential(*layers)

    # initialize the weights

    for m in layers:
        init_weights(m)


    return layers


def convt_bn_relu(in_channels, out_channels, kernel_size,
                  stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2DTranspose(in_channels, out_channels, kernel_size, stride, padding, output_padding)
    )
    if bn:
        layers.append(nn.BatchNorm2D(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2))  # not sure about the parameter inplace = True
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
            self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)

        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)

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

        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768, out_channels=128, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128), out_channels=64, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64), out_channels=64, kernel_size=kernel_size,
                                    stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64, kernel_size=kernel_size,
                                    stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1,
                                   stride=1, bn=False, relu=False)

    def forward(self, x):
        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(x['d'])
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = paddle.concat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        print(conv1[0,0,0,:10])
        conv2 = self.conv2(conv1)
        print(conv2[0, 0, 0, :10])
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        print(conv3[0, 0, 0, :10])
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        print(conv4[0, 0, 0, :10])
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        print(conv5[0, 0, 0, :10])
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76
        print(conv6[0, 0, 0, :10])

        # decoder
        convt5 = self.convt5(conv6)
        y = paddle.concat((convt5, conv5), 1)
        print(convt5[0, 0, 0, :10])
        convt4 = self.convt4(y)
        y = paddle.concat((convt4, conv4), 1)
        print(convt4[0, 0, 0, :10])
        convt3 = self.convt3(y)
        y = paddle.concat((convt3, conv3), 1)
        print(convt3[0, 0, 0, :10])
        convt2 = self.convt2(y)
        y = paddle.concat((convt2, conv2), 1)
        print(convt2[0, 0, 0, :10])
        convt1 = self.convt1(y)
        print(convt1[0, 0, 0, :10])
        y = paddle.concat((convt1, conv1), 1)
        y = self.convtf(y)
        print(y[0, 0, 0, :10])
        y += (x['d'] / 100.0)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m
