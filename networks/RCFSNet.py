import torch
from torchvision import models
import torch.nn as nn
import numpy as np
# from resnet import resnet34
# import resnet
from torch.nn import functional as F
#import torchsummary
from torch.nn import init
from functools import partial

nonlinearity = partial(F.relu,inplace=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)

# 521 对第二层进行双注意力
class CDAM2(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=9):
        super(CDAM2, self).__init__()
        self.h = 256
        self.w = 256

        self.relu1 = nn.ReLU()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(256, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(256, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(64 * 5 * 4, 64*5, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=64*5*2, out_channels=64*5*2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=64*5*2, out_channels=64*5*2, kernel_size=1, padding=0, stride=1)

        # 横卷
        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.h, 1), padding=(0, 0), stride=1)
        # 竖卷
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.w), padding=(0, 0), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        n, c, h, w = x.size()
        y1 = self.avg_pool_x(x)
        y1 = y1.reshape(n, c, h)
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        y2 = self.avg_pool_y(x)
        y2 = y2.reshape(n, c, w)

        # Two different branches of ECA module
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        yac = self.conv111(torch.cat([x * y1.expand_as(x), x * y2.expand_as(x)],dim=1))

        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max,_ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)],dim=1))

        out = self.convout(torch.cat([yac, yap], dim=1))

        return out

# 531 （第三层双注意力设置）
class CDAM3(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=7):
        super(CDAM3, self).__init__()
        self.h = 128
        self.w = 128

        self.relu1 = nn.ReLU()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(128, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(128, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(64*4*5, 64*5, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=64*5*2, out_channels=64*5*2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=64*5*2, out_channels=64*5*2, kernel_size=1, padding=0, stride=1)

        #横卷
        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.h, 1), padding=(0, 0), stride=1)
        #竖卷
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.w), padding=(0, 0), stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        n, c, h, w = x.size()

        y1 = self.avg_pool_x(x)
        #y1=torch.squeeze(y1)
        y1 = y1.reshape(n, c, h)
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        y2 = self.avg_pool_y(x)
        y2 = y2.reshape(n, c, w)

        # Two different branches of ECA module
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        yac = self.conv111(torch.cat([x * y1.expand_as(x), x * y2.expand_as(x)], dim=1))

        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max,_ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)],dim=1))

        out = self.convout(torch.cat([yac, yap], dim=1))

        return out

# 541 对第四层进行双通道
class CDAM4(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=5):
        super(CDAM4, self).__init__()
        self.h = 64
        self.w = 64
        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(64, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(64, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(64 * 4 * 5, 64*5, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=64*5*2, out_channels=64*5*2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=64*5*2, out_channels=64*5*2, kernel_size=1, padding=0, stride=1)

        # 横卷
        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.h, 1), padding=(0, 0), stride=1)
        # 竖卷
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.w), padding=(0, 0), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        n, c, h, w = x.size()

        y1 = self.avg_pool_x(x)
        # y1=torch.squeeze(y1)
        y1 = y1.reshape(n, c, h)
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        y2 = self.avg_pool_y(x)
        y2 = y2.reshape(n, c, w)

        # Two different branches of ECA module
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2.transpose(-1, -2)))).transpose(-1, -2).reshape(n, c, 1, 1))

        yac = self.conv111(torch.cat([x * y1.expand_as(x), x * y2.expand_as(x)],dim=1))

        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)], dim=1))
        out = self.convout(torch.cat([yac, yap], dim=1))
        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class RCFSNet(nn.Module):
    def __init__(self, num_classes=1, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                 base_channel=32):
        super(RCFSNet, self).__init__()
        filters = [64, 64, 128, 256, 512]

        ## -------------Encoder--------------
        super().__init__()
        resnet = models.resnet34(pretrained=False)
        #本地加载resnet34权重
        #resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.up = nn.Upsample(scale_factor=2)
        self.ConvBnRelu = ConvBnRelu(in_planes=512)

        self.CDAM2 = CDAM2()
        self.CDAM3 = CDAM3()
        self.CDAM4 = CDAM4()

        self.hd5_d1 = nn.Upsample(scale_factor=16)
        self.hd4_d1 = nn.Upsample(scale_factor=8)
        self.hd3_d1 = nn.Upsample(scale_factor=4)
        self.hd2_d1 = nn.Upsample(scale_factor=2)
        self.MSCE = MSCE(channel=512)

        self.decoder5 = DecoderBlock(filters[-1], filters[-2], relu=False, last=True)  # 256
        self.decoder4 = DecoderBlock(filters[-2], filters[-3], relu=False)  # 128
        self.decoder3 = DecoderBlock(filters[-3], filters[-4], relu=False)  # 64
        self.decoder2 = DecoderBlock(filters[-4], filters[-4], relu=False)  # 32

        self.FSFF_2 = FSFF_2([filters[0], filters[1], filters[4]], width=filters[1], up_kwargs=2)
        self.FSFF_3 = FSFF_3([filters[1], filters[2], filters[4]], width=filters[1], up_kwargs=2)
        self.FSFF_4 = FSFF_4([filters[2], filters[3], filters[4]], width=filters[1], up_kwargs=2)
        self.main_head = BaseNetHead(filters[0], num_classes, 2,
                                     is_aux=False, norm_layer=norm_layer)
        self.conv5 = nn.Conv2d(in_channels=filters[-1],out_channels=filters[1],kernel_size=3,
                               stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[-2],out_channels=filters[1],kernel_size=3,
                               stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[-3],out_channels=filters[1],kernel_size=3,
                               stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[-4],out_channels=filters[1],kernel_size=3,
                               stride=1,padding=1)
        self.relu = nn.ReLU()

        self.conv256 = nn.Conv2d(in_channels=512, out_channels=256,kernel_size=3,
                               stride=1,padding=1)
        self.conv128 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,
                                 stride=1, padding=1)
        self.conv64_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
                                 stride=1, padding=1)
        self.conv64_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
                                  stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        h1 = self.firstconv(inputs)
        h1 = self.firstbn(h1)
        h1 = self.firstrelu(h1)
        h2 = self.firstmaxpool(h1)
        # print('input:',input.shape)
        # print('x0_0:',x0_0.shape)
        h2 = self.encoder1(h2)
        # print('x1_0:',x1_0.shape)
        h3 = self.encoder2(h2)
        # print('x2_0:',x2_0.shape)
        h4 = self.encoder3(h3)
        # print('x3_0:',x3_0.shape)
        h5 = self.encoder4(h4)
        hd5 = self.MSCE(h5)
        '''
        m2 = self.mce_2(h1, h2, h5)
        m3 = self.mce_3(h1, h3, h5)
        m4 = self.mce_4(h1, h4, h5)
        '''

        m2 = self.FSFF_2(h1, h2, h3, h4, h5) #320  256
        m3 = self.FSFF_3(h1, h2, h3, h4, h5) #320  128
        m4 = self.FSFF_4(h1, h2, h3, h4, h5) #320   64

        # d_bottom=self.bottom(c5)
        # d5=d_bottom+c5           #512
        d4 = self.relu(self.conv256(torch.cat([self.decoder5(hd5), m4], dim=1)))  # 256  64   1
        d3 = self.relu(self.conv128(torch.cat([self.decoder4(d4), m3], dim=1)))  # 256  64   1
        d2 = self.relu(self.conv64_1(torch.cat([self.decoder3(d3), m2], dim=1)))  # 256  64   1
        d1 = self.relu(self.conv64_2(torch.cat([self.decoder2(d2), h1], dim=1)))  # 256  64   1
        
        main_out = F.sigmoid(self.main_head(d1 + self.conv5(self.hd5_d1(hd5)) + self.conv4(self.hd4_d1(d4)) + self.conv3(
                self.hd3_d1(d3)) + self.conv2(self.hd2_d1(d2))))

        return main_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)


class FSFF_2(nn.Module):
    def __init__(self, in_channels, width=64, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(FSFF_2, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(width*5, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
        self.CDAM = CDAM2()

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4]), self.conv1(inputs[-5])]
        _, _, h, w = feats[-2].size()
        feats[-1] = F.interpolate(feats[-1], (h, w))
        feats[-3] = F.interpolate(feats[-3], (h, w))
        feats[-4] = F.interpolate(feats[-4], (h, w))
        feats[-5] = F.interpolate(feats[-5], (h, w))
        feat1 = torch.cat((feats[-1], feats[-2], feats[-3], feats[-4], feats[-5]), dim=1)
        feat2 = self.conv_out(self.CDAM(feat1))
        return feat2

class FSFF_3(nn.Module):
    def __init__(self, in_channels, width=64, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(FSFF_3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(width*5, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(width*2))

        self.CDAM = CDAM3()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.normal_(m.weight.data, 1.0, 0.02)
                    init.constant_(m.bias.data, 0.0)
                    # m.weight.data.fill_(1)
                    # m.bias.data.zero_()

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4]), self.conv1(inputs[-5])]
        _, _, h, w = feats[-3].size()
        feats[-1] = F.interpolate(feats[-1], (h, w))
        feats[-2] = F.interpolate(feats[-2], (h, w))
        feats[-4] = F.interpolate(feats[-4], (h, w))
        feats[-5] = F.interpolate(feats[-5], (h, w))
        feat1 = torch.cat((feats[-1], feats[-2], feats[-3], feats[-4], feats[-5]), dim=1)
        feat2 = self.conv_out(self.CDAM(feat1))
        return feat2


class FSFF_4(nn.Module):
    def __init__(self, in_channels, width=64, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(FSFF_4, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(5 * width, 256, 1, padding=0, bias=False),
            nn.BatchNorm2d(width*4))

        self.CDAM=CDAM4()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.normal_(m.weight.data, 1.0, 0.02)
                    init.constant_(m.bias.data, 0.0)
                    # m.weight.data.fill_(1)
                    # m.bias.data.zero_()

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4]), self.conv1(inputs[-5])]
        _, _, h, w = feats[-4].size()
        feats[-1] = F.interpolate(feats[-1], (h, w))
        feats[-2] = F.interpolate(feats[-2], (h, w))
        feats[-3] = F.interpolate(feats[-3], (h, w))
        feats[-5] = F.interpolate(feats[-5], (h, w))
        feat1 = torch.cat((feats[-1], feats[-2], feats[-3], feats[-4], feats[-5]), dim=1)
        feat2 = self.conv_out(self.CDAM(feat1))
        return feat2

class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                           has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear',
                              align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output

class MSCE(nn.Module):
    def __init__(self, channel):
        super(MSCE, self).__init__()
        self.dilate11 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate22 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate33 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate44 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=(3,1), dilation=1, padding=(1, 0))
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=(3,1), dilation=2, padding=(2, 0))
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=(3,1), dilation=4, padding=(4, 0))
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=(3,1), dilation=8, padding=(8, 0))
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=(1,3), dilation=1, padding=(0, 1))
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=(1,3), dilation=2, padding=(0, 2))
        self.dilate7 = nn.Conv2d(channel, channel, kernel_size=(1,3), dilation=4, padding=(0, 4))
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=(1,3), dilation=8, padding=(0, 8))
        self.dconv = nn.Conv2d(channel*5, channel, kernel_size=(1,1), stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ASPPH = ASPPPoolingH(in_channels=channel,out_channels=channel)
        self.ASPPW = ASPPPoolingW(in_channels=channel, out_channels=channel)

        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate11_out = nonlinearity(self.dilate11(x))
        dilate21_out = nonlinearity(self.dilate22(dilate11_out))
        dilate31_out = nonlinearity(self.dilate33(dilate21_out))
        dilate41_out = nonlinearity(self.dilate44(dilate31_out))

        dilate1_out = self.conv1(dilate11_out+dilate21_out+dilate31_out+dilate41_out)

        dilate12_out = nonlinearity(self.dilate1(x))
        dilate22_out = nonlinearity(self.dilate2(dilate12_out))
        dilate32_out = nonlinearity(self.dilate3(dilate22_out))
        dilate42_out = nonlinearity(self.dilate4(dilate32_out))

        dilate2_out = self.conv2(dilate12_out+dilate22_out+dilate32_out+dilate42_out)

        dilate13_out = nonlinearity(self.dilate5(x))
        dilate23_out = nonlinearity(self.dilate6(dilate13_out))
        dilate33_out = nonlinearity(self.dilate7(dilate23_out))
        dilate43_out = nonlinearity(self.dilate8(dilate33_out))

        dilate3_out = self.conv3(dilate13_out+dilate23_out+dilate33_out+dilate43_out)

        dilateH_out = self.ASPPH(x)
        dilateW_out = self.ASPPW(x)

        outsum = torch.cat([dilate1_out,dilate2_out,dilate3_out,dilateH_out,dilateW_out], dim=1)

        out = self.dconv(outsum)
        out = self.gamma*out+x*(1-self.gamma)
        return out


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes=512, out_planes=512, ksize=3, stride=1, pad=1, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last == False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs


class ASPPPoolingH(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingH, self).__init__(
            nn.AdaptiveAvgPool2d((32,1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPPoolingW(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingW, self).__init__(
            nn.AdaptiveAvgPool2d((1,32)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

def lunwen3():
    return RCFSNet()



if __name__ == "__main__":
    import numpy as np
    a = np.random.rand(1, 3, 1024, 1024)
    a = torch.from_numpy(a).to(torch.float32)
    RCFSNet = RCFSNet()
    predicted = RCFSNet(a)
    print(predicted.shape)

