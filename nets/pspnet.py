import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from nets.mobilenetv3 import mobilenetv3
from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50


class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        #--------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.relu3 = model.relu3
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x


class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetV3, self).__init__()
        from functools import partial

        model = mobilenetv3(pretrained)
        self.features = model.features

        self.total_idx = len(self.features)  # 16 (0~15层)
        self.down_idx = [1, 2, 4, 7, 11, 13]

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        # --------------------------------------------------------------------------------------------#
        # if downsample_factor == 16:  # 3次长和宽压缩
            # for i in range(self.down_idx[-2], self.total_idx):  # (11~15层)
            #     self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):  # 膨胀卷积
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x208 = self.features[:2](x)    # 第1个bneck
        x104 = self.features[:4](x)    # 第2个bneck
        x52 = self.features[:7](x)     # 第3个bneck
        x26_1 = self.features[:11](x)  # 第4个bneck
        x26_2 = self.features[:13](x)  # 第5个bneck
        x = self.features[:16](x)      # 第6个bneck

        return x208, x104, x52, x26_1, x26_2, x
 
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):  # pool_sizes=[1, 2, 3, 6]
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)  # 272/4=68
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #   13, 13, 272 + 13, 13, 68 + 13, 13, 68 + 13, 13, 68 + 13, 13, 68 = 13, 13, 544
        #-----------------------------------------------------#
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])
        
        # 13, 13, 544 -> 13, 13, 68
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):  # 平均池化
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)  # 自适应划分网格平均池化
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]  # 416*416
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])  # 调整尺寸叠加主干特征
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

def conv_dw(filter_in, filter_out, stride = 1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0   #保证卷积不会对尺寸产生变化
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_three_conv(filters_list, in_filters):  #   三次卷积块
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="mobilenetv3", pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone=="resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        elif backbone=="mobilenetv3":
            #----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [26,26,80]
            #   o为主干部分     [13,13,272]
            #----------------------------------#
            self.backbone = MobileNetV3(downsample_factor, pretrained)
            # aux_channel = 96
            out_channel = 112+160  # 272
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        #--------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   13,13,272 -> 13,13,68 -> 13,13,numclass
        #--------------------------------------------------------------#
        self.master_branch0 = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),  #   13,13,272 -> 13,13,68
            # nn.Conv2d(out_channel//4, num_classes, kernel_size=1)  #  30,30,80 -> 30,30,num_classes
        )

        self.master_branch1 = nn.Sequential(
            _PSPModule(148, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),  # 26*26*148 -> 26*26*37
            # nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )

        self.final_classification = nn.Sequential(
            # _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(24, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv52 = make_three_conv([38, 77], 77)
        self.conv104 = make_three_conv([31, 62], 62)
        self.conv208 = make_three_conv([24, 47], 47)

        if self.aux_branch:
            #---------------------------------------------------#
            #	利用特征获得预测结果
            #   26, 26, 80 -> 26, 26, 68 -> 26, 26, numclasses
            #---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(80, out_channel//8, kernel_size=3, padding=1, bias=False),  # 26, 26, 80 -> 26, 26, 68
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)  # 26, 26, 68 -> 26, 26, numclasses
            )

        # self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])  # 416, 416
        x208, x104, x52, x26_1, x26_2, x = self.backbone(x)
        # print("x208:", "x104:", "x52:", "x26_1:", "x26_2: x:", x208.size(), x104.size(), x52.size(), x26_1.size(), x26_2.size(), x.size())
        x26_2 = self.maxpool(x26_2)  # 26, 26, 112 -> 13, 13, 112
        # print('x26_2.size:', x26_2.size())
        # print("x:", x.size())
        x = torch.cat([x, x26_2], 1)  # 13, 13, 160+112=272

        x = self.master_branch0(x)  # 13, 13, 272 -> 13, 13, 68
        x = self.upsample(x)        # 13, 13, 68 -> 26, 26, 68
        x_cat0 = torch.cat([x, x26_1], dim=1)  # 26, 26, 80+68=148

        x_cat0 = self.master_branch1(x_cat0)   # 26, 26, 148 -> 26, 26, 37
        x_cat0 = self.upsample(x_cat0)  # 26, 26, 37 -> 52, 52, 37
        x_cat1 = torch.cat([x_cat0, x52], 1)  # 52, 52, 37+40=77

        x_cat1 = self.conv52(x_cat1)     # 52, 52, 77 -> 52, 52, 38
        x_cat1 = self.upsample(x_cat1)   # 52, 52, 38 -> 104, 104, 38
        x_cat2 = torch.cat([x_cat1, x104], 1)  # 104, 104, 38+24=62

        x_cat2 = self.conv104(x_cat2)    # 104, 104, 62 -> 104, 104, 31
        x_cat2 = self.upsample(x_cat2)   # 104, 104, 31 -> 208, 208, 31
        x_cat3 = torch.cat([x_cat2, x208], 1)  # 208, 208, 31+16=47

        x_cat3 = self.conv208(x_cat3)    # 208, 208, 47 -> 208, 208, 24
        x_cat3 = self.upsample(x_cat3)   # 208, 208, 24 -> 416, 416, 24
        output = self.final_classification(x_cat3)  # 416, 416, 24 -> 416, 416, num_classes

        # output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)  # 调整到输入图片大小416*416
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x26_1)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)  # 调整到输入图片大小416*416
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
