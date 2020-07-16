# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:59:10 2019

@author: Fsl
"""
#import _init_paths
import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param
import torchsummary
from torch.nn import functional as F
from torch.nn import init
class SE_unet(nn.Module):

    def __init__(self, in_channels=1, n_classes=4, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(SE_unet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.SE_Block_1 = SE_Block(planes=filters[0],r = 16)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.SE_Block_2 = SE_Block(planes=filters[1],r = 16)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.SE_Block_3 = SE_Block(planes=filters[2],r = 16)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.SE_Block_4 = SE_Block(planes=filters[3],r = 16)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        

    def forward(self, inputs):
        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.SE_Block_1(conv1)
        maxpool1 = self.maxpool(maxpool1)       # 16*256*256
        
        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.SE_Block_2(conv2)
        maxpool2 = self.maxpool(maxpool2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.SE_Block_3(conv3)
        maxpool3 = self.maxpool(maxpool3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.SE_Block_4(conv4)
        maxpool4 = self.maxpool(maxpool4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final = self.final(up1)
#        final=F.sigmoid(final)
        final=F.log_softmax(final,dim=1)

        return final
    
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)#上采样，反卷积
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_size, out_size, 1))
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)
    
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
        
class SE_Block(nn.Module):#自写的
    def __init__(self, planes,r = 16, stride=1, downsample=None):
        super(SE_Block, self).__init__()
#中间是se模块的部分，其他是resnet正常部分
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes , planes // r, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // r, planes , kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
#
    def forward(self, x):
        input = x
        out1 = self.global_pool(x)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)
        res = out1 * input

        return res
#model = UNet()
#torchsummary.summary(model, (1, 512, 512))
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 512, 512)).cuda()

    unet_plus_plus = SE_unet(in_channels=1, n_classes=2).cuda()
    output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))