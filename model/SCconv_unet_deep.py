# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:35:39 2020

@author: 45780
"""

import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param
import torchsummary
from torch.nn import functional as F
from torch.nn import init

class SCconv_UNet_deepsup(nn.Module):

    def __init__(self, in_channels=1, n_classes=4, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(SCconv_UNet_deepsup, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv1(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv1(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv1(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv1(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv1(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        
        #deep Supervision

        self.deepsup_3 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)
        self.output_3_up = nn.Upsample(scale_factor=8, mode='bilinear')
        self.deepsup_2 = nn.Conv2d(filters[2], n_classes, kernel_size=1, stride=1, padding=0)
        self.output_2_up = nn.Upsample(scale_factor=4, mode='bilinear')
        self.deepsup_1 = nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1, padding=0)
        self.output_1_up = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256
        
        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up4_deep = self.deepsup_3(up4)
        up4_deep = self.output_3_up(up4_deep)
        
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up3_deep = self.deepsup_2(up3)
        up3_deep = self.output_2_up(up3_deep)
        
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up2_deep = self.deepsup_1(up2)
        up2_deep = self.output_1_up(up2_deep)
        
        up1 = self.up_concat1(up2,conv1)     # 16*512*512
        

        final = self.final(up1)
#        final=F.softmax(final,dim=1)#对每一行使用Softmax
        up4_deep = F.log_softmax(final,dim=1)
        up3_deep = F.log_softmax(final,dim=1)
        up2_deep = F.log_softmax(final,dim=1)
        final=F.log_softmax(final,dim=1)

        return up4_deep,up3_deep,up2_deep,final

class unetConv1(nn.Module):#编码部分用SCnet，解码不用
    def __init__(self, in_size, out_size, n=2, ks=3, stride=1, padding=1):
        super(unetConv1, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding

        for i in range(1, n+1):
            conv = SCnet(in_size, out_size)
            setattr(self, 'conv%d'%i, conv)
            in_size = out_size

##         initialise the blocks
#        for m in self.children():
#            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

 
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
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_size, out_size, 1))
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue#continue语句可用于循环中，用于跳过当前循环的剩余代码，然后继续进行下一轮的循环。
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
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SCConv(nn.Module):#文章中虚线框部分
    def __init__(self, planes, stride, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    conv3x3(planes, planes), 
                    nn.BatchNorm2d(planes),
                    )
        self.k3 = nn.Sequential(
                    conv3x3(planes, planes), 
                    nn.BatchNorm2d(planes),
                    )
        self.k4 = nn.Sequential(
                    conv3x3(planes, planes, stride), 
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    )
        #         initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class SCnet(nn.Module):
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1):
        super(SCnet, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = conv1x1(inplanes, planes)
        self.bn1_a = nn.BatchNorm2d(planes)

        self.k1 = nn.Sequential(
                    conv3x3(planes, planes, stride), 
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    )

        self.conv1_b = conv1x1(inplanes, planes)
        self.bn1_b = nn.BatchNorm2d(planes)
        
        self.scconv = SCConv(planes, stride, self.pooling_r)

        self.bn3 = nn.BatchNorm2d(planes * 2 )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
    #         initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('SCConv') != -1: continue#continue语句可用于循环中，用于跳过当前循环的剩余代码，然后继续进行下一轮的循环。
            init_weights(m, init_type='kaiming')
 
    def forward(self, x):

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)


        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = torch.cat([out_a, out_b], dim=1)
        out = self.bn3(out)

        out = self.relu(out)

        return out
        
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 512, 512)).cuda()

    unet_plus_plus = SCconv_UNet_deepsup(in_channels=1, n_classes=2).cuda()
    a,b,c,output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))