# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:12:48 2020

@author: 45780
"""

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchsummary
from torch.nn import functional as F
from torch.nn import init

from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
from Synchronized.CC import CC_module 

BatchNorm2d = SyncBN#functools.partial(InPlaceABNSync, activation='identity')

class CCUNet_deepsup(nn.Module):

    def __init__(self, in_channels=1, n_classes=4, feature_scale=2, is_deconv=False, is_batchnorm=True):
        super(CCUNet_deepsup, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp_2(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp_2(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp_2(filters[2], filters[1], self.is_deconv)
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
        self.RCCAModule =RCCAModule(out_size,out_size )
        self.DANetHead =DANetHead(out_size,out_size,norm_layer=nn.BatchNorm2d )
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
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
            outputs0 = self.conv(outputs0)
            outputs0 = self.RCCAModule(outputs0)
            
        return outputs0
    
class unetUp_2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_2, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        self.CAM_Module = CAM_Module(out_size )
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
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
            outputs0 = self.conv(outputs0)
            outputs0 = self.CAM_Module(outputs0)
            
        return outputs0

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

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels   # in_channels=2018，通道数缩减为512
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False), norm_layer(inter_channels), nn.ReLU())
 
        self.sa = RCCAModule(inter_channels,inter_channels)  # 空间注意力模块
        self.sc = CAM_Module(inter_channels)  # 通道注意力模块
        
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        
        # nn.Dropout2d(p,inplace)：p表示将元素置0的概率；inplace若设置为True，会在原地执行操作。
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
 
    def forward(self, x):
        
        sa_feat = self.sa(x)      # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sa_conv = self.conv51(sa_feat)        # 经过一个1×1卷积降维后，再送入通道注意力模块
#        sa_output = self.conv6(sa_conv)  
        
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sc_conv = self.conv52(sc_feat)
#        sc_output = self.conv7(sc_conv)
 
        feat_sum = sa_conv+sc_conv  # 两个注意力模块结果相加       
        sasc_output = self.conv8(feat_sum)  # 最后再送入1个有dropout的1×1卷积中
 
        output = sasc_output

        return output  # 输出模块融合后的结果，以及两个模块各自的结果
    
class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax  = Softmax(dim=-1)  # 对每一行进行softmax
    def forward(self,x):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)# A -> (N,C,HW)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)# A -> (N,HW,C)
        energy = torch.bmm(proj_query, proj_key)# 矩阵乘积，通道注意图：X -> (N,C,C)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)
        
        out = self.gamma*out + x
        return out

    
class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels //4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.cca = CC_module(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, inter_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(inter_channels),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output  
    
#class RCCAModule(nn.Module):#原版RCCA
#    def __init__(self, in_channels, out_channels, num_classes):
#        super(RCCAModule, self).__init__()
#        inter_channels = in_channels // 4
#        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
#        self.cca = CrissCrossAttention(inter_channels)
#        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
#
#        self.bottleneck = nn.Sequential(
#            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
#            BatchNorm2d(out_channels),nn.ReLU(inplace=False),
#            nn.Dropout2d(0.1),
#            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
#            )
#
#    def forward(self, x, recurrence=1):
#        output = self.conva(x)
#        for i in range(recurrence):
#            output = self.cca(output)
#        output = self.convb(output)
#
#        output = self.bottleneck(torch.cat([x, output], 1))
#        return output
    
if __name__ == '__main__':
    inputs = torch.rand((1, 1, 512, 512)).cuda()

    unet_plus_plus = CCUNet_deepsup(in_channels=1, n_classes=2).cuda()
    a,b,c,output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))