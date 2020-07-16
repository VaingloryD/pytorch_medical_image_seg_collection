# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:19:51 2020

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

torch_ver = torch.__version__[:3]
 
__all__ = ['PAM_Module', 'CAM_Module']
 
class DAUNet_deepsup(nn.Module):

    def __init__(self, in_channels=1, n_classes=4, feature_scale=2, is_deconv=False, is_batchnorm=True):
        super(DAUNet_deepsup, self).__init__()
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
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp_2(filters[1], filters[0], self.is_deconv)
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

class unetUp_2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_2, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        self.DANetHead = DANetHead(out_size,out_size,norm_layer=nn.BatchNorm2d )
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
            outputs0 = self.DANetHead(outputs0)
            
        return outputs0

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


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
 
        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
 
        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()     
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)# B -> (N,C,HW) -> (N,HW,C)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)# C -> (N,C,HW)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)#计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3.
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)       # D -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        out = out.view(m_batchsize, C, height, width)        # output -> (N,C,H,W)
 
        out = self.gamma*out + x
        return out
 
 
class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax  = Softmax(dim=-1)  # 对每一行进行softmax
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
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
 

class DANetHead_2(nn.Module):#融合
    def __init__(self, in_size, out_size, stride=1):
        super(DANetHead_2, self).__init__()
        inter_channels = in_size // 4  # in_channels=2018，通道数缩减为512
        
        self.conv1 = conv3x3(in_size, inter_channels, stride)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        

        self.sa = PAM_Module(inter_channels)  # 空间注意力模块
        self.sc = CAM_Module(inter_channels)  # 通道注意力模块
        # nn.Dropout2d(p,inplace)：p表示将元素置0的概率；inplace若设置为True，会在原地执行操作。
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_size, 1))
 
    def forward(self, x):
        # 经过一个1×1卷积降维后，再送入空间注意力模块
        feat1 = self.conv1(x)
        feat1 = self.bn1(feat1)
        feat1 = self.relu(feat1)
        sa_feat = self.sa(feat1)  
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sa_conv = self.conv1(sa_feat)
        sa_conv = self.bn1(sa_conv)
        sa_conv = self.relu(sa_conv)
 
        # 经过一个1×1卷积降维后，再送入通道注意力模块
        feat2 = self.conv1(x)
        feat2 = self.bn1(feat2)
        feat2 = self.relu(feat2)
        sc_feat = self.sc(feat2)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sc_conv = self.conv1(sc_feat)
        sc_conv = self.bn1(sc_conv)
        sc_conv = self.relu(sc_conv)
 
        feat_sum = sa_conv+sc_conv  # 两个注意力模块结果相加       
        sasc_output = self.conv8(feat_sum)  # 最后再送入1个有dropout的1×1卷积中
 
        output = sasc_output

        return output  # 输出模块融合后的结果
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4  # in_channels=2018，通道数缩减为512
        
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())       
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
 
        self.sa = PAM_Module(inter_channels)  # 空间注意力模块
        self.sc = CAM_Module(inter_channels)  # 通道注意力模块
        
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        
        # nn.Dropout2d(p,inplace)：p表示将元素置0的概率；inplace若设置为True，会在原地执行操作。
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
 
    def forward(self, x):
        # 经过一个1×1卷积降维后，再送入空间注意力模块
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)  
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sa_conv = self.conv51(sa_feat)
#        sa_output = self.conv6(sa_conv)  
 
        # 经过一个1×1卷积降维后，再送入通道注意力模块
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sc_conv = self.conv52(sc_feat)
#        sc_output = self.conv7(sc_conv)
 
        feat_sum = sa_conv+sc_conv  # 两个注意力模块结果相加       
        sasc_output = self.conv8(feat_sum)  # 最后再送入1个有dropout的1×1卷积中
 
        output = sasc_output

        return output  # 输出模块融合后的结果，以及两个模块各自的结果
    
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 512, 512)).cuda()

    unet_plus_plus = DAUNet_deepsup(in_channels=1, n_classes=2).cuda()
    a,b,c,output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))    