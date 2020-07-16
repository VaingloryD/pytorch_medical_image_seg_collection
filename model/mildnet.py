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
class Mild_net(nn.Module):

    def __init__(self, in_channels=1, n_classes=4, feature_scale=2, is_deconv=False,is_batchnorm=True):
        super(Mild_net, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024, 640]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.Mil_unit1 = Mil_unit(filters[0], filters[1], self.is_batchnorm)
        self.Residual_unit1 = Residual_unit(filters[1], filters[1], self.is_batchnorm)
        
        self.Mil_unit2 = Mil_unit(filters[1], filters[2], self.is_batchnorm)
        self.Residual_unit2 = Residual_unit(filters[2], filters[2], self.is_batchnorm)
        
        self.Mil_unit3 = Mil_unit(filters[2], filters[3], self.is_batchnorm)
        self.Residual_unit3 = Residual_unit(filters[3], filters[3], self.is_batchnorm)
        
        self.Mil_unit4 = Mil_unit(filters[3], filters[4], self.is_batchnorm)
        self.Residual_unit4 = Residual_unit(filters[4], filters[4], self.is_batchnorm)
        
        
        # ASPP
        self.aspp = ASPP(filters[4], filters[5])
        self.conv10 = nn.Conv2d(filters[5], filters[4], 1, stride = 1)#将原始图通过1*1卷积变换通道，最左边的1是因为灰度图
        
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
#        self.drop = nn.Dropout2d(p=0.5)#dropout
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        

    def forward(self, inputs):
        conv1 = self.conv1(inputs)           # 前两个3*3,64*256*512
        maxpool1 = self.maxpool(conv1)       # 64*128*256
        mil1 = self.Mil_unit1(maxpool1, inputs)#128*128*256
        Residual1 = self.Residual_unit1(mil1)#128*128*256
        maxpool2 = self.maxpool(Residual1)#128*64*128
        mil2 = self.Mil_unit2(maxpool2, inputs)#256*64*128
        Residual2 = self.Residual_unit2(mil2)#256*64*128
        maxpool3 = self.maxpool(Residual2)#256*32*64
        mil3 = self.Mil_unit3(maxpool3, inputs)#512*32*64
        Residual3 = self.Residual_unit3(mil3)#512*32*64
        maxpool4 = self.maxpool(Residual3)#512*16*32        
        mil4 = self.Mil_unit4(maxpool4, inputs)#1024*16*32
        Residual4 = self.Residual_unit4(mil4)#1024*16*32
        
        aspp = self.aspp(Residual4)       # 640*32*64
        aspp = self.conv10(aspp)
        
        up4 = self.up_concat4(aspp ,Residual3 )
        up3 = self.up_concat3(up4 ,Residual2 )     # 128*128*256
        up2 = self.up_concat2(up3,Residual1)     # 64*256*512
        up1 = self.up_concat1(up2,conv1)     # 64*256*512
        final = self.final(up1)

#        final=F.sigmoid(final)
        final=F.log_softmax(final,dim=1)

        return up1,final
    
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
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_size, out_size, 1))
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue          #unetConv2已经是一个初始化好的类，不需要再初始化
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
        
        
class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))#平均池化
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)#1*1卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)#3*3卷积，膨胀率6
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)#3*3卷积，膨胀率12
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)#3*3卷积，膨胀率18
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

    
class Residual_unit(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Residual_unit, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Dilated_Residual_unit_1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernal=3, stride=1):
        super(Dilated_Residual_unit_1, self).__init__()
        self.kernal = kernal
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Dilated_Residual_unit_2(nn.Module):
    def __init__(self, inplanes, planes, kernal=3, stride=1):
        super(Dilated_Residual_unit_2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm2d(planes)
        self.kernal = kernal
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Mil_unit(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(Mil_unit, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.conv1x = conv3x3(inplanes, inplanes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3( planes,  planes, stride)
        self.stride = stride
        self.conv1_1 = nn.Conv2d(1, inplanes, 1, stride)#将原始图通过1*1卷积变换通道，最左边的1是因为灰度图

    def forward(self, x, origin):

        size = x.shape[2:]#******

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)#特征图经过两个3*3卷积
        
        out2 = F.interpolate(origin, size, mode="bilinear")
        out2 = self.conv1_1(out2)#将原图通道变为与特征图输入通道一致
        
        out2 = self.conv1x(out2)
        out2 = self.bn1(out2)#输入通道
        out2 = self.relu(out2)#原图下采样，经过3*3卷积
        out3 = torch.cat([x, out2], 1)#将上一层的特征图和原图concat，通道翻倍
        out3 = self.conv3(out3)
        out3 = self.bn2(out3)
        out3 = self.relu(out3)#特征图与原图下采样后concat经过3*3卷积
        
        out = out3 + out1#两条线相加
        out = self.relu(out)
        
        return out
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


    

    
#model = Mild_net().cuda()
#torchsummary.summary(model, (1, 512, 512))
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 512, 512)).cuda()

    unet_plus_plus = Mild_net(in_channels=1, n_classes=2).cuda()
    output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))