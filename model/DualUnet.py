# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:16:06 2019

@author: Administrator
"""

#import _init_paths
import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param
import torchsummary
from torch.nn import functional as F
from torch.nn import init
class Dual_Unet(nn.Module):

    def __init__(self, in_channels=1, n_classes=4, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(Dual_Unet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        

        filters = [64, 128, 256, 512, 1024, 2048]
        filters = [int(x / self.feature_scale) for x in filters]


        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        
        self.Context_1 = Context_Path(self.in_channels, filters[0])
        self.Context_2 = Context_Path(filters[0], filters[1])
        self.Context_3 = Context_Path(filters[1], filters[2])
        self.Context_4 = Context_Path(filters[2], filters[3])
        
        self.Attention_1 = Attention_Skip(filters[0])
        self.Attention_2 = Attention_Skip(filters[1])
        self.Attention_3 = Attention_Skip(filters[2])
        self.Attention_4 = Attention_Skip(filters[3])
        
        self.Feature_Fusion = Feature_Fusion(filters[4],filters[4])
        
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        
        self.final = nn.Conv2d(filters[1], n_classes, 1)
        
        self.conv1_1 = nn.Conv2d(filters[5], filters[4], 1, 1)
             
        # 1*1 conv (without any concat)
#        self.final = Multiscale_Predict(filters[2], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)           # 64*512*512
        maxpool1 = self.maxpool(conv1)       # 64*256*256        
        conv2 = self.conv2(maxpool1)         # 128*256*256
        maxpool2 = self.maxpool(conv2)       # 128*128*128
        conv3 = self.conv3(maxpool2)         # 256*128*128
        maxpool3 = self.maxpool(conv3)       # 256*64*64
        conv4 = self.conv4(maxpool3)         # 512*64*64
        
        conx1 = self.Context_1(inputs)#64
        conx1_1 = self.maxpool(conx1)
        conx2 = self.Context_2(conx1_1)#128
        conx2_1 = self.maxpool(conx2)
        conx3 = self.Context_3(conx2_1)#256
        conx3_1 = self.maxpool(conx3)
        conx4 = self.Context_4(conx3_1)#512

        Attention1 = self.Attention_1(conv1,conx1) #128    
        Attention2 = self.Attention_2(conv2,conx2) #256    
        Attention3 = self.Attention_3(conv3,conx3) #512   
        Attention4 = self.Attention_4(conv4,conx4) #1024    
        
        Feature_Fusion = self.Feature_Fusion(conv4,conx4)#1024
        
        lay1 = torch.cat([Feature_Fusion, Attention4], dim=1)#2048
        lay1 = self.conv1_1(lay1)#1024
        
        lay2 = self.up_concat4(lay1,Attention3)  # 512
        lay3 = self.up_concat3(lay2,Attention2)     # 256
        lay4 = self.up_concat2(lay3,Attention1)     # 128
        final = self.final(lay4)
        
#        final = self.final(lay3,lay4)
#        final=F.softmax(final,dim=1)#对每一行使用Softmax
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
        
        
class Feature_Fusion(nn.Module):#自写的
    def __init__(self, inplanes, planes,r = 16, stride=1, downsample=None):
        super(Feature_Fusion, self).__init__()
#中间是se模块的部分，其他是resnet正常部分
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes , planes // r, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // r, planes , kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(planes)
        
    def forward(self, x, y):
        
        out = torch.cat([x,y], dim=1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        input = out
        
        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)
        
        res_1 = out1 * input
        res = res_1 + input
        
        return res

    
class Attention_Skip(nn.Module):#自写的
    def __init__(self, planes,r = 16, stride=1, downsample=None):
        super(Attention_Skip, self).__init__()
#中间是se模块的部分，其他是resnet正常部分
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            2*planes , 2*planes // r, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            2*planes // r, 2*planes , kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(2*planes)
        
    def forward(self, x, y):
        
        input = torch.cat([x,y], dim=1)
        
        out1 = self.global_pool(input)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)
        out1 = self.bn1(out1)
        
        res = out1 * input

        return res

    
class Context_Path(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(Context_Path, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.stride = stride
        self.conv1_1 = nn.Conv2d(inplanes, planes, 1, stride)
        self.conv1_2 = nn.Conv2d(inplanes, inplanes, 1, stride)
        self.conv1_3 = nn.Conv2d(3*planes, planes, 1, stride)

    def forward(self, x):

        out1 = self.conv1_1(x)#d = 64
        
        out2 = self.conv1_1(x)
        out2 = self.conv2(out2)#将原图通道变为与特征图输入通道一致,d = 32
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        
        out3 = self.conv1_2(x)
        out3 = self.conv1(out3)
        out3 = self.bn1(out3)
        out3 = self.relu(out3)#特征图与原图下采样后concat经过3*3卷积
        out3 = self.conv2(out3)
        out3 = self.bn1(out3)
        out3 = self.relu(out3)#特征图与原图下采样后concat经过3*3卷积
        
        out = torch.cat([out1, out2, out3], dim=1)#通道数变plane的3倍
        out = self.conv1_3(out)
        
        return out




class Multiscale_Predict(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(Multiscale_Predict, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.stride = stride

    def forward(self, x ,y):

        out1 = self.conv1(x)#layer3
        out1 = self.pixelshuffle(out1)
        
        out2 = torch.cat([out1, y], dim=1)
        out2 = self.conv2(out2)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        
        return out2


    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

    
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 512, 512)).cuda()

    unet_plus_plus = Dual_Unet(in_channels=1, n_classes=2).cuda()
    output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))