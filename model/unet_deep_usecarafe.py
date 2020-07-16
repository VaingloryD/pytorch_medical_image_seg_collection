# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:56:19 2020

@author: Administrator
"""

import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param
import torchsummary
from torch.nn import functional as F
from torch.nn import init

class UNet_deepsupusecarafe(nn.Module):

    def __init__(self, in_channels=1, n_classes=3, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet_deepsupusecarafe, self).__init__()
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
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        
        #deep Supervision
        self.up_3 = CARAFE_3(filters[3])
        self.deepsup_3 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)
        self.up_2 = CARAFE_3(filters[2])
        self.deepsup_2 = nn.Conv2d(filters[2], n_classes, kernel_size=1, stride=1, padding=0)
        self.up_1 = CARAFE_2(filters[1])
        self.deepsup_1 = nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1, padding=0)


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
        up4_1 = self.up_3(up4)
        up4_deep = self.deepsup_3(up4_1)
        
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up3_1 = self.up_2(up3)
        up3_deep = self.deepsup_2(up3_1)

        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up2_1 = self.up_1(up2)
        up2_deep = self.deepsup_1(up2_1)
        
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
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
     
        self.up = CARAFE_3(in_size)
        self.conv1 = nn.Conv2d(in_size, out_size, 1)
        
#        self.up = Carafe(in_size)

#         initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue#continue语句可用于循环中，用于跳过当前循环的剩余代码，然后继续进行下一轮的循环。
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        outputs0 = self.conv1(outputs0)
        
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
            
        return self.conv(outputs0)
    
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
        

class CARAFE_3(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        super(CARAFE_3, self).__init__()
        self.scale = scale

        self.comp = nn.Conv2d(c, c_mid,kernel_size=1, stride=1,
                               padding=0, dilation=1)#降低通道数量
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.relu = nn.ReLU(inplace=True)
        
        self.enc = nn.Conv2d(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1)
        self.bn2 = nn.BatchNorm2d((scale * k_up) ** 2)
        
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='bilinear')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)
        
                # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.bn1(W)
        W = self.relu(W)
        
        W = self.enc(W)  # b * 100 * h * w
        W = self.bn2(W)
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X
    
class CARAFE_2(nn.Module):
    def __init__(self, c, c_mid=32, scale=2, k_up=5, k_enc=3):
        super(CARAFE_2, self).__init__()
        self.scale = scale

        self.comp = nn.Conv2d(c, c_mid,kernel_size=1, stride=1,
                               padding=0, dilation=1)#降低通道数量
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.relu = nn.ReLU(inplace=True)
        
        self.enc = nn.Conv2d(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1)
        self.bn2 = nn.BatchNorm2d((scale * k_up) ** 2)
        
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='bilinear')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)
        
                # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.bn1(W)
        W = self.relu(W)
        
        W = self.enc(W)  # b * 100 * h * w
        W = self.bn2(W)
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X
        
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 256, 512)).cuda()

    unet_plus_plus =  UNet_deepsupusecarafe(in_channels=1, n_classes=2).cuda()
    a,b,c,output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))