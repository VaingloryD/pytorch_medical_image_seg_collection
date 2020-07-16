# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:46:32 2020

@author: Administrator
"""

import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param
import torchsummary
from torch.nn import functional as F
from torch.nn import init
class UNet_deepsup(nn.Module):

    def __init__(self, in_channels=1, n_classes=3, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet_deepsup, self).__init__()
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
        
class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                   norm_type,
                                                   psp_size=psp_size)
        
class APNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), norm_type=None,psp_size=(1,3,6,8)):
        super(APNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BNReLU(out_channels, norm_type=norm_type),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    self.psp_size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output
    
class AFNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout,
                 sizes=([1]), norm_type=None,psp_size=(1,3,6,8)):
        super(AFNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage([low_in_channels, high_in_channels], out_channels, key_channels, value_channels, size) for
             size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BatchNorm2d(norm_type=self.norm_type)(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels[0],
                                    in_channels[1],
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    psp_size=self.psp_size)

    def forward(self, low_feats, high_feats):
        priors = [stage(low_feats, high_feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, high_feats], 1))
        return output

class asymmetric_non_local_network(nn.Sequential):
    def __init__(self, configer):
        super(asymmetric_non_local_network, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        # low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout
        self.fusion = AFNB(1024, 2048, 2048, 256, 256, dropout=0.05, sizes=([1]), norm_type=self.configer.get('network', 'norm_type'))
        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, norm_type=self.configer.get('network', 'norm_type')),
            APNB(in_channels=512, out_channels=512, key_channels=256, value_channels=256,
                         dropout=0.05, sizes=([1]), norm_type=self.configer.get('network', 'norm_type'))
        )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, norm_type=self.configer.get('network', 'norm_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
       
    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.fusion(x[-2], x[-1])
        x = self.context(x)
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x     
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 256, 512)).cuda()

    unet_plus_plus = UNet_deepsup(in_channels=1, n_classes=2).cuda()
    a,b,c,output = unet_plus_plus(inputs)
    print('# parameters:', sum(param.numel() for param in unet_plus_plus.parameters()))
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print('# parameters:', get_parameter_number(unet_plus_plus))