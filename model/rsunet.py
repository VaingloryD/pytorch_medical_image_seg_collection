#import _init_paths
import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param
import torchsummary
from torch.nn import functional as F
from torch.nn import init

class RSUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=4, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(RSUNet, self).__init__()                                           #子类继承父类，子类的构造方法的第一行，系统会默认编写super()，在调用子类的构造方法时，先调用父类的无参数构造方法
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.unetDown1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.unetDown2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.unetDown3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.unetDown4 = unetDown(filters[2], filters[3], self.is_batchnorm)
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


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                                       #判别目标类型,为了防止该类里有单独的Conv2d也需要初始化
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.unetDown1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256      
        conv2 = self.unetDown2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128
        conv3 = self.unetDown3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64
        conv4 = self.unetDown4(maxpool3)         # 128*64*64
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
        final=F.log_softmax(final,dim=1)
        up4_deep = F.log_softmax(final,dim=1)
        up3_deep = F.log_softmax(final,dim=1)
        up2_deep = F.log_softmax(final,dim=1)

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
        for m in self.children():                                              #两个conv子模块
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.sb = SBU_Block(out_size)
        
    def forward(self, inputs):   
        x = self.conv(inputs)
        shortcut = x.clone()
        sbx = self.sb(x)
        outputs = shortcut+sbx
        
        return outputs
        
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

class SBU_Block(nn.Module):
    def __init__(self, channel):
        super(SBU_Block, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        y = x.clone()
        agap = self.avg_pool(torch.abs(x))#torch.abs参数的绝对值作为输出
        alpha = self.fc(agap)
        sigma = agap * alpha
        soft_threshold = torch.abs(sigma.expand_as(y))

        y[torch.abs(y) < soft_threshold]=0
        return y
    
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)                                        #apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:                                       #全连接层
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
#model = RSUNet().cuda()
#torchsummary.summary(model, (1, 512, 512))