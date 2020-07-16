import torch
import glob
import os
from torchvision import transforms
from torchvision.transforms import functional as F
#import cv2
from PIL import Image
# import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
#from utils import get_label_info, one_hot_it
import random
import matplotlib.pyplot as plt 

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass

def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class LinearLesion(torch.utils.data.Dataset):
    def __init__(self, dataset_path,scale,k_fold_test=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path=dataset_path+'\\img'#训练集路径
        self.mask_path=dataset_path+'\\mask'#标签路径
        self.image_lists,self.label_lists=self.read_list(self.img_path,k_fold_test=k_fold_test)
        self.flip =iaa.SomeOf((2,4),[
             iaa.Fliplr(0.5),
             iaa.Flipud(0.5),
             iaa.Affine(rotate=(-30, 30)),
             iaa.AdditiveGaussianNoise(scale=(0.0,0.08*255))], random_order=True)
        # resize
        self.resize_label = transforms.Resize(scale, Image.NEAREST)#重置标签图像分辨率，插值方法选择，Image.NEAREST为低质量插值
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)#重置原图图像分辨率，插值方法选择，Image.BILINEAR为双线性插值
        # normalization
        self.to_tensor = transforms.ToTensor()#将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]，归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_lists[index])

        
        img = np.array(img)#转化为numpy格式
        labels=self.label_lists[index]

        #load label
        if self.mode !='test':
            label = Image.open(self.label_lists[index])
            label = np.array(label) 
        
            label[label==255]=1
            label[label==190]=2
#            label[label==105]=3
            
            # label=np.argmax(label,axis=-1)
            # label[label!=1]=0
            # augment image and label
            
            if self.mode == 'train':
                
                
                seq_det = self.flip.to_deterministic()#确定一个数据增强的序列
                segmap = ia.SegmentationMapOnImage(label, shape=label.shape, nb_classes=3)
                img = seq_det.augment_image(img)#将方法应用在原图像上
                
#                plt.imshow(img.astype(np.float32))#显示原图
#                plt.show()
                
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)# 将方法应用在分割标签上，并且转换成np类型，这里尺度（256,512）
#                plt.imshow(label.astype(np.float32))
#                plt.show()
                

#            label=np.reshape(label,(1,)+label.shape)
#                
#                plt.imshow(label.astype(np.float32))#显示label图片
#                plt.show()
#            label=torch.from_numpy(label.copy()).float()#二分类用float

            labels = torch.from_numpy(label.copy()).long()#多分类label用long
           
        img=np.reshape(img,img.shape+(1,))       # 如果输入是1通道需打开此注释 ******
        
        img = self.to_tensor(img.copy()).float() 
        
        
        

        return img, labels

    def __len__(self):
        return len(self.image_lists)
    def read_list(self,image_path,k_fold_test=1):
        fold=sorted(os.listdir(image_path))#对列表进行排序
        # print(fold)
        os.listdir()#指定的文件夹包含的文件或文件夹的名字的列表。
        img_list=[]
        if self.mode=='train':
            fold_r=fold
            fold_r.remove('f'+str(k_fold_test))# 移除测试数据，因为命名方式为f加数字
            for item in fold_r:
                img_list+=glob.glob(os.path.join(image_path,item)+'\\*.png')#这里是原图片的图片列表，将每个文件夹的图片加入列表

            label_list=[x.replace('img','mask').split('.')[0]+'.png' for x in img_list]#标签列表，其中对应的名称与原图一致

            
        elif self.mode=='val' or self.mode=='test':
            fold_s=fold[k_fold_test-1]
            img_list=glob.glob(os.path.join(image_path,fold_s)+'\\*.png')
            label_list=[x.replace('img','mask').split('.')[0]+'.png' for x in img_list]
            
        return img_list,label_list


if __name__ == '__main__':
    data = LinearLesion(r'C:\Users\Administrator\Desktop\model file\Pytorch_Medical_Segmention-multi-deep_spie\Dataset\Linear_lesion', (256, 256),mode='train')
    
    from torch.utils.data import DataLoader
    dataloader_test = DataLoader(
        data,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False 
    )
    for i, (img, label) in enumerate(dataloader_test):
        
        label_colors = torch.unique(label.view(label.size(0), -1) .type(torch.LongTensor))
        image_colors = torch.unique(img.view(img.size(0), -1), dim=1)
        
#        print(label.shape)
#        print(img.shape)
        
        label_arr = np.squeeze(label.numpy())#去除维度中是1的维度，例如(1, 3, 256, 512)变(3, 256, 512)，（1, 1, 256, 512)变(256, 512)
        image_arr = np.squeeze(img.numpy())
#        print(label_arr.shape)
#        print(image_arr.shape)
        
        #显示图片
        plt.imshow(label_arr.astype(np.float32))#这里由于torch中tensor的格式问题，转化为np后(3, 256, 512)是不能显示的，一定要（256, 512，3)
        plt.show()#这里标签只有0,1,2
        plt.imshow(image_arr.astype(np.float64))
        plt.show()
        
        #显示图片存在的灰度
#        print(image_colors)
#        print(label_colors)
#        print(list(label))
#        print(list(img.size()))
        break
        if i>3:
            break
