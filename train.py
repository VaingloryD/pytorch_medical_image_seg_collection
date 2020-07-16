import argparse
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.Linear_lesion import LinearLesion
import socket
from datetime import datetime

import os

from model.unet_deep import UNet_deepsup
from model.CBAMunet_deep import CBAMUNet_deepsup
from model.SCconv_unet_deep import SCconv_UNet_deepsup
from model.GCunet_deep_2 import GCUNet_deepsup_2
from model.lipunet_deep import lipUNet_deepsup
from model.diao_improve import diao_deepsup_improve
from model.unet_deep_improve import unet_deepsup_improve
from model.unet_deepsup_stip import unet_deepsup_strip
from model.unet_carafe_deep import UNet_carafe_deepsup
from model.diao import diao_deepsup_origin
from model.unet_deep_usecarafe import UNet_deepsupusecarafe

import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from  PIL import Image
#from utils import poly_lr_scheduler
#from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy,batch_intersection_union,batch_pix_accuracy
import utils.utils as u
import utils.loss as LS
from utils.config import DefaultConfig
import torch.backends.cudnn as cudnn
import imgaug as ia
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     np.random.seed(seed)
     ia.seed(seed)
     random.seed(seed)

def val(args, model, dataloader):#计算dice等指标
    print('\n')
    print('Start Validation!')
    with torch.no_grad():
        model.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0

        total_Dice_1=[]
        total_Acc_1=[]
        total_jaccard_1=[]
        total_Sensitivity_1=[]
        total_Specificity_1=[]
        
        total_Dice_2=[]
        total_Acc_2=[]
        total_jaccard_2=[]
        total_Sensitivity_2=[]
        total_Specificity_2=[]
        

        for i, (data, label) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()#将tensor从cpu –> gpu
                label = label.cuda()
            
            # get RGB predict image
            up4_deep,up3_deep,up2_deep,predict = model(data)
            
            Dice_1,Dice_2,Acc_1,Acc_2,Jaccard_1,Jaccard_2,Sensitivity_1,Sensitivity_2,Specificity_1,Specificity_2 = u.eval_multi_seg(predict,label)

            total_Dice_1+=Dice_1
            total_Acc_1+=Acc_1
            total_jaccard_1+=Jaccard_1
            total_Sensitivity_1+=Sensitivity_1
            total_Specificity_1+=Specificity_1
            
            total_Dice_2+=Dice_2
            total_Acc_2+=Acc_2
            total_jaccard_2+=Jaccard_2
            total_Sensitivity_2+=Sensitivity_2
            total_Specificity_2+=Specificity_2
            

            dice_1=sum(total_Dice_1) / len(total_Dice_1)
            acc_1=sum(total_Acc_1) / len(total_Acc_1)
            jac_1=sum(total_jaccard_1) / len(total_jaccard_1)
            sen_1=sum(total_Sensitivity_1) / len(total_Sensitivity_1)
            spe_1=sum(total_Specificity_1) / len(total_Specificity_1)
            
            dice_2=sum(total_Dice_2) / len(total_Dice_2)
            acc_2=sum(total_Acc_2) / len(total_Acc_2)
            jac_2=sum(total_jaccard_2) / len(total_jaccard_2)
            sen_2=sum(total_Sensitivity_2) / len(total_Sensitivity_2)
            spe_2=sum(total_Specificity_2) / len(total_Specificity_2)
            
            tbar.set_description(
                'Dice_1: %.3f,Dice_2: %.3f, Acc_1: %.3f, Acc_2: %.3f, Jac_1: %.3f, Jac_2: %.3f,Sen_1: %.3f, Sen_2: %.3f, Spe_1: %.3f, Spe_2: %.3f'
                % (dice_1,dice_2,acc_1,acc_2,jac_1,jac_2,sen_1,sen_2,spe_1,spe_2))


        print('Dice_1:',dice_1)
        print('Dice_2:',dice_2)

        print('Acc_1:',acc_1)
        print('Acc_2:',acc_2)

        print('Jac_1:',jac_1)
        print('Jac_2:',jac_2)

        print('Sen_1:',sen_1)
        print('Sen_2:',sen_2)

        print('Spe_1:',spe_1)
        print('Spe_2:',spe_2)

        return dice_1,dice_2,acc_1,acc_2,jac_1,jac_2,sen_1,sen_2,spe_1,spe_2
    


def train(args, model, optimizer,criterion, dataloader_train, dataloader_val,writer,k_fold):
    
    step = 0
    best_pred=0.0
    for epoch in range(args.num_epochs):
        lr = u.adjust_learning_rate(args,optimizer,epoch) 
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)#在 Python 长循环中添加一个进度提示信息
        tq.set_description('fold %d,epoch %d, lr %f' % (int(k_fold),epoch, lr))
        loss_record = []
        train_loss=0.0
#        is_best=False        for i,(data, label) in enumerate(dataloader_train):
            # if i>len(dataloader_train)-2:
            #     break
        
        for i,(data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
               data = data.cuda()#输入数据
               label = label.cuda()
            optimizer.zero_grad()#梯度清零
            
            up4_deep,up3_deep,up2_deep,main_out = model(data)
     
            # get weight_map
            weight_map=torch.zeros(args.num_classes)
            weight_map=weight_map.cuda()
            for j in range(args.num_classes):
                weight_map[j]=1/(torch.sum((label==j).float())+1.0)
                
                
            loss_aux_1=F.nll_loss(main_out,label)
            loss_aux_2=F.nll_loss(up4_deep,label)
            loss_aux_3=F.nll_loss(up3_deep,label)
            loss_aux_4=F.nll_loss(up2_deep,label)

            loss_main_1= criterion[1](main_out, label)#criterion就是设置loss
            loss_main_2= criterion[1](up4_deep, label)#criterion就是设置loss
            loss_main_3= criterion[1](up3_deep, label)#criterion就是设置loss
            loss_main_4= criterion[1](up2_deep, label)#criterion就是设置loss
            
            loss =loss_main_1+loss_aux_1+loss_main_2+loss_aux_2+loss_main_3+loss_aux_3+loss_main_4+loss_aux_4
            loss.backward()#反向传播
            optimizer.step()
            tq.update(args.batch_size)#每隔一个batchsize更新一次            
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss/(i+1)))
            step += 1
#            if step%10==0:
#                writer.add_scalar('Train/loss_step_{}'.format(int(k_fold)), loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)#对损失函数求平均
        writer.add_scalar('Train/loss_epoch_{}'.format(int(k_fold)), float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        

        if epoch % args.validation_step == 0:
            Dice_1,Dice_2,Acc_1,Acc_2,Jaccard_1,Jaccard_2,Sensitivity_1,Sensitivity_2,Specificity_1,Specificity_2= val(args, model, dataloader_val)
            writer.add_scalar('Valid_{}/Dice_val_1'.format(int(k_fold)), Dice_1, epoch)
#            writer.add_scalar('Valid_{}/Acc_val_1'.format(int(k_fold)), Acc_1, epoch)
            writer.add_scalar('Valid_{}/Jac_val_1'.format(int(k_fold)), Jaccard_1, epoch)
            writer.add_scalar('Valid_{}/Sen_val_1'.format(int(k_fold)), Sensitivity_1, epoch)
            writer.add_scalar('Valid_{}/Spe_val_1'.format(int(k_fold)), Specificity_1, epoch)
            
            writer.add_scalar('Valid_{}/Dice_val_2'.format(int(k_fold)), Dice_2, epoch)
#            writer.add_scalar('Valid_{}/Acc_val_2'.format(int(k_fold)), Acc_2, epoch)
            writer.add_scalar('Valid_{}/Jac_val_2'.format(int(k_fold)), Jaccard_2, epoch)
            writer.add_scalar('Valid_{}/Sen_val_2'.format(int(k_fold)), Sensitivity_2, epoch)
            writer.add_scalar('Valid_{}/Spe_val_2'.format(int(k_fold)), Specificity_2, epoch)
            
            is_best=(Dice_1+Dice_2)/2 > best_pred
            best_pred = max(best_pred, (Dice_1+Dice_2)/2)
            checkpoint_dir_root = args.save_model_path
            checkpoint_dir=os.path.join(checkpoint_dir_root,str(k_fold))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest =os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
            u.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dice': best_pred,
                    }, best_pred,epoch,is_best, checkpoint_dir,filename=checkpoint_latest)
                    

    
def test(model,dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()                                                           #测试模式，自动把BN和DropOut固定住，不会取平均，而是用训练好的值
        #仅预测结果，不计算指标
        tq = tqdm.tqdm(dataloader,desc='\r')
        tq.set_description('test')                                             #进度条及前缀
        comments=os.getcwd().split('/')[-1]                                    #方法用于返回当前工作目录即网络名称，仅train.py所在最后一个目录U-Net
     
        for i, (data, label_path) in enumerate(tq):                            #1个1个预测
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                
            up4_deep,up3_deep,up2_deep,predict = model(data)
            predict=torch.argmax(torch.exp(predict),dim=1)                     #nchw,寻找通道维度上的最大值变为nhw
             
            pred=predict.data.cpu().numpy().astype(np.uint8)                   #predict是Variable，predict.data是把Variable里的tensor取出来放到cpu上，转为numpy
#            pred_RGB=OCT_Fluid.COLOR_DICT[pred.astype(np.uint8)]              #彩色三通道 nhwc
            pred_gray=np.zeros(pred.shape,dtype='uint8')
            pred_gray[pred == 1] = 255
            pred_gray[pred == 2] = 190                                         #nch灰度图
            
            for index,item in enumerate(label_path):                           #这个循环应该没用，batchsize=1则labelpath=1
                print(label_path)
                save_img_path=label_path[index].replace('mask','_predict_mask')
                print(save_img_path)
                if not os.path.exists(os.path.dirname(save_img_path)):         #创建目录文件夹
                    os.makedirs(os.path.dirname(save_img_path))
#                img=Image.fromarray(pred_RGB[index].squeeze().astype(np.uint8))#hwc squeeze()删除为1的维度
                
                mask_scale=Image.open(label_path[index])
                pedict=Image.fromarray(pred_gray[index].squeeze().astype(np.uint8))
                predict_resize=pedict.resize(mask_scale.size)
                predict_resize.save(save_img_path)

                tq.set_postfix({'save_name':'{}'.format(save_img_path)})                         #设置进度条后缀
        tq.close()
            
def main(mode='train',args=None,writer=None,k_fold=1):


    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)#文件路径
    dataset_train = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width),k_fold_test=k_fold,mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )
    
    dataset_val = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width),k_fold_test=k_fold,mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )

    dataset_test = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width),k_fold_test=k_fold,mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )

    # build model
#    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
      
    #load model
    model_all={'UNet_deepsupusecarafe':UNet_deepsupusecarafe(in_channels=1, n_classes=args.num_classes)}
    model=model_all[args.net_work]
    cudnn.benchmark = True
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists
    if args.pretrained_model_path and mode=='test':
        print("=> loading pretrained model '{}'".format(args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')
        
        
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),eps=1e-08, weight_decay=args.weight_decay)
    criterion_aux=nn.NLLLoss(weight=None)#输出层 用了log_softmax 则需要用这个误差函数，那么多分类要用这个
    #criterion_aux=nn.BCEWithLogitsLoss(weight=None)
    
    criterion_main=LS.Multi_DiceLoss()
    criterion=[criterion_aux,criterion_main]
    if mode=='train':
        train(args, model, optimizer,criterion, dataloader_train, dataloader_val,writer,k_fold)
    if mode=='test':
        test(model,dataloader_test, args)
    if mode=='train_test':
        train(args, model, optimizer,criterion, dataloader_train, dataloader_val)
        eval(model,dataloader_test, args)




if __name__ == '__main__':
    seed=1234
    setup_seed(seed)

    args=DefaultConfig()
    modes=args.mode

    if modes=='train':
        comments=os.getcwd().split('/')[-1]
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(args.log_dirs, comments+'_'+current_time + '_' + socket.gethostname())#路径拼接
        writer = SummaryWriter(logdir=log_dir)
        for i in range(args.k_fold):
            main(mode='train',args=args,writer=writer,k_fold=int(i+1))
            
    elif modes=='test':
         main(mode='test',args=args,writer=None,k_fold=args.test_fold)

