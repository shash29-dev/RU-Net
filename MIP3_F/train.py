import os
import numpy as np 
import torch
import pdb
import arch_utils
import torch.nn as nn
import torch.nn.functional as F
import os
from time import time
# from models_mi import MIPSPnet
from t2 import ParallelNet
from metrics import RunningConfusionMatrix
import SimpleITK as sitk

batch_size=2
num_workers=8
pin_memory=True
learning_rate_decay = [500, 750]
learning_rate = 1e-4
train_ds = arch_utils.GetData(ct_dir='./processed_data/train/ct',seg_dir='./processed_data/train/seg')
train_dl = torch.utils.data.DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

val_ds = arch_utils.GetData(ct_dir='./processed_data/test/ct',seg_dir='./processed_data/test/seg')
val_dl = torch.utils.data.DataLoader(val_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

os.environ['CUDA_VISIBLE_DEVICES']='0'

        
# pdb.set_trace() 
if not os.path.exists('./module'): os.makedirs('./module')
if not os.path.exists('./log/balanced'): os.makedirs('./log/balanced')
net = ParallelNet(size=train_ds.size,training=True).cuda()
net.load_state_dict(torch.load('./module/val_net15-0.719.pth'))
net.train()
opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, learning_rate_decay)
alpha=0.33
# class_weights=torch.FloatTensor(np.load('weights.npy')).cuda()
loss_func=nn.CrossEntropyLoss(weight=None)
start=time()

def train_net():
    # net.training=True
    net.train()
    mean_loss = []
    for step, (ct, seg) in enumerate(train_dl):
        ct = ct.cuda()
        seg = seg.cuda()

        liver_pred,tumor_pred=net(ct)
        loss=0
        liver_pred=liver_pred.permute(0,2,1,3,4)
        tumor_pred=tumor_pred.permute(0,2,1,3,4)

        # output=outputs.permute(0,2,1,3,4)
        for bs in range(liver_pred.shape[0]):
            predL=liver_pred[bs]
            predT=tumor_pred[bs]
            Liver_label=seg[bs].clone()
            Liver_label[Liver_label>=1]=1

            Tumor_label=seg[bs].clone()
            Tumor_label[Tumor_label!=2]=0
            Tumor_label[Tumor_label==2]=1
            loss_liver=loss_func(predL,Liver_label.long())
            loss_tumor=loss_func(predT,Tumor_label.long())
            loss+= (alpha*loss_liver+ (1-alpha)*loss_tumor)
            # pred_max=torch.argmax(pred,dim=1)
            # current_cm=cm2.update_matrix(target.flatten().cpu().numpy(),pred_max.flatten().cpu().numpy())
            # current,overall=cm2.get_results_current_and_overall(current_cm)
        mean_loss.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        string='Training: epoch:{}, step:{}/{}, loss: {:.3f}, time:{:.3f} min'.format(epoch,step,len(train_dl) ,loss.item(), (time() - start) / 60)
        print(string)
        with open('log/balanced/log_train_iter.txt','a') as f: f.write(string+'\n')
    print('\t\t<------------------>\n')
    return mean_loss

def val_net(epoch,save=False):
    # net.training=False
    net.eval()
    mean_loss = []
    cm_liver=RunningConfusionMatrix()
    cm_tumor=RunningConfusionMatrix()
    for step, (ct, seg) in enumerate(val_dl):
        ct = ct.cuda()
        seg = seg.cuda()
        with torch.no_grad():
            liver_pred,tumor_pred=net(ct)
        loss=0
        liver_pred=liver_pred.permute(0,2,1,3,4)
        tumor_pred=tumor_pred.permute(0,2,1,3,4)

        # output=outputs.permute(0,2,1,3,4)
        for bs in range(liver_pred.shape[0]):
            predL=liver_pred[bs]
            predT=tumor_pred[bs]
            Liver_label=seg[bs].clone()
            Liver_label[Liver_label>=1]=1

            Tumor_label=seg[bs].clone()
            Tumor_label[Tumor_label!=2]=0
            Tumor_label[Tumor_label==2]=1
            loss_liver=loss_func(predL,Liver_label.long())
            loss_tumor=loss_func(predT,Tumor_label.long())
            loss+= (alpha*loss_liver+ (1-alpha)*loss_tumor)
            predL_max=torch.argmax(predL,dim=1)
            predT_max=torch.argmax(predT,dim=1)
            _=cm_liver.update_matrix(Liver_label.flatten().cpu().numpy(),predL_max.flatten().cpu().numpy())
            _=cm_tumor.update_matrix(Tumor_label.flatten().cpu().numpy(),predT_max.flatten().cpu().numpy())
            if save:
                path_Save='./prediction_files/'+str(epoch)
                if not os.path.exists(path_Save): os.makedirs(path_Save)
                out_prob = sitk.GetImageFromArray(pred_max.cpu().numpy())
                gt_labels= sitk.GetImageFromArray(target.cpu().numpy())
                sitk.WriteImage(out_prob, path_Save+'/prob_'+str(step)+'_'+str(bs)+'.nii')
                sitk.WriteImage(gt_labels, path_Save+'GTlabel_'+str(step)+'_'+str(bs)+'.nii')

        overall_liver=cm_liver.get_results_cm()
        overall_tumor=cm_tumor.get_results_cm()
        mean_loss.append(loss.item())
        string='Validation: epoch:{}, step:{}/{}, loss: {:.4f}, time:{:.2f} min\n'.format(epoch,step,len(val_dl) ,loss.item(), (time() - start) / 60)
        string+='\tLiver Acc:{:.1f}, mIoU: {:.1f}, Acc: {:.1f}, {:.1f}\n'.format(overall_liver['Overall Acc']*100, overall_liver['Mean IoU']*100, overall_liver['Mean Acc'][0]*100, overall_liver['Mean Acc'][1]*100)
        string+='\tTumor Acc:{:.1f}, mIoU: {:.1f}, Acc: {:.1f}, {:.1f}'.format(overall_tumor['Overall Acc']*100, overall_tumor['Mean IoU']*100, overall_tumor['Mean Acc'][0]*100, overall_tumor['Mean Acc'][1]*100)
        print(string)
        with open('log/balanced/log_val_iter.txt','a') as f: f.write(string+'\n')
    print('\t\t<------------------>\n')


for epoch in range(16,20000):
    if (epoch+0)%5==0:
        validate=True
    else:
        validate=False

    mean_loss=train_net()
    mean_loss = sum(mean_loss) / len(mean_loss)
    string='Train Epoch {} : Mean Loss: {:.3f}\n'.format(epoch,mean_loss)
    print(string)
    with open('log/balanced/log_train_epoch.txt','a') as f: f.write(string+'\n')
    if epoch>50:
        validate=True
    if validate:
        val_net(epoch,save=False)
        torch.save(net.state_dict(), './module/val_net{}-{:.3f}.pth'.format(epoch, mean_loss))
        # lr_decay.step()
        


