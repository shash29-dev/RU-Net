import os
import numpy as np 
import torch
import pdb
import arch_utils
import torch.nn as nn
import torch.nn.functional as F
import os
from time import time
from t2 import ParallelNet
from metrics import RunningConfusionMatrix
import SimpleITK as sitk
import pickle
import matplotlib.pyplot as plt

batch_size=1
num_workers=8
pin_memory=True

val_ds = arch_utils.GetEvalData(ct_dir='./processed_data/test/ct',seg_dir='./processed_data/test/seg')
val_dl = torch.utils.data.DataLoader(val_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

os.environ['CUDA_VISIBLE_DEVICES']='0'

def view(labelgtL,predictionL,inputct,labelgtT,predictionT,step,stepidx):
    for idx in range(labelgtL.shape[0]):
        ax=plt.subplot(1,3,1)
        ax.imshow(labelgtL[idx])
        ax.axis('off')
        ax.set_title('GT Liver')
        ax=plt.subplot(1,3,2)
        ax.imshow(predictionL[idx])
        ax.axis('off')
        ax.set_title('Liver pred')
        ax=plt.subplot(1,3,3)
        ax.imshow(inputct[idx])
        ax.axis('off')
        ax.set_title('Input')
        plt.savefig('./prediction_files/images/image_{}_{}_{}.png'.format(step,stepidx,idx),dpi=720)
        plt.close('all')
        
# pdb.set_trace() 
if not os.path.exists('./module'): os.makedirs('./module')
net = ParallelNet(size=val_ds.size,training=True).cuda()
net.load_state_dict(torch.load('./module/val_net92-0.629.pth'))
pdb.set_trace()
net.eval()
cm_liver=RunningConfusionMatrix()
cm_tumor=RunningConfusionMatrix()
epoch=10
for step, (ct, seg) in enumerate(val_dl):
    ct=ct.cuda()
    seg=seg.cuda()
    in_ct=torch.zeros(ct[0,0].shape).cuda()
    final_outL=torch.zeros(ct[0,0].shape).cuda()
    final_outT=torch.zeros(ct[0,0].shape).cuda()
    seg_out=torch.zeros(seg[0].shape).cuda()
    slices=list(range(0,ct.shape[2]+1,16))
    if slices[-1]!=ct.shape[2]:
        slices.append(ct.shape[2])
    for idx in range(len(slices)-1):
        ct_slice=ct[:,:,slices[idx]:slices[idx+1],:,:]
        seg_slice=seg[:,slices[idx]:slices[idx+1],:,:]
        if ct_slice.shape[2]<3:
            break
        liver_pred,tumor_pred=net(ct_slice)
        liver_pred=liver_pred.permute(0,2,1,3,4)
        tumor_pred=tumor_pred.permute(0,2,1,3,4)
        predL=liver_pred[0]
        predT=tumor_pred[0]
        Liver_label=seg_slice[0].clone()
        Liver_label[Liver_label>=1]=1

        Tumor_label=seg_slice[0].clone()
        Tumor_label[Tumor_label!=2]=0
        Tumor_label[Tumor_label==2]=1
        predL_max=torch.argmax(predL,dim=1)
        predT_max=torch.argmax(predT,dim=1)
        # _=cm_liver.update_matrix(Liver_label.flatten().cpu().numpy(),predL_max.flatten().cpu().numpy())
        # _=cm_tumor.update_matrix(Tumor_label.flatten().cpu().numpy(),predT_max.flatten().cpu().numpy())
        # overall_liver=cm_liver.get_results_cm()
        # overall_tumor=cm_tumor.get_results_cm()
        # string='Validation: {}/{}, \t{}/{}\n'.format(step,len(val_dl),idx,len(slices) )
        # string+='\tLiver Acc:{:.1f}, mIoU: {:.1f}, Acc: {:.1f}, {:.1f}\n'.format(overall_liver['Overall Acc']*100, overall_liver['Mean IoU']*100, overall_liver['Mean Acc'][0]*100, overall_liver['Mean Acc'][1]*100)
        # string+='\tTumor Acc:{:.1f}, mIoU: {:.1f}, Acc: {:.1f}, {:.1f}'.format(overall_tumor['Overall Acc']*100, overall_tumor['Mean IoU']*100, overall_tumor['Mean Acc'][0]*100, overall_tumor['Mean Acc'][1]*100)
        # print(string)
        # with open('./prediction_files/log_val_iter.txt','a') as f: f.write(string+'\n')
        
        save_screenshots=True
        if save_screenshots:
            labelgtL=Liver_label.cpu().numpy()
            predictionL=predL_max.cpu().numpy()
            inputct=ct_slice[0,0].cpu().numpy()
            labelgtT=Tumor_label.cpu().numpy()
            predictionT=predT_max.cpu().numpy()

            if labelgtL.max()==1 or labelgtT.max()==1:
                print('Max Tumor: ',labelgtT.max())
                view(labelgtL,predictionL,inputct,labelgtT,predictionT,step,idx)

        
    # save=False
    # if save:
    #     path_Save='./prediction_files/15_module'
    #     if not os.path.exists(path_Save): os.makedirs(path_Save)
    #     out_probL = sitk.GetImageFromArray(final_outL.cpu().numpy())
    #     out_probT = sitk.GetImageFromArray(final_outT.cpu().numpy())
    #     in_ct_array=sitk.GetImageFromArray(in_ct.cpu().numpy())
    #     gt_labels= sitk.GetImageFromArray(seg_out.cpu().numpy())
    #     sitk.WriteImage(out_probL, path_Save+'/probL_'+str(step)+'.nii')
    #     sitk.WriteImage(out_probT, path_Save+'/probT_'+str(step)+'.nii')
    #     sitk.WriteImage(gt_labels, path_Save+'/GTlabel_'+str(step)+'.nii')
    #     sitk.WriteImage(in_ct_array, path_Save+'/inct_'+str(step)+'.nii')
    print('-----------------------***---------------------------\n')

# overall=cm_liver.get_results_cm()
# print(overall)
# pickle.dump(overall, open('./prediction_files/overall_cm.pkl','wb'))
# pdb.set_trace()