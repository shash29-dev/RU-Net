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
from t2 import MIPSPnet_t2
from metrics import RunningConfusionMatrix
import SimpleITK as sitk

batch_size=8
num_workers=8
pin_memory=True
learning_rate_decay = [500, 750]
learning_rate = 1e-4
train_ds = arch_utils.GetData(file_dir='./GlasData')
train_dl = torch.utils.data.DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

val_ds = arch_utils.GetData(file_dir='./GlasData',data='test')
val_dl = torch.utils.data.DataLoader(val_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

os.environ['CUDA_VISIBLE_DEVICES']='0'

        
# pdb.set_trace() 
if not os.path.exists('./module'): os.makedirs('./module')
if not os.path.exists('./log/balanced'): os.makedirs('./log/balanced')
net = MIPSPnet_t2(size=16,training=True).cuda()
# net.load_state_dict(torch.load('./module/val_net5.pth'))
net.train()
opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, learning_rate_decay)
alpha=0.33
# class_weights=torch.FloatTensor(np.load('weights.npy')).cuda()
loss_func=nn.BCELoss(weight=None)
start=time()

def train_net():
    # net.training=True
    net.train()
    mean_loss = []
    correct_grade_pred=0
    total_grades=0
    for step, (image, seg, benign) in enumerate(train_dl):
        image = image.cuda()
        seg = seg.cuda()
        benign=benign.cuda()
        gland_pred, grade_pred=net(image)

        loss_benign=loss_func(grade_pred.flatten(),benign.float().cuda())
        loss_label=loss_func(gland_pred,seg.unsqueeze(1))
        loss= (0.2* loss_benign) + (0.8* loss_label)
        correct_pred=np.sum(np.round(grade_pred.detach().cpu().numpy().flatten())==benign.flatten().cpu().numpy())
        correct_grade_pred+=correct_pred
        total_grades+=batch_size
        mean_loss.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        string='Training: epoch:{}, step:{}/{}, loss: {:.3f}, Grade Acc: {:.3f}, time:{:.3f} min'.format(epoch,step,len(train_dl) ,loss.item(),correct_grade_pred/total_grades ,(time() - start) / 60)
        print(string)
        with open('log/balanced/log_train_iter.txt','a') as f: f.write(string+'\n')
    print('\t\t<------------------>\n')
    return mean_loss

def val_net(epoch,save=False):
    # net.training=False
    net.eval()
    mean_loss = []
    correct_grade_pred=0
    total_grades=0
    cm_gland=RunningConfusionMatrix()
    for step, (image, seg, benign) in enumerate(val_dl):
        image = image.cuda()
        seg = seg.cuda()
        with torch.no_grad():
            gland_pred,grade_pred=net(image)

        loss=loss_func(gland_pred,seg.unsqueeze(1))
        mean_loss.append(loss.item())
        gland_pred_labels=np.round(gland_pred.cpu().numpy())
        _=cm_gland.update_matrix(gland_pred_labels.flatten(),seg.cpu().numpy().flatten())
        correct_pred=np.sum(np.round(grade_pred.detach().cpu().numpy().flatten())==benign.flatten().cpu().numpy())
        correct_grade_pred+=correct_pred
        total_grades+=batch_size
        overall=cm_gland.get_results_cm()
        string='Validation: epoch:{}, step:{}/{}, loss: {:.4f}, Grade Acc: {:.3f},time:{:.2f} min\n'.format(epoch,step,len(val_dl) ,loss.item(), correct_grade_pred/total_grades,(time() - start) / 60)
        string+='\tGland Acc:{:.1f}, mIoU: {:.1f}, Acc: {:.1f}, {:.1f}\n'.format(overall['Overall Acc']*100, overall['Mean IoU']*100, overall['Mean Acc'][0]*100, overall['Mean Acc'][1]*100)
        print(string)
        with open('log/balanced/log_val_iter.txt','a') as f: f.write(string+'\n')
    print('\t\t<------------------>\n')


for epoch in range(20000):
    if (epoch+0)%100==0:
        validate=True
    else:
        validate=False

    mean_loss=train_net()
    mean_loss = sum(mean_loss) / len(mean_loss)
    string='Train Epoch {} : Mean Loss: {:.3f}\n'.format(epoch,mean_loss)
    print(string)
    with open('log/balanced/log_train_epoch.txt','a') as f: f.write(string+'\n')
    if epoch>1000:
        validate=True
    if validate:
        val_net(epoch,save=False)
        torch.save(net.state_dict(), './module/val_net{}.pth'.format(epoch))
        # lr_decay.step()
        


