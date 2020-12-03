import os
import numpy as np 
import torch
import pdb
import arch_utils
import torch.nn as nn
import torch.nn.functional as F
import os
from time import time
from t2 import MIPSPnet_t2
from metrics import RunningConfusionMatrix
import SimpleITK as sitk
import pickle
import matplotlib.pyplot as plt

batch_size=1
num_workers=8
pin_memory=True

val_ds = arch_utils.GetEvalData(file_dir='./GlasData',data='test')
val_dl = torch.utils.data.DataLoader(val_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
os.environ['CUDA_VISIBLE_DEVICES']='0'

def view(seg,pred,image,step):
    ax=plt.subplot(1,3,1)
    ax.imshow(seg)
    ax.axis('off')
    ax.set_title('GT Glands')
    ax=plt.subplot(1,3,2)
    ax.imshow(pred)
    ax.axis('off')
    ax.set_title('Pred Glands')
    ax=plt.subplot(1,3,3)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Input Image')
    plt.savefig('./prediction_files/images/image_{}.png'.format(step),dpi=720)
    plt.close('all')

  
net = MIPSPnet_t2(size=16,training=True).cuda()
net.load_state_dict(torch.load('./module/val_net300.pth'))
net.eval()
cm_gland=RunningConfusionMatrix()
loss_func=nn.BCELoss(weight=None)


mean_loss = []
cm_gland=RunningConfusionMatrix()
epoch=300
start=time()

for step, (image, seg) in enumerate(val_dl):
    image = image.cuda()
    seg = seg.cuda()
    with torch.no_grad():
        gland_pred=net(image)

    loss=loss_func(gland_pred,seg.unsqueeze(1))
    mean_loss.append(loss.item())
    
    gland_pred_labels=np.round(gland_pred.cpu().numpy())
    _=cm_gland.update_matrix(gland_pred_labels.flatten(),seg.cpu().numpy().flatten())

    overall=cm_gland.get_results_cm()
    string='Validation: epoch:{}, step:{}/{}, loss: {:.4f}, time:{:.2f} min\n'.format(epoch,step,len(val_dl) ,loss.item(), (time() - start) / 60)
    string+='\tGland Acc:{:.1f}, mIoU: {:.1f}, Acc: {:.1f}, {:.1f}\n'.format(overall['Overall Acc']*100, overall['Mean IoU']*100, overall['Mean Acc'][0]*100, overall['Mean Acc'][1]*100)
    print(string)
    with open('./prediction_files/log_val_iter.txt','a') as f: f.write(string+'\n')
    view(seg[0].cpu().numpy(),gland_pred_labels[0,0], image[0].permute(1,2,0).cpu().numpy(),step)

print('\t\t<------------------>\n')






overall=cm_gland.get_results_cm()
print(overall)
pdb.set_trace()
pickle.dump(overall, open('./prediction_files/overall_cm.pkl','wb'))
pdb.set_trace()