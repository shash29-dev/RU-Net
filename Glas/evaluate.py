import os
import numpy as np 
import torch
import pdb
import arch_utils
import torch.nn as nn
import torch.nn.functional as F
import os
from time import time
from t2_with_grade import MIPSPnet_t2
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

def view(seg,pred,image,step,grade_gt_label,grade_pred_label):
    if grade_gt_label==0:
        string='Malignant'
    else:
        string= 'Benign'

    if grade_pred_label==0:
        stringP='Malignant'
    else:
        stringP='Benign'

    ax=plt.subplot(1,3,1)
    ax.imshow(seg)
    ax.axis('off')
    ax.set_title('GT Glands\n{}'.format(string))
    ax=plt.subplot(1,3,2)
    ax.imshow(pred)
    ax.axis('off')
    ax.set_title('Pred Glands\n{}'.format(stringP))
    ax=plt.subplot(1,3,3)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Input Image')
    plt.savefig('./prediction_files/images/image_{}.png'.format(step),dpi=720)
    plt.close('all')
    # pdb.set_trace()

  
net = MIPSPnet_t2(size=16,training=True).cuda()


net.load_state_dict(torch.load('./module/val_net1008.pth'))
net.eval()
pdb.set_trace()
cm_gland=RunningConfusionMatrix()
loss_func=nn.BCELoss(weight=None)


mean_loss = []
cm_gland=RunningConfusionMatrix()
epoch=563
start=time()
correct_grade_pred=0
total_grades=0
for step, (image, seg,benign) in enumerate(val_dl):
    image = image.cuda()
    seg = seg.cuda()
    with torch.no_grad():
        gland_pred, grade_pred=net(image)
    loss=loss_func(gland_pred,seg.unsqueeze(1))
    mean_loss.append(loss.item())
    
    gland_pred_labels=np.round(gland_pred.cpu().numpy())
    grade_pred_label= np.round(grade_pred.cpu().numpy())
    grade_gt_label=benign.long().cpu().numpy()
    correct_pred=np.sum(np.round(grade_pred.detach().cpu().numpy().flatten())==benign.flatten().cpu().numpy())
    correct_grade_pred+=correct_pred
    total_grades+=batch_size
    # pdb.set_trace()

    _=cm_gland.update_matrix(gland_pred_labels.flatten(),seg.cpu().numpy().flatten())

    overall=cm_gland.get_results_cm()
    string='Validation: epoch:{}, step:{}/{}, loss: {:.4f}, Grade Acc: {:.3f},time:{:.2f} min\n'.format(epoch,step,len(val_dl) ,loss.item(), correct_grade_pred/total_grades,(time() - start) / 60)
    string+='\tGland Acc:{:.1f}, mIoU: {:.1f}, Acc: {:.1f}, {:.1f}\n'.format(overall['Overall Acc']*100, overall['Mean IoU']*100, overall['Mean Acc'][0]*100, overall['Mean Acc'][1]*100)
    print(string)
    with open('./prediction_files/log_val_iter.txt','a') as f: f.write(string+'\n')
    view(seg[0].cpu().numpy(),gland_pred_labels[0,0], image[0].permute(1,2,0).cpu().numpy(),step,grade_gt_label,grade_pred_label)

print('\t\t<------------------>\n')






overall=cm_gland.get_results_cm()
print(overall)
pdb.set_trace()
pickle.dump(overall, open('./prediction_files/overall_cm.pkl','wb'))
pdb.set_trace()