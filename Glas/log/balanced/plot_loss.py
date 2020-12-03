import numpy as np 
import matplotlib.pyplot as plt 
import pdb


with open('./log_train_epoch.txt') as f:
	lines=f.readlines()

lines_t=[x for x in lines if x.startswith('Train')]
tline= [float(x.split()[-1].strip()) for x in lines_t]


with open('./log_val_iter.txt') as f:
	val_lines=f.readlines()

epoch_list=[]
loss_list=[]
acc_list=[]
bg_Acc_list=[]
liver_acc_list=[]
mIoU_list=[]
grade_acc_list=[]
for idx,line in enumerate(val_lines):
	if line.startswith('Validation'):
		if line.split('step:')[1].startswith('9/10'):
			tmp=val_lines[idx:idx+3]
			grade_acc_list.append(float(tmp[0].split('Grade Acc: ')[1].split(',')[0]))
			loss_list.append(float(tmp[0].split('loss: ')[-1].split(',')[0]))
			epoch_list.append(int(tmp[0].split('epoch:')[-1].split(',')[0]))
			acc_list.append(float(tmp[1].split('Acc:')[1].split(',')[0]))
			bg_Acc_list.append(float(tmp[1].split('Acc:')[-1].split(',')[0].strip()))
			liver_acc_list.append(float(tmp[1].split('Acc:')[-1].split(',')[1].strip()))
			mIoU_list.append(float(tmp[1].split('mIoU: ')[1].split(',')[0]))


epoch_list=epoch_list[:-8]+epoch_list[-1:]
loss_list=loss_list[:-8]+loss_list[-1:]
acc_list=acc_list[:-8]+acc_list[-1:]
bg_Acc_list=bg_Acc_list[:-8]+bg_Acc_list[-1:]
liver_acc_list=liver_acc_list[:-8]+liver_acc_list[-1:]
mIoU_list=mIoU_list[:-8]+mIoU_list[-1:]
grade_acc_list=grade_acc_list[:-8]+grade_acc_list[-1:]




plt.plot(epoch_list,loss_list, label='Validation Loss')
plt.plot(tline, label='Training Loss')
plt.legend()
plt.title('BCE Loss')

plt.figure()
plt.plot(epoch_list, acc_list, label='Average Accuracy')
plt.plot(epoch_list, bg_Acc_list, label='Background Accuracy')
plt.plot(epoch_list, liver_acc_list, label='Gland Accuracy')
plt.plot(epoch_list,mIoU_list, label='mIoU')
plt.plot(epoch_list,np.array(grade_acc_list)*100, label='Grade Accuracy')
plt.title('Metric score on Validation Set')


plt.legend()



plt.show()



pdb.set_trace()
