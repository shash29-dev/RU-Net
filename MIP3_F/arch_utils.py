import os
import sys
import random
import numpy as np 
import SimpleITK as sitk 
import torch
import pdb
import glob

class GetData(torch.utils.data.Dataset):
	def __init__(self, ct_dir, seg_dir):
		self.cts= glob.glob(ct_dir+'/vol*.nii')
		self.size=16


	def __getitem__(self, index):
		ct_path=self.cts[index]
		seg_path=ct_path.replace('ct/volume','seg/segmentation')

		ct_array= sitk.GetArrayFromImage(sitk.ReadImage(ct_path, sitk.sitkInt16))
		seg_array= sitk.GetArrayFromImage(sitk.ReadImage(seg_path, sitk.sitkInt16))

		ct_array=ct_array.astype(np.float32)/200
		start_slice = random.randint(0, ct_array.shape[0] - self.size)
		end_slice = start_slice + self.size - 1
		ct_array = ct_array[start_slice:end_slice + 1, :, :]
		seg_array = seg_array[start_slice:end_slice + 1, :, :]
		ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
		seg_array = torch.FloatTensor(seg_array)

		return ct_array, seg_array

	def __len__(self):
		return len(self.cts)


class GetEvalData(torch.utils.data.Dataset):
	def __init__(self, ct_dir, seg_dir):
		self.cts= glob.glob(ct_dir+'/vol*.nii')
		self.size=48


	def __getitem__(self, index):
		ct_path=self.cts[index]
		seg_path=ct_path.replace('ct/volume','seg/segmentation')

		ct_array= sitk.GetArrayFromImage(sitk.ReadImage(ct_path, sitk.sitkInt16))
		seg_array= sitk.GetArrayFromImage(sitk.ReadImage(seg_path, sitk.sitkInt16))

		ct_array=ct_array.astype(np.float32)/200
		# start_slice = random.randint(0, ct_array.shape[0] - self.size)
		# end_slice = start_slice + self.size - 1
		# ct_array = ct_array[start_slice:end_slice + 1, :, :]
		# seg_array = seg_array[start_slice:end_slice + 1, :, :]
		ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
		seg_array = torch.FloatTensor(seg_array)

		return ct_array, seg_array

	def __len__(self):
		return len(self.cts)



