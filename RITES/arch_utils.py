import os
import sys
import random
import numpy as np 
import SimpleITK as sitk 
import torch
import pdb
import glob
import Transforms_utils
import PIL
import matplotlib.pyplot as plt
import pandas as pd

class GetData(torch.utils.data.Dataset):
    def __init__(self, file_dir, data='training'):

        self.gt= glob.glob(file_dir+'/'+data+'/vessel/*.png')
        self.transformList_train=[
                        Transforms_utils.RandomHorizontalFlip(p=0.5),
                        Transforms_utils.RandomVerticalFlip(p=0.5),
                        Transforms_utils.Rescale(520),
                        Transforms_utils.RandomCrop(512),
                        Transforms_utils.ToTensor()
                        ]
        
    def __getitem__(self, index):
        label=self.gt[index]
        image=label.replace('vessel','images')
        # gt=PIL.Image.open(label)
        gt_av=PIL.Image.open(label.replace('vessel','av'))
        image=PIL.Image.open(image.replace('png','tif'))
        x=np.array(gt_av)
        gt_label=np.zeros(x.shape[:-1])
        gt_label[np.where(x[:,:,0]==255)]=1
        gt_label[np.where(x[:,:,2]==255)]=2
        gt= PIL.Image.fromarray(gt_label)
        sample={'image':image,'gt':gt}
        trans= Transforms_utils.transformImage(sample)
        sample=trans(self.transformList_train)
        sample['gt']=(sample['gt']>0).float()

        return sample['image'], sample['gt']

    def __len__(self):
        return len(self.gt)


class GetEvalData(torch.utils.data.Dataset):
    def __init__(self, file_dir, data='train'):
        self.gt= glob.glob(file_dir+'/'+data+'/vessel/*.png')
        self.transformList_train=[
                        # Transforms_utils.RandomHorizontalFlip(p=0.5),
                        # Transforms_utils.RandomVerticalFlip(p=0.5),
                        Transforms_utils.Rescale(520),
                        Transforms_utils.RandomCrop(512),
                        Transforms_utils.ToTensor()
                        ]
        # self.grade=pd.read_csv('./GlasData/Grade.csv')
        
    def __getitem__(self, index):
        label=self.gt[index]
        image=label.replace('vessel','images')
        # gt=PIL.Image.open(label)
        gt_av=PIL.Image.open(label.replace('vessel','av'))
        image=PIL.Image.open(image.replace('png','tif'))
        x=np.array(gt_av)
        gt_label=np.zeros(x.shape[:-1])
        gt_label[np.where(x[:,:,0]==255)]=1
        gt_label[np.where(x[:,:,2]==255)]=2
        gt= PIL.Image.fromarray(gt_label)
        sample={'image':image,'gt':gt}
        trans= Transforms_utils.transformImage(sample)
        sample=trans(self.transformList_train)
        sample['gt']=(sample['gt']>0).float()

        return sample['image'], sample['gt']

    def __len__(self):
        return len(self.gt)



