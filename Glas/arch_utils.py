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
    def __init__(self, file_dir, data='train'):
        self.gt= glob.glob(file_dir+'/'+data+'*_anno.bmp')
        self.transformList_train=[
                        Transforms_utils.RandomHorizontalFlip(p=0.5),
                        Transforms_utils.RandomVerticalFlip(p=0.5),
                        Transforms_utils.Rescale(520),
                        Transforms_utils.RandomCrop(512),
                        Transforms_utils.ToTensor()
                        ]
        self.grade=pd.read_csv('./GlasData/Grade.csv')

        
    def __getitem__(self, index):
        label=self.gt[index]
        name=label.split('/')[-1].split('.')[0].split('_anno')[0]
        benign=self.grade[self.grade['name']==name][' grade (GlaS)']
        benign=np.array(benign).tolist()[0].strip()=='benign'
        image=label.replace('_anno','')
        gt=PIL.Image.open(label)
        image=PIL.Image.open(image)
        sample={'image':image,'gt':gt}
        trans= Transforms_utils.transformImage(sample)
        sample=trans(self.transformList_train)
        sample['gt']=(sample['gt']>0).float()
        sample['benign']=benign
        return sample['image'], sample['gt'], sample['benign']

    def __len__(self):
        return len(self.gt)


class GetEvalData(torch.utils.data.Dataset):
    def __init__(self, file_dir, data='train'):
        self.gt= glob.glob(file_dir+'/'+data+'*_anno.bmp')
        self.transformList_train=[
                        # Transforms_utils.RandomHorizontalFlip(p=0.5),
                        # Transforms_utils.RandomVerticalFlip(p=0.5),
                        Transforms_utils.Rescale(520),
                        Transforms_utils.RandomCrop(512),
                        Transforms_utils.ToTensor()
                        ]
        self.grade=pd.read_csv('./GlasData/Grade.csv')
        
    def __getitem__(self, index):
        label=self.gt[index]
        name=label.split('/')[-1].split('.')[0].split('_anno')[0]
        benign=self.grade[self.grade['name']==name][' grade (GlaS)']
        benign=np.array(benign).tolist()[0].strip()=='benign'
        image=label.replace('_anno','')
        gt=PIL.Image.open(label)
        image=PIL.Image.open(image)
        sample={'image':image,'gt':gt}
        trans= Transforms_utils.transformImage(sample)
        sample=trans(self.transformList_train)
        sample['gt']=(sample['gt']>0).float()
        sample['benign']=benign
        return sample['image'], sample['gt'], sample['benign']

    def __len__(self):
        return len(self.gt)



