import os
import numpy as np 
import torch
import pdb
import arch_utils
import torch.nn as nn
import torch.nn.functional as F
import os
from time import time

class Block(nn.Module):
    def __init__(self, in_planes,out_planes,kernel,stride=1,padding=1,pool_kernel=2,training=True):
        super(Block,self).__init__()
        self.training=training
        self.pool_kernel=pool_kernel
        self.inconv=nn.Conv3d(in_planes,in_planes,kernel,stride,padding)
        self.bn1= nn.BatchNorm3d(in_planes)
        self.relu1=nn.PReLU(in_planes)
        self.conv1=nn.Conv3d(in_planes,out_planes,kernel,stride,padding)
        self.bn2= nn.BatchNorm3d(out_planes)
        self.relu2=nn.PReLU(out_planes)

    def forward(self,x,y=None):
        dr=0.0 #After ep 50
        out1=F.dropout(self.relu1(self.bn1(self.inconv(x))),dr,self.training)
        out_res=x+out1
        out2=F.dropout(self.relu2(self.bn2(self.conv1(out_res))),dr,self.training)
        if y==None:
            outp=F.max_pool3d(out2,kernel_size=self.pool_kernel)
        else:
            upL=nn.Upsample(size=y.shape[2:], mode='trilinear', align_corners=True)
            # pdb.set_trace()
            outp=torch.cat((upL(out2),y),dim=1)
        return out2,outp


class MIPSPnet_t2(torch.nn.Module):
    def __init__(self,size,training=True):
        super(MIPSPnet_t2,self).__init__()
        self.training=training
        planes=[4,8,16,32,64,128,256]
        pool=[(1,2,2),(2,1,1),2]
        # self.conv1=nn.Conv3d(1,planes[0],3,1,1)
        self.block1= Block(in_planes=planes[0],out_planes=planes[1],kernel=3,pool_kernel=pool[0],training=self.training)
        self.block2= Block(in_planes=planes[1],out_planes=planes[2],kernel=3,pool_kernel=pool[0],training=self.training)
        self.block3_1= Block(in_planes=planes[2],out_planes=planes[3],kernel=3,pool_kernel=pool[1],training=self.training)
        self.block3_2= Block(in_planes=planes[3],out_planes=planes[3],kernel=3,pool_kernel=pool[2],training=self.training)

        self.block4_1= Block(in_planes=planes[3],out_planes=planes[4],kernel=3,pool_kernel=pool[1],training=self.training)
        self.block4_2= Block(in_planes=planes[4],out_planes=planes[4],kernel=3,pool_kernel=pool[2],training=self.training)

        self.block5_1= Block(in_planes=planes[4],out_planes=planes[5],kernel=3,pool_kernel=pool[1],training=self.training)
        self.block5_2= Block(in_planes=planes[5],out_planes=planes[5],kernel=3,pool_kernel=pool[2],training=self.training)

        self.block6= Block(in_planes=planes[5],out_planes=planes[6],kernel=3,pool_kernel=pool[0],training=self.training)

        self.up_block6=Block(in_planes=planes[6],out_planes=planes[5],kernel=3,pool_kernel=pool[0],training=self.training)
        self.up_block5=Block(in_planes=planes[5]*2,out_planes=planes[4],kernel=3,pool_kernel=pool[0],training=self.training)
        self.up_block4=Block(in_planes=planes[4]*2,out_planes=planes[3],kernel=3,pool_kernel=pool[0],training=self.training)
        self.up_block3=Block(in_planes=planes[3]*2,out_planes=planes[2],kernel=3,pool_kernel=pool[0],training=self.training)
        self.up_block2=Block(in_planes=planes[2]*2,out_planes=planes[1],kernel=3,pool_kernel=pool[0],training=self.training)
        self.up_block1=Block(in_planes=planes[1]*2,out_planes=planes[0],kernel=3,pool_kernel=pool[0],training=self.training)

        self.final=nn.Sequential(
                nn.Conv3d(planes[0]*2,planes[0]*2,3,1,1),
                nn.PReLU(planes[0]*2),
                nn.BatchNorm3d(planes[0]*2),
                nn.Conv3d(planes[0]*2,2,3,1,1),
                nn.Sigmoid()
            )

    
    def forward(self,x):

        # x1=self.conv1(x)
        x1=x
        _,x2p=self.block1(x1)
        _,x3p=self.block2(x2p)

        x4,x4p=self.block3_1(x3p)
        _,x5p=self.block3_2(torch.cat((x4p,x4),dim=2))

        x6,x6p=self.block4_1(x5p)
        _,x7p=self.block4_2(torch.cat((x6p,x6),dim=2))

        x8,x8p=self.block5_1(x7p)
        _,x9p=self.block5_2(torch.cat((x8p,x8),dim=2))

        _,x10p=self.block6(x9p)

        _,x9up=self.up_block6(x10p,x9p)
        _,x7up=self.up_block5(x9up,x7p)
        _,x5up=self.up_block4(x7up,x5p)
        _,x3up=self.up_block3(x5up,x3p)
        _,x2up=self.up_block2(x3up,x2p)
        _,x1up=self.up_block1(x2up,x1)
        final=self.final(x1up)
        return x1up,final

class ParallelNet(torch.nn.Module):
    def __init__(self,size,training=True):
        super(ParallelNet,self).__init__()
        self.conv1=nn.Conv3d(1,4,3,1,1)
        self.conv2=nn.Conv3d(1,4,3,1,1)
        self.liver_Seg=MIPSPnet_t2(size=size)
        self.tumor_Seg=MIPSPnet_t2(size=size)
    def forward(self,x):
        conv1_liver=self.conv1(x)
        _,liver_pred=self.liver_Seg(conv1_liver)
        
        conv2_liver=self.conv1(x)
        _,tumor_pred=self.tumor_Seg(conv2_liver)
        return liver_pred,tumor_pred