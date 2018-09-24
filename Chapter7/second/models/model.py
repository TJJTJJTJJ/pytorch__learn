
# coding: utf-8

# In[1]:


from torch import nn
from .BasicModule import BasicModule
from torch import autograd
import torch


# ###  定义生成器  init(),forward()

# In[2]:


class NetG(BasicModule):
    """
    生成器定义
    __init__()
    forward()
    """
    def __init__(self,opt):
        super(NetG,self).__init__()
        ngf = opt.ngf
        self.main = nn.Sequential(
            # 输入 1*nz*1*1维的噪声
            nn.ConvTranspose2d(opt.nz,ngf*8,  4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # (ngf*8)*4*4
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # (ngf*4)*8*8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2)*16*16
            nn.ConvTranspose2d(ngf*2, ngf*1, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*1),
            nn.ReLU(True),
            # (ngf*1)*32*32
            nn.ConvTranspose2d(ngf, 3, 5,3,1,bias=False),
            nn.Tanh()
            # 3*96*96  range(-1,1)
        )
    def forward(self,x):
        return self.main(x)


# ### 定义判别器

# In[3]:


class NetD(BasicModule):
    """
    判别器
    __init__()
    forward()
    """
    def __init__(self,opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # 与生成器正好相反
            # 3*96*96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*32*32
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            # batch*1*1*1
            nn.Sigmoid()
            
        )
    def forward(self,x):
        return self.main(x).view(-1) # batch
        

