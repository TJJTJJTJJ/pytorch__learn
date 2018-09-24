
# coding: utf-8

# # 风格迁移网络
# 主体网络TransformerNet由三个小网络组成,并且小网络内部的模块是单一的，分别是
# * 小网络
# * * 小网络的模块
# * downconv_layers
# * * 3*DownConv
# * residulablock_layers
# * * 5* ResidulaBlock
# * upconv_layers
# * * UpConv+UpConv+Conv
# 
# 在这章，所有的卷积层都变成了ReflectionPad2d+Conv2d，所以我们用一个Conv模块进行代替。 这一块还能继续优化，待续
# 

# In[1]:


from torch import nn
import torch as t
import numpy as np


# In[2]:


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet,self).__init__()
        
        # 下卷积
        # H-->H-->H/2-->H/4
        self.downconv_layer = nn.Sequential(
            DownConv( 3, 32,kernel_size=9,stride=1),
            DownConv(32, 64,kernel_size=3,stride=2), 
            DownConv(64,128,kernel_size=3,stride=2)  

        )
        
        
        #  Residual layers
        # 这里应该可以改进，待续
        self.residualblock_layers = nn.Sequential(
            ResidualBlock(128,128,3,1),
            ResidualBlock(128,128,3,1),
            ResidualBlock(128,128,3,1),
            ResidualBlock(128,128,3,1),
            ResidualBlock(128,128,3,1)
        )
        
        # Upsampling Layers
        # 第八章的和第七章的从小变大的方法不一样，第七章的变大是通过逆卷积的方法ConvTranspose2d()，但这次是通过Unsampling的方法。


        self.upconv_layer = nn.Sequential(
            UpConv(128,64,3,1,upsample = 2),
            UpConv(64,32,3,1,upsample=2),
            Conv(32,3,9,1)       
        )

    def forward(self,x):
        x = self.downconv_layer(x)
        x = self.residualblock_layers(x)
        x = self.upconv_layer(x)
        return x
        


# In[3]:


class DownConv(nn.Module):
    """
    下卷积
    使用边界反射补充，原因未知
    当stride=2的时候，为什么还要边界反射补充
    每一个下卷积层的block都是一样的 relected+InstanceNorm2d+relu
    upsample上采样，是pooling的逆过程
    """
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(DownConv,self).__init__()
        self.main = nn.Sequential(   
            Conv(in_channels,out_channels,kernel_size,stride),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(inplace=True)
        )
        
        
    
    def forward(self,x):
        return self.main(x)
        


# In[4]:


class ResidualBlock(nn.Module):
    """
    残差层，用于过渡
    这个残差层写的也不一样，第六章实现的block是conv+x->relu,但作者给出的代码是conv+x->输出
    有待结果验证
    RelU也没有设置True
    
    """
    def __init__(self,input_channels,out_channels,kernels,stride):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            Conv(input_channels,out_channels,kernels,stride),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(inplace=True),
            Conv(out_channels,out_channels,kernels,stride),
            nn.InstanceNorm2d(out_channels,affine=True)
            
        )
        
        
    
    def forward(self,x):
        # zhelikeneng youcuo 
        residual = x
        out = self.left(x)
        out = out + residual
        return out
        


# In[5]:


class UpConv(nn.Module):
    """
    上卷积
    这里使用边界反射填充
    这里有疑问，不知道怎么实现的
    ref:http://distill.pub/2016/deconv-checkerboard/
    待续
    upsample上采样，是pooling的逆过程，刚刚看了看说明，感觉不太对，还有一个align_corner也没有想明白
    待续
    
    """
    
    def __init__(self,input_channels,out_channels,kernel_size,stride, upsample=None):
        super(UpConv,self).__init__()
        self.upsample = upsample
        if self.upsample:
        # wo jued upsample yinggai 
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.main = nn.Sequential(
            Conv(input_channels,out_channels,kernel_size,stride),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(inplace=True)
        )
        
      
    
    def forward(self,x):
        #if self.upsample:
        #    x = self.upsample_layer(x)
        # zheli yaoxie zhushi jide
        if self.upsample:
           x = t.nn.functional.interpolate(x, scale_factor=self.upsample)
        x = self.main(x)
        return x
        


# In[6]:
class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    默认的卷积的padding操作是补0，这里使用边界反射填充
    先上采样，然后做一个卷积(Conv2d)，而不是采用ConvTranspose2d，这种效果更好，参见
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = t.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class Conv(nn.Module):
    # 相当于在写ReflectionPad+Conv,因为所有的conv都变成了这个样子，所以我们专门拿出来，避免造轮子
    # 这一块也可以重新优化，先根据作者的来，如果不行，再换。
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(Conv,self).__init__()
        self.main = nn.Sequential(   
            nn.ReflectionPad2d(int(kernel_size/2)),
            nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        )
        
    def forward(self,x):
        return self.main(x)

class ConvLayer(nn.Module):
    """
    add ReflectionPad for Conv
    默认的卷积的padding操作是补0，这里使用边界反射填充
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

