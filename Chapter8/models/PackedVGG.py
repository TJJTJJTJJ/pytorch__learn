
# coding: utf-8

# # Vgg16
# * 使用预训练好的网络VGG
# * 修改网络的前向传播过程，获取中间层的输出
# * 删除后面的不需要的层

# In[11]:


from torch import nn
from torchvision.models import vgg16
from collections import namedtuple


# In[12]:


class Vgg16(nn.Module):
    # __init__
    # forward()
    # 这里取特征的方法还有研究
    def __init__(self):
        super(Vgg16,self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()
        
    
    def forward(self,x):
        # 输出两种损失 
        # -content loss:relu3_3
        # -style loss:relu1_2,relu2_2,relu3_3,relu3_4 分别对应的是{3,8,15,22}
        # 输出 list:[relu1_2,relu2_2,relu3_3,relu3_4], 其中 relu1_2：b*c*h*w
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22}:
                results.append(x)
        vgg_outputs = namedtuple('VggOutputs',['relu1_2','relu2_2','relu3_3','relu4_3'])
        out = vgg_outputs(*results)
        return out
                

