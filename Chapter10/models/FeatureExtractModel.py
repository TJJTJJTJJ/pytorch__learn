
# coding: utf-8

# In[3]:


from torch import nn
import torchvision as tv


# In[ ]:

__all__ = ['FeatureExtractModel']

class FeatureExtractModel(nn.Module):
    def __init__(self):
        super(FeatureExtractModel, self).__init__()
        resnet50 = tv.models.resnet50(pretrained=True)
        del resnet50.fc
        resnet50.fc = lambda x:x
        self.resnet50 = resnet50
        
    def forward(self, x):
        
        return self.resnet50(x)

