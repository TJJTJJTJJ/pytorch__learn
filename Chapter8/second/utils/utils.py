
# coding: utf-8

# In[1]:


import torchvision as tv
import torchvision.transforms as T


#     # 加载图片可以使用第六章的方式
#     # data = Image.open(path), transform(data)
#     # 作者这次用了新的方式tv.datasets.folder.default_loader
#     # 官网没有找到，只能去看源码了
#     # 给出几个解释
#     # https://www.jianshu.com/p/db61875b73fb
#     # 虽然python里面自带一个PIL（python images library), 但这个库现在已经停止更新了，所以使用Pillow, 它是由PIL发展而来的。
#     # 看源码，发现源码是这么加载图片的，
#     # with open(path,'rb') as f: img = Image.open(f), return img.convert('RGB') 看起来要比咱们的高级一些，但也要给作者提提问题

# In[2]:


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# In[3]:


def get_style_data(path):
    """
    获取单张风格图片，并将其转化为tensor
    path：
    vgg网络的输入是在Image上训练得到的，我们需要对其以同样的方式进行归一化
    return： 1*c*H*W,分布-2~2 tensor
    """
    
    
    style_image = tv.datasets.folder.default_loader(path) # 0-1
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])
    style_tensor = transforms(style_image)
    return style_tensor.unsqueeze(0) 


# In[7]:


def normalize_batch(batch_data):
    """
    data:0-255, b*3*h*w, cuda, Variable
    return:-2~2,b*3*h*w, Variable
    """
    # new 创建一个相同类型的数据，主要针对与cuda
    mean = batch_data.data.new(IMAGENET_MEAN).view(1,-1,1,1)
    std = batch_data.data.new(IMAGENET_STD).view(1,-1,1,1)
    mean = mean.expand_as(batch_data.data)
    std = std.expand_as(batch_data.data)
    return (batch_data/255-mean)/std
    
    


# In[16]:


def gram_matrix(x):
    """
    求y的gram矩阵
    输入 b*c*h*w cuda, tensor
    输出 b*c*c cuda tensor
    """
    (b,c,h,w) = x.size()
    features = x.view(b,c,h*w)
    features_t = features.transpose(1,2)
    # 我觉得这里应该除以 h*w就行，因为b*c*c中的每一个元素只代表了H×W个元素， 但是先按照作者的思路写
    gram = features.bmm(features_t)/(c*h*w)
    return gram
    


# In[8]:


import torch 


# In[14]:


a  = torch.Tensor(range(24))


# In[15]:


a.view(2,3,4)

