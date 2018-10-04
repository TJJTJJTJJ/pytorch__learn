
# coding: utf-8

# 因为还没有下载到数据集，所以先写代码，

# 可以适当地把一些疑问写在外面，把注释写在里面，方便看


# 作者也开始直接在文件里写运行代码而不是以函数的形式
# 第一次以dataloader的形式加载data，似乎在尽可能地把train中的东西放外面
# 当某个参数不停地往里面传时，应该关注这个参数第一次传进去的地方和实际使用的地方
# 对于ResNet的_make_layer还是不能很好地理解其写法

# def feature_extract----def get_dataloader----class CaptionDataset


"""
利用resnet50提取图片的语义信息
tensor 200k*2048 200k张图片，每张图片2048维
并保存成results.pth

"""


# In[ ]:

import sys
sys.path.append('..')
from models import FeatureExtractModel

# In[106]:


import torch as t
from torch.utils import data
from torchvision import transforms as T
import os
from PIL import Image
import torchvision as tv
import tqdm


# In[6]:
__all__ = ['CaptionDataset', 'get_dataloader', 'feature_extract']

class Config(object):
    
# feature_extract.py
    # def feature_extract
    # step: configuration
    use_gpu = True
    # step: data 
    caption_data_path = '../dataset/caption.pth'  # 经过预处理后的人工描述信息  class CaptionDataset 
    batch_size = 8
    num_workers = 4 

    # class CaptionDataset
    img_path = '/home/zbp/Linux/tjj/data/ai_challenger_caption_train_20170902/caption_train_images_20170902'
    # img_path='/home/zbp/Linux/tjj/data/ai_challenger_caption_train_20170902/caption_train_images_20170902/XXX.jpg'

    save_path = '../dataset/results.pth'
    
    def parse(self,**kwargs):
        
        for k,v in kwargs.items():
            setattr(self,k,v)
            
        print('User Config')
        for k in dir(self):
            if not k.startswith('__'):
                print('{0}====={1}'.format(k,getattr(self,k)))
        
opt = Config()


# In[3]:


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# In[130]:


class CaptionDataset(data.Dataset):
    
    def __init__(self, caption_data_path):
        
        
        self.transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
        
        data = t.load(caption_data_path)
        ix2id = data['ix2id']
        # 所有图片的路径
        self.imgs = [os.path.join(opt.img_path, ix2id[i]) for i in range(len(ix2id))]
        
    
    def __getitem__(self,index):
        
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        
        return img, index
    
        
    def __len__(self):
        return len(self.imgs)
    


# In[131]:


def get_dataloader(opt):
    dataset = CaptionDataset(opt.caption_data_path)
    dataloader = data.DataLoader(dataset, 
                                 batch_size=opt.batch_size,
                                 num_workers=opt.num_workers,
                                 shuffle=False)
    
    return dataloader


# In[ ]:

@t.no_grad()
def feature_extract(**kwargs):
    #step: configure
    opt.parse(**kwargs)
    opt.batch_size = 256
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    t.set_grad_enabled(False)
    

    # step: data
    dataloader = get_dataloader(opt)
    
    # step: model
    resnet50 = FeatureExtractModel()
    resnet50 = resnet50.to(device)
    
    # step: 前向传播
    
    results = t.Tensor(len(dataloader.dataset), 2048).fill_(0)

    for ii, (imgs, indexs) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        # 确保序号没有错 不知道这个意义在哪里，难道还有错的不成,
        # 因为在保存200K*2048维的特征的时候，没有保留图片名字，所以需要保证其顺序和ix2id中的顺序一致
        assert indexs[0] == opt.batch_size*ii
        imgs = imgs.to(device)
        features = resnet50(imgs) # tensor batch*2048 
        results[ii*opt.batch_size:(ii+1)*opt.batch_size] = features.data.cpu()
    
    # 200k*2048 200k张图片，每张图片2048维
    t.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)


if __name__== "__main__":
    import fire
    fire.Fire()
