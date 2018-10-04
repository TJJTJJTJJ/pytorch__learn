
# coding: utf-8

# 这里有一些作者在代码中声明的功能没有太看懂，等写完回来想想
# 在一定程度上，作者开始把dataloader写在dataset里，写在这里的话只会影响一些Config的操作，但opt不会影响最终结果才对
# 这里的装饰器还有待研究


# def get_dataloader()----class CaptionDataset()
#                   |----def create_collate_fn()


# In[7]:


import torch.utils.data

import torch as t
from torch.utils import data
import numpy as np


# In[9]:

__all__ = ['CaptionDataset', 'create_collate_fn', 'get_dataloader']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CaptionDataset(data.Dataset):
    # 暂时还没有看到作者为什么作者还要返回index
    # 是为了知道是第几张图片
    def __init__(self, opt):
        
        self.opt = opt
        data = t.load(opt.caption_data_path)
        self.data = data
        self.captions = data['caption']
        self.word2ix = data['word2ix']
        self.padding = self.word2ix[data.get('padding')]
        self.end = self.word2ix[data.get('end')]
        self._data = data
        self.ix2word = data['ix2word']
        self.all_imgs = t.load(opt.img_feature_path) # 200k*2048
        
    def __getitem__(self, index):
        """
        @return: img(tensor):2048
                 caption(list): 当前图片的描述的id
                 index: int
        """

        img = self.all_imgs[index]
        caption = self.captions[index]
        # 5句话随便选一句
        rdn_index = np.random.choice(len(caption), 1)
        caption = caption[rdn_index[0]]
        caption = t.LongTensor(caption)
        return img, caption, index
        
    def __len__(self):
        return len(self.captions)


# In[1]:

#  ????????????????????????????????????????????????????/
def create_collate_fn(padding, eos, max_length=50):
    # 这个是为了Dataloader服务的，用于如何将多个样本拼接成一个batch_size
    # 主要涉及到到caption的长短不一致的问题
    
    def collate_fn(img_cap):
        # 还是希望尽可能在句子末尾加上</EOS>，但是如果恰好相等，那么就不加，所以在求句子的batch_length的时候，句子长度加1就是为了这个。万一句子都很短，就可以有空闲的地方加上末尾
        # cap_tensor: 词+padding+eos
        # 
        """
        输入 list of data,
        [(img1, cap1, index1), (img2, cap2, index2)]
        img(tensor) 1*2048
        cap(list) 第某张图片的第某个描述
        index(int)
        @return： imgs(tensor)： b*2048
                 cap_tensor(tensor): seq_len*b
                 lengths(list): b int
                 indexs(tuple): b int
        """

        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps, indexs =zip(*img_cap) # imgs: b*2048
        imgs = t.stack(imgs,dim=0) # imgs: b*2048
        lengths = [min(len(c)+1, max_length) for c in caps ]
        batch_length = max(lengths)
        cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding) # tensor seq_len*batch_size
        for i, c in enumerate(caps):
            end_cap = lengths[i]-1
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
            if end_cap<batch_length:
                cap_tensor[end_cap, i] = eos
        # 输入值是list:[d1, d2, d3,...],d是tuple，是都是Dataset以元组的形式的返回值，d=(img,label)
        # 即 [(img1, label1), (img2, label2), ...] img1 维度(*)
        # 输出值应该是tuple或者其他形式，因为要进行解压，tuple应该是比较好的，用于tuple中的每一个值的形式应该是这样的。
        # （imgs， labels） imgs: （b,*）的维度， tensor, labels: (b,*)的维度 tensor
        # 一般用 imgs, labels = zip(*list), imgs = t.stack(imgs,dim=0)
        
        # 
        return (imgs, (cap_tensor, lengths), indexs)
            
    return collate_fn
        
        
        


# In[80]:


def get_dataloader(opt):
    """
    (imgs, (cap_tensor, lengths), indexs)
    
    @return： imgs(tensor)： b*1*2048
             cap_tensor(tensor): seq_len*b
             lengths(list): b int
             indexs(tuple): b int
             
    """

    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
                                batch_size=opt.batch_size, 
                                num_workers=opt.num_workers, 
                                shuffle=opt.shuffle,
                                collate_fn=create_collate_fn(dataset.padding, dataset.end)
                                )
    # 
    return dataloader


# In[ ]:


# 这个只是为了测试用，是对该文件涉及到的函数的一个正确性的验证
if __name__ == '__main__':
    # from config import opt
    dataloader = get_dataloader(opt)
    
    for ii, data in enumerate(dataloader):
        print(ii, data)
        break

