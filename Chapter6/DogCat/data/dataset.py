
# -*- coding:utf-8 -*-
# @Time  : 18-8-29 下午10:21
# Author : TJJ
from torch.utils  import data
import os
from torchvision import transforms as T
from PIL import Image

class DogCat(data.Dataset):
    # The main purpose of the class is to process data
    """
    __init__(self,root,transforms,train,test):self.imgs self.transform
    __getitem__(sefl,index)
    __len__(self)
    """

    def __init__(self, root, transform=None, train=True, test=False):

        # test:  data/test/8973.jpg
        # train: data/train/cat.1004.jpg
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        self.test = test
        self.train = train
        if self.test: # test
            imgs = sorted(imgs, key= lambda x: x.split('.')[-2].split('/')[-1])
            self.imgs= imgs
        else:
            imgs = sorted(imgs, key = lambda x: x.split('.')[-2])
            num  = len(imgs)
            if self.train:
                self.imgs = imgs[:int(0.7*num)] # train
            else:
                self.imgs = imgs[int(0.7*num):] # val

        if transform is None:
            normalize = T.Normalize(mean=[0.483,0.456,0.406], std=[0.229,0.224,0.225])
            if test or not train:  # test val
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:  # train
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        img_path = self.imgs[item]
        if self.test: # test
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:  # train val
            label = 1 if 'dog' in img_path else 0
        self.label = label

        data = Image.open(img_path)
        data = self.transform(data)
        return data,label

    def __len__(self):
        return len(self.imgs)