#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-8-30 下午4:19
# Author : TJJ
import  warnings

class DefaultConfig(object):
    env = 'default'       # visdom

    use_gpu = True
    # model
    num_class = 2
    model = 'ResNet34' # ResNet34

    load_model_path = None # main.py/ train test

    # data
    train_data_path = '/home/tjj/pytorchtest/chapter6/first/data/train/'  # train val
    test_data_path = '/home/tjj/pytorchtest/chapter6/first/data/test/'
    # data_loader
    train_transform = None
    test_val_transform = None
    batch_size = 2
    shuffle = True # train 打乱
    num_workers = 4

    #train/# step3: criterion and optimizer
    lr = 0.001
    lr_decay = 0.5
    weight_decay = 0e-5

    # step5: train
    max_epoch = 10
    print_freq = 1 # print info every N batch

    # test:
    result_file = 'result.csv'


def parse(self, **kwargs):
    """
    根据字典更新参数
    :param self:
    :param kwargs:
    :return:
    """
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribute {k}".format(k=k))
        setattr(self, k, v)

    print('user config:')
    for k,v in self.__class__.__dict__.items():
        if (not k.startswith('__')) and (not k.startswith('parse')):
            print('{k}:  {v}'.format(k=k,v=v))

DefaultConfig.parse = parse
opt = DefaultConfig()