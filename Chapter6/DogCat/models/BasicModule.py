#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-8-30 上午10:55
# Author : TJJ
import torch as t
import time

class BasicModule(t.nn.Module):
    """"
    对nn.Module的简单封装， 主要是为了save和load，因为module的保存是state_dict的形式

    __init__(self):
    save(self,name=None):
    load(self,path):
    """
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) # 默认名字

    def save(self,name=None):
        # 保持模型，默认使用模型名字+时间作为文件名
        if name is None:
            prefix = 'checkpoints/'+self.model_name+'_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth',time.localtime())

        t.save(self.state_dict(),name)
        return name



    def load(self,path):
        """

        :param path:
        :return:加载模型
        """
        self.load_state_dict(t.load(path))
