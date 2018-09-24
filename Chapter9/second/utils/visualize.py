#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-8-30 下午3:18
# Author : TJJ

from visdom import Visdom
import time
import numpy as np

class Visualizer(object):
    """
    封装visdom的基本操作
    """

    def __init__(self, env='default', **kwargs):
        self.vis = Visdom(env=env,use_incoming_socket=False, **kwargs)
        # 以文字和图的形式展示
        # {'loss1':23,'loss2':24}
        self.index = {}
        self.log_text = ''
    def reinit(self, env='default', **kwargs):
        self.vis = Visdom(env=env, **kwargs)

    def img(self, name ,img_,  **kwargs):
        # img_: batchsize*channels*H*W
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, win='log_text', info='' ):
        """
        self.log({'loss':1,'lr':0.0001})
        :param info:
        :param win:
        :return:
        """
        self.log_text +=(
            '[{time}]{info}<br>'.format(
                time = time.strftime('%m%d_%H:%M:%S'),
                info = info
                                        )
        )
        self.vis.text(self.log_text,win)

    def plot(self, win, y,  **kwargs):
        """
        plot('loss',2)
        :param win:
        :param y:
        :param kwargs:
        :return:
        """

        x = self.index.get(win,0)
        self.vis.line(
            X=np.array([x]), Y=np.array([y]),
            win = win,
            opts= dict(title=win),
            update = None if x ==0 else 'append',
            **kwargs
        )
        self.index[win] = x+1


    def img_many(self,d):
        # d: {'1.jpg':b*c*H*W,'2.jpg':b*c*H*W}
        for k,v in d.items():
            self.img(k,v)

    def plot_many(self,d):
        # d:{'loss1':2,'loass2':4}
        for k,v in d.items():
            self.plot(k,v)

    def __getattr__(self,name):
        # self,function->self.vis.funtion
        return getattr(self.vis, name)