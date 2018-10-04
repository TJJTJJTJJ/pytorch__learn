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

    def __init__(self, env , **kwargs):
        self.vis = Visdom(env=env,use_incoming_socket=False, **kwargs)
        # 以文字和图的形式展示
        # {'loss1':23,'loss2':24}
        self.plot_index = {}
        self.log_index = {}

    def reinit(self, env='default', **kwargs):
        self.vis = Visdom(env=env, **kwargs)

    def img(self, name,  img_, **kwargs):
        # img_(tensor or numpy): batchsize*channels*H*W或者C*H*W 0-1
        # 这里代码已经发生了变化，
        # caption 是在图片的左下角，神奇
        opts = dict(title=name).update(kwargs)
        self.vis.images(img_.cpu(),
                        win=name,
                        opts=opts,
                        )

    def log(self, win='log_text', info='', update=True):
        """
        self.log({'loss':1,'lr':0.0001})
        @param info:
        @param win:
        @return:
        """
        log_text = self.log_index.get(win,'')
        opts = dict(title=win)
        if update:
            log_text += (
                '[{time}]{info}<br>'.format(
                    time=time.strftime('%m%d_%H:%M:%S'),
                    info=info))
        else:
            log_text = (
                '[{time}]{info}<br>'.format(
                    time=time.strftime('%m%d_%H:%M:%S'),
                    info=info))

        self.vis.text(log_text, win=win, opts=opts)
        self.log_index[win] = log_text

    def plot(self, win, y,  **kwargs):
        """
        plot('loss',2)
        :param win:
        :param y:
        :param kwargs:
        :return:
        """

        x = self.plot_index.get(win,0)
        self.vis.line(
            X=np.array([x]), Y=np.array([y]),
            win = win,
            opts= dict(title=win),
            update = None if x ==0 else 'append',
            **kwargs
        )
        self.plot_index[win] = x+1


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
