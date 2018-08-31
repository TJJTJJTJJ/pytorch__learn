#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-8-31 下午1:53
# Author : TJJ
import ipdb
def sum(x):
    r=0
    for ii in x:
        r+=ii
    return r

def mul(x):
    r=1
    for ii in x:
        r*=ii
    return r

ipdb.set_trace()
x = [1,2,3,4,5]
r = sum(x)
rr = mul(x)
