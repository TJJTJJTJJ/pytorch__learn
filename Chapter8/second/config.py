
# coding: utf-8

# In[1]:


import warnings


# In[2]:


class DefaultConfig(object):
#----main.py
    # train()
    # step1:config
    vis = True
    env = 'test1' # visdom 环境
    use_gpu = True  # gpu
    
    # step2:data
    data_root = './data/' # 数据集存放路径：'data/coco/a.jpg'
    image_size = 256
    batch_size = 8
    num_workers = 4      # 多线程加载数据
    style_path = 'style.jpg' # 风格图片存放路径
    # step3:model
    
    # step4:criterion and optimizer
    lr = 1e-3
    
    # step5:train
    epoches = 2  
    # step5.2 loss
    content_weight = 1e5 # content_loss的权重
    style_weight = 1e10 # style_loss的权重
    
    # step5.3: visualize
    print_freq = 10 # 每10个batch可视化一次
    debug_file = 'tmp/debug' # jinru debug moshi

    # step5.4: save
    save_every = 1 # 每个epoch保存一次
    
    # stylize()
    content_path = './content_img/input/input.png'  # 需要进行迁移的图片
    model_path = None  # 预训练模型的路径
    result_path = './content_img/output/output.png'  # 图片迁移后的保存


# 自己第一次写parse，忽略了self
# hasattr用于判断一个对象是否有某个属性，一般常见于类的实例是否有个方法或者值，刚刚用字典测试，发现不适用字典。这一点要注意
# hasattr和setattr分别是指XX.XX这种形式的
# 这个warning模块有点绕，找了几个博客都没有看清楚怎么用，之后要看看基本的用法,或者直接用dir()产生的
# 刚刚查了查怎么查询类或者实例的属性值，有一个还行https://www.cnblogs.com/scolia/p/5582268.html,实例的属性和类的属性的不是一个东西，而是通过一个类似指针的东西从实例指向类
def parse(self,**kwargs):
    for k,v in kwargs:
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribute {k}".format(k=k))
        setattr(self,k,v)
    print("user config")
    
    # print的方式就表示我太low
    # 先打印实例属性，再打印类属性
    keys = []
    for k,v in self.__dict__.items():
        if (not k.startswith('__')) and (k not in keys):
            keys.append(k)
            print('{k}：{v}'.format(k=k,v=v))
    for k,v in self.__class__.__dict__.items():
        if (not k.startswith('__')) and (k not in keys):
            keys.append(k)
            print('{k}：{v}'.format(k=k,v=v))
    
    
    


# In[3]:


DefaultConfig.parse = parse
opt = DefaultConfig()

