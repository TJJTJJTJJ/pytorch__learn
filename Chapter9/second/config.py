
# coding: utf-8

# In[1]:


import warnings


# In[2]:


class Config(object):
    # data.py
    # get_data():
    pickle_path = 'tangtest.npz' # 'tang.npz' # 预处理好的二进制文件
    # ----_parseRawData()
    data_path = '/home/tjj/pytorchtest/chapter9/chinese-poetry/chinese-poetry/json/simplified/' # 诗歌的存放路径 /poetry/poet.song.0.json  
    category = 'poet.tang'  # 诗歌类别，分为： 'poet.song'，'poet.tang'，None：None时为全学
    constrains =  None # 在取诗歌时，只取长度为constrains的单句，这里的单句指的是按照，。！进行分割之后的单句。为None时为全可以。
    constrain = None
    author = None # 作者名字，为None时学习所有作者的诗歌，具体的作者名字可以在 authors.song.json 和 authors.tang.json 中查到，只能是单个作者
    # ----pad_sequences()
    maxlen = 125 # 使每个诗歌都截断或者补充成长度为125的诗歌
    
    # main.py
    # train()
    use_gpu = True
    env = 'default' # visdom.env  None or 'test'
    # step:data
    batch_size = 128
    num_workers = 1
    # step: model
    model_path = None
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    lr = 1e-3
    # step: train
    epoch = 20
    model_prefix = 'checkpoints/tang'  # 模型保存路径
    # step： visualize and validate
    print_freq = 20 # 每20次batch_size就可视化一次
    debug_file = './tmp/debug' # 
    
    # generate()  gen_acrostic()
    max_gen_len = 200  # 生成诗歌最长长度
    
    # gen()
    acrostic = False
    def _parse(self, **kwargs):
        # 记得测试一下fire，每次光是会用，但是具体是个啥情况还是很懵
        # items():字典的items方法以列表的形式返回可遍历的（k,v）元组，或者说对于字典的遍历的三种情况，分别是dict.keys(),dict.values(),dict.items()s
        # hasattr,getattr,setattr是针对类的属性和方法
        # 回去在了解一下git@怎么不能用了
        """
        @param: self
        @param: kwargs
        """
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("don't have the attribute {k}".format(k=k))
            else:
                setattr(self, k, v)
        # 发现了一个更好的打印实例属性的方法，还不用担心会不会被覆盖之类的。
        print("User Config")
        for k in dir(self):
            if not k.startswith('_'):
                print(k, getattr(self,k))
    
    


# In[3]:


opt = Config()

