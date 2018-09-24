
# coding: utf-8

# In[1]:


class Config(object):
#---model.py
    # model/netG
    ngf = 64 # 生成器倒数第二层的feature map
    nz = 100 # 生成器的输入图片的特征通道数
    #model/netD
    ndf = 64  # 判别器第二层的feature map
    

#----main.py
    # train()
    # step1: configure
    # visdom
    vis = True
    env = 'default'
    #gpu
    use_gpu = True
    
    # step2: data
    # transform
    image_size = 96 # 生成的图片大小
    # dataset & dataloader
    data_path = 'data/' #放图片的文件夹
    batch_size = 256
    num_workers = 4
    # label
    

    # step3: model
    # 训练的加载模型
    netg_path = None # './checkpoints/netg_.pth'
    netd_path = None # './checkpoints/netd_211.pth'
    
    # step4: criterion and optimizer
    lrg = 2e-4
    lrd = 2e-4
    beta1 = 0.5 
    
    # step5: train
    max_epoch = 200
    d_every = 1 # 每n个batch_size训练一次判别器
    g_every = 5 # 每m个batch_size训练一次生成器
    ## step5.2: validate and visualize
    print_freq = 20 # 每20个batch画一次图 
    ## step5.3 model save
    save_freq = 10 # 每10个epoch保存一次模型
    img_save_path = 'imgs/' # 生成图片的保存路径
    model_save_path = './checkpoints'
    debug_file = './debug'

    # generate()
    gen_mean = 0
    gen_std  = 1
    gen_num = 64 # 从(batch_size)256张图片中选出64张好的图片
    
    
    
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

Config.parse = parse
opt = Config()

