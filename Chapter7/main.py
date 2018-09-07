
# coding: utf-8

# ### 需要实现四个函数  train、val、test、help

# In[1]:


# import Ipynb_importer
from config import opt
from utils.visualize import Visualizer
from torchvision import transforms as T
import torchvision as tv
import torch as t
from torch.utils.data import DataLoader
from models import NetG,NetD
from torch.autograd import Variable
from torchnet import meter
import os
import numpy
import ipdb
from tqdm import tqdm


# In[12]:


def train(**kwargs):
    # step1: configure
    opt.parse(**kwargs)
    if opt.vis:
        vis = Visualizer(opt.env)
    # step2: data
    normalize = T.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5] )
    transforms = T.Compose(
    [
        T.Resize(opt.image_size),
        T.CenterCrop(opt.image_size),
        T.ToTensor(),
        normalize
    ])
    # 对于这个模型 transform对于train和test没有区别
    dataset = tv.datasets.ImageFolder(opt.data_path,transform=transforms)
    dataloader = DataLoader(dataset,
                            batch_size = opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            drop_last=True)                      # 加载图片，用于训练NetD模型
    
    true_labels = Variable(t.ones(opt.batch_size))
    fake_labels = Variable(t.zeros(opt.batch_size))
    fix_noises = Variable(t.randn(opt.batch_size, opt.nz, 1, 1))  # 固定噪声，用于验证NetG模型
    noises = Variable(t.randn(opt.batch_size, opt.nz, 1, 1))      # 随机噪声，用于训练和测试NetG模型
    
    # step3: model
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc:storage
    if opt.netg_path:
        netg.load(opt.netg_path)
    if opt.netd_path:
    	netd.load(opt.netd_path)
        
    
    # step4: criterion and optimizer
    optimizer_g = t.optim.Adam(params=netg.parameters(), lr = opt.lrg, betas=(opt.beta1,0.999))
    optimizer_d = t.optim.Adam(params=netd.parameters(), lr = opt.lrd, betas=(opt.beta1,0.999))
    criterion = t.nn.BCELoss()
    
    # step: meters
    errord_meter = meter.AverageValueMeter()
    errorg_meter = meter.AverageValueMeter()
    
    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()
    
    # step5: train
    for epoch in range(opt.max_epoch):
        ## step5.1 train
        for ii,(data, _) in tqdm(enumerate(dataloader)):
            real_img = Variable(data)
            if opt.use_gpu:
                real_img = real_img.cuda()
            if (ii+1) % opt.d_every ==0:
                # 判别器
                optimizer_d.zero_grad()
                # 真图片
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()
                # 假图片
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img =  netg(noises).detach()
                fake_output = netd(fake_img)
                error_d_fake = criterion(fake_output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()
                
                error_d = error_d_real+error_d_fake
                errord_meter.add(error_d.data)
                
            if (ii+1) % opt.g_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                
                errorg_meter.add(error_g.data)
                
            ## step5.2 validate and visualize on batch_size  
            # 我们可以看到，损失函数并不是一个epoch画一次，而是几个batch画一次
            if (ii+1) % opt.print_freq == 0 and opt.vis:
                if os.path.exists(opt.debug_file):
                    # import ipdb
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises) # batch_size*nz*1*1 --> batch_size(256)*3*96*96 # 可以认为是在验证模型
                vis.img('fix_fake',fix_fake_imgs.data[:64]*0.5+0.5)
                vis.img('real', real_img.data[:64]*0.5+0.5)
                vis.plot(win = 'errord',y= errord_meter.value()[0])
                vis.plot(win = 'errorg',y= errorg_meter.value()[0])
                
            
        ## step5.3 validate and save model on epoch 
        # 模型保存是每几个epoch保存一次，
        # 按理来说模型验证也应该是每次或这每几次验证一次，这一点和这一章的模型验证有所不一样，不过不用太在意，因为这一章的模型验证没有指标。
        if (epoch+1)%opt.save_freq == 0:
            netg.save(opt.model_save_path,'netg_%s' %epoch)
            netd.save(opt.model_save_path,'netd_%s' %epoch)
            fix_fake_imgs = val(netg,fix_noises)
            tv.utils.save_image(fix_fake_imgs,'%s/%s.png' % (opt.img_save_path, epoch),normalize=True, range=(-1,1))
            # 和作者沟通后，因为数据集少，所以为了避免每次重置的噪声，多几个epoch再重置，等下试试每次重置的话这个误差的变化情况
            errord_meter.reset()
            errorg_meter.reset()
            """
            和作者沟通后，作者本意是为了做梯度衰减，这样就带来第二个问题，学习率下降是否可以通过重构优化器就达到，我觉得不行，
            因为在Adam甚至是其他的简单的优化器，里面都有另外一个隐含变量v或者m，详情可以查询优化器的原理，所以重构优化器之后，
            这写隐含变量就会因此而重新初始化
            """
            # 和作者沟通后，作者本意是为了做梯度衰减，这样就带来第二个问题，学习率下降是否可以通过重构优化器就达到，我觉得不行，因为在Adam甚至是其他的简单的优化器，里面都有另外一个隐含变量v或者m，详情可以查询优化器的原理，所以重构优化器之后，这写隐含变量就会因此而重新初始化
            # optimizer_g = t.optim.Adam(params=netg.parameters(), lr = opt.lrg, betas=(opt.beta1,0.999))
            # optimizer_d = t.optim.Adam(params=netd.parameters(), lr = opt.lrd, betas=(opt.beta1,0.999))

            
        
        
        
        ## 5.4 update learning rate
        
            
            
    


# In[3]:

# step5.3 
def val(netg, fix_noises):
	netg.eval()
	fix_fake_imgs = netg(fix_noises)
	return fix_fake_imgs.data[:64]
	# tv.utils.save_image和tv.utils.make_grid基本是一个东西，输入是tensor,[batch, fear,H,W]
	# 此处的normalize是为了使黑白对比不那么明显
	

    # 验证在固定噪声上生成的图片的正确率。
    # 为了能够实现对模型进行验证，我们开始修改代码
    


# In[4]:


def test(**kwargs):
    # 在这里，更名为generate
    pass


# In[5]:


def help():
    print("""
    	uasge: python file.py <function> [--args=value]
    	<function> := train | generate |help
    	example:
    			python {0} train 
    			python {0} generate
    			python {0} help
    	aviable args:""".format(__file__)
    	)
# __file__ 表示某个文件的路径
    from inspect import getsource
    source = getsource(opt.__class__)
    print(source)


# In[8]:


def generate(**kwargs):
    opt.parse(**kwargs)
    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    
    if opt.netd_path:
        netg.load(opt.netg_path)
    if opt.netg_path:
        netd.load(opt.netd_path)
    
    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        noises = noises.cuda()
    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).data
    # 选好的
    index = scores.topk(opt.gen_num)[1]
    result = []
    for ii in index:
        # tensor的截取与合并  cat, stack,cat+view=stack,stack 新增维度进行合并
        result.append(fake_img.data[ii])
    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1,1))
    


# In[10]:


if __name__ == '__main__':

    import fire
    fire.Fire()
# __name__ 模块名称，即import后面跟的那个
"""
示例：
import numpy
numpy.__name__  == numpy
numpy.random.__name__ == numpy.random
from numpy import random
random.__name__ == random
""" 

