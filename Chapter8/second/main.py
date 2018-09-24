
# coding: utf-8

# 从第七章和第八章的作者的写作风格来看，作者开始把Config开始放在了main.py之内，而不再是重新放在一个.py文件内，我开始思考这样放在一起和分开放的区别，两者在getsorce（）方面没有任何区别，单纯地从代码上看，写在一起未尝不好，这样在调试的时候，直接打开train就可以看到配置文件，但是在属性输入方面就要差一些，没有opt.parse()来得爽快一些，而且作者开始停止了help函数的设置，我决定继续使用分开的方式，因为我感觉很条理，在后期修改其他配置的时候，是不需要打开train.py的，不需要担心会污染训练的代码。

# In[1]:


# import Ipynb_importer
from config import opt 
from utils import Visualizer
from models import TransformerNet, Vgg16
from utils import get_style_data
from utils import normalize_batch
from utils import gram_matrix


# 主文件：
# train(**kwargs):  
# val(model,dataloader):  
# test(**kwargs):   
# help():  
#  

# 这里新介绍一种作者用的怎么使用cuda的方法  
# device=t.device('cuda') if opt.use_gpu else t.device('cpu')  
# vgg.to(device)  
# 代替了之前一直使用的很不方便的
# if opt.use_gpu:
#     model.cuda()
#     
# 但是一直还没有使用 t.cuda.set_device(1)这个方法，
# 刚刚查了下好像可以,torch.device('cuda')采用的是逻辑GPU，
# torch.cuda.current_device()用来查询当前使用的默认GPU
# 我感觉这么写可以，无法验证
# torch.cuda.set_device(1)
# device = torch.device('cuda')
# models.to(device)
# 但是这一种方法只是用单GPU方法，查看document，我们可以发现set_device并不好用，不如CUDA_VISIBLE_DEVICES好用，这个适用于多GPU，待补坑。
# 
# 还要一种方法是 
# with torch.cuda.device(1):   
# 请注意一下torch.device和torch.cuda.device的区别
# 

# In[2]:


from torch import nn
import torch as t
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader 
from torchnet import meter
from tqdm import tqdm
from torch.nn import functional as F
import os
# 只在jupyter中见过tqdm的显示


# In[3]:


def train(**kwargs):
    # step1:config
    opt.parse(**kwargs)
    vis = Visualizer(opt.env)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    
    # step2:data
    # dataloader, style_img
    # 这次图片的处理和之前不一样，之前都是normalize,这次改成了lambda表达式乘以255，这种转化之后要给出一个合理的解释
    # 图片共分为两种，一种是原图，一种是风格图片，在作者的代码里，原图用于训练，需要很多，风格图片需要一张，用于损失函数
    
    transforms = T.Compose([
        T.Resize(opt.image_size),
        T.CenterCrop(opt.image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x*255)    
    ])
    # 这次获取图片的方式和第七章一样，仍然是ImageFolder的方式，而不是dataset的方式
    dataset = tv.datasets.ImageFolder(opt.data_root,transform=transforms)
    dataloader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,drop_last=True)
    
    style_img = get_style_data(opt.style_path) # 1*c*H*W
    style_img = style_img.to(device)
    vis.img('style_image',(style_img.data[0]*0.225+0.45).clamp(min=0,max=1)) # 个人觉得这个没必要，下次可以实验一下
    
    # step3: model：Transformer_net 和 损失网络vgg16
    # 整个模型分为两部分，一部分是转化模型TransformerNet，用于转化原始图片，一部分是损失模型Vgg16，用于评价损失函数，
    # 在这里需要注意一下，Vgg16只是用于评价损失函数的，所以它的参数不参与反向传播，只有Transformer的参数参与反向传播，
    # 也就意味着，我们只训练TransformerNet，只保存TransformerNet的参数，Vgg16的参数是在网络设计时就已经加载进去的。
    # Vgg16是以验证model.eval()的方式在运行，表示其中涉及到pooling等层会发生改变
    # 那模型什么时候开始model.eval()呢，之前是是val和test中就会这样设置，那么Vgg16的设置理由是什么？
    # 这里加载模型的时候，作者使用了简单的map_location的记录方法，更轻巧一些
    # 发现作者在写这些的时候越来越趋向方便的方式
    # 在cuda的使用上，模型的cuda是直接使用的，而数据的cuda是在正式训练的时候才使用的，注意一下两者的区别
    # 在第七章作者是通过两种方式实现网络分离的，一种是对于前面网络netg,进行 fake_img = netg(noises).detach(),使得非叶子节点变成一个类似不需要邱求导的叶子节点
    # 第四章还需要重新看，
    
    transformer_net = TransformerNet()
    
    if opt.model_path:
        transformer_net.load_state_dict(t.load(opt.model_path,map_location= lambda _s, _: _s))    
    transformer_net.to(device)
    

    
    # step3： criterion and optimizer
    optimizer = t.optim.Adam(transformer_net.parameters(),opt.lr)
    # 此通过vgg16实现的，损失函数包含两个Gram矩阵和均方误差，所以，此外，我们还需要求Gram矩阵和均方误差
    vgg16 = Vgg16().eval() # 待验证
    vgg16.to(device)
    # vgg的参数不需要倒数，但仍然需要反向传播
    # 回头重新考虑一下detach和requires_grad的区别
    for param in vgg16.parameters():
        param.requires_grad = False
    criterion = t.nn.MSELoss(reduce=True, size_average=True)
    
    
    # step4: meter 损失统计
    style_meter = meter.AverageValueMeter()
    content_meter = meter.AverageValueMeter()
    total_meter = meter.AverageValueMeter()
    
    # step5.2：loss 补充
    # 求style_image的gram矩阵
    # gram_style:list [relu1_2,relu2_2,relu3_3,relu4_3] 每一个是b*c*c大小的tensor
    with t.no_grad():
        features = vgg16(style_img)
        gram_style = [gram_matrix(feature) for feature in features]
    # 损失网络 Vgg16
    # step5： train
    for epoch in range(opt.epoches):
        style_meter.reset()
        content_meter.reset()
        
        # step5.1: train
        for ii,(data,_) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            # 这里作者没有进行 Variable(),与之前不同
            # pytorch 0.4.之后tensor和Variable不再严格区分，创建的tensor就是variable
            # https://mp.weixin.qq.com/s?__biz=MzI0ODcxODk5OA==&mid=2247494701&idx=2&sn=ea8411d66038f172a2f553770adccbec&chksm=e99edfd4dee956c23c47c7bb97a31ee816eb3a0404466c1a57c12948d807c975053e38b18097&scene=21#wechat_redirect
            data = data.to(device)
            y = transformer_net(data)
            # vgg对输入的图片需要进行归一化
            data = normalize_batch(data)
            y = normalize_batch(y)

           
            feature_data = vgg16(data)
            feature_y = vgg16(y) 
            # 疑问？？现在的feature是一个什么样子的向量？
            
            # step5.2: loss:content loss and style loss
            # content_loss
            # 在这里和书上的讲的不一样，书上是relu3_3,代码用的是relu2_2
            # https://blog.csdn.net/zhangxb35/article/details/72464152?utm_source=itdadao&utm_medium=referral
            # 均方误差指的是一个像素点的损失，可以理解N*b*h*w个元素加起来，然后除以N*b*h*w
            # 随机梯度下降法本身就是对batch内loss求平均后反向传播
            content_loss = opt.content_weight*criterion(feature_y.relu2_2,feature_data.relu2_2)
            # style loss
            # style loss:relu1_2,relu2_2,relu3_3,relu3_4 
            # 此时需要求每一张图片的gram矩阵
            
            style_loss = 0
            # tensor也可以 for i in tensor:,此时只拆解外面一层的tensor
            # ft_y:b*c*h*w, gm_s:1*c*h*w
            for ft_y, gm_s in zip(feature_y, gram_style):
                gram_y = gram_matrix(ft_y)
                style_loss += criterion(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight
            
            total_loss = content_loss + style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            #import ipdb
            #ipdb.set_trace()
            # 获取tensor的值 tensor.item()   tensor.tolist()
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())
            total_meter.add(total_loss.item())
            
            # step5.3: visualize
            if (ii+1)%opt.print_freq == 0 and opt.vis:
                # 为什么总是以这种形式进行debug
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()
                vis.plot('content_loss',content_meter.value()[0])
                vis.plot('style_loss',style_meter.value()[0])
                vis.plot('total_loss',total_meter.value()[0])
                # 因为现在data和y都已经经过了normalize，变成了-2~2，所以需要把它变回去0-1
                vis.img('input',(data.data*0.225+0.45)[0].clamp(min=0,max=1))
                vis.img('output',(y.data*0.225+0.45)[0].clamp(min=0,max=1))
            
        # step 5.4 save and validate and visualize
        if (epoch+1) % opt.save_every == 0:
            t.save(transformer_net.state_dict(), 'checkpoints/%s_style.pth' % epoch)
            # 保存图片的几种方法，第七章的是 
            # tv.utils.save_image(fix_fake_imgs,'%s/%s.png' % (opt.img_save_path, epoch),normalize=True, range=(-1,1))
            # vis.save竟然没找到  我的神   
            vis.save([opt.env])


# In[4]:


@t.no_grad()
def stylize(**kwargs):
    opt.parse(**kwargs)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    # data
    # 对单张图片进行加载验证
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    # 这个transform更看不懂了，竟然不需要resize和centrocrop，这图片大小不一致，能送进去么？？？？
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()
    
    # model
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style.to(device)
    
    # stylize
    output = style_model(content_image) # 0-255~
    output_data = (output.cpu().data[0]/255).clamp(min=0, max=1)
    tv.utils.save_image(output_data,opt.result_path)


# In[12]:


def help():
    print("""
    usage: print file.py <function> [--args=value]
    <function>: train | stylize
    example:
        python {0} train
        python {0} stylize
        python {0} help
    avaiable args :
    """.format(__file__))
    
    from inspect import getsource
    source = getsource(opt.__class__)
    print(source)
    


# In[8]:


if __name__ == '__main__':
    import fire
    fire.Fire()
    

