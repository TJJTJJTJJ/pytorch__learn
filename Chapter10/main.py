
# coding: utf-8

# git+https://github.com/pytorch/tnt.git@master
#     什么意思

# In[9]:
from config import opt
from utils import Visualizer
from data import get_dataloader
from models import CaptionModel, FeatureExtractModel


# In[25]:


import torch as t
from torchnet import meter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from PIL import Image
import torchvision as tv
import fire
import tqdm


# In[ ]:


import torchvision.transforms as T
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

__all__ = ['train', 'generate']

def train(**kwargs):
    
    # step: configure

    opt._parse(**kwargs)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = Visualizer(env=opt.env)
    
    # step: data  这里是指要从数据加载过程中加载所需要的数据，所有数据只加载一次，也可以通过self的方式获取，

    dataloader = get_dataloader(opt)
    _data = dataloader.dataset.data
    # word2ix, ix2word = _data.word2ix, _data.ix2word    
    # ix2id, id2ix = _data.ix2id, _data.id2ix
    word2ix, ix2word = _data['word2ix'], _data['ix2word']
    ix2id, id2ix, end = _data['ix2id'], _data['id2ix'], _data['end']
    eos_id = word2ix[end]
    
    # step: model 
    # 刚刚看了看作者写的模型，在保存模型的时候把opt也一并保存了，这是要做什么的，貌似是为了在进行生成的时候用的
    # 为了避免以后有漏洞，在这里定义模型的时候输入参数暂且按照作者的来，然后在生成的时候再返回来看各个参数的作用
    # 因为word2ix，ix2word是定义数据集的时候用的，按理来说跟模型没有关系才对
    model = CaptionModel(opt, len(ix2word) )
    if opt.model_ckpt:
        model.load(opt.model_ckpt)
    model.to(device)
    
    # step: meter criterion optimizer
    loss_meter = meter.AverageValueMeter()
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    model.save()
    # step: train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        
        for ii,(imgs, (captions, lengths), indexes) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
            
            optimizer.zero_grad()
            imgs = imgs.to(device)
            captions =  captions.to(device)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions, lengths).data # len
            score, _ = model(imgs, input_captions, lengths) # len*vocab
            loss = criterion(score, target_captions)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data.item())
            
            # step: visulize
            if(ii+1)%opt.print_freq == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                    
                vis.plot('loss', loss_meter.value()[0])
                # picture+caption
                # indexes在这里用到了，因为要可视化图片，就需要知道当前是第几张图片，而主模型的输入是直接2048维特征。没有直接指示第几行图片，
                # 同时也说明了，有序的重要性，所以在提取图片特征的时候，不是直接读取文件，而是从id2ix中获取，来使得标题和图片都可以从id2ix中找到对应关系
                # 如果是我，我肯定不会想到用id和序号做一个对应关系，说不定直接用列表存储所有图片名称，
                # 用列表终归不如用dict好，因为dict是可以反推回他是第几张图片 ix2id和id2ix,而列表只能是知道第几张图片的位置，不能反推。

                img_path = os.path.join(opt.img_path, ix2id[indexes[0]]) 
                raw_img = Image.open(img_path).convert('RGB')
                raw_img = tv.transforms.ToTensor()(raw_img)
                
                # captions_np = np.array(captions) # zheli shi weile bimian ziji buzhuyi er daozhi jisuantu de cunzai suoyi duiyu moxing de shuru he shuchu zuo qi ta caozuo shi douxian jiangqi xianshi zhuanhua cheng meiyou jisuantu biru detach() biru with t.no_grad() biru t_.data.tolist() biru t_.data biru qita
                raw_caption = captions.data[:,0].tolist()  #
               
                raw_caption = ''.join([ix2word[i] for i in raw_caption])
                # vis.img('raw_img', raw_img, caption=raw_caption)
                
                info = '<br>'.join([ix2id[indexes[0]],raw_caption])
                vis.log(u'raw_caption', info, False)
                results, scores = model.generate(img=imgs.data[0], eos_id=eos_id)
                cap_sentences = [ ''.join([ix2word[ix.item()] for ix in sentence]) for sentence in results]
                info = '<br>'.join(cap_sentences)
                info = '<br>'.join([ix2id[indexes[0]],info])
                vis.log(u'val', info, False)               
 
        model.save()

                
            
    


# In[ ]:


@t.no_grad()
def generate(**kwargs):
    
    # step: configure
    opt._parse(**kwargs)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = Visualizer(env=opt.env)
    
    # step: data
    # 这里加载数据的方式和train的略微不一样，
    # 这个问题其实从第七章就应该注意到，因为gen和val的区别主要在与data上，val的data和train的data是差不多的，但是gen的data一般只是一个
    # train直接获取了dataloader，但是这里
    # 这里的data有两个，一个是字， 一个是图片
    # word

# train he generate dui shuju de chuli ,s shifou keyi fangzai yiqi ????????????????????????????
    data = t.load(opt.caption_data_path, map_location = lambda s, _:s)
    word2ix, ix2word, end = data['word2ix'], data['ix2word'], data['end']
     
    # picture
    # 这里的代码和feature_extract中的代码有很大的相似性。但是离main已经隔了一层了，所以重新写吧
    # 
    transforms = T.Compose([
        T.Resize(opt.scale_size),
        T.CenterCrop(opt.img_size),
        T.ToTensor(),
        normalize
    ])
    
    img = Image.open(opt.test_img).convert('RGB')
    img = transforms(img).unsqueeze(0)
    img = img.to(device)
    
    # step: model: FeatureExtractModel:resnet50 CaptionModel:caption_model
    resnet50 = FeatureExtractModel()
    resnet50.to(device)
    resnet50.eval()

    
    caption_model = CaptionModel(opt, len(ix2word))
    caption_model.load(opt.model_ckpt_g)
    caption_model.to(device)
    caption_model.eval()
    
    # step: generate
    img_feats = resnet50(img) # 1*2048
    img_feats = img_feats.data[0] # 2048
    eos_id = word2ix[end]
    
    cap_sentences, cap_scores = caption_model.generate(img = img_feats,eos_id = eos_id) # 
    cap_sentences = [ ''.join([ix2word[ix.item()] for ix in sentence]) for sentence in cap_sentences]
    
    # vis.img('image', img)
    info = '<br>'.join(cap_sentences)
    vis.log(u'generate caption', info, False)
    
    return(cap_sentences,cap_scores)
    
def help():
    print("""
    usage: print file.py <function> [--args=value]
    <function>: train | generate
    example:
        python {0} train
        python {0} generate
        python {0} help
    avaiable args :
    """.format(__file__))
    
    from inspect import getsource
    source = getsource(opt.__class__)
    print(source)
    



if __name__ == '__main__':
    
    fire.Fire()

