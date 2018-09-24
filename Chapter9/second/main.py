
# coding: utf-8

# In[1]:

# train()----generate()
# gen()----generate()
#     |----gen_acrostic()

# In[12]:
from config import opt
from dataset import get_data
from models import PoetryModel
from utils import Visualizer


import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torchnet import meter
from tqdm import tqdm
import os
import ipdb


# In[22]:


def train(**kwargs):
    # step: configure
    opt._parse(**kwargs)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    if opt.env:
        vis = Visualizer(env=opt.env)
    # step: data
    data, word2ix, ix2word = get_data(opt) # data numpy二维数组， word2ix, ix2word 字典
    # from_numpy共享内存，一个数字的变化也会影响另一个，但是t.tensor不会共享内存，两个基本完全独立
    data = t.from_numpy(data)
    # 这里是因为鸭子类型，还需要考虑考虑
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    
    # step: model && criterion && meter && optimizer 

    model = PoetryModel(len(word2ix), opt.embedding_dim, opt.hidden_dim, opt.num_layers)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(device)
    
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    
    criterion = nn.CrossEntropyLoss()
    
    loss_meter = meter.AverageValueMeter()
    
    
    # step: train
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, x in tqdm(enumerate(dataloader)):
            # embedding层的输入必须是LongTensor型
            # 现在x是tensor (batchsize*seq_len)，LSTM的输入需要是(seq_len, batch_size, embedding_dim)
            # 矩阵的转置会导致存储空间不连续， 需要调用.contiguous()方法使其连续
            x = x.long().transpose(1,0).contiguous()
            x = x.to(device)
            optimizer.zero_grad()
            input, target = x[:-1,:], x[1:,:]  # target :(seq_len, batch_size)
            # 运行的时候这里要看一下大小
            output,_ = model(input)  # output size (seq_len*batch_size, vocab_size)
            loss = criterion(output, target.view(-1)) # 交叉熵损失的定义
            # 这里需要重新想明白，这个lstm是怎么个输入输出
            loss.backward()
            optimizer.step()
            # 这里的loss是一个只有一个数字的tensor,
            # loss.item()返回一个新的Python的对应的类型，不共享内存，改变不会影响彼此
            # 经师兄提醒，才注意到计算评价loss的时候，需要想办法去除掉loss.backward等特性，避免时间长了占内存，
            # 这里没有loss.data，loss.data也会有backward等特性，还是属于tensor系列，突然感觉自己还是遗漏了好多点。
            # 现在只能一边做，一边查缺补漏，看到哪里，学到哪里，对于一些细节要经常去查。
            # 这里需要重新解释一下，每一个tensor代表一个计算图，如果直接使用tensor进行累加的话，会造total_loss的计算图不断累加的
            # 有点乱了，我去，不管了，先记住，对于损失累加，我们只使用loss.item,这是种完全截断计算图的方法
            loss_meter.add(loss.item())
            # step: visualize and validate
            if (ii+1)%opt.print_freq == 0 and opt.env:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                
                vis.plot('loss',loss_meter.value()[0])
                # 诗歌原文
                # x tensor size (seq_len, batch_size）
                # 二重列表生成式, poetrys:[['我''你'],[..]]

                # poetries = [[ix2word(word_) for word_ in x[:,j_]] for j_ in range(x.shape[1])]
                origin_poetries =[]
                origin_poetries_tmp = []
                #  range(data_.shape[1]
                for j_ in range(3):
                    for word_ in x[:,j_].tolist():
                        origin_poetries_tmp.append(ix2word[word_]) 
                    origin_poetries.append(origin_poetries_tmp)
                    origin_poetries_tmp = []
                vis.log( '<br/>'.join([''.join(origin_poe) for origin_poe in origin_poetries]), win = u'origin_poem'  )
                
                # 生成的诗歌
                gen_poetris = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗 验证模型
                # gen_poetris 二重list，每一个list都是一首诗 [['我','你'],[]]
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = generate(model, word, ix2word, word2ix)
                    gen_poetris.append(gen_poetry)
                # gen_poetris 二重列表，与poetries一致
                vis.log( '<br/>'.join([''.join(gen_poe) for gen_poe in gen_poetris]), win = u'gen_poem'  ) 

        t.save(model.state_dict(), '{0}_{1}.pth'.format(opt.model_prefix, epoch))
    
            
            
        
            
    
    
    
    


# In[ ]:


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定一段词，，根据这几个字接着生成一首完整的诗歌
    @model: 模型
    @start_words: u'春江潮水连海平'
    @return： results list ['春'，'江','潮','水'...] 是一首诗
    """
    # 第一个输入的字，模型输入 input tensor size (seq_len, batch_size)
    input = t.Tensor([word2ix['<START>']]).view(1,1).long()
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    input = input.to(device)
    
    # 暂时不知道这个prefix_words的用法
    # 这里应该是为了保留hidden，
    hidden = None
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1,1)
            
    # 这一段的逻辑是什么？？？ 
    # 每次取给定的字作为输入，求hidden和output，对于output，每次取概率最大的
    results = list(start_words)
    start_word_len = len(start_words)
    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1,1)
        else:
            # output size 1×vocab_size [[1,2,3,...]]
            # 这里应该看一下，输出output是个什么东西
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([word2ix[w]]).view(1,1)
        if w == '<EOP>':
            del results[-1]
            break
            
    return results


# In[ ]:


def gen(**kwargs):
    """
    等会儿回来补补
    """
    # step： configure
    opt._parse(**kwargs)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    # step: data model
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256)
    model.load_state_dict(t.load(opt.model_path,map_location= lambda s_,_:s_))
    model.to(device)
    # 解码和编码问题应该解决一下了
    # python2和python3 字符串兼容
    
    # step: main
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，').replace('.', u'。').replace('?', u'？')   
    
    if opt.acrostic:
        result = gen_acrostic(model, start_words, ix2word, word2ix, prefix_words)
    else:
        result = generate(model, start_words, ix2word, word2ix, prefix_words)
    


# In[ ]:


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words):
    """
    生成藏头诗
    @start_words: u'深度学习' 不能为空
    @prefix_words: 前缀
    @return： results list ['春'，'江','潮','水'...] 是一首诗
    
    """
    
    # step: configure
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    
    # step: data and model
    # 第一个字一定是'<START>'
    # 模型输入 input tensor size (seq_len, batch_size)
    input = t.Tensor([word2ix['<START>']]).view(1,1).long()
    input = input.to(device)
    model.to(device)
    hidden = None
        
    # step： 对prefix_words进行输入
    prefix_words = '' if prefix_words==None else prefix_words    
    for word in prefix_words:
        output, hidden = model(input, hidden)
        input = input.data.new(word2ix[word]).view(1,1)
        
    
    # step: 对start_words进行输入
    # 这里可以看出来，作者假设前面的字只提供hidden的信息，在prefix_words之后，才正式开始作诗
    start_words = list(start_words)+[None]
    results = []
    # 变成列表，方便后续的操作，因为start_words的每个字用过之后就没用了，
    # 用pop不行,因为对于空列表会报错,用None作为结尾标志。可以看出，如果我们想让某个序列正常退出，可以通过设置特殊的结尾来实现。
    # 这一段的逻辑有点乱，因为prefix_words可能没有，所以对于start_words，必须先进行一个模型生成。
    # 对于或有或无的perfix_words，为了消除其存在对代码和思路的影响，应该保证prefix_words前后的代码状态不变，即
    """
    第一种
    这种保证了output,hidden的状态不变
    output, hidden = model(input, hidden)
    
    # step： 对prefix_words进行输入
    prefix_words = '' if prefix_words==None else prefix_words    
    for word in prefix_words:
        input = input.data.new(word2ix[word]).view(1,1)
        output, hidden = model(input, hidden)
        
    for i in range(opt.max_gen_len-1):
        top_index = output[0].topk(1)[1][0].item()    
        ...
        output, hidden = model(input, hidden)
    
    第二种
    这种保证了input的状态不变
    for word in prefix_words:
        output, hidden = model(input, hidden)
        input = (input.data.new([word2ix[word]])).view(1, 1)
    
    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        
        
    决定采用第二种，因为代码的主体思路是for i in range(opt.max_gen_len)，prefix_word是插入部分，是可有可无部分。
    第一种会造成 top_index与model的切分，不利于后期分析。
    或者说，以后碰到这种类型的代码，可以直接跳过中间部分，对后面进行分析。
    
    """
    words_pre = {u'。',u'！',u'<START>'}
    w = '<START>' # pre_word总是等于当前模型输入的字
    """
    对于w的分析，w在第一次循环时，表示输入的字，
    """
    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output[0].topk(1)[1][0].item()
        
        # 当前输入的词input是句号或者感叹号或者<START>,送入藏头诗的字,否则正常输入
        # 对w的更新
        w = start_words.pop(0) if w in words_set else ix2word(top_index)
        
        # None表示藏头字都已经取完并且当前字是
        if w is None:
            break
        input = input.data.new(word2ix[w]).view(1,1)
        
        results.append(w)
    
    return results
    
if __name__ == '__main__':
    import fire
    fire.Fire()

