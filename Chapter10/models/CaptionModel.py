
# coding: utf-8


# 这里有个很尴尬的地方就是在这里定义的opt实际上在这用不了，因为传不进去类里面，但是函数的可以直接用，例如data_preprocess就是直接用的。尴尬。
# 这样有一个问题就出现了，类和函数的到底什么才能知道什么时候用哪个呢

# In[75]:


from torch import nn
import torch as t
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

import sys
sys.path.append('..')
from utils import CaptionGenerator


# In[121]:

__all__ = ['CaptionModel', 'state', 'save', 'load', 'get_optimizer', 'generate']

class CaptionModel(nn.Module):
    # 且看一下这个模型怎么走，有点晕了都
    def __init__(self, opt, vocab_size):
        
        super(CaptionModel, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(2048, opt.rnn_hidden)
        
        # vocab_size = len(ix2word)
        self.embedding = nn.Embedding(vocab_size, opt.embedding_dim)
        self.rnn = nn.LSTM(opt.embedding_dim, opt.rnn_hidden, num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.rnn_hidden, vocab_size)
        
        
    def forward(self, img_feats, captions, lengths):
        """
        img_feats(tensor): b*2048
        caption(tensor): seq_len*b
        """

        img_feats = self.fc(img_feats).unsqueeze(0) # 1*b*256
        
        embeddings = self.embedding(captions) #seq_len* b* 256
        embeddings = t.cat([img_feats, embeddings], dim=0)
        packed_embeddings = pack_padded_sequence(embeddings, lengths) # packed_embeddings: len*embedding_dim
        outputs, state = self.rnn(packed_embeddings) # outputs: len * embedding_dim
        pred = self.classifier(outputs.data) # 
        
        # 在第九章的时候也返回了state，但第九章没有用，这是一个标准流程吗？
        return pred, state
    
    # 下面还有几个函数，先不准备写，看看在哪里会用到
    """
    ________________________________________________________________________________________________
    """
    
    
    
    # 以下是保存和加载模型的相关函数，可以独立成章，做个记录
    # 怎么才能让别人知道我的保存模型的结构呢？ 因为写在这里一般人还是看不见啊
    """
    @return all_state_dict(dict):
    all_state_dict = dict( opt_state_dict = opt_state_dict,
                         model_state_dict = self.state_dict() )
    
    """
   
    def state_all(self):
        # 单独分出来是因为万一需要查的时候也方便查
        """

        """
        opt_state_dict = { k:getattr(self.opt, k) 
                           for k in dir(self.opt) 
                           if not k.startswith('_')}

        all_state_dict = {'opt':opt_state_dict,
                          'state_dict' : self.state_dict() }
        
        return all_state_dict
    # 从第六章到第九章纠结到现在， 结果忘了还能这么写，直接写个load就好了， 非得等继承， 伤心
    # 这里的save有个问题就是不能直接捕捉到epoch。只能以时间的形式捕捉先后
    # 这里又使用了**kwargs，感觉哪哪都用，但是也没看见有啥用啊
    
    def save(self, path=None, **kwargs):

        if path is None:
            path = '{prefix}_{time}.pth'.format(prefix=self.opt.prefix,
                                                time=time.strftime('%m%d_%H%M')
                                               )
        state = self.state_all()
        state.update(kwargs)
        t.save(state, path)
        
        return path
        
    def load(self, path, load_opt=False):
        data = t.load(path, map_location= lambda s, l: s)
        model_state_dict = data['state_dict']
        self.load_state_dict(model_state_dict)
        
        if load_opt:
            for k, v in data['opt_state_dict']:
                setattr(self.opt, k, v)
        # 不知道这里为什么要返回自己 
        return self
    
    
    
    
    """
    ________________________________________________________________________________________________
    """
    # 没想通为什么要在这里写优化器，也就一行啊  感觉没必要啊
    
    def get_optimizer(self, lr):
        return t.optim.Adam(self.parameters(), lr)
    @t.no_grad()
    def generate(self, 
                 img,
                 eos_id, 
                 beam_size = 3, 
                 max_caption_length = 30, 
                 length_normalization_factor = 0.0):
        """
        为什么要在模型里写，其实这个generate可以理解成forward，因为模型的rnn,classifier都是不应该被外界访问的，
        所以只能写在这个里面，准确地说，其实应该是在在这里面直接写beam_searching的，但是采用了调用的手法进行了简略
        
        Args:
          img: 2048
          eos_id: int
        
        Return:
          cap_sentences        list(tensor)    beam_size*max_captions_length
          cap_scores           list(tensor)    beam_size

        """
        # 我觉得这里不需要来判断是不是cuda，因为在上一层输入的时候，就应该输入和模型是一致的。同理在下一层，也不需要判断，应该保证输入是一致的
        img = img.unsqueeze(0) # 1*2048
        img = self.fc(img).unsqueeze(0) # 1*1*256
        cap_gen = CaptionGenerator(embedder = self.embedding, 
                                   rnn = self.rnn, 
                                   classifier = self.classifier,
                                   eos_id = eos_id,
                                   max_caption_length = max_caption_length,
                                   length_normalization_factor = length_normalization_factor)
        
        cap_sentences, cap_scores = cap_gen.beam_search(img) 
        
        return cap_sentences,  cap_scores
