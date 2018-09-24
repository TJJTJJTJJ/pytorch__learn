
# coding: utf-8

# In[1]:


from torch import nn


# embedding需要重新看看说明，是一个什么样的原理
# 
# LSTM的理解：(https://www.zybuluo.com/hanbingtao/note/581764)
# 
# LSTM的理解：colah.github.io
# 
# 晚上去找了同学，才知道了一点关于LSTM的编码器和解码器，不过对于loss的定义还是不清晰，这里还是等写完main函数再看看

# In[62]:


import torch
m = nn.Linear(20,30)
m.bias.shape
input = torch.randn(1,3,2,20)
input.shape
output = m(input)
output.shape


# In[2]:


class PoetryModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        @vocab_size: int 表示输入字的个数
        @embedding_dim: int 表示希望这些字可以被多少维的向量表示
        这两个参数的用法  nn.Embedding(vocab_size, embedding_dim)
        @hidden_dim:  int 是LSTM的隐藏单元的维度
        @return: 
        """
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 这个lstm的定义有疑问
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers)
        # Linear层还有异议，需要重新看一下定义
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        
    def forward(self, input, hidden=None):
        """
        @input: tensor, size： (seq_len, batch_size)  
        @hidden: tuple,(tensor, tensor) 是LSTM模型的初始输入
        """
        seq_len, batch_size = input.size()
        
        if hidden is None:
            # 这里的new方法存疑,已经搞清楚了，就相当于生成同类型的tensor
            # view()和reshape的功能差不多，前者只能是连续内存，后者都行， 
            # view_as(t.tensor(4,3))
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        
        embeds = self.embeddings(input) # embeds size: (seq_len, batch_size, embeding_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0)) 
        # output size (seq_len, batch_size, hidden_dim)
        # hidden tuple,每一个元素的维度 (num_layers, batch_size, hidden_dim)
        # 这里感觉上下两个维都对不上啊，感觉
        # 这里维度能对应上了， Linear层的输入是(batch_size,*,in_features) 输出是（batch_size, *, out_features)
        output = self.linear1(output.view(-1, self.hidden_dim))
        # output size (seq_len*batch_size, vocab_size)
        # hidden tuple,每一个元素的维度 (num_layers, batch_size, hidden_dim)
        return output, hidden
        

