
# coding: utf-8

# class CaptionGenerator----def beam_search----class TopN
#                                         |----def get_topk_data
#                                         |----class Caption
#                                         

# 其中CaptionGenerator用于生成语句，TopN是beam集合，Caption是每个元素的数据结构

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[35]:


import torch as t
from torch import nn
import heapq


# In[29]:
__all__ = ['Caption', 'TopN', 'push', 'extract', 'reset', 'size', 'CaptionGenerator', 'beam_search']

class Caption(object):
    """
    现在不太确定这个集合是hash_table还是set，感觉是hash_tale,是因为set不需要专门的存储结构。再看看吧
    这里应该不是那三个集合，而是集合中的每一个元素，比如G(i),这种，作者应该是重新创建了一种数据结构来用，来进行存储
    
    Args:
      sentence: list(int)
      state: tuple(hn, cn) hn:1*1*hidden_dim
      logprob: probability
      score: 等于logprb或者logprb/len(sentence)
    
    """
    
    def __init__(self, sentence, state, logprob, score, metadata=None):
        """
        Args:
          sentence(list): 
        
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata
    
    # 这里我猜是为了实现堆排序的比较。尽管知道是，但是还是不知道为什么
    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score
    


# In[51]:


class TopN(object):
    """
    Maintains the top N elements of an incrementally provided set.
    就是相当于beam_searching中beam集合，但是注意，在添加元素的时候，我不太确定这里是否要去重，是否需要和hash_table去重，
    """
    def __init__(self, n):
        self._n = n
        self._data = []
    
    def push(self, x):
        # 这里是相当于对元素从set到beam中的优化，在原文中，set到beam中是从小取到大，但在这里是通过小顶堆实现的，把所有的元素压入只保留beam_size大小的小顶堆中。
        assert self._data is not None
        
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)
        
    def extract(self, sort=False):
        """
        
        Args:
          sort: 从大到小排序
        
        Returns:
          data(list): n <= beam_size
          self._data = None
        
        """
        # 注意：；list的复制是共享内存的，元素的修改会影响，但是修改为None不会修改，所以在赋值之后，不是设置为[], 而是设置为None
        assert self._data is not None
        
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        
        return data
        
    def reset(self):
        # 应该和extract在一起的，但是没写在一起就没写在一起吧
        # 的确不能写在一起，None是一个标志位，可以用于其他函数的一个判断
        
        self._data = []
    
    def size(self):
        assert self._data is not None
        return len(self._data)


# In[69]:


class CaptionGenerator(object):
    """
    Class to generate captions from an image-to-text model
    """
    
    def __init__(self, 
                 embedder,
                 rnn, 
                 classifier, 
                 eos_id,
                 beam_size=3,
                 max_caption_length=20,
                 length_normalization_factor=0.0):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, state) and outputs len(vocab) values
          beam_size: Beam size to use when generating captions.
          max_caption_length: The maximum caption length before stopping the search.
          length_normalization_factor: If != 0, a number x such that captions are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of captions depending on their lengths. For example, if
            x > 0 then longer captions will be favored.
        """
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, state) and outputs len(vocab) values
          beam_size: Beam size to use when generating captions.
          max_caption_length: 在生成语句中，异常终止的条件不是hash_table is full,而是语句长度。
          正常终止条件： eos_id
          length_normalization_factor: If != 0, a number x such that captions are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of captions depending on their lengths. For example, if
            x > 0 then longer captions will be favored.
          重新思考一下终止条件：对于单个标题，eos_id是判断标题生成终止条件，
          对于整个循环，当达到max_caption_length或者所有的标题都是eos_id，则停止
        """
        self.beam_size = beam_size
        self.embedder = embedder # 三个模型之一 embedder
        self.rnn = rnn  # 三个模型之一  rnn
        self.classifier = classifier # 三个模型之一  classifier
        
        self.max_caption_length = max_caption_length
        self.eos_id = eos_id
        self.length_normalization_factor = length_normalization_factor
        
    @t.no_grad()
    def beam_search(self, rnn_input, initial_state=None):
        """Runs beam search caption gene
        Args:
          rnn_input(tensor): 1*1*embedding_dim  1*1*256  (已经经过embedder)
          initial_state: (h0, c0)
        
        Returns:
          caps_sentences    list(int)    beam_size*max_captions_length
          caps_scores       list(tensor) beam_size

        """
                
        def get_topk_data(rnn_input, state):
            """
            这里就相当于一个小型的forward,或者说是beam_search中的启发式函数
            不对，这里的state是受限制的，因为state是num_layers*batch_size*hidden_dim的，
            所以这里应该都是一个词进来的
            Args:
              rnn_input(tensor): 1*1*embedding_dim  1*beam_size*embedding_dim
              state:             1*1*hidden_dim     1*beam_size*embedding_dim
              
            Returns:
              id_words(tensor): 1*beam_size   beam_size*beam_size 
              logprobs(tensor): 1*beam_size   beam_size*beam_size
              state:         1*1*hidden_dim   1*beam_size*hidden_dim
            """
            output, state = self.rnn(rnn_input, state) # output:1*1*hidden
            output = self.classifier(output.squeeze(0)) # output: 1*vocabsize
            logprobs = nn.functional.log_softmax(output.data , dim=1) ## 暂时不清楚这里为什么用log_softmax，是负数啊，大哥，不过大小好像不变
            logprobs, id_words = logprobs.topk(self.beam_size, dim=1) # 1*beam_size
            
            return id_words, logprobs, state
        

        # 为了能够更加方便地对数据进行处理，专门创建了一个数据结构来保存当前的已经生成的caption，暂时不清楚这么写创建是否真的很有效率
        partial_captions = TopN(self.beam_size)  # Beam集合 用于中间迭代
        complete_captions = TopN(self.beam_size) 
        """
        Beam集合 用于记录已经成功的标题 因为对于已经成功的标题，不再需向后生成，这个时候把标题放在partial_captions不合适，这是和原始的
        beam_search不同的，beam_search只要找到终点就会停止，而现在要找到多条语句，所以把已经写完的句子放在另一个里面
        
        """
        
        id_words, logprobs, state = get_topk_data(rnn_input, initial_state) # id_words就是set集合
        # 这是对于第一个字
        for k in range(self.beam_size):
            cap = Caption(
                sentence = [id_words[0,k]],
                state = state,
                logprob = logprobs[0,k],
                score = logprobs[0,k]
            )
            partial_captions.push(cap)
            
        # 开始beam searching
        # 只是循环K次
        for j in range(self.max_caption_length):
            #if j>24:
            #    import ipdb
            #    ipdb.set_trace()
         
            partial_captions_list = partial_captions.extract() # list beam_size
            partial_captions.reset()
            input_feed = t.LongTensor([w.sentence[-1] for w in partial_captions_list]) # LongTensor: beam_size 
            state_feed = [w.state for w in partial_captions_list] # 
            if rnn_input.is_cuda:
                input_feed = input_feed.cuda()
            if isinstance(state_feed[0], tuple):
                state_feed_h, state_feed_c = zip(*state_feed)
                state_feed = (t.cat(state_feed_h, dim=1), t.cat(state_feed_c, dim=1))
            else:
                state_feed = t.cat(state_feed, dim=1)
                
            
            embeddings = self.embedder(input_feed).view(1, input_feed.shape[0], -1) # 1*beam_size*embedding_dim
            id_words, logprobs, new_state = get_topk_data(embeddings, state_feed) # 
            # id_words: ?beam_size*beam_size
            # import ipdb
            # ipdb.set_trace()
            for i, partial_caption in enumerate(partial_captions_list):
                if isinstance(new_state, tuple):
                    state = (new_state[0].narrow(1,start=i, length=1), new_state[1].narrow(1,start=i, length=1))
                else:
                    state = new_state[i]
                
                for k in range(self.beam_size):
                    w = id_words[i, k]
                    sentence = partial_caption.sentence+[w]
                    logprob = partial_caption.logprob + logprobs[i,k]
                    score = logprob
                    if w == self.eos_id:
                        if self.length_normalization_factor > 0:
                            score = (score/len(sentence)) ** self.length_normalization_factor
                        cap = Caption(sentence, state, logprob, score)
                        complete_captions.push(cap)
                    else:
                        cap = Caption(sentence, state, logprob, score)
                        partial_captions.push(cap)
            
            # 另外一个终止条件: 所有的标题都生成了eos_id
            if not partial_captions.size():
                break
        # 注意，此时的结束标题有两种可能，一种是到了eos_id，一种是到了max_caption_len
        # 如果是都到了eos_id,那么complete_captions应该是满的，如果是max_caption_len，那么complete_captions可空可不满，可满
        # 那么complete_captions有三种状态，一种是空，把部分标题给complete，一种是半满，从半满中进行取舍，一种是全满，从全满中进行取舍
        
        if not complete_captions.size():
            complete_captions = partial_captions
            
        caps = complete_captions.extract()
        caps_sentences = [ cap.sentence for cap in caps]  # list(tensor)  beam_size*max_captions_length
        caps_scores = [cap.score for cap in caps]         # list(tensor) beam_size
        
        return caps_sentences, caps_scores 
