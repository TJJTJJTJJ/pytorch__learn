
# coding: utf-8

# 这次预处理的代码因为么有数据集，所以只能根据作者的代码来猜

# python data_preprocess.py process --annotation-file=/data/annotation.json --max-words=5000

# In[9]:


import torch as t
import json 
import numpy as np
import tqdm
import jieba


# In[10]:


class Config(object):
    annotation_file = '/home/zbp/Linux/tjj/data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
    max_words = 10000
    min_appear = 2
    unknown = '</UNKNOWN>'
    end = '</EOS>'
    padding = '</PAD>'
    save_path = '../test/caption.pth'
    
    def parse(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


# In[30]:


def process(**kwargs):
    # 低频的词用unknown代替
    opt = Config()
    opt.parse(**kwargs)
    with open(opt.annotation_file) as f:
        data = json.load(f)  # list
    id2ix = {dic['image_id']:ix for ix, dic in enumerate(data)}
    ix2id = {v:k for k, v in id2ix.items()}
    assert id2ix[ix2id[10]] == 10
    
    # 分词
    captions = [ _dic['caption'] for _dic in data]
    # [['','','','',''],
    # ['','','','','']]

    cut_captions = [[list(jieba.cut(ii, cut_all = False)) for ii in item] for item in tqdm.tqdm(captions)] # 三重列表
    
    # 词列表
    word_nums = {} # '快乐'：1000次
    def update(word_nums):
        def updata(word):
            word_nums[word] = word_nums.get(word,0)+1
        return updata
    lamb = update(word_nums)
    _ = [ lamb(word) for sentences in cut_captions for sentence in sentences for word in sentence]
    _word_nums = [(num, word) for word, num in word_nums.items()]
    word_num_lists = sorted( _word_nums, key= lambda x:x[0], reverse=True ) # [(1000,'很好'),...]
    
    # 只取前N个词，并过滤低频

    words = [ word[1] for word in word_num_lists[:opt.max_words] if word[0]>=opt.min_appear]
    words = [opt.unknown, opt.padding, opt.end] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}   
    assert word2ix[ix2word[123]] == 123
    unknown2ix = word2ix[opt.unknown]
    ix_captions = [[[ word2ix.get(word, unknown2ix) 
                     for word in sentence] 
                    for sentence in item]  
                   for item in cut_captions]
    
    readme = u"""
    word：词
    ix:index
    id:图片名
    caption: 分词之后的描述，通过ix2word可以获得原始中文词
    """
    results = {
        'caption': ix_captions,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'ix2id': ix2id,
        'id2ix': id2ix,
        'padding': '</PAD>',
        'end': '</EOS>',
        'readme': readme
    }
    
    t.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)
    


# In[ ]:


if __name__ == '__main__':
    import fire
    fire.Fire()


# python data_preprocess.py process --annotation-file=/data/annotation.json --max-words=5000
