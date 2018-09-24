
# coding: utf-8

# 这次对数据处理的方式和之前完全不一样，以前是通过这种格式来获取数据的
# ```
# # 这种是自己命名一个类
# # dataset.py
# from torch.utils import data
# class DogCat(data.Dataset):
#     def __init__(self):
#     def __getitem__(self,index):
#          return data, label
#     def len(self):
# # main.py
# from data.dataset import DogCat
# dataset = DogCat()
# dataloader = DataLoader(dataset, batch_size,shuffle, num_workers)
# ```
# 或者是这样：
# ```
# ＃　这种是针对图片已经按文件夹分类放好的情况
# # main.py
# import torchvision as tv
# from torch.utils.data import DataLoader
# dataset = tv.datasets.ImageFolder(opt.data_path, transform = transforms)
# dataloader = DataLoader(dataset, batch_size,shuffle, num_workers)
# ```
# 
# 而这次对数据的方式直接定义函数，等写完看看情况

# 函数：
# 主函数：
# def get_data(): 获取数据 data,word2ix,ix2word

# In[1]:



# import Ipynb_importer
# from config import opt


# In[2]:


import os
import json
import numpy as np
import ipdb
import re

get_data()----_parseRawData（）----main()---handlejson----isConstrains
                                                   |----setenceParse
         |----pad_sequences（）
# In[3]:


def _parseRawData( src, category,constrains=None,author=None):

    """
    code from https://github.com/justdark/pytorch-poetry-gen/blob/master/dataHandler.py
    处理json文件，返回诗歌内容
    
    @param: author： 作者名字,为了限制取出的诗歌，如果为None，那么取出所有的作者
    @param: constrains: 长度限制，比如只取五言或者七言，要求是对每句话（以逗号感叹号等分割）都必须是这个长度，不利于词
    @param: src: json 文件存放路径
    @param: category: 类别，有poet.song 和 poet.tang
    @ src+category:要解析所有类型的文件
    @return: data list
    [
    '床前明月光，疑是地上霜，举头望明月，低头思故乡。',
    '一去二三里，烟村四五家，亭台六七座，八九十支花。',
    ......
    ]
    所有文件的诗歌。
    """
    
    """
    文件夹组织形式:分为两种，authors和poet；又分为两个年代，song和tang
    
    chinese-poetry-zhCN/poetry/authors.song.json
    chinese-poetry-zhCN/poetry/poet.song.0.json
    ...
    chinese-poetry-zhCN/poetry/authors.tang.json
    chinese-poetry-zhCN/poetry/poet.tang.0.json
    ...
    
    """
    
    """
    poet.json文件内容示例
    data： list
    [
  {
    "strains": [
      "仄仄仄仄平仄仄，平平仄平○仄仄。", 
      "平平仄仄平仄平，仄仄平平？仄仄。"
    ], 
    "author": "宋太祖", 
    "paragraphs": [
      "欲出未出光辣达，千山万山如火发。", 
      "须臾走向天上来，逐却残星赶却月。"
    ], 
    "title": "日诗"
  }, 
  ...
  ]
  
  poetry:
  [
  {
    "strains": [
      "仄仄仄仄平仄仄，平平仄平○仄仄。", 
      "平平仄仄平仄平，仄仄平平？仄仄。"
    ], 
    "author": "宋太祖", 
    "paragraphs": [
      "欲出未出光辣达，千山万山如火发。", 
      "须臾走向天上来，逐却残星赶却月。"
    ], 
    "title": "日诗"
  }
  ]
  
  poem:
  [
      "欲出未出光辣达，千山万山如火发。", 
      "须臾走向天上来，逐却残星赶却月。"
      ]
  s:
  "欲出未出光辣达，千山万山如火发。"
  pdata:
  '欲出未出光辣达，千山万山如火发。须臾走向天上来，逐却残星赶却月。'
  sp:
  ['欲出未出光辣达','千山万山如火发','']
  tr: '欲出未出光辣达'或者''
  json数据格式：
  1.[],中间是value
  2.{},中间是”键/值“对
  3.可循环嵌套
    """
    def sentenceParse(pdata):
        # 按照作者的意思，这一块pdata可能会是这个样子的
        # pdata：’-181-早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。‘
        # 我观察了一两个唐诗，没有发现，不过先按照作者的来，因为对数据集也不是很清楚，也不影响后续内容
        # 函数的目的是删除pdata内的一些不合法字符
        """
        @pdata:’-181-早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。‘
        @return:result:’早嘗甘蔗淡，生摘琵琶酸。好是去塵俗，煙花長一欄。‘，特殊情况下可能是空
        """
        result = re.sub(u'（.*）','',pdata)
        result = re.sub(u'{.*}','',result)
        result = re.sub(u'《.*》','',result)
        result = re.sub(u'[\]\[]','',result)
        result = re.sub(u'[\d-]','',result)
        result = re.sub(u'。。',u'。',result)
        return result
        
    def isConstrains(poem):
        """
        poem:示例如上
        用于判断poem中的每句话是否等于其要求的长度
        @return： False：表示不满足，应该舍去
                 True：表示满足，应该保留
        """
        for s in poem:
            sp = re.split(u'[，！。]',s)
            for tr in sp:
                if (constrains is not None) and (len(tr) != constrains) and (len(tr) != 0):
                    return False
        return True
    
    
    
    def handleJson(file):
        # 对一个文件内数据进行处理
        """
        @return: ['欲出未出光辣达，千山万山如火发。须臾走向天上来，逐却残星赶却月。',
                  '欲出未出光辣达，千山万山如火发。须臾走向天上来，逐却残星赶却月。'
                  ]
        """
        rst = []
        with open(file,'r') as f:
            data = json.load(f) # data示例如上
        for poetry in data:
            if (author is not None) and (author != poetry['author']):
                continue
            poem = poetry["paragraphs"] # list,但此时长度可能不一致
            # 是否满足长度限制
            if not isConstrains(poem):
                continue
            pdata = ''.join(poem) # '欲出未出光辣达，千山万山如火发。须臾走向天上来，逐却残星赶却月。'
            # 删除pdata内的不合法字符
            pdata = sentenceParse(pdata) # '欲出未出光辣达，千山万山如火发。须臾走向天上来，逐却残星赶却月。'
            if pdata != '':
                rst.append(pdata)
        return rst
        
        
    # main()   
    data = []
    # src = chinese-poetry-zhCN/poetry 
    # [handleJson(os.path.join(src,f)) for f in os.listdir(src) if f.startswith(category)]
    for filename in os.listdir(src):
        if filename.startswith(category):
            path = os.path.join(src,filename)#path = chinese-poetry-zhCN/poetry/poet.song.0.json
            data.extend(handleJson(path))       # 读取文件并进行解析
            
    return data


# In[4]:


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating = 'post',value=0):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    """
    @ sequences:[[1,2,3],[1,2,3,4],[1,2,3,4,5]]
    @ return: x:[[0,1,2,3,],[1,2,3,4],[1,2,3,4]] np.array
    由于这段代码本身作者也是参照了别人写的，里面会有一些没有意义的东西，对此，我选择基本保留作者的原意，因为这是涉及到更多的扩展性，
    
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
            lengths.append(len(x))
    # 一共有num_samples个子列表
    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple() # 
    # 第一个非空子列表的内有sample_shape个列表，这里我们是1.
    # 这里的sequences其实是一个多维这个形状的:num_samples*max_len*sample_shape.并且sample_shape表明，每一个子子列表是必须可以表示成相同大小的矩阵形式的才行。
    # 所以sample_shape其实记录的子子列表的维度,各个np的大小应该是一样的。
    # 类似 [[np,np],[np,np],[np,np]], sample_shape = np.shape = turnc.shape[1:]
    # 在此之所以用np.array是因为list本身只有len方法，没法返回其size，所以我们要想返回其size，可以采用np作为辅助
    for s in sequences:
        if len(s) >0:
            sample_shape = np.asarray(s).shape[1:]
            break
    # tuple的拼接
    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx,:len(trunc)] = trunc
        elif padding == 'pre':
            x[idx,-len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
            
    return x


# In[5]:


def get_data(opt):
    """
    @param  opt: Config对象，因为太多了，所以用opt来传入需要的参数
    @return data: numpy二维数组，每一行是一首诗对应的字的下标 np.array([[1,2,3],[1,2,3]])
    @return word2ix:  元素是 dict 字对应序号，　类似　'月'：１
    @return ix2word:  dict 序号对应字，　类似　１：'月'
    
    # 编码问题值得注意一下
    # 保存成.pth时，为了能保存word2ix这种dict，需要先把他们转化成array，同理，在取出来的时候，也需要.tolist()进行一次转化。
    """

    # 如果已经存在处理好的数据
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path)
        data, word2ix_array, ix2word_array = data['data'], data['word2ix'], data['ix2word']
        word2ix = word2ix_array.tolist()
        ix2word = ix2word_array.tolist()
        return data, word2ix, ix2word
    # 如果没有处理好的数据
    
    # 处理原始的json文件，将全部的poem取出来，形成list
    # data: [
    # '床前明月光，疑是地上霜，举头望明月，低头思故乡。',
    # '一去二三里，烟村四五家，亭台六七座，八九十支花。'
    # ...]
    data = _parseRawData( opt.data_path, opt.category, opt.constrains, opt.author) #　读取文件夹下的所有文件并进行解析成句子
    # 接下来要形成word2ix和ix2word,</s>在html中不显示
    words = {_word for _sentence in data for _word in _sentence}
    word2ix = {_word:_ix for _ix,_word in enumerate(words)} # dict
    word2ix['<EOP>'] = len(word2ix)
    word2ix['<START>'] = len(word2ix)
    word2ix['</s>'] = len(word2ix)
    ix2word = {_ix:_word for _word,_ix in list(word2ix.items())}
    # 将诗歌转化为数字
    # 接下来把data想要写成这样子的
    # [['<START>','床', '前', '明', '月', '光', '，', '疑', '是', '地', '上', '霜','<EOP>'],
    # ['<START>','床', '前', '明', '月', '光', '，', '疑', '是', '地', '上', '霜','<EOP>'],
    # ...]
    # data = [ ''.join(['<START>',_data,'<EOP>']) for _data in data]
    data = [ ['<START>']+list(_data)+['<EOP>'] for _data in data]
    # data: [
    # '床前明月光，疑是地上霜，举头望明月，低头思故乡。',
    # '一去二三里，烟村四五家，亭台六七座，八九十支花。'
    # ...]
    sequences = [ [word2ix[_word]  for _word in _data ]for _data in data]
    # sequences:[[1,2,3],[1,2,3],[1,2,3]]
    # 对诗歌进行增删成固定长度的诗歌
    data = pad_sequences(sequences, 
                         maxlen=opt.maxlen, 
                         dtype='int32', 
                         padding='pre', 
                         truncating = 'post',
                         value=len(word2ix)-1)
    # pad_data:[[0,1,2,3,],[1,2,3,4],[1,2,3,4]]
    # 保存成文件
    word2ix_array = np.array(word2ix)
    ix2word_array = np.array(ix2word)
    np.savez_compressed(opt.pickle_path, 
                        data=data, word2ix = word2ix_array, ix2word = ix2word_array)
    
    return data, word2ix, ix2word

