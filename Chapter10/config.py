
# coding: utf-8

# In[28]:

__all__ = ['Config','__parse']

class Config:

# main.py
    # train()

    # step: configure
    use_gpu = False
    env = 'caption'

    # step: data  # data/data.py
    caption_data_path = './dataset/caption.pth'  # 经过预处理后的人工描述信息
    img_feature_path = './dataset/results.pth'  # 所有图片的features,20w*2048的向量
    
    batch_size = 8
    shuffle = True
    num_workers = 4

    #step: model
    model_ckpt = None
    embedding_dim = 256
    rnn_hidden = 256
    num_layers = 2
    prefix = './checkpoints/caption'  # 模型保存前缀

    # step: meter criterion optimizer
    lr = 1e-3

    # step: train
    max_epoch = 1
    print_freq = 10
    debug_file = './debug'

    # step: visulize
    img_path = '/home/zbp/Linux/tjj/data/ai_challenger_caption_train_20170902/caption_train_images_20170902'
    # img_path='/home/zbp/Linux/tjj/data/ai_challenger_caption_train_20170902/caption_train_images_20170902/XXX.jpg'
  
    # generate()
    scale_size = 224
    img_size = 224
    model_ckpt_g = './checkpoints/caption_1003_1419.pth'
    test_img = './img/example.jpeg'


   # generate
    scale_size = 300
    img_size = 224

#data/data.py

    



    def _parse(self,**kwargs):
       
        for k,v in kwargs.items():
            setattr(self,k,v)
            
        print('User Config')
        for k in dir(self):
            if not k.startswith('_'):
                print('{0}====={1}'.format(k,getattr(self,k)))

opt = Config()
