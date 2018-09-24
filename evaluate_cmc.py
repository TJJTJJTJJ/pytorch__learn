
# coding: utf-8

# main()----evaluate()----compute_cmc
#      |----get_id()

# In[ ]:


import 


# In[ ]:


def evaluate(qf,ql,qc,gf,gl,gc):
    """
    qf: list [1,2,3]
    ql: 1
    qc: 1
    gf: list [[1,2,3],[1,2,3]]
    gl: [1,2,3]
    gc: [1,2,3]
    len(gf)==len(gl)==len(gc)
    """
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large # 表示位置，[4,3,1,0,2]
    index = index[::-1] # 表示
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql) # list [[1],[2]] # 表示位置，即galley中的第几个样本是相同的id
    camera_index = np.argwhere(gc==qc) # list [[1],[2]] # 表示位置

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True) # [2,3]表示同一个id不同摄像头的图片的位置
    junk_index1 = np.argwhere(gl==-1) #  [[1],[2]] 表示id为-1的图片的位置
    junk_index2 = np.intersect1d(query_index, camera_index) #  [1,2] 表示同一个id同一个摄像头的图片的位置
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


# In[ ]:


def compute_cmc(index, good_index, junk_index):
    """
    index: list [4,3,1,0,2]，已经排序，数字表示第几张图片
    good_index: [3,1] list 位置 数字表示第几张图片
    junk_index: [4,2]list 位置 数字表示第几张图片
    """
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]  # [3,1,0]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index) # [t,t,f]
    rows_good = np.argwhere(mask==True) # 
    rows_good = rows_good.flatten() # [0,1]
    
    cmc[rows_good[0]:] = 1

    return cmc


# In[ ]:


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


# In[ ]:


def main():
    image_list_path_gallery = './evaluation_list/market_gallery.txt'
    image_list_path_query = './evaluation_list/market_query.txt'
    with open(image_list_path_gallery,'r') as f1, open(image_list_path_query,'r') as f2:
        gallery_path = f1.readlines()
        query_path = f2.readlines()

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)
    
    
    feature_gallery_path =  './evaluation_features/ReID10Dx_@50000_market_gallery.csv'
    feature_query_path = './evaluation_features/ReID10Dx_@50000_market_query.csv'
    
    import numpy as np
    gallery_feature = np.loadtxt(feature_gallery_path, delimiter=',')
    query_feature = np.loadtxt(feature_query_path, delimiter=',')

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    for i in range(len(query_label)):
        CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('top1:%f top5:%f top10:%f'%(CMC[0],CMC[4],CMC[9])


# In[ ]:


if __name__ == '__main__':
    main()

