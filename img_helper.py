from math import pi, sin, cos
from scipy import misc
import numpy as np

def getPaddingNum(total,gap):
    div = total / gap
    if div == int(div):
        return 0
    else:
        return (int(div) + 1) * gap - total

def imgSegmentation(img):
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    h_gap = 200
    w_gap = 200
    h_cnt = h / h_gap
    w_cnt = w / w_gap
    # 在高度上padding
    if h_cnt != int(h_cnt):
        padding_matrix = [255] * getPaddingNum(h,h_gap) * w
        padding_matrix = np.array(padding_matrix)
        padding_matrix = np.reshape(padding_matrix,(getPaddingNum(h,h_gap),w))
        img = np.vstack((img,padding_matrix))
        h = img.shape[0]
        h_cnt = int(h_cnt)+1
    # 在宽度上padding
    if w_cnt != int(w_cnt):
        padding_matrix = [255] * h * getPaddingNum(w,w_gap)
        padding_matrix = np.array(padding_matrix)
        padding_matrix = np.reshape(padding_matrix,([h,getPaddingNum(w,w_gap)]))
        img = np.concatenate((img,padding_matrix),1)
        w = img.shape[1]
        w_cnt = int(w_cnt)+1
    i = 0
    imgs = []
    while(i < h):
        h_next = min(i+h_gap, h)
        j = 0
        while(j < w):
            w_next = min(j+w_gap, w)
            imgs.append(img[i:h_next,j:w_next])
            j = w_next
        i = h_next
    imgs = np.reshape(imgs, [h_cnt,w_cnt,h_gap,w_gap])
    return np.array(imgs)

def imgMerge(imgs):
    h,w,_,__ = imgs.shape
    res = []
    for i in imgs:
        res.append(np.concatenate(i,axis = 1))
    return np.vstack(res)

def newPredict(img_path, predict_func, output_path, crf):
    img = misc.imread(img_path, mode='L')
    h, w = img.shape
    # 切图
    imgs = imgSegmentation(img)
    predictions_all = []
    for i in imgs:
        # 将灰度值改为单通道
        i = np.expand_dims(i, axis=3)
        i = np.expand_dims(i, axis=0)
        predictions = predict_func([i])[0]
        predictions = np.reshape(predictions, (i.shape[0], i.shape[1], cfg.class_num))
        print(predictions.shape)
        predictions_all.append(predictions)
    
    print(predictions_all)
    predictions = np.resize(predictions_all, (h, w, cfg.class_num))