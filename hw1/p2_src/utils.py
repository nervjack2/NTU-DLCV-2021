import torch
import numpy as np

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(7):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if tp_fp + tp_fn - tp == 0:
            print('class #%d : %1.5f'%(i, 0))
            continue
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))

    return mean_iou

def color_to_label(labels):
    labels = np.array(labels)
    if ((labels != 255) * (labels != 0)).sum():
        print('error')
    rgb_mask = (labels >= 128).astype(int)
    rgb_mask = 4*rgb_mask[:,:,0]+2*rgb_mask[:,:,1]+rgb_mask[:,:,2]
    y = np.empty(rgb_mask.shape,dtype=int)
    y[rgb_mask == 3] = 0 
    y[rgb_mask == 6] = 1
    y[rgb_mask == 5] = 2
    y[rgb_mask == 2] = 3
    y[rgb_mask == 1] = 4
    y[rgb_mask == 7] = 5 
    y[rgb_mask == 0] = 6
    y[rgb_mask == 4] = 6
    return y

def label_to_seg_img(label): 
    """
    label: numpy array, shape=(512,512)
    """
    ans = np.empty((512,512,3),dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            y = label[i,j]
            if y == 0:
                ans[i,j,:] = [0,255,255]
            elif y == 1:
                ans[i,j,:] = [255,255,0]
            elif y == 2:
                ans[i,j,:] = [255,0,255]
            elif y == 3:
                ans[i,j,:] = [0,255,0]
            elif y == 4:
                ans[i,j,:] = [0,0,255]
            elif y == 5:
                ans[i,j,:] = [255,255,255]
            elif y == 6:
                ans[i,j,:] = [0,0,0]
    return ans