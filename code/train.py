import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
import pickle


def calc_logist_reg( im, mask):
    
    data = im2input(im)    
    label = (mask.ravel()).astype(int)
    logr = LogisticRegression(class_weight='balanced', max_iter=10000)
    logr.fit( data, label)
    return logr


def im2input(im, msk_radius=5):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    shift_list = []
    for i in range(-msk_radius,msk_radius+1):
        for j in range(-msk_radius,msk_radius+1):
            if not i==j==0:
                shift_list.append((i,j))
                
    
    #shift_list =  [(-1,0), (0,-1), (0,1), (1,0)]
    n_shifts =  len(shift_list)   
    n_channels = n_shifts + 1
    
    im_neighbor = im[..., None]
    
    for shift in shift_list:
        im_shifted = shift_im(im, shift[0], shift[1])
        im_shifted = im_shifted[..., None]
        im_neighbor = np.concatenate( (im_neighbor, im_shifted), axis=2 )
    
    data = im_neighbor.reshape((-1,n_channels)).astype(float)
    
    return data


def shift_im(im, shift_x, shift_y):
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])    
    (rows, cols) = im.shape[:2] 
    out = cv2.warpAffine(im, M, (cols, rows))   
    return out


if __name__ == '__main__':
    folder_im = 'images_small'   
    f_msk1 = os.path.join(folder_im, 'IMAG0013_mask1.png')
    f_msk2 = os.path.join(folder_im, 'IMAG0013_mask2.png')
    f_im = os.path.join(folder_im, 'IMAG0013.jpg')
            
    f_model = './model.p'
    
    msk1, msk2 = cv2.imread(f_msk1), cv2.imread(f_msk2)
    im = cv2.imread(f_im)
        
    msk = np.logical_or( msk1[...,0]==255, msk2[...,0]==255 ).astype('uint8')
    
    
    logr = calc_logist_reg(im, msk)
    
    f = open(f_model,'wb')
    pickle.dump(logr, f)
    f.close()

    
    
    
    

    
    

    
    