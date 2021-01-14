import cv2
import pickle
import os
import numpy as np
import glob
from train import im2input


def instance_seg(im, model):   
    data =  im2input(im)    
    prob_target = model.predict_proba( data )[:,1]
    prob = prob_target.reshape(im.shape[0:2])
    
    msk = np.zeros_like(prob, 'uint8')
    instance = np.zeros_like(prob, 'uint8')
    
    msk[prob>0.5]=255
    msk[prob<=0.5]=0

    connectivity = 4  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(msk , connectivity , cv2.CV_32S)
    
    
    areas = np.sort(stats[1:,4])   # excluding background
   
    intensity_list = [127, 255]
    
    thres_area = 0.1 * instance.shape[0] * instance.shape[1]
    
    for i, intensity in enumerate(intensity_list):
        idx = np.where(stats[:,4] == areas[-i-1])
        if np.sum(labels == idx[0]) > thres_area:
            instance[ labels == idx[0] ] = intensity
        
    return instance



if __name__ == '__main__':
    folder_im = 'images_small'
    folder_out = 'instance_label'
    model = pickle.load(open('./model.p','rb'))   
    im_names = glob.glob(os.path.join(folder_im, '*.jpg'))
    
    for name in im_names:
        
        im = cv2.imread(name)      
        instance = instance_seg(im, model)
        path_out = os.path.join(folder_out, os.path.basename(name) )
        cv2.imwrite(path_out, instance)
        
    
    
    
    
