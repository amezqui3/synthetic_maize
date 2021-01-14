
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def cal_width(row):
      n = len(row)
      n_start = n_end = 0
      flag = True
      for i in range(n):
            if row[i] and flag:
                  flag = False
                  n_start = i
            if row[i]:
                  n_end = i
      
      return n_end - n_start + 1
  
    
def cal_mid_point(row):
    '''
    calculate the middle point of a cross section of leaf width
    '''
    n = len(row)
    n_start = n_end = 0
    flag = True
    for i in range(n):
          if row[i] and flag:
                flag = False
                n_start = i
          if row[i]:
                n_end = i     
                
    return ( n_start + n_end ) / 2
       


def cal_max_leaf_height_width(im_folder):
    f_names = glob.glob(im_folder)
      
    w_max = 0
    h_max = 0
     
    for name in f_names:
        im = io.imread(name)
        h_im, w_im = im.shape
        if h_im < w_im: 
            im = im.T
            h_im, w_im = w_im, h_im
         
        if h_im > h_max: h_max = h_im
        if w_im > w_max: w_max = w_im
      
    return h_max, w_max
     
                  
            
if __name__ == '__main__':   
    
    
      
    dir_msk = '../split_grayscale/*.tif'
    dir_out = '../width_hist'
           
      
    if not os.path.exists(dir_out): 
        os.makedirs(dir_out)

      
    f_names = glob.glob(dir_msk)
    
    h_max, w_max = cal_max_leaf_height_width(dir_msk)
    
    
    width_array = np.zeros((len(f_names), h_max)) # each row: width historgram
    symmetry_list = []
      
    for idx, name in enumerate(f_names[1:3+1]):
        im = io.imread(name)
        
        h, w = im.shape
        if h>w: 
            im = im.T
            h, w = w, h
        
        msk = im > 0
                       
        if cal_width( msk[:, int(w/4)] ) > cal_width( msk[:, int(w*3/4)] ):
              msk = msk[:, ::-1]
              
        basename = os.path.basename(name)[:11]
        
        out_name = os.path.join(dir_out, basename + '_sym.png')
        
               
        cutoff = int(w/8)
        mid_point_list = np.array( [cal_mid_point(msk[:,i]) for i in range(cutoff + 1, w - cutoff + 1)] )          
        offset_list = ( mid_point_list - np.mean(mid_point_list) ) / w_max
        symmetry_index = np.mean( np.absolute(offset_list) )
        
        symmetry_list.append( symmetry_index )
        
        plt.figure(figsize=[16,12])
        plt.imshow((msk*255).astype('uint8'), cmap='gray')
        plt.hlines(np.mean(mid_point_list), cutoff + 1, w - cutoff, color='r', linewidth=2)
        plt.plot(range(cutoff + 1, w - cutoff + 1), mid_point_list, c='b', linewidth=2)
        plt.xlabel('x',fontsize=15)
        plt.ylabel('y',fontsize=15)        
        plt.savefig(out_name)
        
        plt.close('all')

  
          
          
         


