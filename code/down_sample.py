import cv2
import glob
import os


if __name__=='__main__':
    folder_im = 'Leaves2018'
    folder_im_small = 'images_small'
    
    im_names = glob.glob( os.path.join(folder_im, '*.jpg') ) 
    scale = 16
    
    for path in im_names:
        im = cv2.imread(path)
        
        h, w = im.shape[0:2]        
        im_small = cv2.resize(im, ( int(w/scale), int(h/scale) ) )
        
        basename = os.path.basename(path)
        path_out = os.path.join(folder_im_small, basename)
        
        cv2.imwrite(path_out, im_small)
        