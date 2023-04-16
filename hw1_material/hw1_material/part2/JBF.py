import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        padded_guidance = cv2.normalize(padded_guidance, 
                                        None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
        
        padded_img = cv2.normalize(padded_img, None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
        
        
        ### TODO ###
        #Range kernel table
        #-(Tp - Tq)^2 / (2sigma_r^2)
        #Tp: Totle of region ixel value
        #Tq: centerixel value
        #Index by abs(Tp-Tq)
        Range_kernel = np.exp(-np.arange(256) * np.arange(256) / 
                              (2 * self.sigma_r ** 2))
        
        #Spacial kernel table
        #Index by abs(Tp-Tq)
        x, y = np.meshgrid(np.arange(2 * self.pad_w + 1) - self.pad_w
                       , np.arange(2 * self.pad_w + 1) - self.pad_w)
    
        Spacial_kernel = np.exp(-(x * x + y * y) 
                      / (2 * self.sigma_s ** 2))
        

        
        
        #Determine the img format
        output = np.zeros_like(img)
        
        if img.ndim == 2:
            spliding = self.wndw_size
            center = (spliding-1) /2
            
            Tp_box = np.lib.stride_tricks.sliding_window_view(padded_guidance, 
                                                     (spliding, spliding))
            img_box = np.lib.stride_tricks.sliding_window_view(padded_img, 
                                                     (spliding, spliding))
            
            #Tp_box = img_sliding
            Tq_boxCenter = Tp_box[:,:,center,center]
            Tq_boxCenter = np.expand_dims(np.expand_dims(Tq_boxCenter, axis=2), axis=3)
            
            wgt = Range_kernel[np.abs(Tp_box - Tq_boxCenter)] * Spacial_kernel
            
            output = np.sum(wgt * img_box ,axis = (2,3)) / np.sum(wgt, axis = (2,3))
        
        else:
            Range_kernel = np.exp(-np.arange(256) * np.arange(256) / (2*self.sigma_r**2))
            Range_kernel = np.vstack([Range_kernel,
                                      Range_kernel,
                                      Range_kernel])
            
            spliding = self.wndw_size
            center = (spliding-1) /2
            
            Tp_box = [np.lib.stride_tricks.sliding_window_view(img, 
                                                     (spliding, spliding, 1))]
            #Tq_box = Tp_box[:,:,center,center]
            #Tq_box = np.expand_dims(np.expand_dims(Tq_box, axis=2), axis=3)
            
            
            
            
            
        
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
def Spacial_kernel(pad_w, sigma_s):
    #generate xp-xq table
    x, y = np.meshgrid(np.arange(2 * pad_w + 1) - pad_w
                       , np.arange(2 * pad_w + 1) - pad_w)
    #generate kernel table
    kernel = np.exp(-(x * x + y * y) 
                      / (2 * sigma_s ** 2))
    
    return kernel

def cov(img, kernel):
    shape =  kernel.shape()
    
    result = (np.lib.stride_tricks.sliding_window_view(img, shape) 
              * kernel).reshape
    return result
    
    