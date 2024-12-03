import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import image
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch 
import torch.nn.functional as F

path = r"D:\Work\codev2\code\work_code\model_2\final_19.9v1\C10CSHUY\090.bmp"



def viz(img):
    plt.figure(figsize=(15, 13))
    plt.imshow(img)
    plt.title('Original')
    plt.show()



def sharpen_image(image_path):
    img = cv2.imread(image_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    kernel = np.array(	
    [[-1, -2, -1],
     [-2, 4, 2],
     [1, 2, 1]], dtype=np.float32)
    
    sharpened = cv2.filter2D(img_rgb, -1, kernel)
    

    return sharpened


result = sharpen_image(path)

result = cv2.cvtColor(result, cv2.IMREAD_GRAYSCALE)

ret,thresh1 = cv2.threshold(result,200,255,cv2.THRESH_BINARY)

viz(thresh1)