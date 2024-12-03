# %%
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# %%
save_path = r"D:\Work\codev2\code\work_code\model_2\output"
path_csv = r"D:\Work\codev2\code\work_code\model_2\final_19.9v1\C05 VS-W\053.csv"
csv_data = pd.read_csv(path_csv)
for index, row in csv_data.iterrows():
    path = r"D:\Work\codev2\code\work_code\model_2\final_19.9v1\C05 VS-W\053.bmp"
    img = cv2.imread(path)
    bbox1 = np.float32([[row['Top_Left_X'],row['Top_Left_Y']], [row['Bottom_Right_X'],row['Bottom_Right_Y']], [row['Top_Right_X'],row['Top_Right_Y']], [row['Bottom_Left_X'], row['Bottom_Left_Y']]])
    
    width, height = math.sqrt((row['Top_Left_X']-row['Top_Right_X'])**2+(row['Top_Left_Y']-row['Top_Right_Y'])**2),\
        math.sqrt((row['Top_Left_X']-row['Bottom_Left_X'])**2+(row['Top_Left_Y']-row['Bottom_Left_Y'])**2)
    # print(width, height)
    width, height = int(width), int(height)
    bbox2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(bbox1, bbox2)

    dst = cv2.warpPerspective(img, M, (width, height))

    plt.imshow(dst)
    plt.imsave(os.path.join(save_path, path[-7:-4] + "_" +str(index)+".bmp"), dst)
    plt.show()
    

# %%


# %%



