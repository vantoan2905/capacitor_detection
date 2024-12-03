# %%
# %matplotlib widget
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import ultralytics 
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import shutil





def main():
    an_path = r"D:\Work\codev2\code\work_code\yolo_masterbox\20.9"

    save = r"D:\Work\codev2\code\work_code\yolo_masterbox\data"

    model = YOLO("yolov8x-obb.pt")
    result = model.train(cfg = r"D:\Work\codev2\code\work_code\yolo_masterbox\default.yaml")


    predict = r"D:\Work\codev2\data\first_data\19.9v2"
    test_path = r"D:\Work\codev2\code\work_code\yolo_masterbox\test_"
 



    model = YOLO(r"D:\Work\codev2\code\work_code\yolo_masterbox\runs\obb\train7\weights\best.pt")

    model.predict(test_path, save=True)

if __name__ == "__main__":
    main()




