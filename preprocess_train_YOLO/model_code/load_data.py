import os 
import shutil

def move_data(an_path, new_path):
    for folder in os.listdir(an_path):
        for json in os.listdir(os.path.join(an_path, folder)):
            if json.endswith(".json"):
                shutil.copy(os.path.join(an_path, folder, json), new_path)
            if json.endswith(".bmp"):
                shutil.copy(os.path.join(an_path, folder, json), new_path)
an_path = "D:/Work/codev2/data/final_data/final_data_19.9v1"
new_path = "D:/Work/codev2/code/work_code/model_code/yolo_data"

move_data(an_path, new_path)