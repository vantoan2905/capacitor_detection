import os
import pandas as pd
import json
import ast

prediction_path =  r"D:\Work\codev2\code\work_code\model_code\runs\detect\predict25\labels"

test_data = r"D:\Work\codev2\code\test_data"

list_file_ = []
temp = ""
for file in os.listdir(test_data):
    
    if file.endswith(".csv"):
        temp = file[:3]
        if file[:3] == temp:
            with open(os.path.join( test_data,file), 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    
                    list_2 = line.split('","')[-1].replace('"', '')  
                    list_file_.append([file,list_2[0:-1]])
                    break
                # break
            # list_file_.append([file])


list_file_ = pd.DataFrame(list_file_)
list_file_.to_csv("test.csv")