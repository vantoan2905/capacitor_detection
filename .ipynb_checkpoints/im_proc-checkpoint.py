import os
import json


import cv2
import numpy as np


# def get_ann_template(im_path, imH, imW):
#     f = open(ann_template_path, 'r')
#     template = json.load(f)
#     f.close()
#     template['imagePath'] = os.path.basename(im_path)
#     template['imageHeight'] = imH
#     template['imageWidth'] = imW
#     template['shapes'] = []
#     return template



def viz_label(im_path):
    im_rgb = cv2.imread(im_path, 1)
    im = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

    bin_thres =  np.mean(im)
    _, thres = cv2.threshold(im, bin_thres, 255, cv2.THRESH_BINARY)


    ks = 15
    thres = cv2.erode(thres, np.ones((ks, ks)))

    ks = 51
    thres = cv2.dilate(thres, np.ones((ks, ks)))

    cnts, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

    thres = cv2.fillPoly(thres, [cnts[0]], 1)
    thres = im * thres

    ks = 27
    thres = cv2.GaussianBlur(thres, (ks, ks), 0)
    thres = np.where(thres > np.mean(thres), 0, thres)
    thres = np.where(thres > np.mean(thres[thres>0]), 0, thres)

    ks = 27
    thres = cv2.GaussianBlur(thres, (ks, ks), 0)
    thres = np.where(thres < np.mean(thres[thres>0]), 0, thres)



    thres = cv2.dilate(thres, np.ones((3, 17)))


    cnts, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_area_list = []
    for cnt in cnts:
        cnt_area_list.append(cv2.contourArea(cnt))


    area_mean = np.mean(cnt_area_list)
    lower_bound = area_mean - area_mean*1/3
    upper_bound = area_mean + area_mean*1/3

    found_cnt = 0
    ann_data = get_ann_template(im_path, *im.shape[:2])
    for cnt_id, cnt in enumerate(cnts):
        if (cnt_area_list[cnt_id] < lower_bound) or (cnt_area_list[cnt_id] > upper_bound):
            continue
        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(im_rgb, (x,y), (x+w,y+h), (0, 255, 0), 2)
            ann_data['shapes'].append({
                "label": "resitor",
                "points": [
                    [x, y],
                    [x+w, y+h]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })
            found_cnt += 1

    save_path_prefix = os.path.splitext(im_path)[0]
    with open(f'{save_path_prefix}.json', 'w') as fp:
        json.dump(ann_data, fp, indent=4)

    thres = thres / np.max(thres) * 255

    cv2.imwrite(f'{im_path}.jpg', thres)
    cv2.imwrite(f'{im_path}_box.jpg', im_rgb)

    print("found_cnt", found_cnt)




if __name__ == "__main__":

    im_path = r"D:\zeiss\data_taiyo\Image_20240915170841363.bmp"
    im_path = r"D:\zeiss\data_taiyo\Image_20240917112908739.bmp"
    im_path = r"D:\Work\data\test\039.bmp"

    ann_template_path = r"D:\Work\data\test\039.json"
    viz_label(im_path)

