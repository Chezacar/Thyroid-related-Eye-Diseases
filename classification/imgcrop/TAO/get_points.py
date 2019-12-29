
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_alignment
import os


# In[2]:


def pad(img,padding_ratio):
    '''
    :param img: an image, BGR or RGB
    :param padding_ratio: padding width/original width
    :return: padded img
    '''
    h,w,_ = img.shape
    h,w = (int(h*padding_ratio),int(w*padding_ratio))
    return cv2.copyMakeBorder(img,h,h,w,w,cv2.BORDER_CONSTANT)

def process_img(fa,img_path,padding_ratio,input_ratio):
    img = cv2.imread(img_path) # an BGR img
    img_padded = pad(img,padding_ratio) # padded BGR img
    img_padded_RGB = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)# padded RGB img

    padded_h,padded_w,_ = img_padded_RGB.shape
    input_h,input_w = int(padded_h*input_ratio),int(padded_w*input_ratio)
    input_RGB = cv2.resize(img_padded_RGB,(input_w,input_h)) # resized padded RGB img

    pred = fa.get_landmarks(input_RGB)[0]

    added_h,added_w = int(img.shape[0]*padding_ratio),int(img.shape[1]*padding_ratio)
    pred_ori = pred/input_ratio
    pred_ori[:,0] = pred_ori[:,0] - added_w
    pred_ori[:,1] = pred_ori[:,1] - added_h
    pred_ori = pred_ori.astype(np.int)
#     print(pred_ori)
    return pred_ori


# In[4]:


selected_root = '/DATA5_DB8/data/zdcheng/hyperthyreosis_eye/selected/'

img_list = os.listdir(selected_root)
img_list = [i for i in img_list if i.endswith('jpg') or i.endswith('JPG')]
all_img_list = [selected_root + i for i in img_list]


# In[5]:


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
padding_ratio = 1
input_ratio = 0.3

pred_list = []

count = 0
for img in all_img_list:
    print(count)
    count += 1
    pred_ori = process_img(fa,img,padding_ratio,input_ratio)
    pred_list.append(pred_ori)


# In[6]:


pred_dict_json = dict()
pred_dict_pk = dict()
for idx,img in enumerate(img_list):
    pred = pred_list[idx]
    tmp_list = []
    for p in pred:
        pair = []
        for num in p:
            pair.append(int(num))
        tmp_list.append(pair)
    pred_dict_json[img] = tmp_list
    pred_dict_pk[img] = pred


# In[7]:


import json
output_json = selected_root + 'points.json'
with open(output_json,'w') as f:
    json.dump(pred_dict_json,f,indent=4)


# In[8]:


import pickle
output_pk = selected_root + 'points.dict'
with open(output_pk,'wb') as f:
    pickle.dump(pred_dict_pk,f)

