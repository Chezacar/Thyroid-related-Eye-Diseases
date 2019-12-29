import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_alignment
import os

def pad(img,padding_ratio):
    h,w,_ = img.shape
    h,w = (int(h*padding_ratio),int(w*padding_ratio))
    return cv2.copyMakeBorder(img,h,h,w,w,cv2.BORDER_CONSTANT)

def iter_process_dir(root_dir,padding_ratio,input_ratio):
    #root_dir should end without '/'
    print('start...')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    root_dir_len = len(root_dir)
    out_root_dir = root_dir + '_out/standard'
    padout_root_dir = root_dir + '_out/pad'
    for root,dirs,files in os.walk(root_dir):
        count = 0
        out_root = out_root_dir + root[root_dir_len:]
        padout_root = padout_root_dir + root[root_dir_len:]
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        if not os.path.exists(padout_root):
            os.makedirs(padout_root)
        imgs = [i for i in files if i.endswith('jpg') or i.endswith('JPG')]
        for i in imgs:
            ori_imgpath = os.path.join(root,i)

            out_imgpath = os.path.join(out_root,i)
            padout_imgpath = os.path.join(padout_root,i)
            
            if os.path.exists(out_imgpath) and os.path.exists(padout_imgpath):
                continue
            ori_out,pad_out = process_img(fa,ori_imgpath,padding_ratio,input_ratio)
            if not os.path.exists(out_imgpath):
                cv2.imwrite(out_imgpath,ori_out)
            if not os.path.exists(padout_imgpath):
                cv2.imwrite(padout_imgpath,pad_out)
            count += 1
            if count%20 == 0:
                print(count)

def process_img(fa,img_path,padding_ratio,input_ratio):
    img = cv2.imread(img_path)
    img_padded = pad(img,padding_ratio)
    img_padded_RGB = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    padded_h,padded_w,_ = img_padded_RGB.shape
    input_h,input_w = int(padded_h*input_ratio),int(padded_w*input_ratio)
    input_RGB = cv2.resize(img_padded_RGB,(input_w,input_h))

    pred = fa.get_landmarks(input_RGB)[0]

    added_h,added_w = int(img.shape[0]*padding_ratio),int(img.shape[1]*padding_ratio)
    pred_ori = pred/input_ratio
    pred_ori[:,0] = pred_ori[:,0] - added_w
    pred_ori[:,1] = pred_ori[:,1] - added_h

    pad_out = cv2.cvtColor(input_RGB,cv2.COLOR_RGB2BGR)
    ori_out = img.copy()

    for p in pred:
        cv2.circle(pad_out,(p[0],p[1]),6,(255,255,255),-1)
    for p in pred_ori:
        cv2.circle(ori_out,(p[0],p[1]),20,(255,255,255),-1)
    return ori_out,pad_out

root_dir = '/DATA5_DB8/data/zdcheng/hyperthyreosis_eye/2017wardTAO'
iter_process_dir(root_dir,padding_ratio=1,input_ratio = 0.1)

