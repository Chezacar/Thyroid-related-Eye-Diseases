import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np

class TAO(data.Dataset):
    def __init__(self, ann_file, img_dir, mode, val_fold = None, fake=False):
        self.img_dir = img_dir
        self.mode = mode
        self.fake = fake
        self.ann_file = ann_file

        # read a img data list from json file
        with open(self.ann_file, 'r') as f:
             self.imgdata_list = json.load(f)

        # deal with fold 0~3 when the mode is 'train' or 'val'
        if mode == 'test':
            pass
        else:
            self.val_fold = val_fold
            assert type(self.val_fold) == int and 3 >= self.val_fold >= 0
            if mode == 'train':
                self.imgdata_list = [i for i in self.imgdata_list if i['fold'] != self.val_fold]
            elif mode == 'val':
                self.imgdata_list = [i for i in self.imgdata_list if i['fold'] == self.val_fold]
            else:
                raise RuntimeError('Invalid mode!')

        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]

        self.crop_aug = transforms.RandomCrop(224,0,False)
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.center_crop = transforms.CenterCrop(224)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

        print('Loaded annotation from {}'.format(ann_file))
        print('Images number: {}'.format(len(self.imgdata_list)))

    def __getitem__(self, index):
        imgdata = self.imgdata_list[index]
        img_path = os.path.join(self.img_dir,imgdata['img'])
        img_path = img_path[:-3] + 'npy'
        label = imgdata['label']

        img_array = np.load(img_path)
        img = Image.fromarray(img_array,mode='RGB')
        if self.mode == 'train':
            img = self.crop_aug(img)
            img = self.flip_aug(img)
        else:
            img = self.center_crop(img)
        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return img,label

    def __len__(self):
        return len(self.imgdata_list)

