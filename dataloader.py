# -*- coding: utf-8 -*-
'''
@file: dataloader.py
@author: fanc
@time: 2025/6/16 13:04
'''
from torch.utils.data import Dataset, DataLoader
import os
import SimpleITK as sitk
import torch
import numpy as np
import pandas as pd
class LUAD2_3D(Dataset):
    def __init__(self, root, phase='train'):
        df = pd.read_csv(os.path.join(root, f'{phase}.csv'))
        self.root = root
        self.bids = df['bid'].tolist()
        self.phase = phase
        self.use_flip = 'train' in phase
        self.labels = torch.tensor(df['label'].apply(lambda x: 2 if x == 3 else x).tolist(), dtype=torch.long)
        # if self.use_slice:
        #     self.ts = transforms.ToTensor()
        cols = list(filter(lambda x: x.startswith('f'),  df.columns.tolist()))
        self.clinical = df[cols].fillna(0)
        # self.bbox = df['bbox64'].tolist()
        self.bbox = df['bbox'].tolist()
        self.use_bbox = False

    # @property
    # def getlabels(self):
    #     return [self.labels[i] for i in range(len(self))]

    def __len__(self):
        return len(self.bids)

    def __getitem__(self, i):
        # ct, mask, clinical, bbox = torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,))
        # ct, mask, clinical, bbox, slice, ct64, ct128, ct256, seg, radiomic = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # bbox32, bbox128 = 0, 0
        bid = self.bids[i]
        label = self.labels[i]
        ct = self.get_ct_32(i)
        clinical = self.get_clinical(i)
        bbox = self.bbox[i]
        if self.use_flip:
            ct, bbox = self.random_flip(ct, bbox)
        res = {'label': label, 'bid': bid, 'ct': ct, 'clinical': clinical, 'bbox': bbox}
        return res

    def get_ct_32(self, i):
        # if self.phase in ['train', 'val', 'all']:
        ct_path = os.path.join(self.root, 'cropped', '32sitk', f'{self.bids[i]}.nii.gz')
        # else:
            # ct_path = os.path.join()
            # pass
        # self.get_nii_file(ct_path)
        return self.get_nii_file(ct_path)

    def get_clinical(self, i):
        values = torch.tensor(self.clinical.iloc[i].values.tolist(), dtype=torch.float)
        return values#.unsqueeze(-1)

    def get_nii_file(self, file, normalize=True):
        if file.endswith('.nii.gz'):
            img = sitk.ReadImage(file)
            img = sitk.GetArrayFromImage(img).transpose(1, 2, 0)
        elif file.endswith('.npy'):
            img = np.load(file)
        if normalize:
            img = self.normalize(img)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    def random_flip(self, img, bbox=None):
        # 随机选择翻转轴
        flip_x = random.choice([True, False])
        flip_y = random.choice([True, False])
        flip_z = random.choice([True, False])

        # 对图像进行翻转
        if flip_x:
            img = torch.flip(img, dims=[2])  # 翻转X轴 (第3个维度，index为2)
            if self.use_bbox:
                bbox[0] = 1 - bbox[3] - bbox[0]  # 反转X坐标 (标准化坐标，无需图像尺寸)

        if flip_y:
            img = torch.flip(img, dims=[1])  # 翻转Y轴 (第2个维度，index为1)
            if self.use_bbox:
                bbox[1] = 1 - bbox[4] - bbox[1]  # 反转Y坐标

        if flip_z:
            img = torch.flip(img, dims=[0])  # 翻转Z轴 (第1个维度，index为0)
            if self.use_bbox:
                bbox[2] = 1 - bbox[5] - bbox[2]  # 反转Z坐标
        return img, bbox

    def normalize(self, img, WL=-600, WW=1500):
        MAX = WL + WW / 2
        MIN = WL - WW / 2
        img[img < MIN] = MIN
        img[img > MAX] = MAX
        img = (img - MIN) / WW
        return img

if __name__ == '__main__':

    root = '/zhangyongquan/fanc/datasets/LC1'
    traindataset = LUAD2_3D(root, phase=f'train_fold{0}')
    trainloader = DataLoader(traindataset, shuffle=True, batch_size=4, num_workers=0)
    for batch in trainloader:
        print(batch['ct'].shape, batch['clinical'].shape)
        break