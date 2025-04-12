import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = ''
DATA_DIR_1 = ''

def load_data_cls(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_group_data(partition):
    all_group = []
    all_group_1 = []
    all_data = []
    all_label = []
    all_seg = []

    if partition == 'trainval':
        files = glob.glob(os.path.join(DATA_DIR, '*train*.h5'))
    else:
        files = glob.glob(os.path.join(DATA_DIR, f'*{partition}*.h5'))

    for h5_name in files:
        with h5py.File(h5_name, 'r') as file:
            group_data = file['groups'][:]
            group_data_1 = file['groups_1'][:]
            data = file['data'][:]
            seg = file['seg'][:]
            label = file['label'][:]


        all_group.append(group_data)
        all_group_1.append(group_data_1)
        all_data.append(data)
        all_seg.append(seg)
        all_label.append(label)

    all_group = np.concatenate(all_group, axis=0)
    all_group_1 = np.concatenate(all_group_1, axis=0)
    all_data = np.concatenate(all_data, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_group, all_group_1, all_data, all_seg, all_label

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition

        # 加载 group_total.h5 数据集
        if partition == '':
            self.group_data = self.load_group_data(
                './data/curve_data/group_train_Modelnet.h5')
        else:
            self.group_data = self.load_group_data(
                './data/curve_data/group_test_Modelnet.h5')

    def load_group_data(self, file_path):
        with h5py.File(file_path, 'r') as file:
            group_data = file['groups'][:]  # 确保这里的键名与实际文件匹配
        return group_data

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        # 确保 group 的索引正确并扩展维度
        group_index = item // 2  # 每个 group 数据项对应两个点云数据项
        group = self.group_data[group_index]

        # 将 group 的第一个维度扩展为 batch_size 的两倍
        group = np.tile(group, (2, 1))
        if self.partition == '':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = rotate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label, group

    def __len__(self):
        return self.data.shape[0]

class ShapeNetPart(Dataset):
    def __init__(self, num_points=2048, partition='',
                 class_choice=None):
        self.group, self.group_1, self.data, self.seg, self.label = load_group_data(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice

        if self.class_choice is not None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        group = self.group[2*item:2*item+2]
        group_1 = self.group_1[2*item:2*item+2]
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg, group, group_1

    def __len__(self):
        return self.data.shape[0]
