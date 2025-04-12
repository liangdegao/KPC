import torch
import numpy as np
import h5py
import os
from tqdm import tqdm
from data import ShapeNetPart
from RRT_datadeal import curve_deal
from torch.utils.data import DataLoader


def save_groups(train_loader, output_file):
    with h5py.File(output_file, 'w') as hf:
        group_total = []
        for data, label, seg in tqdm(train_loader, desc='Processing data'):
            data_1 = torch.tensor(data, dtype=torch.float32)
            group = curve_deal(data_1)
            if isinstance(group, list):
                group_np = np.concatenate([np.array(g) for g in group], axis=0)
            else:
                group_np = np.array(group)

            group_total.append(group_np)

            # Concatenate all groups and save as a dataset in the HDF5 file
        group_total_np = np.concatenate(group_total, axis=0)
        hf.create_dataset('groups', data=group_total_np)


if __name__ == '__main__':
    train_dataset = ShapeNetPart(partition='./data/hdf5_data', num_points=2048,
                                 class_choice=None)  # Adjust parameters as needed
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=2, shuffle=True, drop_last=True)
    save_groups(train_loader, '').device('cpu')