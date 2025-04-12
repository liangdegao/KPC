from models.KPC_util import index_points,farthest_point_sample,query_ball_point,farthest_sampling
from models.ops import group
import numpy as np
from models.RRT_1 import rrt_star_m
import random
import torch
import laspy
from einops import rearrange


def rand():
    random_value = random.uniform(0.2, 0.4)

    # 保留到小数点后两位
    random_value_rounded = round(random_value, 2)

    return random_value_rounded
def rrt_walk(start_point_3,end_point_3, xyz_1, xyz, curve_num, curve_pointnum,xunhuan):
    path_total = []
    # path_vis = []
    c = 0
    # vis_point_cloud_2(xyz_1.squeeze(0),start_point_3,end_point_3)
    distance = np.linalg.norm(start_point_3-end_point_3, axis=1)
    while len(path_total) < curve_num:
        step = distance / 10
        search = step * 2.
        path_1 = rrt_star_m(np.array(start_point_3), np.array(end_point_3), xyz_1, num_iterations=120, search_radius=search,
                            step_size=step)
        i = len(path_1)
        path_total = []
        c = c + 1
        for i in range(i):
            path = path_1[i]
            path = np.reshape(path, (-1, 3))
            num = path.shape[0]
            if num >= curve_pointnum: #线上点的个数大于3
                path_total.append(path)
                if len(path_total) >= curve_num: #线的条数2
                    break
    path_2 = path_total[0]
    path_2 = torch.tensor(np.expand_dims(path_2, axis=0))
    path_2 = index_points(path_2, farthest_point_sample(path_2, npoint=curve_pointnum))  # 一条线几个点
    for j in range(len(path_total) - 1):
        path_3 = torch.tensor(np.expand_dims(path_total[j + 1], axis=0))
        if path_3.shape[1] > curve_pointnum:  # 判断一条线上的点是否大于3
            path_3 = index_points(path_3, farthest_point_sample(path_3, npoint=curve_pointnum))  # 一条线几个点
        path_total_1 = torch.cat([path_2, path_3], dim=0)
        path_2 = path_total_1
    path_2 = path_2.view(1, -1, 3)  # c,n,t
    return path_2
def curve(xyz, xyz_num, npoint, curve_num, curve_pointnum,r):
    b, _, _ = xyz.size()
    group_3 = []
    group_4 = []
    for i in range(len(xyz_num)):
        num = xyz_num[i]
        xyz_norm = xyz.clone()
        xyz = index_points(xyz.transpose(1, 2), farthest_point_sample(xyz.transpose(1, 2), npoint=num)).transpose(1,2)
        start_point = index_points(xyz.transpose(1, 2), farthest_point_sample(xyz.transpose(1, 2), npoint=npoint))
        start_point_2, end_point_2 = farthest_sampling(start_point,r=r[i])
        for q in range(b):
            start_point_3 = start_point_2[q, :, :]
            end_point_3 = end_point_2[q, :, :]
            xyz_norm_1 = xyz_norm[q, :, :].unsqueeze(0)
            xyz_1 = xyz[q, :, :].unsqueeze(0)
            group_1 = rrt_walk(start_point_3, end_point_3, xyz_1, xyz_norm_1, curve_num=curve_num,
                               curve_pointnum=curve_pointnum,xunhuan=1)  # 进算法
            while (group_1.shape[1] < int(curve_num * curve_pointnum)):   # 一共有几个点
                group_1 = rrt_walk(start_point_3, end_point_3, xyz_1, xyz_norm_1, curve_num=curve_num,
                                   curve_pointnum=curve_pointnum,xunhuan=i)
            if q == 0:
                group_3.append(group_1)
            else:
                group_4.append(group_1)

    group_3 = torch.stack(group_3,dim=0).squeeze(1)
    group_4 = torch.stack(group_4, dim=0).squeeze(1)

    group_total = torch.cat((group_3,group_4),dim=0)
    return group_total

def read_pts_file_1(file_path):
    """
    读取.pts点云文件
    """
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 忽略空行
                point = [float(x) for x in line.strip().split()]
                points.append(point)
    return np.array(points)

def curve_deal(point):
    point_num = [512,256]
    r = [0.3,0.6,0.8,1.2]
    group_1 = curve(torch.tensor(point.transpose(1, 2), dtype=torch.float32), point_num, 256, curve_num=40, curve_pointnum=4,r=r)

    return group_1
