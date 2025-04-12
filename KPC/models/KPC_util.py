import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import matplotlib.pyplot as plt
# import open3d as o3d
from sklearn.preprocessing import LabelEncoder
import scipy.spatial.distance as dist

def knn(x, k):
    k = k + 1
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def normal_knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def random_start_end_points(extracted_xyz):
    """
    从提取出的坐标信息中随机选择两个不同的点作为起始点和结束点

    参数：
        extracted_xyz: 提取出的坐标信息，形状为 (3, N)

    返回值：
        start_point: 起始点坐标，形状为 (3, 1)
        end_point: 结束点坐标，形状为 (3, 1)
    """
    while True:
        # 从坐标信息中随机选择两个不同的点的索引
        start_index = torch.randint(0, extracted_xyz.shape[1], (1,))
        end_index = torch.randint(0, extracted_xyz.shape[1], (1,))
        if start_index != end_index:
            break

    # 提取随机选择的起始点和结束点的坐标
    start_point = extracted_xyz[:, start_index]
    end_point = extracted_xyz[:, end_index]

    return start_point, end_point


def compute_curvature_knn_with_indices(point_cloud_knn):
    """
    计算点云数据的曲率，并返回每个点的曲率和索引，使用KNN获取的邻域点。

    参数：
        point_cloud_knn (np.array): 点云数据，形状为 (B, C, N, K)

    返回值：
        curvatures (np.array): 每个点的曲率，形状为 (B, N)
        indices (np.array): 每个点的索引，形状为 (B, N)
    """
    B, C, N, K = point_cloud_knn.shape
    curvatures = np.zeros((B, N))
    indices = np.zeros((B, N), dtype=int)

    for b in range(B):
        for n in range(N):
            # 获取当前点的邻域点
            neighbors = point_cloud_knn[b, :, n, :].T  # 形状为 (K, C)
            # 计算协方差矩阵
            covariance_matrix = np.cov(neighbors, rowvar=False)
            # 计算特征值
            eigenvalues, _ = np.linalg.eigh(covariance_matrix)
            # 曲率定义为最小特征值除以特征值之和
            curvature = eigenvalues[0] / np.sum(eigenvalues)
            curvatures[b, n] = curvature
            indices[b, n] = n  # 保存当前点的索引

    return curvatures, indices

def index_points(points, idx): #根据索引获取对应的点
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) * 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):#查询固定半径后，每个点的邻居点
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint)) #使用最远距离采样获取新的点坐标
    torch.cuda.empty_cache()

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()

    new_points = index_points(points, idx)#N，S，C
    torch.cuda.empty_cache()

    if returnfps:
        return new_xyz, new_points, idx
    else:
        return new_xyz, new_points

def sample_and_group_1(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points,idx

def farthest_sampling(data,r):
    b, n, _ = data.shape

    if n % 2 != 0:
        raise ValueError("每个样本的数据点数量 n 必须是偶数")

    start_points_batch = []
    end_points_batch = []

    # 对每个 batch 进行处理
    for i in range(b):
        sample_data = data[i]  # 取出第 i 个样本
        # 计算所有点对之间的距离
        dist_matrix = cdist(sample_data, sample_data)

        # 只取上三角部分的距离（不包括对角线）
        dist_matrix = np.triu(dist_matrix, k=1)

        # 将距离矩阵展平并排序，得到降序的索引
        dist_vector = dist_matrix.flatten()
        dist_vector[dist_vector >= r] = 0
        sorted_indices = np.argsort(-dist_vector)  # 降序排列索引

        used_rows = set()
        used_cols = set()

        start_points = []
        end_points = []

        # 遍历排序后的距离索引
        for idx in sorted_indices:
            row = idx // n
            col = idx % n

            # 确保行列索引不重复
            if row not in used_rows and col not in used_cols and row not in used_cols and col not in used_rows:
                start_points.append(row)
                end_points.append(col)
                used_rows.add(row)
                used_cols.add(col)

            # 如果找到了足够的点对（n//2个），就退出循环
            if len(start_points) >= 60:#50
                break

        start_point = sample_data[start_points]
        end_point = sample_data[end_points]
        start_point = start_point.unsqueeze(0)
        end_point = end_point.unsqueeze(0)

        start_points_batch.append(start_point)
        end_points_batch.append(end_point)

    s = np.concatenate(start_points_batch)
    e = np.concatenate(end_points_batch)
    return s,e