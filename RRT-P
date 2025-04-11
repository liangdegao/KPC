import numpy as np
from scipy.spatial import KDTree
import concurrent.futures
import random

def get_point_cloud_bounds(point_cloud):
    """计算点云数据的边界范围（最小和最大坐标）。"""
    point_cloud = np.array(np.reshape(point_cloud, (1, -1)))
    min_bounds = np.min(point_cloud, axis=1)
    max_bounds = np.max(point_cloud, axis=1)
    return min_bounds, max_bounds

class Node:
    """定义节点类，代表每一个搜索点。"""
    def __init__(self, point, cost=0, parent=None):
        self.point = np.array(point, dtype=np.float32)
        self.cost = float(cost)
        self.parent = parent

def distance(point1, point2):
    """计算两点之间的欧几里得距离。"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest(nodes, q_rand):
    """在现有节点中找到离随机点q_rand最近的节点。"""
    return min(nodes, key=lambda node: distance(node.point, q_rand))


def cosine_similarity(vector1, vector2):
    """计算两个向量之间的余弦相似度，处理零向量的情况。"""
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        # 如果任意一个向量是零向量，返回 0 或 NaN（根据需求选择）
        return 0  # 或者 return np.nan

    return np.dot(vector1, vector2) / (norm1 * norm2)

def steer(q_near, q_rand, step_size, xyz, goal_direction):
    """从q_near朝向q_rand方向生成一个新的节点，但距离不超过step_size。"""
    q_near_1 = q_near.point
    direction = np.array(q_rand) - np.array(q_near.point)
    length = np.linalg.norm(direction)
    if length == 0:
        direction = direction
        length = 0.06
    else:
        direction = direction / length
    cos = cosine_similarity(direction.squeeze(0), goal_direction.squeeze(0))
    if -1 < cos < -0.2:#-0.2
        direction = goal_direction / 2
    length = min(step_size, length)
    new_point = q_near.point + direction * length
    point = np.expand_dims(np.array(find_nearest_point(xyz, new_point)), axis=0)
    return Node(point)

def is_within_point_cloud(point, node,tolerance=0.0001):
    """检查给定的节点是否在点云数据中。"""
    node = node.point
    point = np.array(point.transpose(0,1))
    # 计算点与点云中所有点的差异
    distances = np.linalg.norm(point - node, axis=1)
    index = np.argmin(distances)

    # 如果最小的距离小于一个很小的阈值（例如 1e-6），认为点在点云中
    return np.min(distances) < tolerance

def is_within_point_cloud_1(point, node,tolerance=0.0001):
    """检查给定的节点是否在点云数据中。"""
    point = np.array(point.transpose(0,1))
    # 计算点与点云中所有点的差异
    distances = np.linalg.norm(point - node, axis=1)
    index = np.argmin(distances)

    # 如果最小的距离小于一个很小的阈值（例如 1e-6），认为点在点云中
    return np.min(distances) < tolerance

def calculate_unit_vector(point1, point2):
    """计算从point1到point2的单位向量。"""
    vector = np.array(point2) - np.array(point1)
    length = np.linalg.norm(vector)
    unit_vector = vector / length
    return unit_vector

def find_nearest_point(point_cloud, path_point):
    """在点云数据中找到离指定点最近的一个点，并用它替换路径点。"""

    point_cloud = point_cloud.transpose(0,1)
    # 计算每个点到目标点的欧氏距离
    distances = np.linalg.norm(point_cloud - path_point, axis=1)

    # 找到距离最小的点的索引
    closest_point_index = np.argmin(distances)

    # 返回最接近的点
    closest_point = point_cloud[closest_point_index]
    return closest_point

def find_path(nodes, start, goal, goal_threshold):
    """寻找从起点到终点的路径。"""
    goal_node = min([node for node in nodes if distance(node.point, goal) < goal_threshold], key=lambda n: n.cost, default=None)
    # goal_node = min(nodes, key=lambda n: distance(n.point, goal), default=None)
    path = []
    if goal_node is None:
        return path
    while goal_node is not None:
        path.append(tuple(goal_node.point)) 
        goal_node = goal_node.parent
    return path[::-1]

def rrt_star_m(start_points, end_points, point_cloud, num_iterations , search_radius , step_size, tolerance=0.0001, tor=0.1):
    """
    执行RRT*算法以找到起点到终点的路径。
    """
    point_cloud = point_cloud.squeeze(0)
    # kdtree = KDTree(point_cloud.T)  # 构建KD树
    # vis_point_cloud_1(point_cloud.transpose(0,1))
    all_paths = []
    a = 0
    for i in range(start_points.shape[0]):
        start = np.expand_dims(start_points[i, :],axis=0)
        goal = np.expand_dims(end_points[i, :],axis=0)
        path = []
        d = 0
        search_radius_1 = search_radius[i]
        step_size_1 = step_size[i]
        while len(path) < 2: # 初始化节点列表，包含起点
            used_indices = set()
            nodes = [Node(start)]
            goal_direction = calculate_unit_vector(start, goal)
            for _ in range(num_iterations):
                # q_rand = np.random.uniform(min_1, max_1, 3)  # 随机生成点
                # q_rand = point_cloud[:, np.random.randint(point_cloud.shape[1])]
                while True:
                    index = np.random.randint(point_cloud.shape[1])
                    if index not in used_indices:
                        used_indices.add(index)
                        break
                q_rand = point_cloud.transpose(0,1)[index,:]# 从点云数据中随机抽取一个点
                q_near = nearest(nodes, q_rand)  # 找到最近的节点
                q_new = steer(q_near, q_rand, step_size_1, point_cloud, goal_direction)  # 生成新节点
                if is_within_point_cloud(point_cloud, q_new, tolerance):  # 确保新节点在点云数据中
                    neighbors = [node for node in nodes if
                                 distance(node.point, q_new.point) < search_radius_1 and is_within_point_cloud(point_cloud,
                                                                                                               q_new,
                                                                                                               tolerance)]
                    q_new.parent = min(neighbors, key=lambda node: node.cost + distance(node.point, q_new.point),
                                       default=q_near) if neighbors else q_near
                    q_new.cost = q_new.parent.cost + distance(q_new.parent.point, q_new.point)
                    nodes.append(q_new)  # 添加新节点到节点列表
                    a = a + 1
            cost = []
            parent = []
            for c,node in enumerate(nodes):
                cost.append(node.cost)
                parent.append(node.parent)
                if not is_within_point_cloud(point_cloud, node, tolerance):
                    print("有点不再点云中")
                    print(c)
            path = find_path(nodes, start, goal, tor)  # 寻找路径
        all_paths.append(path)
    # print(a)
    return all_paths
