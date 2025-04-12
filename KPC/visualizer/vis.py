import open3d as o3d
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

def visualize_point_cloud(points):
    """
    使用 Open3D 可视化点云数据和路径。

    参数:
    - points: np.array，形状为 (N, 3)，表示点云数据，颜色为蓝色。
    - group: np.array，形状为 (N, 3)，表示分组数据，颜色为黑色。
    - paths: list，包含多个 np.array，每个路径为形状为 (m, 3) 的点集。
    """
    # 创建Open3D的点云对象
    point_cloud_points = o3d.geometry.PointCloud()
    point_cloud_points.points = o3d.utility.Vector3dVector(points)
    point_cloud_points.paint_uniform_color([0, 0, 1])  # 将点云颜色设置为蓝色

    o3d.visualization.draw_geometries([point_cloud_points])

def visualize_point_cloud_with_labels(point_cloud_data, labels):
    """
    可视化带有标签的点云数据，使用不同的颜色表示不同的标签。

    参数:
    - point_cloud_data (numpy.ndarray): 点云数据，形状为(N, 3)，N为点的数量。
    - labels (numpy.ndarray): 标签数据，形状为(N,)。
    """
    # 获取标签的个数（不同标签的数量）
    num_labels = len(np.unique(labels))
    print(f"Number of unique labels: {num_labels}")

    # 使用LabelEncoder将标签映射到颜色
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 为每个标签分配一个颜色
    colors = plt.cm.jet(encoded_labels / float(num_labels))[:, :3]  # 使用jet颜色映射并去掉alpha通道

    # 创建PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization with Labels")

def visualize_point_cloud_with_colors(points,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 设置点云坐标
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置点云颜色

    # 可视化
    o3d.visualization.draw_geometries([pcd])

def visualize_point_cloud_with_paths(point_cloud_data, start_points, end_points, paths, group):
    """
    可视化点云数据、起点、终点和路径。

    参数:
    - point_cloud_data (np.array): 点云数据，形状为 (N, 3)
    - start_points (np.array): 起点，形状为 (n, 3)
    - end_points (np.array): 终点，形状为 (n, 3)
    - paths (list): 路径列表，每个路径是形状为 (m, 3) 的 np.array

    返回:
    - None
    """
    point_cloud_data = point_cloud_data.transpose(0,1)
    point_cloud_data[:,0] += 0
    # 确保起点和终点形状正确
    start_points = np.reshape(start_points, (-1, 3))
    end_points = np.reshape(end_points, (-1, 3))

    # 可视化点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # pcd_group = o3d.geometry.PointCloud()
    # pcd_group.points = o3d.utility.Vector3dVector(group)
    # pcd_group.paint_uniform_color([0,1,1])

    # 将起点和终点作为不同颜色的点
    start_pcd = o3d.geometry.PointCloud()
    start_pcd.points = o3d.utility.Vector3dVector(start_points)
    start_pcd.paint_uniform_color([1, 0, 0])  # 红色

    end_pcd = o3d.geometry.PointCloud()
    end_pcd.points = o3d.utility.Vector3dVector(end_points)
    end_pcd.paint_uniform_color([0, 1, 0])  # 绿色

    # 绘制路径
    lines = []
    colors = []
    all_paths = []  # 用于存储所有路径点

    total_points = 0  # 用于记录点的总数，计算线的索引

    for path in paths:
        # 获取路径长度
        m = len(path)
        path = np.reshape(path, (-1, 3))
        all_paths.append(path)

        # 为每条路径添加连接线
        path_color = [random.random(), random.random(), random.random()]  # 为当前路径生成颜色
        for j in range(m - 1):
            lines.append([total_points + j, total_points + j + 1])
            colors.append(path_color)  # 为该路径的所有线条添加相同的颜色

        total_points += m  # 更新点的总数

    # 合并所有路径点
    all_paths = np.vstack(all_paths)

    # 生成路径的LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(all_paths)  # 合并所有路径点
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)

    # 可视化所有元素
    # o3d.visualization.draw_geometries([pcd, start_pcd, end_pcd, lineset])
    o3d.visualization.draw_geometries([pcd, lineset])

def visualize_3d_rrt_star_animation_1(point_cloud, nodes, path, start, goal, save_path=r'./data/rrt_star_animation.gif'):
    """
    3D 动画可视化 RRT* 过程
    参数：
        point_cloud: 点云数据 (N,3) numpy数组
        nodes: 算法生成的节点列表
        path: 找到的路径坐标列表 (M,3)
        start: 起点坐标 (3,)
        goal: 终点坐标 (3,)
        save_path: 保存动图的文件名
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云（半透明灰色）
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               c='green', alpha=0.2, label='Point Cloud')

    # 标记起点（绿色立方体）
    ax.scatter(start[0], start[1], start[2],
               c='limegreen', s=200, marker='s', edgecolor='black', label='Start')

    # 标记终点（紫色星形）
    ax.scatter(goal[0], goal[1], goal[2],
               c='darkviolet', s=300, marker='*', edgecolor='black', label='Goal')

    # 设置坐标轴
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title('3D RRT* Path Planning', fontsize=16)

    ax.legend(loc='upper right', fontsize=10)
    ax.view_init(elev=25, azim=45)  # 初始视角

    # 记录绘制的元素
    tree_lines = []  # 记录搜索树的线条
    path_lines = []  # 记录最终路径的线条

    def update(frame):
        """动画更新函数"""
        nonlocal tree_lines, path_lines

        # 绘制搜索树（前 len(nodes) 帧）
        if frame < len(nodes):
            node = nodes[frame]
            if node.parent:
                a = [node.point[0], node.parent.point[0]]
                a1, a2 = a[0], a[1]
                x = [a1[0], a2[0]]
                y = [a1[1], a2[1]]
                z = [a1[2], a2[2]]
                line, = ax.plot(x, y, z, color='royalblue', linewidth=0.8, alpha=0.7)
                tree_lines.append(line)

        # 绘制最终路径（后 len(path) 帧）
        if frame >= len(nodes) and len(path) > 1:
            path_idx = frame - len(nodes)
            if path_idx < len(path) - 1:
                p1 = torch.tensor(path[path_idx])
                p2 = torch.tensor(path[path_idx + 1])
                p1 = p1.detach().cpu().numpy().flatten()
                p2 = p2.detach().cpu().numpy().flatten()
                line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                color='red', linewidth=3, marker='o', markersize=4, markerfacecolor='yellow')
                path_lines.append(line)

        # Ensure the lines remain on the plot by returning all of them
        return tree_lines + path_lines

    # 创建动画
    total_frames = len(nodes) + len(path)  # 总帧数
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

    # 保存动画
    ani.save(save_path, writer='ffmpeg', fps=10)
    plt.show()
