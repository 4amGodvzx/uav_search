import airsim
import time
import numpy as np
import math
from skimage.transform import resize
from collections import defaultdict, Counter

from uav_search.to_map_test import to_map_xyz

def add_masks(masks):
    added_masks = []
    for mask_group in masks:
        if not mask_group:  # 如果当前目标的掩码列表为空
            added_masks.append(None)
        else:
            # 堆叠所有掩码并计算逻辑或（OR）
            stacked = np.stack(mask_group, axis=0)  # shape: [N, H, W]
            combined_mask = np.any(stacked, axis=0)
            added_masks.append(combined_mask)
    return added_masks

def downsample_masks(masks, scale_factor):
    downsampled = []
    for mask in masks:
        if mask is not None:
            downsampled_mask = resize(mask, (mask.shape[0] // scale_factor, mask.shape[1] // scale_factor), order=0, anti_aliasing=False)
            downsampled.append(downsampled_mask)
        else:
            downsampled.append(None)
    return downsampled

def exploration_rate(depth, max_depth, decay_factor):
    # 确保depth不为负
    depth = max(0.0, depth)
    # 归一化距离
    normalized_depth = depth / max_depth
    # 指数衰减公式
    rate = math.exp(-decay_factor * normalized_depth)
    # 确保结果在[0,1]范围内
    return max(0.0, min(1.0, rate))

def trace_ray_in_grid(start_world, end_world, map_origin, grid_size):
    ray_start = np.asarray(start_world).flatten()
    ray_end = np.asarray(end_world).flatten()
    
    direction = ray_end - ray_start
    distance = np.linalg.norm(direction)
    
    if distance < 1e-6:  # 距离太近，无法形成射线
        return set()
        
    direction_normalized = direction / distance
    
    # 采样步长略小于栅格尺寸的一半，以确保不会跳过栅格
    step_size = grid_size[0] * 0.5 
    num_steps = int(np.ceil(distance / step_size))
    
    traversed_grids = set()
    
    for i in range(num_steps + 1):
        # 计算当前采样点的世界坐标
        current_point = ray_start + i * step_size * direction_normalized
        
        # 将世界坐标转换为地图坐标，再转换为栅格索引
        map_coords = current_point - map_origin
        grid_indices = tuple((map_coords / grid_size).astype(int))
        
        traversed_grids.add(grid_indices)
        
    return traversed_grids

def map_update(attraction_map, exploration_map, prepared_masks, attraction_scores, start_position, depth_image, camera_fov, camera_position, camera_orientation):
    depth_image = depth_image
    camera_fov = camera_fov
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    
    new_attraction_map = attraction_map.copy()
    new_exploration_map = exploration_map.copy()

    # 地图参数
    map_size = np.array([200.0, 200.0, 50.0])  # 地图尺寸 (meters)
    grid_size = np.array([5.0, 5.0, 5.0])      # 网格尺寸 (meters)
    map_resolution = (map_size / grid_size).astype(int)  # 地图分辨率 [40, 40, 10]

    # 计算地图原点在世界坐标系中的位置 (start_position 是地图的中心)
    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0

    # --- 1. 更新吸引力图 (Attraction Map) ---

    # 步骤 1.1: 收集所有掩码像素点对地图网格的贡献
    # 字典的键是网格索引 (gx, gy, gz)，值是 (深度, 对象ID) 的列表
    grid_contributions = defaultdict(list)

    for object_id, mask in enumerate(prepared_masks):
        if mask is None:
            continue

        # 获取掩码为 True 的所有像素坐标
        pixels_v, pixels_u = np.where(mask)

        for v, u in zip(pixels_v, pixels_u):
            depth = depth_image[v, u]
            if depth >= 250:  # 只处理有效深度范围内的点
                continue

            # 将像素坐标转换为世界坐标
            world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)

            # 将世界坐标转换为地图坐标，再转换为网格索引
            map_coords = world_coords - map_origin
            grid_indices = (map_coords / grid_size).astype(int)
            gx, gy, gz = grid_indices

            # 边界检查，确保网格索引在地图范围内
            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                grid_contributions[(gx, gy, gz)].append((depth, object_id))

    # 步骤 1.2: 根据收集到的贡献点，更新每个网格的深度和吸引力分数
    for (gx, gy, gz), points_list in grid_contributions.items():
        if not points_list:
            continue

        # --- 深度更新 ---
        # 找到当前帧所有贡献到此网格的点中的最小深度
        min_depth_in_frame = min(p[0] for p in points_list)
        
        # 与地图中存储的旧深度比较
        stored_depth = attraction_map[gx, gy, gz, 1]
        if min_depth_in_frame < stored_depth or stored_depth < 0:
            new_attraction_map[gx, gy, gz, 1] = min_depth_in_frame

        # --- 吸引力分数更新 (投票) ---
        # 统计哪个对象ID在此网格中出现的次数最多
        object_ids = [p[1] for p in points_list]
        if object_ids:
            # Counter(...).most_common(1) 返回 [(元素, 次数)] 形式的列表
            winner_object_id = Counter(object_ids).most_common(1)[0][0]
            winning_score = attraction_scores[winner_object_id]
            new_attraction_map[gx, gy, gz, 0] = winning_score

    # --- 2. 更新探索图 (Exploration Map) ---
    
    add_exploration_map = np.zeros_like(exploration_map)
    observed_points_cnt = 0 # Only used for trainning
    drone_world_pos = camera_position.to_numpy_array()
    
    # 遍历整个深度图
    for v in range(image_shape[0]):
        for u in range(image_shape[1]):
            depth = depth_image[v, u]
            if depth >= 250:
                continue

            endpoint_world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)
            ray_grids = trace_ray_in_grid(drone_world_pos, endpoint_world_coords, map_origin, grid_size)
            
            endpoint_map_coords = endpoint_world_coords - map_origin
            endpoint_grid_indices = (endpoint_map_coords / grid_size).astype(int)
            gx, gy, gz = endpoint_grid_indices

            for grid_indices in ray_grids:
                gx, gy, gz = grid_indices

                # 边界检查
                if not (0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]):
                    continue
                
                # 计算该栅格中心到无人机的距离
                grid_center_world_pos = (np.array(grid_indices) + 0.5) * grid_size + map_origin

                dist_to_grid = np.linalg.norm(grid_center_world_pos - drone_world_pos)

                # 根据到栅格的距离计算探索率
                rate = exploration_rate(dist_to_grid, max_depth=200, decay_factor=5) # Hyperparameter: 最大距离和衰减因子

                # 更新探索地图（只保留更大的探索率）
                if rate > add_exploration_map[gx, gy, gz]:
                    add_exploration_map[gx, gy, gz] = rate

            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                observed_points_cnt += 1

    new_exploration_map = exploration_map + add_exploration_map
    # Hyperparameter: 遗忘因子
    # new_exploration_map *= 0.95

    return new_attraction_map, new_exploration_map, observed_points_cnt
