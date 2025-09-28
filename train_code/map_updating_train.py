import numpy as np
import math

from uav_search.to_map_test import to_map_xyz
from uav_search.to_map_numpy import depth_image_to_world_points

def exploration_rate(distance: np.ndarray, max_depth=200, decay_factor=5) -> np.ndarray:
    rates = np.where(
        distance > max_depth, 
        0.0, 
        np.exp(-decay_factor * (distance / max_depth))
    )
    return rates

def trace_rays_vectorized(drone_world_pos, endpoints_world, map_origin, grid_size, map_resolution):

    start_points = np.expand_dims(drone_world_pos, axis=0)
    
    directions = endpoints_world - start_points
    distances = np.linalg.norm(directions, axis=1)

    # 确定最长的射线
    max_distance = np.max(distances)
    step_size = grid_size[0] * 0.5
    num_steps = int(np.ceil(max_distance / step_size))
    
    # 创建采样步长
    t_values = np.linspace(0.0, 1.0, num_steps)
    
    sampled_points = start_points[:, np.newaxis, :] + \
                     directions[:, np.newaxis, :] * t_values[np.newaxis, :, np.newaxis]
    
    # 将所有采样点坐标转换为栅格索引
    map_coords = sampled_points.reshape(-1, 3) - map_origin
    grid_indices = (map_coords / grid_size).astype(np.int32)
    
    # 过滤和去重
    valid_mask = np.all((grid_indices >= 0) & (grid_indices < np.array(map_resolution)), axis=1)
    grid_indices = grid_indices[valid_mask]
    
    Rx, Ry, Rz = map_resolution
    unique_ids = grid_indices[:, 0] * Ry * Rz + grid_indices[:, 1] * Rz + grid_indices[:, 2]
    _, unique_idx = np.unique(unique_ids, return_index=True)
    unique_grid_indices = grid_indices[unique_idx]
    
    return unique_grid_indices
'''
ground_truth_attraction_map:
It is from human-labeled data for training.
The data type of ground_truth_attraction_map:
ground_truth_attraction_map[task_id] = np.zeros((40, 40, 10), dtype=np.float32)
'''
def map_update(attraction_map, exploration_map, obstacle_map, current_ground_truth_attraction_map, start_position, depth_image, camera_fov, camera_position, camera_orientation):
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    new_attraction_map = attraction_map.copy()
    new_exploration_map = exploration_map.copy()
    new_obstacle_map = obstacle_map.copy()
    
    # 地图参数
    map_size = np.array([200.0, 200.0, 50.0])  # 地图尺寸 (meters)
    grid_size = np.array([5.0, 5.0, 5.0])      # 网格尺寸 (meters)
    map_resolution = (map_size / grid_size).astype(int)  # 地图分辨率 [40, 40, 10]
    
    REWARD_DISTANCE_THRESHOLD = 50.0 # 奖励计算的最大距离
    KEY_THRESHOLD = 0.9 # 关键兴趣点的阈值
    KEY_REWARD = 10.0 # 关键兴趣点的额外奖励    

    # 计算地图原点在世界坐标系中的位置 (start_position 是地图的中心)
    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0
    
    attraction_reward = 0.0
    
    for v in range(image_shape[0]):
        for u in range(image_shape[1]):
            depth = depth_image[v, u]
            if depth >= 250:
                continue
    
            world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)
            map_coords = world_coords - map_origin
            grid_indices = (map_coords / grid_size).astype(int)
            gx, gy, gz = grid_indices
            
            # 边界检查
            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                new_attraction_map[gx, gy, gz] = current_ground_truth_attraction_map[gx, gy, gz]
                new_obstacle_map[gx, gy, gz] = 1  # 标记为障碍物
                
                # 计算兴趣奖励
                if depth < REWARD_DISTANCE_THRESHOLD:
                    attraction_distance_weights = (REWARD_DISTANCE_THRESHOLD - nearby_distances) / REWARD_DISTANCE_THRESHOLD
                    attraction_reward += attraction_map[gx, gy, gz] * attraction_distance_weights
                    if attraction_map[gx, gy, gz] >= KEY_THRESHOLD:
                        attraction_reward += KEY_REWARD * attraction_distance_weights
                
    add_exploration_map = np.zeros_like(exploration_map)
    drone_world_pos = camera_position.to_numpy_array()

    # 过滤掉无效深度值
    valid_depth_mask = depth_image < 250
    
    # 向量化计算所有有效像素对应的世界坐标
    all_endpoints_world = depth_image_to_world_points(
        depth_image, camera_fov, camera_position, camera_orientation
    )

    flat_mask = valid_depth_mask.flatten()
    valid_endpoints = all_endpoints_world[flat_mask]
    drone_world_pos = camera_position.to_numpy_array().flatten()

    # 得到所有被视线穿过的、不重复的、在边界内的栅格索引
    traversed_grids_indices = trace_rays_vectorized(
        drone_world_pos, valid_endpoints, map_origin, grid_size, map_resolution
    )

    if traversed_grids_indices.shape[0] == 0:
        exploration_reward = 0.0
        return new_attraction_map, new_exploration_map, new_obstacle_map
    
    # 计算所有这些栅格到无人机的距离
    grid_centers_world = (traversed_grids_indices + 0.5) * grid_size + map_origin
    distances_to_grids = np.linalg.norm(grid_centers_world - drone_world_pos, axis=1)
    
    # 计算探索奖励
    exploration_reward = 0.0
 
    positive_gain = 1.0  # 探索未知区域的基础奖励增益
    negative_gain = 1.0  # 重复探索已知区域的基础惩罚增益

    nearby_mask = distances_to_grids < REWARD_DISTANCE_THRESHOLD

    if np.any(nearby_mask):
        nearby_indices = traversed_grids_indices[nearby_mask]
        nearby_distances = distances_to_grids[nearby_mask]

        # 获取这些近距离栅格在原探索地图中的探索值
        gx_nearby, gy_nearby, gz_nearby = nearby_indices.T
        existing_exploration_values = exploration_map[gx_nearby, gy_nearby, gz_nearby]

        # 距离衰减权重
        exploration_distance_weights = (REWARD_DISTANCE_THRESHOLD - nearby_distances) / REWARD_DISTANCE_THRESHOLD

        rewards = np.zeros_like(nearby_distances)
        positive_mask = existing_exploration_values < 1.0
        rewards[positive_mask] = positive_gain * exploration_distance_weights[positive_mask]
        negative_mask = existing_exploration_values > 1.0
        rewards[negative_mask] = -negative_gain * exploration_distance_weights[negative_mask]

        exploration_reward = np.sum(rewards)

    rates = exploration_rate(distances_to_grids, max_depth=200, decay_factor=5)

    gx, gy, gz = traversed_grids_indices.T # .T是转置，得到三个 (M,) 的数组
    current_rates = exploration_map[gx, gy, gz]
    update_mask = rates > current_rates
    add_exploration_map[gx[update_mask], gy[update_mask], gz[update_mask]] = rates[update_mask]
    
    new_exploration_map = exploration_map + add_exploration_map

    return new_attraction_map, new_exploration_map, new_obstacle_map, attraction_reward, exploration_reward