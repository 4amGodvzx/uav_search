import numpy as np
from uav_search.to_map_test import to_map_xyz

def map_update_simple(attraction_map, exploration_map, obstacle_map, current_ground_truth_attraction_map, start_position, depth_image, camera_fov, camera_position, camera_orientation, ori):
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    new_attraction_map = attraction_map.copy()
    new_exploration_map = exploration_map.copy()
    new_obstacle_map = obstacle_map.copy()
    
    # 地图参数
    map_size = np.array([200.0, 200.0, 50.0])  # 地图尺寸 (meters)
    grid_size = np.array([5.0, 5.0, 5.0])      # 网格尺寸 (meters)
    map_resolution = (map_size / grid_size).astype(int)  # 地图分辨率 [40, 40, 10]
    obstacle_grid_size = np.array([5.0, 5.0, 5.0])
    obstacle_map_resolution = (map_size / obstacle_grid_size).astype(int)
    
    
    REWARD_DISTANCE_THRESHOLD = 30.0 # 奖励计算的最大距离
    KEY_THRESHOLD = 0.9 # 关键兴趣点的阈值
    KEY_REWARD = 5.0 # 关键兴趣点的额外奖励

    # 计算地图原点在世界坐标系中的位置 (start_position 是地图的中心)
    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0
    
    observed_grids = {}

    for v in range(image_shape[0]):
        for u in range(image_shape[1]):
            depth = depth_image[v, u]
            if depth >= 250:
                continue

            world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)
            map_coords = world_coords - map_origin
            grid_indices = tuple((map_coords / grid_size).astype(int))
            gx, gy, gz = grid_indices
            obstacle_grid_indices = (map_coords / obstacle_grid_size).astype(int)
            gx_o, gy_o, gz_o = obstacle_grid_indices            

            if 0 <= gx_o < obstacle_map_resolution[0] and 0 <= gy_o < obstacle_map_resolution[1] and 0 <= gz_o < obstacle_map_resolution[2]:
                new_obstacle_map[gx_o, gy_o, gz_o] = 1.0  # 标记为障碍物

            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                if grid_indices not in observed_grids or depth < observed_grids[grid_indices]:
                    observed_grids[grid_indices] = depth
    
    attraction_reward = 0.0
    exploration_reward = 0.0
    
    if not observed_grids:
        return new_attraction_map, new_exploration_map, new_obstacle_map, attraction_reward, exploration_reward

    for grid_indices, min_depth in observed_grids.items():
        gx, gy, gz = grid_indices

        ground_truth_value = current_ground_truth_attraction_map[gx, gy, gz]
        new_attraction_map[gx, gy, gz, 0] = ground_truth_value

        if min_depth < REWARD_DISTANCE_THRESHOLD:
            attraction_distance_weights = (REWARD_DISTANCE_THRESHOLD - min_depth) / REWARD_DISTANCE_THRESHOLD
            attraction_reward += ground_truth_value * attraction_distance_weights
            if ground_truth_value >= KEY_THRESHOLD:
                attraction_reward += KEY_REWARD * attraction_distance_weights
                
    VIEW_DEPTH = 50.0   # 视线距离
    VIEW_HEIGHT = 20.0  # 垂直视野高度
    RATE_CENTER = 0.4 # 探索奖励零点
    EXPLORATION_GAIN = 0.5 # 每次观测，探索值增加的基础量

    drone_world_pos = camera_position.to_numpy_array().flatten()

    # 2. 获取地图中所有栅格的索引和世界坐标中心点
    # 使用 np.indices 高效生成所有栅格的索引
    indices_x, indices_y, indices_z = np.indices(map_resolution)
    all_indices = np.stack([indices_x.ravel(), indices_y.ravel(), indices_z.ravel()], axis=-1)
    all_centers_world = (all_indices + 0.5) * grid_size + map_origin

    # 3. 筛选出在三角柱体视锥内的栅格
    relative_coords = all_centers_world - drone_world_pos
    rel_x, rel_y, rel_z = relative_coords.T

    # 根据无人机朝向(ori)，将所有栅格旋转到无人机的局部坐标系
    ori_conditions = [ori == 0, ori == 1, ori == 2, ori == 3]
    local_x_choices = [-rel_y, -rel_x, rel_y, rel_x]
    local_y_choices = [rel_x, -rel_y, -rel_x, rel_y]
    local_x = np.select(ori_conditions, local_x_choices)
    local_y = np.select(ori_conditions, local_y_choices)

    in_height_mask = np.abs(rel_z) <= VIEW_HEIGHT / 2
    in_depth_mask = (local_y > 0) & (local_y <= VIEW_DEPTH)
    in_angle_mask = np.abs(local_x) <= local_y # 90度FOV的核心条件
    in_prism_mask = in_height_mask & in_depth_mask & in_angle_mask

    # 获取所有被视线穿过的栅格索引
    traversed_indices_prism = all_indices[in_prism_mask]
    if traversed_indices_prism.shape[0] > 0:
        # 4. 计算探索奖励 (逻辑和之前一样，作用于新找到的栅格)
        distances_to_grids = np.linalg.norm(relative_coords[in_prism_mask], axis=1)
        nearby_mask = distances_to_grids < REWARD_DISTANCE_THRESHOLD

        if np.any(nearby_mask):
            nearby_indices = traversed_indices_prism[nearby_mask]
            nearby_distances = distances_to_grids[nearby_mask]

            gx, gy, gz = nearby_indices.T
            existing_exploration_values = exploration_map[gx, gy, gz]

            distance_weights = (REWARD_DISTANCE_THRESHOLD - nearby_distances) / REWARD_DISTANCE_THRESHOLD
            rewards = (RATE_CENTER - existing_exploration_values) * distance_weights
            exploration_reward = np.sum(rewards)

        # 5. 更新探索地图
        # 计算探索值的增加量，距离越近增加越多
        gx, gy, gz = traversed_indices_prism.T
        distances_for_update = np.linalg.norm(relative_coords[in_prism_mask], axis=1)
        distances_for_update = np.clip(distances_for_update, 0, VIEW_DEPTH)

        # 距离越远，增加量越小 (线性衰减)
        exploration_increase = (1 - distances_for_update / VIEW_DEPTH) * EXPLORATION_GAIN

        new_exploration_map[gx, gy, gz] += exploration_increase

    return new_attraction_map, new_exploration_map, new_obstacle_map, attraction_reward, exploration_reward

class MockVector:

    def __init__(self, x, y, z):

        # 增加 x_val, y_val, z_val 属性以更好地模拟真实对象

        self.x_val = x

        self.y_val = y

        self.z_val = z

    def to_numpy_array(self):

        return np.array([self.x_val, self.y_val, self.z_val], dtype=np.float64)


# 模拟 AirSim 的 Quaternionr 或类似对象 (修正版)

class MockQuaternion:

    def __init__(self, w, x, y, z):

        # 修正：直接存储 w_val, x_val, y_val, z_val 属性

        self.w_val = w

        self.x_val = x

        self.y_val = y

        self.z_val = z

    def to_numpy_array(self):

        return np.array([self.w_val, self.x_val, self.y_val, self.z_val], dtype=np.float64)
    
if __name__ == "__main__":

    print("--- 开始测试 map_update_simple 函数 ---")


    # --- 3.1 定义测试参数 ---

    MAP_SIZE = np.array([200.0, 200.0, 50.0])

    GRID_SIZE = np.array([5.0, 5.0, 5.0])

    MAP_RESOLUTION = (MAP_SIZE / GRID_SIZE).astype(int) # [40, 40, 10]

    IMG_SHAPE = (128, 128)

    

    # --- 3.2 创建模拟输入数据 ---

    print("1. 创建模拟输入数据...")

    

    # a. 地图 (全部初始化为0)

    # 假设 attraction_map 有一个额外的通道维度

    initial_attraction_map = np.zeros(list(MAP_RESOLUTION) + [1], dtype=np.float32)

    initial_exploration_map = np.zeros(MAP_RESOLUTION, dtype=np.float32)

    initial_obstacle_map = np.zeros(MAP_RESOLUTION, dtype=np.float32)

    

    ground_truth_map = np.zeros(MAP_RESOLUTION, dtype=np.float32)



    # c. 无人机状态


    map_center_position = MockVector(0.0, 0.0, 0.0) 


    drone_position = MockVector(0.0, 0.0, 0.0)

    drone_orientation = MockQuaternion(1.0, 0.0, 0.0, 0.0) # 水平姿态

    drone_ori = 1 # 朝向西

    print(f"    - 无人机位置: {drone_position.to_numpy_array()}, 朝向: 西 (ori=1)")


    # d. 模拟深度图 (前方15米处有一堵墙)

    depth_image = np.full(IMG_SHAPE, 250.0, dtype=np.float32)

    depth_image[:, :] = 15.0 # 所有像素深度都设为15米

    print(f"    - 创建了一个 {IMG_SHAPE} 的深度图，前方15米处有障碍物。")

    

    # e. 相机参数

    camera_fov = 90.0


    # --- 3.3 调用函数 ---

    print("\n2. 调用 map_update_simple 函数...")

    

    # 记录更新前的地图状态

    sum_exp_before = np.sum(initial_exploration_map)

    sum_obs_before = np.sum(initial_obstacle_map)

    

    new_att_map, new_exp_map, new_obs_map, attr_reward, exp_reward = map_update_simple(

        attraction_map=initial_attraction_map,

        exploration_map=initial_exploration_map,

        obstacle_map=initial_obstacle_map,

        current_ground_truth_attraction_map=ground_truth_map,

        start_position=map_center_position,

        depth_image=depth_image,

        camera_fov=camera_fov,

        camera_position=drone_position,

        camera_orientation=drone_orientation,

        ori=drone_ori

    )

    

    print("    - 函数执行完毕。")

    

    # --- 3.4 验证结果 ---

    print("\n3. 验证输出结果...")

    

    # a. 检查奖励值

    print(f"    - 计算出的吸引力奖励: {attr_reward:.4f}")

    print(f"    - 计算出的探索奖励: {exp_reward:.4f}")

    

    # b. 检查地图更新

    sum_exp_after = np.sum(new_exp_map)

    sum_obs_after = np.sum(new_obs_map)

    sum_att_after = np.sum(new_att_map)

    

    print(f"    - 探索地图总值: {sum_exp_before:.4f} -> {sum_exp_after:.4f}")

    print(f"    - 障碍地图总值: {sum_obs_before:.4f} -> {sum_obs_after:.4f}")

    print(f"    - 吸引力地图总值: 0.0 -> {sum_att_after:.4f}")



    drone_position = MockVector(10.0, 0.0, 0.0)

    drone_orientation = MockQuaternion(1.0, 0.0, 0.0, 0.0) # 水平姿态

    drone_ori = 0

    print(f"    - 无人机位置: {drone_position.to_numpy_array()}, 朝向: 西 (ori=1)")


    print("\n2. 调用 map_update_simple 函数...")

    

    # 记录更新前的地图状态

    sum_exp_before = np.sum(initial_exploration_map)

    sum_obs_before = np.sum(initial_obstacle_map)

    

    new_att_map, new_exp_map, new_obs_map, attr_reward, exp_reward = map_update_simple(

        attraction_map=new_att_map,

        exploration_map=new_exp_map,

        obstacle_map=initial_obstacle_map,

        current_ground_truth_attraction_map=ground_truth_map,

        start_position=map_center_position,

        depth_image=depth_image,

        camera_fov=camera_fov,

        camera_position=drone_position,

        camera_orientation=drone_orientation,

        ori=drone_ori

    )

    

    print("    - 函数执行完毕。")

    

    # --- 3.4 验证结果 ---

    print("\n3. 验证输出结果...")

    

    # a. 检查奖励值

    print(f"    - 计算出的吸引力奖励: {attr_reward:.4f}")

    print(f"    - 计算出的探索奖励: {exp_reward:.4f}")

    

    # b. 检查地图更新

    sum_exp_after = np.sum(new_exp_map)

    sum_obs_after = np.sum(new_obs_map)

    sum_att_after = np.sum(new_att_map)

    

    print(f"    - 探索地图总值: {sum_exp_before:.4f} -> {sum_exp_after:.4f}")

    print(f"    - 障碍地图总值: {sum_obs_before:.4f} -> {sum_obs_after:.4f}")

    print(f"    - 吸引力地图总值: 0.0 -> {sum_att_after:.4f}")