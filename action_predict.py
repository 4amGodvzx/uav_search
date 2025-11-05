import numpy as np

def action_predict(attraction_map, exploration_map, obstacle_map):
    # 地图和无人机参数
    map_shape = np.array([40, 40, 10])
    drone_pos_local = np.array([20, 20, 5])
    
    # 视野模拟参数
    VIEW_DEPTH = 100.0   # 视线距离 (meters)
    VIEW_HEIGHT = 30.0  # 垂直视野高度 (meters)
    grid_size = 5.0     # 单个网格的尺寸 (meters)

    # 奖励计算参数
    REWARD_DISTANCE_THRESHOLD = 50.0
    KEY_THRESHOLD = 0.9
    KEY_REWARD = 5.0
    RATE_CENTER = 0.4
    
    # 动作定义
    actions = {
        0: 'forward', 
        1: 'rotate_left', 
        2: 'rotate_right', 
        3: 'rotate_backward', 
        4: 'ascend', 
        5: 'descend'
    }
    
    # 定义奖励权重 (可以调整吸引和探索的优先级)
    W_ATTRACTION = 1.0
    W_EXPLORATION = 0.4

    # 生成局部地图中所有网格的索引和中心点坐标
    # 这只需要计算一次，后续对每个动作的模拟都可复用
    indices_x, indices_y, indices_z = np.indices(map_shape)
    all_indices = np.stack([indices_x.ravel(), indices_y.ravel(), indices_z.ravel()], axis=-1)
    # 局部地图的中心点坐标就是它们的索引值（因为原点在(0,0,0)）
    all_centers_local = all_indices.astype(float) 

    predicted_rewards = {}
    current_pos_int = drone_pos_local.astype(int) # 获取当前整数坐标

    for action_idx, action_name in actions.items():
        
        # 1. 模拟动作，确定无人机 hypothetical (假设的) 下一个状态
        hypothetical_pos = drone_pos_local.copy()
        hypothetical_ori = 0  # 0:前, 1:左, 2:右, 3:后 (相对于当前局部地图坐标系)

        is_movement_action = False
        if action_name == 'forward':
            hypothetical_pos[0] += 2
            is_movement_action = True
        elif action_name == 'ascend':
            hypothetical_pos[2] -= 1 # z越大越下，所以上升是z减小
            is_movement_action = True
        elif action_name == 'descend':
            hypothetical_pos[2] += 1
            is_movement_action = True
        elif action_name == 'rotate_left':
            hypothetical_ori = 1
        elif action_name == 'rotate_right':
            hypothetical_ori = 2
        elif action_name == 'rotate_backward':
            hypothetical_ori = 3
            
        px, py, pz = int(hypothetical_pos[0]), int(hypothetical_pos[1]), int(hypothetical_pos[2])

        if not (1 < px < map_shape[0] - 1 and 1 < py < map_shape[1] - 1 and 1 < pz < map_shape[2] - 1):
            predicted_rewards[action_idx] = -np.inf
            continue

        if action_name in ['ascend', 'descend']:
            predicted_rewards[action_idx] -= 10.0
        
        is_safe = True
        
        if is_movement_action:
            if action_name == 'forward':
                start_x = current_pos_int[0]
                end_x = px + 2
                if np.any(obstacle_map[start_x:end_x, current_pos_int[1], current_pos_int[2]] >= 1.0):
                    is_safe = False
            elif action_name == 'ascend':
                start_z = pz + 2
                end_z = current_pos_int[2]
                if np.any(obstacle_map[px, py, start_z:end_z] >= 1.0):
                    is_safe = False
            elif action_name == 'descend':
                start_z = current_pos_int[2]
                end_z = pz + 2
                if np.any(obstacle_map[px, py, start_z:end_z] >= 1.0):
                    is_safe = False

        if obstacle_map[px, py, pz] >= 1.0:
            is_safe = False

        if not is_safe:
            predicted_rewards[action_idx] = -np.inf
            continue

        # 3. 预测从 hypothetical 状态能观测到的网格
        # 计算所有网格中心相对于 hypothetical 位置的坐标
        relative_coords = (all_centers_local - hypothetical_pos) * grid_size
        rel_x, rel_y, rel_z = relative_coords.T
        
        # 根据 hypothetical 朝向，将相对坐标旋转到无人机的局部坐标系
        # local_y 是前向, local_x 是右向
        ori_conditions = [hypothetical_ori == 0, hypothetical_ori == 1, hypothetical_ori == 2, hypothetical_ori == 3]
        local_y_choices = [rel_x, -rel_y, rel_y, -rel_x] # 前, 左, 右, 后
        local_x_choices = [rel_y, rel_x, -rel_x, -rel_y] # 右, 前, 后, 左
        local_y = np.select(ori_conditions, local_y_choices)
        local_x = np.select(ori_conditions, local_x_choices)

        # 使用三角柱体筛选视锥内的网格
        in_height_mask = np.abs(rel_z) <= VIEW_HEIGHT / 2
        in_depth_mask = (local_y > 0) & (local_y <= VIEW_DEPTH)
        in_angle_mask = np.abs(local_x) <= local_y # 90度FOV
        
        in_prism_mask = in_height_mask & in_depth_mask & in_angle_mask
        
        # 获取所有将被观测到的网格的索引
        observed_indices = all_indices[in_prism_mask]
        
        if observed_indices.shape[0] == 0:
            predicted_rewards[action_idx] = 0.0
            continue

        # 4. 计算预测奖励
        attraction_reward = 0.0
        exploration_reward = 0.0
        
        # 只考虑 REWARD_DISTANCE_THRESHOLD 范围内的网格来计算奖励
        distances_to_grids = np.linalg.norm(relative_coords[in_prism_mask], axis=1)
        nearby_mask = distances_to_grids < REWARD_DISTANCE_THRESHOLD

        if np.any(nearby_mask):
            nearby_indices = observed_indices[nearby_mask]
            nearby_distances = distances_to_grids[nearby_mask]
            
            gx, gy, gz = nearby_indices.T
            
            # 边界检查：筛掉那些因为填充而值为-1的无效网格
            valid_mask = (attraction_map[gx, gy, gz] != -1.0)
            if not np.any(valid_mask):
                predicted_rewards[action_idx] = 0.0
                continue
            
            # 应用掩码
            gx, gy, gz = gx[valid_mask], gy[valid_mask], gz[valid_mask]
            nearby_distances = nearby_distances[valid_mask]

            # 计算吸引力奖励
            attraction_values = attraction_map[gx, gy, gz]
            distance_weights = (REWARD_DISTANCE_THRESHOLD - nearby_distances) / REWARD_DISTANCE_THRESHOLD
            
            attraction_reward = np.sum(attraction_values * distance_weights)
            # 关键点额外奖励
            key_point_mask = attraction_values >= KEY_THRESHOLD
            attraction_reward += np.sum(KEY_REWARD * distance_weights[key_point_mask])

            # 计算探索奖励
            exploration_values = exploration_map[gx, gy, gz]
            exploration_rewards = (RATE_CENTER - exploration_values) * distance_weights
            exploration_reward = np.sum(exploration_rewards)

        # 5. 计算总奖励并存储
        total_predicted_reward = W_ATTRACTION * attraction_reward + W_EXPLORATION * exploration_reward
        predicted_rewards[action_idx] = total_predicted_reward

    # 6. 对动作按奖励排序
    rewards_list = list(predicted_rewards.values())
    all_rewards_are_same = np.max(rewards_list) - np.min(rewards_list) < 1e-2
    if all_rewards_are_same:
        sorted_actions = [1, 2, 0, 3, 4, 5]
    else:
        sorted_actions = sorted(predicted_rewards.keys(), key=lambda k: predicted_rewards[k], reverse=True)
    action = sorted_actions[0]  # 选择奖励最高的动作作为最终动作
    print("Predicted Rewards:", predicted_rewards, " Selected Action:", action)
    
    return int(action)

