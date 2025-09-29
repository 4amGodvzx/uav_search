import numpy as np

from uav_search.to_map_test import to_map_xyz

def obstacle_update(obstacle_map, start_position, depth_image, camera_fov, camera_position, camera_orientation):
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    new_obstacle_map = obstacle_map.copy()
    
    # 地图参数
    map_size = np.array([200.0, 200.0, 50.0])  # 地图尺寸 (meters)
    grid_size = np.array([2.5, 2.5, 2.5])      # 网格尺寸 (meters)
    map_resolution = (map_size / grid_size).astype(int)  # 地图分辨率 [80, 80, 20]

    # 计算地图原点在世界坐标系中的位置 (start_position 是地图的中心)
    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0
    
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
                new_obstacle_map[gx, gy, gz] = 1.0  # 标记为障碍物
    
    return new_obstacle_map

def _crop_rotate_and_pad(full_map: np.ndarray, center_coords: np.ndarray, crop_size: tuple, padding_value: float, orientation: int) -> np.ndarray:
    local_map = np.full(crop_size, padding_value, dtype=full_map.dtype)
    center_indices = np.round(center_coords).astype(int)

    map_dims = full_map.shape
    half_size = np.array(crop_size) // 2

    src_start = center_indices - half_size
    src_end = center_indices + half_size

    dest_start = np.zeros_like(half_size)
    dest_end = np.array(crop_size, dtype=int)

    for i in range(3):
        if src_start[i] < 0:
            dest_start[i] = -src_start[i]
            src_start[i] = 0

        if src_end[i] > map_dims[i]:
            dest_end[i] -= (src_end[i] - map_dims[i])
            src_end[i] = map_dims[i] 

    src_slice = tuple(slice(s, e) for s, e in zip(src_start, src_end))
    dest_slice = tuple(slice(s, e) for s, e in zip(dest_start, dest_end))

    if all(s.start < s.stop for s in src_slice):
        local_map[dest_slice] = full_map[src_slice]

    rotated_map = np.rot90(local_map, k=orientation, axes=(0, 1))
    return rotated_map

def map_input_preparation(attraction_map, exploration_map, obstacle_map, uav_pose: dict):
    position = np.array(uav_pose['position'])
    orientation = uav_pose['orientation']

    attraction_map_input = _crop_rotate_and_pad(
        full_map=attraction_map[:, :, :, 0],
        center_coords=position,
        crop_size=(20, 20, 10),
        padding_value=-1.0,
        orientation=orientation
    )

    exploration_map_input = _crop_rotate_and_pad(
        full_map=exploration_map,
        center_coords=position,
        crop_size=(20, 20, 10),
        padding_value=-1.0,
        orientation=orientation
    )

    obstacle_center_coords = position * 2
    obstacle_map_input = _crop_rotate_and_pad(
        full_map=obstacle_map,
        center_coords=obstacle_center_coords,
        crop_size=(40, 40, 8),
        padding_value=1.0,
        orientation=orientation
    )
    
    map_input = {
        'attraction_map_input': np.transpose(attraction_map_input,(2,0,1)),
        'exploration_map_input': np.transpose(exploration_map_input,(2,0,1)),
        'obstacle_map_input': np.transpose(obstacle_map_input,(2,0,1))
    }
    
    return map_input
'''
original_shape_attraction = (40, 40, 10, 2)
original_shape_exploration = (40, 40, 10)
original_shape_obstacle = (80, 80, 20)

uav_pose = {
    'position': [5, 5, 7],
    'orientation': 1 # 0: north, 1: west, 2: south, 3: east
}

attraction_flat = np.loadtxt(f"test_images/attraction_map_15.txt")
attraction_map = attraction_flat.reshape(original_shape_attraction)

exploration_flat = np.loadtxt(f"test_images/exploration_map_15.txt")
exploration_map = exploration_flat.reshape(original_shape_exploration)

obstacle_flat = np.loadtxt(f"test_images/obstacle_map_15.txt")
obstacle_map = obstacle_flat.reshape(original_shape_obstacle)

map_input = map_input_preparation(attraction_map, exploration_map, obstacle_map, uav_pose)

np.savetxt(f"test_images/attraction_map_input_test.txt", map_input['attraction_map_input'].flatten())
np.savetxt(f"test_images/exploration_map_input_test.txt", map_input['exploration_map_input'].flatten())
np.savetxt(f"test_images/obstacle_map_input_test.txt", map_input['obstacle_map_input'].flatten())
'''