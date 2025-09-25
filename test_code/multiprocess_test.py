import threading
import time
import airsim
import numpy as np

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

from airsim_utils import get_images
from grounded_sam_test import grounded_sam
from uav_search.test_code.map_updating_test import add_masks, downsample_masks, map_update
from detection_test import detection_test
from action_model_inputs_test import obstacle_update

device = "cuda:3"

# 初始化模型和处理器
dino_model_directory = "/data/zhangdaoxuan/models/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
dino_processor = AutoProcessor.from_pretrained(dino_model_directory)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_directory).to(device)

sam_model_directory = "/data/zhangdaoxuan/models/models--facebook--sam-vit-huge/snapshots/87aecf0df4ce6b30cd7de76e87673c49644bdf67"
sam_processor = SamProcessor.from_pretrained(sam_model_directory)
sam_model = SamModel.from_pretrained(sam_model_directory).to(device)

detection_dino_processor = AutoProcessor.from_pretrained(dino_model_directory)
detection_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_directory).to(device)

# Action list for testing
Action_list = [0,0,0,0,0,0,0,0,2,2,1,1,1,1,3,1] # 0: west, 1: south, 2: rotate left, 3: rotate right 

# 无人机初始位置(From dataset)
start_position = airsim.Vector3r(-60, 20, -8)
start_orientation = airsim.utils.to_quaternion(0, 0, 0)

# 无人机当前位置和朝向(网格坐标系),以及碰撞状态
uav_pose = {
    'position': [20, 20, 5],
    'orientation': 0,  # 0: north, 1: west, 2: south, 3: east
    'is_collided': False
}

# 地图记录
maps_history = []

attraction_map = np.zeros((40, 40, 10, 2), dtype=np.float32) # 空的3D吸引力图,每个网格额外存储最近一次更新时的观测距离
attraction_map[:, :, :, 1] = -1 # -1 表示未被观察过的网格
exploration_map = np.zeros((40, 40, 10), dtype=np.float32) # 空的3D探索图
obstacle_map = np.zeros((80, 80, 20), dtype=np.uint8) # 空的3D障碍物图

# 共享地图数据
maps = {
    'attraction_map': attraction_map,
    'exploration_map': exploration_map,
    'obstacle_map': obstacle_map,
    'observed_points_cnt': 0
}

# 共享检测信息
detection_information = {
    'detection_success': False,
    'detected_position': [0.0, 0.0, 0.0],
    'reach_max_steps': False
}

# 用于保护 uav_pose 的锁
uav_pose_lock = threading.Lock()
# 用于保护 maps 的锁
data_lock = threading.Lock()
# 用于保护 detection_information 的锁
detection_information_lock = threading.Lock()
# 用于保护 AirSim Client API 调用的锁，特别是飞行控制和图像获取
#airsim_lock = threading.Lock()

def action_thread_func():
    action_client = airsim.MultirotorClient()
    action_client.confirmConnection()
    action_client.enableApiControl(True)
    action_client.armDisarm(True)
    
    print("[Action] Taking off...")
    action_client.takeoffAsync().join()
    pose = airsim.Pose(start_position, start_orientation)
    action_client.simSetVehiclePose(pose, True)
    print("[Action] Takeoff complete.")
    
    for i in range(16):
        if detection_information['detection_success']:
            with detection_information_lock:
                detection_pos = detection_information['detected_position']
            print(f"[Action] Detection success! Moving to detected position: {detection_pos}")
            action_client.moveToZAsync(-30, 2).join()
            action_client.moveToPositionAsync(detection_pos[0], detection_pos[1], -30, 5).join()
            print("[Action] Reached detected position.")
            break
        else:
            with data_lock:
                maps_copy = maps.copy()
            
            _, depth_image, camera_position, camera_orientation, rgb_vis, _ = get_images(action_client)
            camera_fov = 90
            obstacle_map = maps_copy['obstacle_map']
            new_obstacle_map = obstacle_update(obstacle_map, start_position, depth_image, camera_fov, camera_position, camera_orientation)
            with data_lock:
                maps['obstacle_map'] = new_obstacle_map
            
            airsim.write_file(f"test_images/action_image_{int(time.time())}.png", rgb_vis)

            with data_lock:
                maps_history.append(maps_copy)
            
            time.sleep(6)
            # Calculate action
            print("[Action] Starting to move forward...")

            #with airsim_lock:
            match Action_list[i]:
                case 0:
                    action_client.moveByVelocityAsync(0, -3, 0, duration=4).join()
                case 1:
                    action_client.moveByVelocityAsync(-3, 0, 0, duration=4).join()
                case 2:
                    action_client.rotateByYawRateAsync(-30, duration=3).join()
                case 3:
                    action_client.rotateByYawRateAsync(30, duration=3).join()
            print(f"[Action] Number {i} movement finished.")
    with detection_information_lock:
        detection_information["reach_max_steps"] = True

def detection_thread_func():
    detection_client = airsim.MultirotorClient()
    detection_client.confirmConnection()
    k = 0
    
    while detection_information["reach_max_steps"] == False and detection_information['detection_success'] == False:
        pil_image, depth_image, camera_position, camera_orientation, rgb_vis, _ = get_images(detection_client)
        camera_fov = 90
        
        airsim.write_file(f"test_images/detection_image_{int(time.time())}.png", rgb_vis)

        point_world = detection_test(detection_dino_processor, detection_dino_model, pil_image, depth_image, camera_fov, camera_position, camera_orientation, rgb_vis)
        if point_world is not None:
            with detection_information_lock:
                detection_information['detection_success'] = True
                detection_information['detected_position'][0] = float(point_world[0])
                detection_information['detected_position'][1] = float(point_world[1])
                detection_information['detected_position'][2] = float(point_world[2])

        print(f"[Detection] Number {k} Detection Done.")
        k += 1

def planning_thread_func():
    planning_client = airsim.MultirotorClient()
    planning_client.confirmConnection()
    j = 0
    
    while detection_information["reach_max_steps"] == False:
        if detection_information['detection_success']:
            print("[Planning] Detection already succeeded, skipping planning.")
            break
        else:
            get_obs_time_start = time.time()
            
            pil_image, depth_image, camera_position, camera_orientation, _, rgb_base64 = get_images(planning_client)
            camera_fov = 90
            
            with data_lock:
                maps_copy = maps.copy()
            attraction_map = maps_copy['attraction_map']
            exploration_map = maps_copy['exploration_map']
            
            get_obs_time_end = time.time()
            print(f"[Planning] Get observation time: {get_obs_time_end - get_obs_time_start:.2f} seconds")

            result_dict, attraction_scores = grounded_sam(dino_processor, dino_model, sam_processor, sam_model, pil_image, rgb_base64)

            if result_dict["success"] == False:
                print("[Planning] objects detection failed, skipping this frame.")
                continue
            
            map_update_time_start = time.time()
            
            added_masks = add_masks(result_dict["masks"])
            prepared_masks = downsample_masks(added_masks, scale_factor=2) # 1024*1024 to 512*512
            new_attraction_map, new_exploration_map, observed_points_cnt = map_update(attraction_map, exploration_map, prepared_masks, attraction_scores, start_position, depth_image, camera_fov, camera_position, camera_orientation)

            with data_lock:
                maps.update({
                    'attraction_map': new_attraction_map,
                    'exploration_map': new_exploration_map,
                    'observed_points_cnt': observed_points_cnt
                })
                
            map_update_time_end = time.time()
            print(f"[Planning] Map update time: {map_update_time_end - map_update_time_start:.2f} seconds")

            print(f"[Planning] Number {j} Planning Done.")
            j += 1

if __name__ == "__main__":

    action_thread = threading.Thread(target=action_thread_func, name="Action")
    planning_thread = threading.Thread(target=planning_thread_func, name="Planning")
    detection_thread = threading.Thread(target=detection_thread_func, name="Detection")

    print("Starting threads...")

    action_thread.start()

    # 稍作延时，确保action线程已经起飞
    time.sleep(4)
    detection_thread.start()
    planning_thread.start()

    try:
        action_thread.join()
        planning_thread.join()
        detection_thread.join()
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, stopping threads.")
        
        # 等待线程结束（带超时）
        action_thread.join(timeout=2.0)
        planning_thread.join(timeout=2.0)
        detection_thread.join(timeout=2.0)

        # 检查线程是否已停止
        if action_thread.is_alive():
            print("Warning: Action thread did not stop gracefully!")
        if planning_thread.is_alive():
            print("Warning: Planning thread did not stop gracefully!")
        if detection_thread.is_alive():
            print("Warning: Detection thread did not stop gracefully!")
    finally:
        # 确保程序退出时无人机恢复初始状态
        print("Resetting AirSim environment...")
        client = airsim.MultirotorClient()
        client.reset()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Done.")
        
        # Save the maps history
        for i in range(len(maps_history)):
            np.savetxt(f"test_images/attraction_map_{i}.txt", maps_history[i]['attraction_map'].flatten())
            np.savetxt(f"test_images/exploration_map_{i}.txt", maps_history[i]['exploration_map'].flatten())
            np.savetxt(f"test_images/obstacle_map_{i}.txt", maps_history[i]['obstacle_map'].flatten())