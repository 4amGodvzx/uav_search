import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

from airsim_utils import get_images
from uav_search.test_code.grounded_sam_train import grounded_sam
from uav_search.test_code.map_updating_test import add_masks, downsample_masks, map_update
from action_model_inputs_test import obstacle_update, map_input_preparation, _crop_rotate_and_pad

task_data = {
    0: {'target_position': np.array([-203.99273681640625,137.1072540283203,-0.6712745428085327]), 'start_position': airsim.Vector3r(-178.84925950766097,140.11065228596107,-10.0), 'object_description': " Tent: Geometric dome-like form, deep blue fabric with yellow trim and tension rods, taut surface with visible seams, used for outdoor shelter or camping."},
    1: {'target_position': np.array([-120.49274444580078,-81.59274291992188,4.728725433349609]), 'start_position': airsim.Vector3r(-145.62128359585262,-89.33258409721105,-10.0), 'object_description': " Truck:  Geometric rectangular form, industrial orange and black body with six wheels and a debris-filled cargo bed, bold \"steel\" branding on cab and sides, used for heavy-duty material transport."},
    2: {'target_position': np.array([-101.89273834228516,-244.19273376464844,3.5287256240844727]), 'start_position': airsim.Vector3r(-140.04161056069867,-232.02151874317585,-10.0), 'object_description': " Playground: Geometric modular structure, yellow-painted wood and red plastic, curved tube slide with rope climbing net and ladder steps, designed for children's recreational play."}
}

class AirSimDroneEnv(gym.Env):
    # metadata is not necessary for AirSim, but we include it for completeness
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

        DEVICE = "cuda:0"
        # Model loading
        model_directory = "/data/zhangdaoxuan/models/models-Qwen2.5-VL-7B-Instruct"
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_directory,
            dtype=torch.float16,
            attn_implementation="sdpa"
        ).to(DEVICE)
        self.qwen_processor = AutoProcessor.from_pretrained(model_directory)
    
        dino_model_directory = "/data/zhangdaoxuan/models/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_directory)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_directory).to(DEVICE)

        sam_model_directory = "/data/zhangdaoxuan/models/models--facebook--sam-vit-huge/snapshots/87aecf0df4ce6b30cd7de76e87673c49644bdf67"
        self.sam_processor = SamProcessor.from_pretrained(sam_model_directory)
        self.sam_model = SamModel.from_pretrained(sam_model_directory).to(DEVICE)
        print("Models loaded.")
        # Action space
        self.action_space = spaces.Discrete(6) 
        # 动作映射
        self.action = [0,1,2,3,4,5] # 0:forward, 1:rotate left, 2:rotate right, 3:rotate backward, 4:ascend, 5:descend
        # Observation Space
        self.observation_space = spaces.Dict({
            "attraction_map_input": spaces.Box(low=0, high=1, shape=(10, 20, 20), dtype=np.float32),
            "exploration_map_input": spaces.Box(low=0, high=1000, shape=(10, 20, 20), dtype=np.float32),
            "obstacle_map_input": spaces.Box(low=0, high=1, shape=(8, 40, 40), dtype=np.uint8)
        })
        # AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # Initial position and orientation
        self.start_position = airsim.Vector3r(0,0,0)
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        # Map
        self.attraction_map = np.zeros((40, 40, 10, 2), dtype=np.float32)
        self.attraction_map[:, :, :, 1] = -1
        self.exploration_map = np.zeros((40, 40, 10), dtype=np.float32)
        self.obstacle_map = np.zeros((80, 80, 20), dtype=np.uint8)
        self.observed_points_cnt = 0 # Only used in training
        self.uav_pose = {
            'position': [20, 20, 5],
            'orientation': 0 # 0: north, 1: west, 2: south, 3: east
            }
        # Task parameters
        self.task_id = 0
        self.object_description = ""
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.episode_step_count = 0
        self.max_steps_per_episode = 200 # 每回合最大步数
        # Distance to target
        self.last_dist_to_target = 0.0
        self.current_dist_to_target = 0.0
        # Log info
        self.reward_log = {'reward': [0,0,0,0,0]}
        print("AirSim environment initialized.")
        
    def _compute_new_position(self, forward_distance, position: airsim.Vector3r, yaw) -> airsim.Vector3r:
        dx = forward_distance * np.cos(yaw)
        dy = forward_distance * np.sin(yaw)

        new_position = airsim.Vector3r(position.x_val + dx, position.y_val + dy, position.z_val)
        return new_position
    
    def _map_update(self):
        # 更新地图
        pil_image, depth_image, camera_position, camera_orientation, _, rgb_base64 = get_images(self.client)
        camera_fov = 90
            
        new_obstacle_map = obstacle_update(self.obstacle_map, self.start_position, depth_image, camera_fov, camera_position, camera_orientation)
        self.obstacle_map = new_obstacle_map
            
        result_dict, attraction_scores = grounded_sam(self.qwen_processor, self.qwen_model, self.dino_processor, self.dino_model, self.sam_processor, self.sam_model, pil_image, self.object_description, rgb_base64)
        if result_dict["success"] == True:
            added_masks = add_masks(result_dict['masks'])
            prepared_masks = downsample_masks(added_masks, scale_factor=2) # 1024*1024 to 512*512
            new_attraction_map, new_exploration_map, observed_points_cnt = map_update(self.attraction_map, self.exploration_map, prepared_masks, attraction_scores, self.start_position, depth_image, camera_fov, camera_position, camera_orientation)
            self.attraction_map = new_attraction_map
            self.exploration_map = new_exploration_map
            self.observed_points_cnt = observed_points_cnt

    def _get_obs(self):
        # Get input map
        map_input = map_input_preparation(self.attraction_map, self.exploration_map, self.obstacle_map, self.uav_pose)

        return map_input

    def _compute_reward(self,terminated):
        STEP_PENALTY = -0.2
        
        # Distance-based reward
        distance_decrease = self.last_dist_to_target - self.current_dist_to_target
        if 10 <= self.current_dist_to_target < 30:
            k = 2.0
        elif 30 <= self.current_dist_to_target < 80:
            k = 1.0
        elif 80 <= self.current_dist_to_target < 150:
            k = 0.5
        else: # >= 150
            k = 0.2
        
        dis_reward = k * distance_decrease + STEP_PENALTY
        
        sparse_reward = 0.0
        if terminated:
            if self.current_dist_to_target < 10.0: # successful termination
                sparse_reward = 200.0
            else: # failure termination
                sparse_reward = -200.0
            
        # Map processing for reward calculation
        cropped_attraction_map = _crop_rotate_and_pad(
            full_map=self.attraction_map[:, :, :, 0],
            center_coords=self.uav_pose['position'],
            crop_size=(8, 16, 6),
            padding_value=-1.0,
            orientation=self.uav_pose['orientation']
        )[:, 8:15, :]
        cropped_exploration_map = _crop_rotate_and_pad(
            full_map=self.exploration_map,
            center_coords=self.uav_pose['position'],
            crop_size=(8, 16, 6),
            padding_value=-1.0,
            orientation=self.uav_pose['orientation']
        )[:, 8:15, :]
        
        # Attraction-based reward
        if cropped_attraction_map.size == 0:
            attraction_reward = 0.0
        else:
            attraction_reward = (np.max(cropped_attraction_map) - 0.3) * 20
        
        # Exploration-based reward
        exploration_reward = -np.mean(cropped_exploration_map) * 2
        
        # Punishment for inactive observation
        observation_reward = 0.0
        if self.observed_points_cnt < 25000:
            observation_reward = (25000 - self.observed_points_cnt) * 0.002
            
        # Log
        self.reward_log['reward'][0] = dis_reward
        self.reward_log['reward'][1] = sparse_reward
        self.reward_log['reward'][2] = attraction_reward
        self.reward_log['reward'][3] = exploration_reward
        self.reward_log['reward'][4] = observation_reward
        
        return dis_reward + sparse_reward + attraction_reward + exploration_reward + observation_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置无人机状态
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.client.takeoffAsync().join()
        self.start_position = task_data[self.task_id]['start_position'] # From dataset
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(self.start_position, self.start_orientation)
        self.client.simSetVehiclePose(pose, True)
        
        self.object_description = task_data[self.task_id]['object_description'] # From dataset
        self.target_position = task_data[self.task_id]['target_position'] # From dataset
        self.last_dist_to_target = 0.0
        self.current_dist_to_target = np.linalg.norm(self.target_position - np.array([self.start_position.x_val, self.start_position.y_val, self.start_position.z_val]))

        self.episode_step_count = 0
        
        # Map reset
        self.attraction_map = np.zeros((40, 40, 10, 2), dtype=np.float32)
        self.attraction_map[:, :, :, 1] = -1
        self.exploration_map = np.zeros((40, 40, 10), dtype=np.float32)
        self.obstacle_map = np.zeros((80, 80, 20), dtype=np.uint8)
        self.observed_points_cnt = 0
        self.uav_pose = {
            'position': [20, 20, 5],
            'orientation': 0
            }
        
        # 获取初始观测值#
        self._map_update()
        initial_observation = self._get_obs()
        info = {} # 初始info为空字典
        
        self.task_id += 1

        return initial_observation, info

    def step(self, action):
        self.episode_step_count += 1
        # Log info
        if self.episode_step_count % 100 == 0:
            print(f"last_uav_pose: {self.uav_pose}")
        
        start_time = time.time()
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        
        np_position = np.array([position.x_val, position.y_val, position.z_val])
        new_position = self._compute_new_position(10, position, yaw)
        
        match action:
            case 0: 
                self.client.moveToPositionAsync(new_position.x_val, new_position.y_val, new_position.z_val, 3).join()
                match self.uav_pose['orientation']:
                    case 0: self.uav_pose['position'][1] += 2
                    case 1: self.uav_pose['position'][0] -= 2
                    case 2: self.uav_pose['position'][1] -= 2
                    case 3: self.uav_pose['position'][0] += 2
            case 1:
                self.client.rotateByYawRateAsync(-30, 3).join()
                self.uav_pose['orientation'] = (self.uav_pose['orientation'] + 1) % 4
            case 2:
                self.client.rotateByYawRateAsync(30, 3).join()
                if self.uav_pose['orientation'] == 0:
                    self.uav_pose['orientation'] = 3
                else:
                    self.uav_pose['orientation'] = self.uav_pose['orientation'] - 1
            case 3:
                self.client.rotateByYawRateAsync(-60, 3).join()
                self.uav_pose['orientation'] = (self.uav_pose['orientation'] + 2) % 4
            case 4:
                self.client.moveToZAsync(position.z_val - 5, 2).join()
                self.uav_pose['position'][2] -= 1
            case 5:
                self.client.moveToZAsync(position.z_val + 5, 2).join()
                self.uav_pose['position'][2] += 1
        time.sleep(0.2) # 等待一下，确保状态更新
        
        # 检查是否结束 (Terminated & Truncated)
        terminated = False
        self.last_dist_to_target = self.current_dist_to_target
        self.current_dist_to_target = np.linalg.norm(self.target_position - np_position)
        collision_info = self.client.simGetCollisionInfo()

        if self.uav_pose['position'][0] < 0 or self.uav_pose['position'][0] >= 40 or \
           self.uav_pose['position'][1] < 0 or self.uav_pose['position'][1] >= 40 or \
           self.uav_pose['position'][2] < 0 or self.uav_pose['position'][2] >= 10:
            terminated = True
            print("Out of bounds! Episode terminated.")

        if collision_info.has_collided:
            terminated = True
            print("Collision detected! Episode terminated.")
        
        if self.current_dist_to_target < 10.0:
            terminated = True
            print("Target reached! Episode terminated.")

        truncated = False
        if self.episode_step_count >= self.max_steps_per_episode:
            truncated = True
            print("Max steps reached. Episode truncated.")
        
        if not terminated and not truncated:
            self._map_update()
            observation = self._get_obs()
        else:
            observation = self._get_obs()
        
        reward = self._compute_reward(terminated)

        end_time = time.time()
        # Log info
        if self.episode_step_count % 100 == 0:
            print(f"Task {self.task_id}, Step {self.episode_step_count}")
            print(f"Action taken: {action}")
            print(f"Reward components: {self.reward_log['reward']}")
            print(f"Current_uav_pose: {self.uav_pose}")
            print(f"Step time: {end_time - start_time} seconds")
            print(f"Start position: {self.start_position}, Current position: {np_position}, Distance to target: {self.current_dist_to_target}")
            
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("AirSim environment closed.")