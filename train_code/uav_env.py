import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time

from uav_search.airsim_utils import get_train_images
from uav_search.action_model_inputs_test import map_input_preparation
from uav_search.train_code.map_updating_train import map_update

task_data = {
    0: {'target_position': np.array([-203.99273681640625,137.1072540283203,-0.6712745428085327]), 'start_position': airsim.Vector3r(-178.84925950766097,140.11065228596107,-10.0), 'object_description': " Tent: Geometric dome-like form, deep blue fabric with yellow trim and tension rods, taut surface with visible seams, used for outdoor shelter or camping."},
    1: {'target_position': np.array([-120.49274444580078,-81.59274291992188,4.728725433349609]), 'start_position': airsim.Vector3r(-145.62128359585262,-89.33258409721105,-10.0), 'object_description': " Truck:  Geometric rectangular form, industrial orange and black body with six wheels and a debris-filled cargo bed, bold \"steel\" branding on cab and sides, used for heavy-duty material transport."},
    2: {'target_position': np.array([-101.89273834228516,-244.19273376464844,3.5287256240844727]), 'start_position': airsim.Vector3r(-140.04161056069867,-232.02151874317585,-10.0), 'object_description': " Playground: Geometric modular structure, yellow-painted wood and red plastic, curved tube slide with rope climbing net and ladder steps, designed for children's recreational play."}
} # Will be loaded from json file in future

class AirSimDroneEnv(gym.Env):
    # metadata is not necessary for AirSim, but we include it for completeness
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

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
        self.client.takeoffAsync().join()

        # Map
        self._map_reset()
        self.grid_size = 5.0
        self.grid_origin_pose = np.array([20, 20, 5])
        # Task parameters
        self.task_id = 0
        self.object_description = ""
        self.start_position = airsim.Vector3r(0,0,0)
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.episode_step_count = 0
        self.max_steps_per_episode = 200 # 每回合最大步数
        # Distance to target
        self.last_dist_to_target = 0.0
        self.current_dist_to_target = 0.0
        # Log info
        self.reward_log = {'reward': [0,0,0,0,0]}
        print("AirSim environment initialized.")
    
    def _map_reset(self):
        self.current_ground_truth_attraction_map = task_data[self.task_id]['ground_truth_attraction_map'] # From dataset
        self.attraction_map = np.zeros((40, 40, 10, 2), dtype=np.float32)
        self.attraction_map[:, :, :, 1] = -1
        self.exploration_map = np.zeros((40, 40, 10), dtype=np.float32)
        self.obstacle_map = np.zeros((80, 80, 20), dtype=np.uint8)
        self.observed_points_cnt = 0 # Only used in training
        self.uav_pose = {
            'position': np.array([20, 20, 5]),
            'orientation': 0 # 0: north, 1: west, 2: south, 3: east
            }
            
    def _compute_new_position(self, forward_distance, position: airsim.Vector3r, yaw) -> airsim.Vector3r:
        dx = forward_distance * np.cos(yaw)
        dy = forward_distance * np.sin(yaw)

        new_position = airsim.Vector3r(position.x_val + dx, position.y_val + dy, position.z_val)
        return new_position
    
    def _map_update(self):
        # 更新地图
        depth_image, camera_position, camera_orientation = get_train_images(self.client)
        camera_fov = 90

        new_attraction_map, new_exploration_map, new_obstacle_map, attraction_reward, exploration_reward = map_update(
            self.attraction_map,
            self.exploration_map,
            self.obstacle_map,
            self.current_ground_truth_attraction_map,
            self.start_position,
            depth_image,
            camera_fov,
            camera_position,
            camera_orientation
        )
            
        self.attraction_map = new_attraction_map
        self.exploration_map = new_exploration_map
        self.obstacle_map = new_obstacle_map
        
        return attraction_reward, exploration_reward

    def _update_uav_pose_from_airsim(self):
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        delta_position = np.array([
            position.y_val - self.start_position.y_val,
            position.x_val - self.start_position.x_val,
            position.z_val - self.start_position.z_val 
        ])
        
        grid_position = np.round(delta_position / self.grid_size).astype(int) + self.grid_origin_pose

        _, _, yaw_rad = airsim.to_eularian_angles(orientation)
        yaw_deg = np.degrees(yaw_rad)
        
        if -45 <= yaw_deg <= 45:
            grid_orientation = 0 # north
        elif -135 < yaw_deg <= -45:
            grid_orientation = 1 # west
        elif yaw_deg > 135 or yaw_deg < -135:
            grid_orientation = 2 # south
        else: # 45 <= yaw_deg <= 135
            grid_orientation = 3 # east
            
        # 更新内部状态
        self.uav_pose['position'] = grid_position
        self.uav_pose['orientation'] = grid_orientation

    def _get_obs(self):
        # Get input map
        map_input = map_input_preparation(self.attraction_map, self.exploration_map, self.obstacle_map, self.uav_pose)

        return map_input

    def _compute_reward(self,terminated, attraction_reward=0.0, exploration_reward=0.0):
        # Reward weights
        W_ATTRACTION = 0.1
        W_EXPLORATION = 0.01
        W_DISTANCE = 1.5
        W_SPARSE = 1.0
        
        # Distance-based reward
        STEP_PENALTY = -0.2
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
            
        # Log
        self.reward_log['reward'][0] = dis_reward
        self.reward_log['reward'][1] = sparse_reward
        self.reward_log['reward'][2] = attraction_reward
        self.reward_log['reward'][3] = exploration_reward

        return W_DISTANCE * dis_reward + W_SPARSE * sparse_reward + W_ATTRACTION * attraction_reward + W_EXPLORATION * exploration_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(f"Resetting environment for task {self.task_id}")
        
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.start_position = task_data[self.task_id]['start_position'] # From dataset
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(self.start_position, self.start_orientation)
        self.client.simSetVehiclePose(pose, True)
        
        self.object_description = task_data[self.task_id]['object_description'] # From dataset
        self.target_position = task_data[self.task_id]['target_position'] # From dataset
        self.current_dist_to_target = np.linalg.norm(self.target_position - np.array([self.start_position.x_val, self.start_position.y_val, self.start_position.z_val]))
        self.last_dist_to_target = self.current_dist_to_target

        self.episode_step_count = 0
        
        # Map reset
        self._map_reset()

        # 获取初始观测值
        _, _ = self._map_update()
        initial_observation = self._get_obs()
        info = {}  # 初始info为空字典

        self.task_id = (self.task_id + 1) % 40

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
        
        # 0:north, 1:west, 2:south, 3:east
        YAW_ANGLES = [0, -90, 180, 90]
        
        match action:
            case 0: 
                self.client.moveToPositionAsync(new_position.x_val, new_position.y_val, new_position.z_val, 3).join()
            case 1:
                current_orientation_idx = self.uav_pose['orientation']
                new_orientation_idx = (current_orientation_idx + 1) % 4
                target_yaw = YAW_ANGLES[new_orientation_idx]
                self.client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            case 2:
                current_orientation_idx = self.uav_pose['orientation']
                new_orientation_idx = (current_orientation_idx - 1 + 4) % 4
                target_yaw = YAW_ANGLES[new_orientation_idx]
                self.client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            case 3:
                current_orientation_idx = self.uav_pose['orientation']
                new_orientation_idx = (current_orientation_idx + 2) % 4
                target_yaw = YAW_ANGLES[new_orientation_idx]
                self.client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            case 4:
                self.client.moveToZAsync(position.z_val - self.grid_size, 2).join()
            case 5:
                self.client.moveToZAsync(position.z_val + self.grid_size, 2).join()
        
        self._update_uav_pose_from_airsim()
        
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
            self.client.reset()
        
        if self.current_dist_to_target < 10.0:
            terminated = True
            print("Target reached! Episode terminated.")

        truncated = False
        if self.episode_step_count >= self.max_steps_per_episode:
            truncated = True
            print("Max steps reached. Episode truncated.")
        
        if not terminated and not truncated:
            attraction_reward, exploration_reward = self._map_update()
            observation = self._get_obs()
        else:
            observation = self._get_obs()

        reward = self._compute_reward(terminated, attraction_reward, exploration_reward)

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