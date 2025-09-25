import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time

class AirSimDroneEnv(gym.Env):
    """
    一个用于AirSim无人机导航的自定义Gymnasium环境。
    任务: 无人机需要飞到一个固定的目标点。
    """
    # metadata用于渲染等，对于AirSim可以简单设置
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

        # 1. 初始化AirSim客户端
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 2. 定义动作空间 (Action Space)
        # 我们定义6个离散动作: 前, 后, 左, 右, 上, 下
        self.action_space = spaces.Discrete(6) 
        # 动作映射
        self.action_map = {
            0: (5, 0, 0),    # 向前飞
            1: (-5, 0, 0),   # 向后飞
            2: (0, 5, 0),    # 向右飞
            3: (0, -5, 0),   # 向左飞
            4: (0, 0, -5),   # 向上飞 (Z轴在AirSim中是向下的)
            5: (0, 0, 5),    # 向下飞
        }

        # 3. 定义观测空间 (Observation Space)
        # 观测值是无人机相对于目标点的向量 (dx, dy, dz)
        # 我们使用Box空间，因为值是连续的
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(3,), dtype=np.float32)

        # 4. 设置任务参数
        self.target_position = np.array([30.0, 0.0, -10.0]) # 目标点坐标 (x, y, z)
        self.episode_step_count = 0
        self.max_steps_per_episode = 200 # 每回合最大步数
        print("AirSim environment initialized.")

    def _get_obs(self):
        """获取当前观测值"""
        # 获取无人机状态
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        
        # 将AirSim坐标转换为numpy数组
        current_position = np.array([position.x_val, position.y_val, position.z_val])
        
        # 计算到目标的相对位置向量
        relative_position = self.target_position - current_position
        return relative_position.astype(np.float32)

    def _compute_reward(self, current_pos):
        """计算奖励"""
        # 目标距离
        dist_to_target = np.linalg.norm(self.target_position - current_pos)

        # 奖励设计:
        # 1. 距离目标越近，奖励越高
        #    鼓励接近
        reward = 30 - dist_to_target

        # 2. 碰撞惩罚
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 500 # 巨大的负奖励

        # 3. 到达目标奖励
        if dist_to_target < 5.0: # 距离小于5米算作到达
            reward += 1000 # 巨大的正奖励

        return reward

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 重置无人机状态
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # 起飞
        self.client.takeoffAsync().join()
        # 飞到一个初始高度
        #self.client.moveToZAsync(-10, 1).join()

        self.episode_step_count = 0
        
        # 获取初始观测值
        initial_observation = self._get_obs()
        info = {} # 初始info为空字典

        return initial_observation, info

    def step(self, action):
        """执行一步动作"""
        self.episode_step_count += 1

        # 1. 将离散action映射到AirSim控制指令
        vx, vy, vz = self.action_map[action]
        # 以速度控制无人机移动1秒
        self.client.moveByVelocityAsync(vx, vy, vz, duration=1).join()
        time.sleep(0.1) # 等待一下，确保状态更新

        # 2. 获取新状态 (Observation)
        observation = self._get_obs()
        
        # 3. 获取无人机当前位置用于计算奖励
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_position = np.array([pos.x_val, pos.y_val, pos.z_val])

        # 4. 计算奖励 (Reward)
        reward = self._compute_reward(current_position)

        # 5. 检查是否结束 (Terminated & Truncated)
        terminated = False
        dist_to_target = np.linalg.norm(self.target_position - current_position)
        collision_info = self.client.simGetCollisionInfo()

        if collision_info.has_collided:
            terminated = True
            print("Collision detected! Episode terminated.")
        
        if dist_to_target < 5.0:
            terminated = True
            print("Target reached! Episode terminated.")

        truncated = False
        if self.episode_step_count >= self.max_steps_per_episode:
            truncated = True
            print("Max steps reached. Episode truncated.")

        # 6. 额外信息 (Info)
        info = {'distance_to_target': dist_to_target}

        return observation, reward, terminated, truncated, info

    def close(self):
        """关闭环境"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
        print("AirSim environment closed.")

