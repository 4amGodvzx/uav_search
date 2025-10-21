import multiprocessing
import subprocess
import time
import signal
import json
import datetime
import os
import airsim
import numpy as np
from gymnasium import spaces

# 地图和栅格化参数
GRID_SCALE = 5.0
UAV_START_GRID_POS = np.array([20, 20, 5])

ACTION_SPACE = spaces.Discrete(6)

# 动作定义
YAW_ANGLES = [0, -90, 180, 90]  # North, West, South, East

class UAVSearchAgent:
    def __init__(self, start_position: airsim.Vector3r, target_position: airsim.Vector3r, object_name: str, object_description: str, log_dir="experiment_logs"):
        """
        初始化智能体和实验环境
        :param start_position: airsim.Vector3r, 无人机起始位置
        :param target_position: airsim.Vector3r, 目标位置
        :param object_name: str, 目标物体名称 (e.g., "Van")
        :param object_description: str, 目标物体详细描述
        :param log_dir: str, 实验日志保存目录
        """
        self.start_position = start_position
        self.start_orientation = airsim.utils.to_quaternion(0, 0, 0)
        self.target_position = target_position
        self.object_name = object_name
        self.object_description = object_description

        # 初始化日志记录
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"experiment_{timestamp}.json")

        # 初始化多进程管理器和共享数据
        self.manager = multiprocessing.Manager()
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        self.experiment_data = self.manager.list()

    def _update_uav_pose_from_airsim(self, state: airsim.MultirotorState):
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        delta_position = np.array([
            position.x_val - self.start_position.x_val,
            position.y_val - self.start_position.y_val,
            position.z_val - self.start_position.z_val
        ])
        
        grid_position = np.round(delta_position / GRID_SCALE).astype(int) + UAV_START_GRID_POS

        _, _, yaw_rad = airsim.to_eularian_angles(orientation)
        yaw_deg = np.degrees(yaw_rad)
        
        if -45 <= yaw_deg <= 45:
            grid_orientation = 0  # north
        elif -135 < yaw_deg <= -45:
            grid_orientation = 1  # west
        elif yaw_deg > 135 or yaw_deg < -135:
            grid_orientation = 2  # south
        else:  # 45 <= yaw_deg <= 135
            grid_orientation = 3  # east
            
        uav_pose = {'position': grid_position, 'orientation': grid_orientation}
        return uav_pose

    def _action_process(self):
        action_client = airsim.MultirotorClient(port=41460)
        action_client.confirmConnection()
        action_client.reset()
        action_client.enableApiControl(True)
        action_client.armDisarm(True)

        action_client.takeoffAsync().join()
        pose = airsim.Pose(self.start_position, self.start_orientation)
        action_client.simSetVehiclePose(pose, True)
        
        uav_pose = {'position': UAV_START_GRID_POS, 'orientation': 0}
        path_length = 0.0
        
        print("[Action] Initialize complete.")

        for i in range(80): # Max steps
            log_entry = {'step': i}
            
            step_start_time = time.time()
            
            action = ACTION_SPACE.sample()

            log_entry['step_duration'] = time.time() - step_start_time
            log_entry['uav_pose_before_action'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in uav_pose.items()}
            log_entry['action_taken'] = action

            state = action_client.getMultirotorState()
            position = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(orientation)
            
            dx = 10.0 * np.cos(yaw)
            dy = 10.0 * np.sin(yaw)
            new_position = airsim.Vector3r(position.x_val + dx, position.y_val + dy, position.z_val)
            
            if action == 0:
                action_client.moveToPositionAsync(new_position.x_val, new_position.y_val, new_position.z_val, 3, timeout_sec=5).join()
                path_length += 10.0
            elif action == 1:
                target_yaw = YAW_ANGLES[(uav_pose['orientation'] + 1) % 4]
                action_client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            elif action == 2:
                target_yaw = YAW_ANGLES[(uav_pose['orientation'] - 1 + 4) % 4]
                action_client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            elif action == 3:
                target_yaw = YAW_ANGLES[(uav_pose['orientation'] + 2) % 4]
                action_client.rotateToYawAsync(target_yaw, timeout_sec=3).join()
            elif action == 4:
                action_client.moveToZAsync(position.z_val - 5.0, 2, timeout_sec=3).join()
                path_length += 5.0
            elif action == 5:
                action_client.moveToZAsync(position.z_val + 5.0, 2, timeout_sec=3).join()
                path_length += 5.0

            print(f"[Action] Step {i} movement finished.")
            
            new_state = action_client.getMultirotorState()
            uav_pose = self._update_uav_pose_from_airsim(new_state)
            
            # Termination check
            terminated = False
            termination_reason = "none"
            collision_info = action_client.simGetCollisionInfo()
            if collision_info.has_collided:
                terminated = True
                termination_reason = "collision"

            log_entry['uav_pose_after_action'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in uav_pose.items()}
            dis_to_target = np.linalg.norm(np.array([position.x_val,position.y_val,position.z_val]) - np.array([self.target_position.x_val, self.target_position.y_val, self.target_position.z_val]))
            log_entry['dis_to_target'] = dis_to_target
            log_entry['oracle_success'] = True if dis_to_target < 10.0 else False
            self.experiment_data.append(log_entry)
            
            if terminated:
                print(f"[Action] Terminated due to: {termination_reason}")
                log_entry['event'] = 'terminated'
                log_entry['termination_reason'] = termination_reason
                self.experiment_data.append({'total_path_length': path_length})

        print("[Action] Reached max steps.")
        self.experiment_data.append({'event': 'max_steps_reached'})
        self.experiment_data.append({'total_path_length': path_length})

    def run(self):
        processes = [
            multiprocessing.Process(target=self._action_process, name="Action"),
        ]

        print("Starting processes...")
        for p in processes:
            p.start()

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, terminating processes.")
            for p in processes:
                p.terminate()
            time.sleep(1) # Give time for processes to terminate
            for p in processes:
                p.join()
        finally:
            print("All processes have finished. Saving log...")
            with open(self.log_file, 'w') as f:
                json.dump(list(self.experiment_data), f, indent=4)
            print(f"Log saved to {self.log_file}")
            
            print("Resetting AirSim environment...")
            client = airsim.MultirotorClient(port=41460)
            client.reset()
            client.armDisarm(False)
            client.enableApiControl(False)
            print("Done.")

def wait_for_airsim_ready(timeout_sec=120):
    time.sleep(10)
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            print("Waiting for AirSim to become ready...")
            client = airsim.MultirotorClient(port=41460)
            client.confirmConnection()
            print("AirSim connection confirmed.")
            # 再等一小会儿，确保场景完全加载
            time.sleep(2)
            return True
        except:
            print("Connection failed, retrying in 5 seconds...")
            time.sleep(5)
    print("Error: AirSim did not become ready within the timeout period.")
    return False

class ExperimentRunner:
    def __init__(self, tasks_file, map_scripts_config, base_log_dir="experiment_logs"):
        """
        初始化实验运行器。
        :param tasks_file: 包含所有任务的JSON文件路径。
        :param map_scripts_config: 一个字典，将地图名称映射到其启动脚本路径。
        :param base_log_dir: 保存所有实验日志的根目录。
        """
        self.tasks_file = tasks_file
        self.map_scripts = map_scripts_config
        self.base_log_dir = base_log_dir
        self.tasks = []
        self.current_map_name = None
        self.airsim_process = None
        os.makedirs(self.base_log_dir, exist_ok=True)

    def load_tasks(self):
        """从JSON文件加载任务列表。"""
        with open(self.tasks_file, 'r') as f:
            self.tasks = json.load(f)
        print(f"Successfully loaded {len(self.tasks)} tasks from {self.tasks_file}")

    def _manage_airsim_map(self, target_map_name):
        # 检查地图是否已经是目标地图且进程正在运行
        if target_map_name == self.current_map_name and self.airsim_process and self.airsim_process.poll() is None:
            print(f"Map '{target_map_name}' is already running.")
            return True

        print(f"Switching map... Current: '{self.current_map_name}', Target: '{target_map_name}'")

        # 如果有正在运行的AirSim进程，则终止它
        if self.airsim_process:
            self.cleanup()

        # 找到目标地图的启动脚本
        script_path = self.map_scripts.get(target_map_name)
        if not script_path:
            print(f"Error: No launch script found for map: {target_map_name}")
            return False
        
        # 启动新的AirSim进程
        print(f"Launching new AirSim process for map '{target_map_name}'...")
        launch_command = ['bash', script_path, '-RenderOffscreen', '-NoSound', '-NoVSync', '-GraphicsAdapter=0']
        self.airsim_process = subprocess.Popen(launch_command, start_new_session=True)
        self.current_map_name = target_map_name
        
        # 等待AirSim完全加载并准备就绪
        return wait_for_airsim_ready()

    def run_all_experiments(self):
        self.load_tasks()
        
        try:
            for task in self.tasks:
                task_id = task['task_id']
                map_name = task['map']
                
                print("\n" + "="*50)
                print(f"STARTING TASK {task_id}: Find '{task['object_name']}' in map '{map_name}'")
                print("="*50)

                # 1. 启动或切换到正确的地图
                if not self._manage_airsim_map(map_name):
                    print(f"Failed to launch map {map_name}. Skipping task {task_id}.")
                    continue

                # 2. 准备Agent的参数
                start_pos_list = task['start_position']
                target_pos_list = task['object_position']
                exp_start_position = airsim.Vector3r(start_pos_list[0], start_pos_list[1], start_pos_list[2])
                exp_target_position = airsim.Vector3r(target_pos_list[0], target_pos_list[1], target_pos_list[2])
                
                # 为每个任务创建独立的日志目录
                task_log_dir = os.path.join(self.base_log_dir, f"task_{task_id}_{map_name}")

                # 3. 创建并运行Agent
                agent = UAVSearchAgent(
                    start_position=exp_start_position,
                    target_position=exp_target_position,
                    object_name=task['object_name'],
                    object_description=task['description'],
                    log_dir=task_log_dir
                )
                agent.run()

                print(f"TASK {task_id} COMPLETED.")
                # 在任务之间留出一点时间，以防万一
                time.sleep(5)

        except Exception as e:
            print(f"An unexpected error occurred during experiments: {e}")
        finally:
            print("All experiments finished. Cleaning up...")
            self.cleanup()

    def cleanup(self):
        if self.airsim_process and self.airsim_process.poll() is None: # 检查进程是否仍在运行
            print(f"Attempting to terminate AirSim process group with PGID: {os.getpgid(self.airsim_process.pid)}")
            try:
                pgid = os.getpgid(self.airsim_process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.airsim_process.wait(timeout=10)
                print("Process group terminated gracefully.")
                time.sleep(2) # 确保进程完全关闭
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(1)
            except subprocess.TimeoutExpired:
                print("Process group did not terminate gracefully, forcing kill (SIGKILL).")
                pgid = os.getpgid(self.airsim_process.pid)
                os.killpg(pgid, signal.SIGKILL)
                self.airsim_process.wait()
                print("Process group killed.")
            except ProcessLookupError:
                print("Process was already gone before termination signal could be sent.")
            except Exception as e:
                print(f"An error occurred while closing the process group: {e}")
        self.airsim_process = None
        self.client = None
        time.sleep(7)
        print("Environment closed.")

if __name__ == "__main__":
    # --- 自动化测试配置 ---

    # 1. 定义你的地图启动脚本路径
    #    你需要根据你的项目结构修改这些路径
    MAP_SCRIPTS_CONFIG = {
        "BrushifyUrban": "BrushifyUrban/BrushifyUrban.sh",
        "CabinLake": "CabinLake/CabinLake.sh",
        "DownTown": "DownTown/DownTown_test1.sh",
        "Neighborhood": "Neighborhood/NewNeighborhood.sh",
        "Slum": "Slum/Slum_test1.sh",
        "UrbanJapan": "UrbanJapan/UrbanJapan.sh",
        "Venice": "Venice/Vinice_test1.sh",
        "WesternTown": "WesternTown/WesternTown_test1.sh",
        "WinterTown": "WinterTown/WinterTown_test1.sh",
        "Barnyard": "Barnyard/Barnyard_test1.sh",
        "CityStreet": "CityStreet/CleanCityStreet.sh",
        "NYC": "NYC/NYC1950.sh"
    }

    # 2. 定义任务配置文件的路径
    TASKS_JSON_PATH = "uav_search/task_map/val_tasks.json"
    
    # 3. 定义日志保存的根目录
    BASE_LOG_DIR = "all_experiment_logs"

    # --- 启动实验 ---
    if not os.path.exists(TASKS_JSON_PATH):
        print(f"Error: Tasks file not found at {TASKS_JSON_PATH}")
    else:
        runner = ExperimentRunner(
            tasks_file=TASKS_JSON_PATH,
            map_scripts_config=MAP_SCRIPTS_CONFIG,
            base_log_dir=BASE_LOG_DIR
        )
        runner.run_all_experiments()

