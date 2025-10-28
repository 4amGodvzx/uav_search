import multiprocessing
import subprocess
import time
import signal
import json
import datetime
import os
import airsim
import numpy as np
from collections import deque
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from gymnasium import spaces

from uav_search.detection_test import detection_test
from uav_search.airsim_utils import get_images

DEVICE = "cuda:0"
DINO_MODEL_DIR = "models/models-grounding-dino-base"

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

MAP_SIZE_METERS = 100  # 地图边长（米）
MAP_RESOLUTION = 0.5   # 地图分辨率（米/像素）
MAP_SHAPE = (int(MAP_SIZE_METERS / MAP_RESOLUTION), int(MAP_SIZE_METERS / MAP_RESOLUTION))

# 地图和栅格化参数
GRID_SCALE = 5.0
UAV_START_GRID_POS = np.array([20, 20, 5])

ACTION_SPACE = spaces.Discrete(7)

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
        
        self.grid_map = np.full(MAP_SHAPE, UNKNOWN, dtype=np.int8)
        self.map_origin = np.array([-MAP_SIZE_METERS / 2, -MAP_SIZE_METERS / 2])

        # 初始化日志记录
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"experiment_{timestamp}.json")

        # 初始化多进程管理器和共享数据
        self.manager = multiprocessing.Manager()
        self._initialize_shared_memory()
        
    def _world_to_grid(self, world_pos):
        if isinstance(world_pos, airsim.Vector3r):
            world_pos = np.array([world_pos.x_val, world_pos.y_val])
        grid_pos = (world_pos - self.map_origin) / MAP_RESOLUTION
        return grid_pos.astype(int)

    def _grid_to_world(self, grid_pos):
        world_pos = grid_pos.astype(float) * MAP_RESOLUTION + self.map_origin
        # 返回一个可以在AirSim中使用的Vector3r，Z值需要单独处理
        return airsim.Vector3r(world_pos[0], world_pos[1], 0)

    def _update_map_with_lidar(self, client, uav_pos_grid):
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            return

        points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))

        pose = lidar_data.pose
        for p in points:
            world_point = pose.position + airsim.Vector3r(p[0], p[1], p[2])
            
            obstacle_grid = self._world_to_grid(world_point)
            
            x0, y0 = uav_pos_grid
            x1, y1 = obstacle_grid
            
            if 0 <= x1 < MAP_SHAPE[0] and 0 <= y1 < MAP_SHAPE[1]:
                dx, dy = abs(x1 - x0), -abs(y1 - y0)
                sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
                err = dx + dy
                while True:
                    if 0 <= x0 < MAP_SHAPE[0] and 0 <= y0 < MAP_SHAPE[1]:
                        self.grid_map[x0, y0] = FREE
                    if x0 == x1 and y0 == y1:
                        break
                    e2 = 2 * err
                    if e2 >= dy:
                        err += dy
                        x0 += sx
                    if e2 <= dx:
                        err += dx
                        y0 += sy
                self.grid_map[x1, y1] = OCCUPIED


    def _detect_frontiers(self):
        frontiers = []
        visited = np.zeros_like(self.grid_map, dtype=bool)
        
        for r in range(MAP_SHAPE[0]):
            for c in range(MAP_SHAPE[1]):
                if self.grid_map[r, c] == FREE and not visited[r, c]:
                    is_frontier_cell = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < MAP_SHAPE[0] and 0 <= nc < MAP_SHAPE[1] and self.grid_map[nr, nc] == UNKNOWN:
                                is_frontier_cell = True
                                break
                        if is_frontier_cell:
                            break
                    
                    if is_frontier_cell:
                        current_frontier = []
                        q = deque([(r, c)])
                        visited[r, c] = True
                        
                        while q:
                            curr_r, curr_c = q.popleft()
                            current_frontier.append((curr_r, curr_c))
                            
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    nr, nc = curr_r + dr, curr_c + dc
                                    if 0 <= nr < MAP_SHAPE[0] and 0 <= nc < MAP_SHAPE[1] and not visited[nr, nc]:
                                        neighbor_is_frontier = False
                                        if self.grid_map[nr, nc] == FREE:
                                            for dr_n in [-1, 0, 1]:
                                                for dc_n in [-1, 0, 1]:
                                                    nnr, nnc = nr + dr_n, nc + dc_n
                                                    if 0 <= nnr < MAP_SHAPE[0] and 0 <= nnc < MAP_SHAPE[1] and self.grid_map[nnr, nnc] == UNKNOWN:
                                                        neighbor_is_frontier = True
                                                        break
                                                if neighbor_is_frontier: break
                                        
                                        if neighbor_is_frontier:
                                            visited[nr, nc] = True
                                            q.append((nr, nc))
                        
                        if len(current_frontier) > 3:
                            frontiers.append(current_frontier)
        return frontiers

    def _select_target_frontier(self, frontiers, uav_pos):
        if not frontiers:
            return None

        min_dist = float('inf')
        best_target = None
        
        uav_pos_np = np.array([uav_pos.x_val, uav_pos.y_val])

        for frontier in frontiers:
            centroid = np.mean(frontier, axis=0)
            centroid_world = self._grid_to_world(centroid)
            centroid_world_np = np.array([centroid_world.x_val, centroid_world.y_val])
            
            dist = np.linalg.norm(uav_pos_np - centroid_world_np)
            
            if dist < min_dist:
                min_dist = dist
                best_target = centroid_world
        
        return best_target

    def _initialize_shared_memory(self):
        self.detection_lock = self.manager.Lock()
        self.experiment_data = self.manager.list()
        self.shared_detection_info = self.manager.dict({
            'detection_success': False,
            'detected_position': self.manager.list([0.0, 0.0, 0.0]),
            'reach_max_steps': False,
            'terminated': False
        })

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
            
            with self.detection_lock:
                is_detected = self.shared_detection_info['detection_success']
            if is_detected:
                detection_pos = list(self.shared_detection_info['detected_position'])
                print(f"[Action] Detection success! Moving to detected position: {detection_pos}")
                log_entry['event'] = 'detection_success'
                log_entry['final_move_target'] = detection_pos
                dis_to_target = np.linalg.norm(np.array(detection_pos) - np.array([self.target_position.x_val, self.target_position.y_val, self.target_position.z_val]))
                log_entry['dis_to_target'] = dis_to_target
                log_entry['success'] = True if dis_to_target < 10.0 else False
                self.experiment_data.append(log_entry)
                    
                action_client.moveToZAsync(-30, 2).join()
                action_client.moveToPositionAsync(detection_pos[0], detection_pos[1], -30, 5).join()
                print("[Action] Reached detected position.")
                    
                with self.detection_lock:
                    self.shared_detection_info['terminated'] = True
                return
            
            state = action_client.getMultirotorState()
            position = state.kinematics_estimated.position
            uav_pos_grid = self._world_to_grid(position)
            self._update_map_with_lidar(action_client, uav_pos_grid)

            # 2. Detect Frontiers
            frontiers = self._detect_frontiers()
            
            if not frontiers:
                print("[Action] No more frontiers to explore. Exploration complete.")
                log_entry['event'] = 'exploration_complete'
                self.experiment_data.append(log_entry)
                break

            # 3. Select Target
            target_position = self._select_target_frontier(frontiers, position)
            
            if target_position is None:
                print("[Action] Could not select a valid target. Stopping.")
                break
                
            print(f"[Action] Step {i}: Moving to nearest frontier at {target_position.to_numpy_array()}")
            log_entry['action_taken'] = f"move_to_frontier_{target_position.to_numpy_array()}"

            current_z = position.z_val
            action_client.moveToPositionAsync(target_position.x_val, target_position.y_val, current_z, 5).join()
            
            new_position = action_client.getMultirotorState().kinematics_estimated.position
            path_length += np.linalg.norm(position.to_numpy_array() - new_position.to_numpy_array())
            
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
                return

        print("[Action] Reached max steps.")
        self.experiment_data.append({'event': 'max_steps_reached'})
        self.experiment_data.append({'total_path_length': path_length})
        
    def _detection_process(self):
        time.sleep(2)
        detection_dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_DIR)
        detection_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_DIR).to(DEVICE)
        print("[Detection] Models loaded.")

        detection_client = airsim.MultirotorClient(port=41460)
        detection_client.confirmConnection()
        k = 0

        while True:
            with self.detection_lock:
                if self.shared_detection_info['terminated']:
                    break
            
            pil_image, depth_image, camera_position, camera_orientation, _, _ = get_images(detection_client)
            camera_fov = 90

            point_world = detection_test(detection_dino_processor, detection_dino_model, pil_image, depth_image, camera_fov, camera_position, camera_orientation, self.object_name)
            
            if point_world is not None:
                with self.detection_lock:
                    if not self.shared_detection_info['detection_success']:
                        self.shared_detection_info['detection_success'] = True
                        self.shared_detection_info['detected_position'][:] = [float(p) for p in point_world]
                        print(f"[Detection] Target '{self.object_name}' found at {point_world}")
                        # 记录检测成功事件
                        self.experiment_data.append({
                            'event': 'target_detected',
                            'timestamp': time.time(),
                            'detected_position': point_world.tolist()
                        })

            print(f"[Detection] Cycle {k} Done.")
            time.sleep(2)
            k += 1
        print("[Detection] Process terminating.")

    def run(self):
        processes = [
            multiprocessing.Process(target=self._action_process, name="Action"),
            multiprocessing.Process(target=self._detection_process, name="Detection")
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

                # 3. 创建并运行Agent
                agent = UAVSearchAgent(
                    start_position=exp_start_position,
                    target_position=exp_target_position,
                    object_name=task['object_name'],
                    object_description=task['description'],
                    log_dir=BASE_LOG_DIR
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
    BASE_LOG_DIR = "random_experiment_logs"

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

