import multiprocessing
import time
import airsim
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

from airsim_utils import get_images
from grounded_sam_test import grounded_sam
from map_updating_numpy import add_masks, downsample_masks, map_update
from detection_test import detection_test
from action_model_inputs_test import obstacle_update

DEVICE = "cuda:6"

# Action list for testing
Action_list = [1,1,1,1,1,1,1,1,2,2,0,0,0,0,3,0]  # 0: east, 1: south, 2: rotate left, 3: rotate right

# 无人机初始位置(From dataset)
start_position = airsim.Vector3r(-250, -140, -12)
start_orientation = airsim.utils.to_quaternion(0, 0, -1.57)

# From dataset
object_name = "Van"
object_description = "Van: A medium-sized vehicle with a boxy shape, white exterior, sliding side door, rear double doors, and windows on the front and sides. Commonly used for transporting goods or passengers."

def action_process_func(shared_maps, shared_detection_info, shared_maps_history, data_lock, detection_lock):
    action_client = airsim.MultirotorClient()
    action_client.confirmConnection()
    action_client.enableApiControl(True)
    action_client.armDisarm(True)

    action_client.takeoffAsync().join()
    pose = airsim.Pose(start_position, start_orientation)
    action_client.simSetVehiclePose(pose, True)
    print("[Action] Initialize complete.")

    for i in range(len(Action_list)):
        with detection_lock:
            if shared_detection_info['detection_success']:
                detection_pos = shared_detection_info['detected_position']
                print(f"[Action] Detection success! Moving to detected position: {detection_pos}")
                action_client.moveToZAsync(-30, 2).join()
                action_client.moveToPositionAsync(detection_pos[0], detection_pos[1], -30, 5).join()
                print("[Action] Reached detected position.")
                break
        # To do: update uav_pose
        
        _, depth_image, camera_position, camera_orientation, rgb_vis, _ = get_images(action_client)
        camera_fov = 90
        
        with data_lock:
            obstacle_map_copy = np.frombuffer(shared_maps['obstacle_map_buffer'], dtype=np.uint8).reshape(80, 80, 20)
        
        new_obstacle_map = obstacle_update(obstacle_map_copy, start_position, depth_image, camera_fov, camera_position, camera_orientation)
        
        with data_lock:
            shared_maps['obstacle_map_buffer'] = new_obstacle_map.tobytes()
            
            maps_copy_for_history = {
                'attraction_map': np.frombuffer(shared_maps['attraction_map_buffer'], dtype=np.float32).reshape(40, 40, 10, 2),
                'exploration_map': np.frombuffer(shared_maps['exploration_map_buffer'], dtype=np.float32).reshape(40, 40, 10),
                'obstacle_map': new_obstacle_map.copy()
            }
            shared_maps_history.append(maps_copy_for_history)

        airsim.write_file(f"test_images/action_image_{int(time.time())}.png", rgb_vis)
        
        # Calculate action
        time.sleep(1)
        print("[Action] Starting to move forward...")
        # Action for testing
        match Action_list[i]:
            case 0:
                action_client.moveByVelocityAsync(0, 3, 0, duration=4).join()
            case 1:
                action_client.moveByVelocityAsync(-3, 0, 0, duration=4).join()
            case 2:
                action_client.rotateByYawRateAsync(-30, duration=3).join()
            case 3:
                action_client.rotateByYawRateAsync(30, duration=3).join()
        
        print(f"[Action] Number {i} movement finished.")
    
    with detection_lock:
        shared_detection_info["reach_max_steps"] = True

def detection_process_func(shared_detection_info, detection_lock):
    dino_model_directory = "../models/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    detection_dino_processor = AutoProcessor.from_pretrained(dino_model_directory)
    detection_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_directory).to(DEVICE)
    print("[Detection] Models loaded.")

    detection_client = airsim.MultirotorClient()
    detection_client.confirmConnection()
    k = 0

    while True:
        with detection_lock:
            if shared_detection_info["reach_max_steps"] or shared_detection_info['detection_success']:
                break
        
        pil_image, depth_image, camera_position, camera_orientation, rgb_vis, _ = get_images(detection_client)
        camera_fov = 90

        airsim.write_file(f"test_images/detection_image_{int(time.time())}.png", rgb_vis)

        point_world = detection_test(detection_dino_processor, detection_dino_model, pil_image, depth_image, camera_fov, camera_position, camera_orientation, object_name)
        
        if point_world is not None:
            with detection_lock:
                if not shared_detection_info['detection_success']:
                    shared_detection_info['detection_success'] = True
                    shared_detection_info['detected_position'] = [float(point_world[0]), float(point_world[1]), float(point_world[2])]

        print(f"[Detection] Number {k} Detection Done.")
        time.sleep(2)  # Avoid too frequent detections
        k += 1

def planning_process_func(shared_maps, shared_detection_info, data_lock, detection_lock):
    print("[Planning] Loading models...")
    '''
    model_directory = "/data/zhangdaoxuan/models/models-Qwen2.5-VL-7B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_directory,
        dtype=torch.float16,
        attn_implementation="sdpa"
    ).to(DEVICE)
    qwen_processor = AutoProcessor.from_pretrained(model_directory)
    '''
    qwen_model = None
    qwen_processor = None # Not used in current version

    dino_model_directory = "../models/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    dino_processor = AutoProcessor.from_pretrained(dino_model_directory)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_directory).to(DEVICE)

    sam_model_directory = "../models/models-sam-vit-base"
    sam_processor = SamProcessor.from_pretrained(sam_model_directory)
    sam_model = SamModel.from_pretrained(sam_model_directory).to(DEVICE)
    print("[Planning] Models loaded.")

    planning_client = airsim.MultirotorClient()
    planning_client.confirmConnection()
    j = 0

    while True:
        with detection_lock:
            if shared_detection_info["reach_max_steps"] or shared_detection_info['detection_success']:
                break
        
        get_obs_time_start = time.time()
        pil_image, depth_image, camera_position, camera_orientation, rgb_vis, rgb_base64 = get_images(planning_client)
        camera_fov = 90
        get_obs_time_end = time.time()
        
        airsim.write_file(f"test_images/planning_image_{int(time.time())}.png", rgb_vis)
        print(f"[Planning] Get observation time: {get_obs_time_end - get_obs_time_start:.2f} seconds")

        result_dict, attraction_scores = grounded_sam(qwen_processor, qwen_model, dino_processor, dino_model, sam_processor, sam_model, pil_image, rgb_base64, object_description)

        if not result_dict["success"]:
            print("[Planning] objects detection failed, skipping this frame.")
            time.sleep(1)
            continue

        map_update_time_start = time.time()

        with data_lock:
            attraction_map = np.frombuffer(shared_maps['attraction_map_buffer'], dtype=np.float32).reshape(40, 40, 10, 2)
            exploration_map = np.frombuffer(shared_maps['exploration_map_buffer'], dtype=np.float32).reshape(40, 40, 10)

        added_masks = add_masks(result_dict["masks"])
        prepared_masks = downsample_masks(added_masks, scale_factor=2)  # 512*512 to 256*256
        new_attraction_map, new_exploration_map, observed_points_cnt = map_update(attraction_map, exploration_map, prepared_masks, attraction_scores, start_position, depth_image, camera_fov, camera_position, camera_orientation)

        with data_lock:
            shared_maps['attraction_map_buffer'] = new_attraction_map.tobytes()
            shared_maps['exploration_map_buffer'] = new_exploration_map.tobytes()
            shared_maps['observed_points_cnt'] = observed_points_cnt

        map_update_time_end = time.time()
        print(f"[Planning] Map update time: {map_update_time_end - map_update_time_start:.2f} seconds")
        print(f"[Planning] Number {j} Planning Done.")
        j += 1

if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        # 共享地图数据
        attraction_map = np.zeros((40, 40, 10, 2), dtype=np.float32)
        attraction_map[:, :, :, 1] = -1
        exploration_map = np.zeros((40, 40, 10), dtype=np.float32)
        obstacle_map = np.zeros((80, 80, 20), dtype=np.uint8)

        shared_maps = manager.dict({
            'attraction_map_buffer': attraction_map.tobytes(),
            'exploration_map_buffer': exploration_map.tobytes(),
            'obstacle_map_buffer': obstacle_map.tobytes(),
            'observed_points_cnt': 0 # Only used in training
        })

        # 共享检测信息
        shared_detection_info = manager.dict({
            'detection_success': False,
            'detected_position': manager.list([0.0, 0.0, 0.0]),
            'reach_max_steps': False
        })
        
        shared_maps_history = manager.list()

        data_lock = manager.Lock()
        detection_lock = manager.Lock()

        action_process = multiprocessing.Process(target=action_process_func, args=(shared_maps, shared_detection_info, shared_maps_history, data_lock, detection_lock), name="Action")
        planning_process = multiprocessing.Process(target=planning_process_func, args=(shared_maps, shared_detection_info, data_lock, detection_lock), name="Planning")
        detection_process = multiprocessing.Process(target=detection_process_func, args=(shared_detection_info, detection_lock), name="Detection")

        print("Starting processes...")
        action_process.start()
        detection_process.start()
        planning_process.start()

        try:
            action_process.join()
            planning_process.join()
            detection_process.join()
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, terminating processes.")
            action_process.terminate()
            time.sleep(3)
            planning_process.terminate()
            detection_process.terminate()
            action_process.join()
            planning_process.join()
            detection_process.join()
        finally:
            print("Resetting AirSim environment...")
            client = airsim.MultirotorClient()
            client.reset()
            client.armDisarm(False)
            client.enableApiControl(False)
            print("Done.")

            # Save map history
            maps_history_list = list(shared_maps_history)
            print(f"Saving {len(maps_history_list)} map history records...")
            for i, maps_record in enumerate(maps_history_list):
                np.savetxt(f"test_images/attraction_map_{i}.txt", maps_record['attraction_map'].flatten())
                np.savetxt(f"test_images/exploration_map_{i}.txt", maps_record['exploration_map'].flatten())
                np.savetxt(f"test_images/obstacle_map_{i}.txt", maps_record['obstacle_map'].flatten())
            print("Map history saved.")
