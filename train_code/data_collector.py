import airsim
import numpy as np
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from collections import defaultdict, Counter

from uav_search.airsim_utils import get_images
from uav_search.grounded_sam_test import grounded_sam
from uav_search.map_updating_numpy import add_masks, downsample_masks
from uav_search.to_map_test import to_map_xyz

DEVICE = "cuda:1"

# Action list for testing
Action_list = [1,6,1,1,5,0,0,0,0,0,0,0,0,4,7,1,1,1,1,1,4,2,2,8]
# 0: north, 1: west, 2: south, 3: east, 4: rotate left, 5: rotate right, 6: ascend, 7: descend, 8: stop, 9: up2, 10: down2, 11: up3, 12: up4

object_description = "Orange Car:Geometric four-wheeled form, matte orange body with black-tinted windows, integrated spoiler and sunroof, used for personal ground transport."

# 初始位置和朝向
start_position = airsim.Vector3r(-4, -55, -10)

# 实际初始位置和朝向
a_start_position = airsim.Vector3r(-114, 28, 3)
a_start_orientation = airsim.utils.to_quaternion(0, 0, -1.57)

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
pose = airsim.Pose(a_start_position, a_start_orientation)
client.simSetVehiclePose(pose, True)
print("Initialize complete.")

attraction_map = np.zeros((40, 40, 10, 2), dtype=np.float32)
attraction_map[:, :, :, 1] = -1

maps_history = []

qwen_model = None
qwen_processor = None # Not used in current version
    
dino_model_directory = "/data_all/zhangdaoxuan/models/models-grounding-dino-base"
dino_processor = AutoProcessor.from_pretrained(dino_model_directory)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_directory).to(DEVICE)

sam_model_directory = "/data_all/zhangdaoxuan/models/models-sam-vit-base"
sam_processor = SamProcessor.from_pretrained(sam_model_directory)
sam_model = SamModel.from_pretrained(sam_model_directory).to(DEVICE)
print("[Planning] Models loaded.")

for i in range(len(Action_list)):
    
    while True:
        pil_image, depth_image, camera_position, camera_orientation, rgb_vis, rgb_base64 = get_images(client)
        camera_fov = 90

        result_dict, attraction_scores = grounded_sam(qwen_processor, qwen_model, dino_processor, dino_model, sam_processor, sam_model, pil_image, rgb_base64, object_description)

        if result_dict["success"]:
            print("[Planning] Objects detection successful. Proceeding with map update.")
            airsim.write_file(f"/data_all/zhangdaoxuan/uav_search/collect_images/collect_image_{int(time.time())}.png", rgb_vis)
            break
        else:
            print("[Planning] Objects detection failed, retrying in 1 second...")
            time.sleep(1)
    
    added_masks = add_masks(result_dict["masks"])
    prepared_masks = downsample_masks(added_masks, scale_factor=2)  # 512*512 to 256*256
    
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    new_attraction_map = attraction_map.copy()
    map_size = np.array([200.0, 200.0, 50.0])  # 地图尺寸 (meters)
    grid_size = np.array([5.0, 5.0, 5.0])      # 网格尺寸 (meters)
    map_resolution = (map_size / grid_size).astype(int)  # 地图分辨率 [40, 40, 10]
    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0
    grid_contributions = defaultdict(list)

    for object_id, mask in enumerate(prepared_masks):
        if mask is None:
            continue
        
        pixels_v, pixels_u = np.where(mask)

        for v, u in zip(pixels_v, pixels_u):
            depth = depth_image[v, u]
            if depth >= 250:  # 只处理有效深度范围内的点
                continue

            world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)

            map_coords = world_coords - map_origin
            grid_indices = (map_coords / grid_size).astype(int)
            gx, gy, gz = grid_indices

            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                grid_contributions[(gx, gy, gz)].append((depth, object_id))

    for (gx, gy, gz), points_list in grid_contributions.items():
        if not points_list:
            continue

        min_depth_in_frame = min(p[0] for p in points_list)

        stored_depth = attraction_map[gx, gy, gz, 1]
        if min_depth_in_frame < stored_depth or stored_depth < 0:
            new_attraction_map[gx, gy, gz, 1] = min_depth_in_frame

        object_ids = [p[1] for p in points_list]
        if object_ids:
            winner_object_id = Counter(object_ids).most_common(1)[0][0]
            winning_score = attraction_scores[winner_object_id]
            new_attraction_map[gx, gy, gz, 0] = winning_score
            
    maps_history.append(new_attraction_map[:, :, :, 0].copy())
    attraction_map = new_attraction_map
    
    match Action_list[i]:
        case 0:
            client.moveByVelocityAsync(3, 0, 0, duration=4).join()
        case 1:
            client.moveByVelocityAsync(0, -3, 0, duration=4).join()
        case 2:
            client.moveByVelocityAsync(-3, 0, 0, duration=4).join()
        case 3:
            client.moveByVelocityAsync(0, 3, 0, duration=4).join()
        case 4:
            client.rotateByYawRateAsync(-30, duration=3).join()
        case 5:
            client.rotateByYawRateAsync(30, duration=3).join()
        case 6:
            client.moveToZAsync(-10, 3).join()
        case 7:
            client.moveToZAsync(4, 3).join()
        case 8:
            print("[Action] Stop action received, ending data collection.")
        case 9:
            client.moveToZAsync(-40, 3).join()
        case 10:
            client.moveToZAsync(16, 3).join()
        case 11:
            client.moveToZAsync(-6, 3).join()
        case 12:
            client.moveToZAsync(-2, 3).join()
    print(f"[Action] Number {i} movement finished.")
    time.sleep(2)

num = len(maps_history)
np.savetxt(f"/data_all/zhangdaoxuan/uav_search/task_map_cache/task_35.txt", maps_history[num-1].flatten())