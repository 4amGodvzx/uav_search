import airsim
import time
import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
from uav_search.object_nav.object_detect import object_detect
import warnings

def calculate_distance(depth_roi: np.ndarray):

    flat_depth = depth_roi.flatten()
    
    # 定义有效深度的范围
    MIN_VALID_DEPTH = 20  # 最小有效深度（米）
    MAX_VALID_DEPTH = 150  # 最大有效深度（米）
    
    # 筛选有效深度值：
    # 1. 大于最小有效深度
    # 2. 小于最大有效深度  
    # 3. 排除NaN和无穷大值
    valid_mask = (
        (flat_depth >= MIN_VALID_DEPTH) & 
        (flat_depth <= MAX_VALID_DEPTH) &
        np.isfinite(flat_depth)
    )
    
    valid_depths = flat_depth[valid_mask]
    
    # 检查是否有有效数据
    if len(valid_depths) == 0:
        print("警告: ROI区域内没有有效的深度数据")
        return None
    
    # 进一步过滤异常值
    if len(valid_depths) > 10:
        Q1 = np.percentile(valid_depths, 25)
        Q3 = np.percentile(valid_depths, 75)
        IQR = Q3 - Q1
        lower_bound = 20
        upper_bound = Q3 + 0 * IQR

        # 筛选在合理范围内的数据
        final_mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
        filtered_depths = valid_depths[final_mask]
        
        if len(filtered_depths) > 0:
            valid_depths = filtered_depths
        else:
            print("警告: 过滤失败")
    
    # 计算有效深度的平均值
    distance = np.mean(valid_depths)

    return distance

def save_image(client: airsim.MultirotorClient):
    rgb_response = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
    ])
    
    image_name = str(int(time.time()))
    airsim.write_file(f"check_images/check_{image_name}.png", rgb_response[0].image_data_uint8)

def checking_nav(client: airsim.MultirotorClient, distance: float):
    if distance is None:
        print("无法计算距离，停止前进")
        return
    
    # 设置前进距离和速度
    forward_distance = distance - 40
    speed = 10
    
    if forward_distance <= 0:
        print("目标距离过近，无需前进")
        return
    
    # 获取当前无人机位置和朝向
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    orientation = state.kinematics_estimated.orientation
    _, _, yaw = airsim.to_eularian_angles(orientation)
    
    # 计算前进方向的增量
    dx = forward_distance * np.cos(yaw)
    dy = forward_distance * np.sin(yaw)
    
    # 计算新的目标位置
    new_position = airsim.Vector3r(position.x_val + dx,
                                   position.y_val + dy,
                                   position.z_val)  # 保持当前高度
    
    # 执行前进动作
    client.moveToPositionAsync(new_position.x_val, new_position.y_val, new_position.z_val, speed).join()
    
    # 下降合适高度
    height = -10
    client.moveToZAsync(height, 2).join()
    time.sleep(2)

    # 再次确认距离
    _, _, depth_image_check = get_image_responses(client)
    check_roi = depth_image_check[216:296, 216:296]
    check_distance = np.percentile(check_roi, 15) - 10
    print(f"The final check distance is: {check_distance} meters")
    
    f_state = client.getMultirotorState()
    f_position = f_state.kinematics_estimated.position
    f_orientation = f_state.kinematics_estimated.orientation
    _, _, f_yaw = airsim.to_eularian_angles(f_orientation)

    f_dx = check_distance * np.cos(f_yaw)
    f_dy = check_distance * np.sin(f_yaw)

    f_new_position = airsim.Vector3r(f_position.x_val + f_dx,
                                   f_position.y_val + f_dy,
                                   f_position.z_val)

    client.moveToPositionAsync(f_new_position.x_val, f_new_position.y_val, f_new_position.z_val, 5).join()

    print("到达目标位置")
    time.sleep(4)
    save_image(client)

    cnt = 0
    while cnt < 3:
        c_state = client.getMultirotorState()
        c_pos = c_state.kinematics_estimated.position
        c_orientation = c_state.kinematics_estimated.orientation
        
        _, _, c_yaw = airsim.to_eularian_angles(c_orientation)
        
        # 右移
        c_dx = 10 * np.cos(c_yaw + np.pi/2)
        c_dy = 10 * np.sin(c_yaw + np.pi/2)
        c_new_position = airsim.Vector3r(c_pos.x_val + c_dx,
                                        c_pos.y_val + c_dy,
                                        height)
        client.moveToPositionAsync(c_new_position.x_val, c_new_position.y_val, c_new_position.z_val, 3).join()
        
        # 再前进
        c_dx = 10 * np.cos(c_yaw)
        c_dy = 10 * np.sin(c_yaw)
        c_new_position = airsim.Vector3r(c_new_position.x_val + c_dx,
                                        c_new_position.y_val + c_dy,
                                        height)
        client.moveToPositionAsync(c_new_position.x_val, c_new_position.y_val, c_new_position.z_val, 3).join()
        
        # 左转90度
        client.rotateByYawRateAsync(-90, 1).join()
        time.sleep(3)
        
        save_image(client)
        
        cnt += 1
        
    print("巡检完成")
    
def get_image_responses(client: airsim.MultirotorClient):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # RGB图像
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)  # 深度图（透视）
    ])
    rgb_response = responses[0]
    depth_response = responses[1]
    
    image_array = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    image_np = image_array.reshape(rgb_response.height, rgb_response.width, 3)
    image_pil = Image.fromarray(image_np)
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image_transformed, _ = transform(image_pil, None)
    #depth_array = np.array(depth_response.image_data_float, dtype=np.float32)
    #depth_image = depth_array.reshape(depth_response.height, depth_response.width)
    depth_image = airsim.get_pfm_array(depth_response)
    return image_np, image_transformed, depth_image

def detect_nav(client: airsim.MultirotorClient):
    # 设置旋转参数
    rotation_increment_find = 90  # 寻找时旋转的角度（度）
    rotation_increment_center = 5  # 对准时旋转的角度（度）
    center_threshold = 0.005  # 目标中心与图像中心的距离阈值
    while True:
        image_np, image_transformed, depth_image = get_image_responses(client)
        boxes, logits, phrases = object_detect(image_np, image_transformed)

        # 如果没有检测到目标，原地顺时针旋转
        if boxes is None or len(boxes) == 0 or logits is None or len(logits) == 0 or phrases != ['windmill']:
            yaw_rate = rotation_increment_find
            client.rotateByYawRateAsync(yaw_rate, 0.5).join()
            time.sleep(1)
            continue
        
        print("检测到目标，开始对准图像中心")
    
        # 选择置信度最高的目标
        max_confidence_idx = np.argmax(logits)
        target_box = boxes[max_confidence_idx]
    
        # 获取图像尺寸
        image_width = 1024
        image_height = 1024
    
        # 计算目标框中心坐标
        box_center_x = target_box[0] * image_width
        box_center_y = target_box[1] * image_height
        
        # 计算框的宽度和高度
        box_width = int(target_box[2] * image_width)
        box_height = int(target_box[3] * image_height)

        # 计算图像中心坐标
        image_center_x = image_width / 2
        image_center_y = image_height / 2
    
        # 计算目标中心与图像中心的偏移
        offset_x = (box_center_x - image_center_x).item()
        offset_y = (box_center_y - image_center_y).item()
        
        # 大致旋转
        init_yaw_rate = (box_center_x - 512) * (90.0 / 1024)
        client.rotateByYawRateAsync(init_yaw_rate.item(), 1).join()

        while abs(offset_x) > center_threshold * image_width:
            # 计算旋转方向和速度
            yaw_rate = np.sign(offset_x) * rotation_increment_center

            # 执行旋转
            client.rotateByYawRateAsync(yaw_rate, 0.1).join()
        
            # 再次获取图像和检测目标
            image_np, image_transformed, depth_image = get_image_responses(client)
            boxes, logits ,phrases = object_detect(image_np, image_transformed)

            if boxes is None or len(boxes) == 0 or logits is None or len(logits) == 0 or phrases != ['windmill']:
                print("旋转过程中丢失目标，重新搜索...")
                break
        
            # 更新目标框和偏移量
            target_box = boxes[np.argmax(logits)]
            box_center_x = target_box[0] * image_width
            box_center_y = target_box[1] * image_height
            offset_x = (box_center_x - image_center_x).item()
            offset_y = (box_center_y - image_center_y).item()
            box_width = int(target_box[2] * image_width)
            box_height = int(target_box[3] * image_height)

            time.sleep(0.1)
        else:
            print("目标已对准图像中心")
            break

    box_up = int((box_center_y - box_height / 2) / 2)
    box_down = int((box_center_y + box_height / 2) / 2)
    box_left = int((box_center_x - box_width / 2) / 2)
    box_right = int((box_center_x + box_width / 2) / 2)
    depth_roi = depth_image[box_up:box_down, box_left:box_right]
    distance = calculate_distance(depth_roi)
    print(f"Distance to target: {distance} meters")
    return distance

def pos_nav(client: airsim.MultirotorClient):
    client.moveToPositionAsync(69, 77, -26, 10).join()
    client.moveToZAsync(-18, 2).join()
    time.sleep(3)

def main():
    warnings.filterwarnings("ignore")  # 关闭所有警告
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    start_position = airsim.Vector3r(264, 274, -26)
    start_orientation = airsim.utils.to_quaternion(0, 0, 2.000)
    pose = airsim.Pose(start_position, start_orientation)
    client.simSetVehiclePose(pose, True)

    time.sleep(3)

    pos_nav(client)
    distance = detect_nav(client)
    checking_nav(client, distance)

main()