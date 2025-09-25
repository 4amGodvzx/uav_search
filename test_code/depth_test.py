import airsim
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# 解锁
client.armDisarm(True)
client.takeoffAsync().join()

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
    airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)
])

depth_perspective = airsim.get_pfm_array(responses[0])
depth_planar = airsim.get_pfm_array(responses[1])
img1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(responses[2].height, responses[2].width, 3)

print("任务完成！")