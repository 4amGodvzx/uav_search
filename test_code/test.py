import airsim
import time

# 连接到 AirSim 模拟器
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# 解锁并起飞
client.armDisarm(True)
client.takeoffAsync().join()

time_start = time.time()
responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
    ])
time_end = time.time()
print(f"Image retrieval took {time_end - time_start:.2f} seconds")
