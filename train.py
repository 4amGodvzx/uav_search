from stable_baselines3 import DQN
from uav_search.test_code.uav_env_old import AirSimDroneEnv
import os

log_dir = "logs/"
model_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

env = AirSimDroneEnv()

model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir,device="cuda:0")

TIMESTEPS = 512
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=1,tb_log_name="DQN_Test_0", progress_bar=True)
model.save(f"{model_dir}/dqn_test_512")

env.close()