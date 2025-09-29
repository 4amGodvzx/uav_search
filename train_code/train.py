from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from uav_search.train_code.uav_env import AirSimDroneEnv
import os
import torch
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

log_dir = "uav_search/logs/"
model_dir = "uav_search/models/"
checkpoint_dir = "uav_search/checkpoints/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

env = AirSimDroneEnv()

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=checkpoint_dir,
                                         name_prefix='dqn_num_0')

model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir,device="cuda:3")

TIMESTEPS = 50000

try:
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        log_interval=10,
        tb_log_name="dqn_num_0",
        progress_bar=True,
        callback=checkpoint_callback
    )
    model.save(f"{model_dir}/dqn_num_0_final_{TIMESTEPS}")
except KeyboardInterrupt:
    print("Training interrupted by user. Saving model...")
    model.save(f"{model_dir}/dqn_num_0_interrupted_{model.num_timesteps}")
finally:
    env.close()
