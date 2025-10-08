from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from uav_search.train_code.uav_env_multi import AirSimDroneEnv
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

if __name__ == '__main__':
    num_cpu = 4
    base_port = 41451

    def make_env(rank, seed=0):
        def _init():
            env = AirSimDroneEnv(worker_index=rank, base_port=base_port)
            env = Monitor(env)
            return env
        return _init

    env_fns = [make_env(i) for i in range(num_cpu)]
    vec_env = SubprocVecEnv(env_fns)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=checkpoint_dir,name_prefix='ppo_num_0')

    model = PPO.load("uav_search/models/ppo_num_0_interrupted_113780.zip", env=vec_env, device="cuda:0")

    TIMESTEPS = 400000

    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            log_interval=1,
            tb_log_name="ppo_num_0",
            callback=checkpoint_callback
        )
        model.save(f"{model_dir}/ppo_num_0_final_{TIMESTEPS}")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
        model.save(f"{model_dir}/ppo_num_0_interrupted_{model.num_timesteps}")
    finally:
        vec_env.close()