from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
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

PRETRAINED_MODEL_PATH = os.path.join(checkpoint_dir, "ppo_num_3_600000_steps.zip") # 或者某个 checkpoint

VEC_NORMALIZE_STATS_PATH = os.path.join(checkpoint_dir, "ppo_num_3_vecnormalize_600000_steps.pkl")

NEW_MODEL_NAME = "f_ppo_num_5" 

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
    base_env = SubprocVecEnv(env_fns)

    print("Loading old VecNormalize stats...")
    old_vec_env = VecNormalize.load(VEC_NORMALIZE_STATS_PATH, base_env)

    print("Creating new VecNormalize environment and transferring observation stats...")
    # 创建一个新的 VecNormalize 环境，它的奖励统计数据是全新的（从零开始）
    vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, gamma=0.99)
    # 将旧环境的观测统计数据 (obs_rms) 复制到新环境中
    vec_env.obs_rms = old_vec_env.obs_rms
    
    # 此时，vec_env 拥有了预训练的观测标准化参数和全新的奖励标准化参数
    print("Observation stats transferred successfully.")

    # 3. 加载预训练的模型权重
    print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
    # 将模型与我们精心准备好的新 vec_env 关联起来
    model = PPO.load(PRETRAINED_MODEL_PATH, env=vec_env, device="cuda:2", tensorboard_log = log_dir)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=checkpoint_dir, name_prefix="f_ppo_num_5", save_vecnormalize=True)
    # 4. 开始新的训练
    # TIMESTEPS 是你希望在微调阶段额外训练的步数
    FINETUNE_TIMESTEPS = 400000 
    
    try:
        model.learn(
            total_timesteps=FINETUNE_TIMESTEPS,
            # 重置步数计数器，因为这是一次新的训练任务
            reset_num_timesteps=True, 
            log_interval=1,
            tb_log_name=NEW_MODEL_NAME,
            callback=checkpoint_callback
        )
        
        # 保存微调后的模型和新的 VecNormalize 统计数据
        model.save(f"{model_dir}/{NEW_MODEL_NAME}_final_{FINETUNE_TIMESTEPS}")
        vec_env.save(os.path.join(model_dir, f"vec_normalize_{NEW_MODEL_NAME}_final.pkl"))

    except KeyboardInterrupt:
        print("Fine-tuning interrupted. Saving...")
        model.save(f"{model_dir}/{NEW_MODEL_NAME}_interrupted_{model.num_timesteps}")
        vec_env.save(os.path.join(model_dir, f"vec_normalize_{NEW_MODEL_NAME}_interrupted_{model.num_timesteps}.pkl"))
    finally:
        vec_env.close()

