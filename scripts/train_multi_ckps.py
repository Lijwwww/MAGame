# 训练-每隔固定频率保存checkpoints，仅训练不评估

from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from envs.vec_env_rlgames import VecEnvRLGames
from envs.vec_env_rlgames2 import VecEnvRLGames2
from utils.task_util import initialize_task

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import re
import os
import glob
import time
import datetime

# 导入自定义训练模型模块 (需包含 main 函数且支持 save_freq 参数)
from models.multi_ckps import TD3_multi_ckps, SAC_multi_ckps, PPO_multi_ckps, DDPG_multi_ckps, TQC_multi_ckps, TRPO_multi_ckps, CrossQ_multi_ckps
# 如果有其他支持多断点保存的算法，在这里导入

# 导入 SB3/Contrib 类用于加载 (包含 .load 函数)
import stable_baselines3
import sb3_contrib

# 训练算法映射 (专门针对支持 multi_ckps 的模块)
TRAIN_MULTI_CKPT_MAP = {
    'TD3': TD3_multi_ckps,
    'SAC': SAC_multi_ckps,
    'PPO': PPO_multi_ckps,
    'DDPG': DDPG_multi_ckps,
    'CrossQ': CrossQ_multi_ckps,
    'TRPO': TRPO_multi_ckps,
    'TQC': TQC_multi_ckps,
    # 如果未来实现了 PPO_set_multi_ckps，可以加在这里
}

# 加载算法映射 (用于评估阶段加载模型)
LOAD_ALGO_MAP = {
    'TD3': stable_baselines3.TD3,
    'SAC': stable_baselines3.SAC,
    'PPO': stable_baselines3.PPO,
    'DDPG': stable_baselines3.DDPG,
    'CrossQ': sb3_contrib.CrossQ,
    'TRPO': sb3_contrib.TRPO,
    'TQC': sb3_contrib.TQC,
}

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless
    env = VecEnvRLGames2(headless=headless, sim_device=cfg.device_id)

    cfg_dict = omegaconf_to_dict(cfg)

    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    # --- 1. 读取命令行参数 ---
    # 用法：/isaac-sim/python.sh scripts/train_multi_ckps.py +model_name=TD3 +checkpoint_name=TD3 +env_type=capture
    env_name = cfg.get("env_name", "default") # 在训练中仅决定测试环境，这里用不到
    model_name = cfg.get("model_name", "TD3") 
    checkpoint_name = cfg.get("checkpoint_name", model_name)
    env_type = cfg.get("env_type", "capture") 
    
    cfg_dict["env_name"] = env_name
    cfg_dict["env_type"] = env_type
    
    # --- 2. 目录设置 ---
    # Checkpoint 保存目录：checkpoints/<env_type>/<algo>
    checkpoints_dir = os.path.join("checkpoints", env_type, model_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    task = initialize_task(cfg_dict, env)

    # 开始训练并按频率保存 Checkpoints
    start_time = time.time()
    
    if model_name in TRAIN_MULTI_CKPT_MAP:
        print(f"Start Training with Checkpoints: {checkpoint_name}...")
        print(f"Saving checkpoints to: {checkpoints_dir}")
        
        TRAIN_MULTI_CKPT_MAP[model_name].main(
            env=env, 
            timesteps=200000, 
            save_freq=20000, 
            save_path=checkpoints_dir, # 传入保存路径
            name_prefix=checkpoint_name    # 传入命名前缀
        )
    else:
        raise ValueError(f"Algorithm {model_name} does not support multi-checkpoint training (not in TRAIN_MULTI_CKPT_MAP).")

    # --- 统计时间 ---
    end_time = time.time()
    seconds = int(end_time - start_time)
    human_readable_time = str(datetime.timedelta(seconds=seconds))
    print(f"Total Time: {human_readable_time}")

if __name__ == '__main__':
    parse_hydra_configs()