from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from envs.vec_env_rlgames import VecEnvRLGames
from envs.vec_env_rlgames2 import VecEnvRLGames2
from utils.task_util import initialize_task
from utils.evaluation_utils.evaluation import evaluation_for_training, evaluation_tianshou, evaluation

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import os

from models import SAC, PPO, TD3, DDPG, A2C, TRPO, NPG, TQC, RecurrentPPO, CrossQ

import stable_baselines3
import sb3_contrib

# 训练算法映射 (用于调用 .main 进行训练)
TRAIN_ALGO_MAP = {
    'TD3': TD3,
    'SAC': SAC,
    'PPO': PPO,
    'DDPG': DDPG,
    'CrossQ': CrossQ,
    'TRPO': TRPO,
    'TQC': TQC,
}

# 加载算法映射 (用于调用 .load 进行回测)
LOAD_ALGO_MAP = {
    'TD3': stable_baselines3.TD3,
    'SAC': stable_baselines3.SAC,
    'PPO': stable_baselines3.PPO,
    'DDPG': stable_baselines3.DDPG,
    'A2C': stable_baselines3.A2C,
    'CrossQ': sb3_contrib.CrossQ,
    'TRPO': sb3_contrib.TRPO,
    'TQC': sb3_contrib.TQC,
    'RecurrentPPO': sb3_contrib.RecurrentPPO,
}

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless
    # env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
    env = VecEnvRLGames2(headless=headless, sim_device=cfg.device_id)

    cfg_dict = omegaconf_to_dict(cfg)

    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    # --- 1. 读取命令行参数 ---
    # 用法示例：python scripts/train.py test=True +model_name=TD3 +checkpoint_name=TD3 +env_type=capture
    env_name = cfg.get("env_name", "default") # 在训练中没用，因为训练永远在默认环境下进行
    model_name = cfg.get("model_name", "TD3") 
    checkpoint_name = cfg.get("checkpoint_name", model_name)
    env_type = cfg.get("env_type", "capture") 

    cfg_dict["env_name"] = env_name
    cfg_dict["env_type"] = env_type

    # --- 2. 目录设置 ---
    # 训练日志目录：results/train/<env_type>/<checkpoint_name>/<env_name>
    train_save_dir = os.path.join("logs", "train", env_type, checkpoint_name, env_name)
    os.makedirs(train_save_dir, exist_ok=True)
    
    # 评估日志目录
    eval_save_dir = os.path.join("logs", "eval", env_type)
    os.makedirs(eval_save_dir, exist_ok=True)

    # 模型保存目录
    checkpoint_dir = os.path.join("checkpoints", env_type, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    task = initialize_task(cfg_dict, env)

    # --- 3. 开始训练 ---
    if model_name in TRAIN_ALGO_MAP:
        print(f"Starting training for Algorithm: {model_name} | Save as: {checkpoint_name}")
        TRAIN_ALGO_MAP[model_name].main(env=env, timesteps=200000, save_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown training algorithm {model_name}. Please check TRAIN_ALGO_MAP.")

    # --- 4. 训练评估 ---
    evaluation_for_training(env, train_save_dir, env_name, checkpoint_name)

    # --- 5. 加载模型并进行完整评估 ---
    if model_name in LOAD_ALGO_MAP:
        print(f"Loading model for final evaluation: {model_name} from {checkpoint_path}")
        AlgoClass = LOAD_ALGO_MAP[model_name]
        model = AlgoClass.load(checkpoint_path)
    else:
        print(f"Warning: {model_name} not found in LOAD_ALGO_MAP, skipping final evaluation.")

    # 评估
    # evaluation(env, model, env_name, checkpoint_name, eval_save_dir, n_eval_episodes=1000)
    # evaluation_tianshou(env)

if __name__ == '__main__':
    parse_hydra_configs()