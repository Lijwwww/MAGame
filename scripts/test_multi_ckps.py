# 测试-每隔固定频率保存的checkpoints，仅评估

from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from envs.vec_env_rlgames import VecEnvRLGames
from envs.vec_env_rlgames2 import VecEnvRLGames2
from utils.task_util import initialize_task
from utils.evaluation_utils.evaluation import evaluation_for_training, evaluation

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import re
import os
import glob
import time
import datetime

# 1. 导入自定义训练模型模块 (需包含 main 函数且支持 save_freq 参数)
from models.multi_ckps import TD3_multi_ckps
# 如果有其他支持多断点保存的算法，在这里导入

# 2. 导入 SB3/Contrib 类用于加载 (包含 .load 函数)
import stable_baselines3
import sb3_contrib

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
    # /isaac-sim/python.sh scripts/test_multi_ckps.py +env_name=obstacle +model_name=TD3 +checkpoint_name=TD3 +env_type=capture
    env_name = cfg.get("env_name", "default")
    model_name = cfg.get("model_name", "TD3")
    checkpoint_name = cfg.get("checkpoint_name", model_name)
    env_type = cfg.get("env_type", "capture") 
    
    cfg_dict["env_name"] = env_name
    cfg_dict["env_type"] = env_type
    
    # 构造命名前缀 (用于识别文件)
    name_prefix = f'{checkpoint_name}_{env_name}'
    
    # --- 2. 目录设置 ---
    # Checkpoint 保存目录：checkpoints/<env_type>/<algo>
    checkpoints_dir = os.path.join("checkpoints", env_type, model_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 评估日志保存目录：results/eval_multi_ckps/<env_type>/<algo>/<env>
    eval_save_dir = os.path.join("logs", "eval_multi_ckps", env_type, checkpoint_name)
    os.makedirs(eval_save_dir, exist_ok=True)

    task = initialize_task(cfg_dict, env)
    
    start_time = time.time()
    
    # --- 批量加载并按顺序测试 ---
    print("\nStart Evaluation on Checkpoints...")

    # 1. 查找所有 zip 文件
    search_pattern = os.path.join(checkpoints_dir, f"{checkpoint_name}_*.zip")
    checkpoint_files = glob.glob(search_pattern)

    # 2. 按步数从小到大排序
    # 提取文件名中的 step 数字进行排序
    def get_step_num(filename):
        match = re.search(r"_(\d+)_steps", filename)
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=get_step_num)

    # 3. 循环评估
    for cp_path in checkpoint_files:
        basename = os.path.basename(cp_path)
        
        # 解析当前 Checkpoint 的标识 (例如 "20000_steps")
        # 逻辑：去除后缀 -> 去除前缀(TD3_default_) -> 剩下的就是 steps 标识
        file_name_no_ext = basename.replace(".zip", "")
        step_identifier = file_name_no_ext.replace(f"{checkpoint_name}_", "")
        
        print(f"Testing Model: {basename} | ID: {step_identifier}")

        if model_name in LOAD_ALGO_MAP:
            AlgoClass = LOAD_ALGO_MAP[model_name]
            # 加载模型
            model = AlgoClass.load(cp_path)

            # 环境名填步数，放在以环境为命名的子文件夹中
            evaluation(env, model, env_name=step_identifier, checkpoint_name=name_prefix, save_dir=eval_save_dir, n_eval_episodes=500)
            
        else:
            print(f"Warning: Cannot load {model_name}, not in LOAD_ALGO_MAP.")

    print("All checkpoints evaluated.")

    # --- 统计时间 ---
    end_time = time.time()
    seconds = int(end_time - start_time)
    human_readable_time = str(datetime.timedelta(seconds=seconds))
    print(f"Total Time: {human_readable_time}")

if __name__ == '__main__':
    parse_hydra_configs()