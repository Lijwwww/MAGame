from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from envs.vec_env_rlgames import VecEnvRLGames
from envs.vec_env_rlgames2 import VecEnvRLGames2
from utils.task_util import initialize_task
from utils.evaluation_utils.evaluation import evaluation
# from utils.evaluation_utils.evaluation import evaluation_tianshou

import hydra
import os
from omegaconf import DictConfig

from stable_baselines3 import SAC, PPO, TD3, DDPG
from sb3_contrib import TRPO, TQC, CrossQ 

# 算法名称到类的映射字典
ALGO_MAP = {
    'TD3': TD3,
    'SAC': SAC,
    'PPO': PPO,
    'DDPG': DDPG,
    'CrossQ': CrossQ,
    'TRPO': TRPO,
    'TQC': TQC,
}

# 合法的环境名列表 
ENV_LIST = {'default', 'obstacle0', 'obstacle1', 'speed0', 'speed1', 'poses0', 'poses1', 'friction0', 'friction1'}

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless
    # env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id) # 旧版gym
    env = VecEnvRLGames2(headless=headless, sim_device=cfg.device_id)

    cfg_dict = omegaconf_to_dict(cfg)

    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    # 读取命令行参数，如果没传就用默认值
    # 用法：/isaac-sim/python.sh scripts/test.py test=True +env_name=default +model_name=TD3 +checkpoint_name=TD3 +env_type=capture
    env_name = cfg.get("env_name", "default")
    model_name = cfg.get("model_name", "TD3")
    checkpoint_name = cfg.get("checkpoint_name", model_name)
    env_type = cfg.get("env_type", "capture") 
    
    if env_name not in ENV_LIST:
        raise ValueError(f"Unknown environment name {env_name}. Please check ENV_LIST.")
    cfg_dict["env_name"] = env_name
    cfg_dict["env_type"] = env_type
    
    # 评估日志目录
    save_dir = os.path.join("logs", "eval", env_type)
    os.makedirs(save_dir, exist_ok=True)

    # 模型保存目录
    checkpoint_dir = os.path.join("checkpoints", env_type, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    task = initialize_task(cfg_dict, env)

    # 载入模型文件
    if checkpoint_name in ALGO_MAP:
        print(f"Testing {model_name} by using checkpoint {checkpoint_name}")
        print(f"Env_type={env_type}, Env_name={env_name}")
        AlgoClass = ALGO_MAP[checkpoint_name]
        model = AlgoClass.load(checkpoint_path)
    else:
        raise ValueError(f"Unknown algorithm name {model_name}. Please check ALGO_MAP.")

    evaluation(env, model, env_name, checkpoint_name, save_dir, n_eval_episodes=1000)
    # evaluation_tianshou(env) # tianshou库的模型测试函数

if __name__ == '__main__':
    parse_hydra_configs()