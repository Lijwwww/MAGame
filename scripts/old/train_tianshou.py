from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from envs.vec_env_rlgames import VecEnvRLGames
from utils.task_util import initialize_task

import hydra
from omegaconf import DictConfig

import numpy as np
import torch

from models import SAC, PPO, TD3, DDPG, TD3_tianshou, PPO_tianshou, A2C, TRPO, NPG

from utils.evaluation_utils.evaluation import evaluation_for_training, evaluation_tianshou, evaluation
from envs.vec_env_rlgames2 import VecEnvRLGames2

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless
    # env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
    env = VecEnvRLGames2(headless=headless, sim_device=cfg.device_id)

    cfg_dict = omegaconf_to_dict(cfg)

    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    task = initialize_task(cfg_dict, env)

    # TD3_tianshou.main(env=env, timesteps=200000)
    # PPO_tianshou.main(env=env, timesteps=200000)
    NPG.main(env=env, timesteps=200000)

    evaluation_for_training(env, 'NPG')
    evaluation_tianshou(env)

if __name__ == '__main__':
    parse_hydra_configs()
