import os
import torch
import numpy as np
import datetime
from torch import nn

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import NPGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.exploration import GaussianNoise
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from torch.distributions import Distribution, Independent, Normal

# def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
#         loc, scale = loc_scale
#         return Independent(Normal(loc, scale), 1)

def dist(logits) -> Distribution:
    if isinstance(logits, tuple):
        loc, scale = logits
    else:
        loc = logits
        scale = torch.full_like(loc, 0.01)
    return Independent(Normal(loc, scale), 1)

def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)

def dist_fn(loc, scale) -> Distribution:
    return Independent(Normal(loc, scale), 1)

def get_policy(env, resume_path=None):
    obs_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    print(f'obs_shape: {obs_shape}\naction_shape: {action_shape}')
    print(f'min_action: {min_action}, max_action: {max_action}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net_actor = Net(state_shape=obs_shape, hidden_sizes=[256, 256], device=device, activation=nn.Tanh)
    actor = ActorProb(preprocess_net=net_actor, action_shape=action_shape, device=device).to(device)

    net_critic = Net(state_shape=obs_shape, action_shape=action_shape, hidden_sizes=[256, 256], device=device, activation=nn.Tanh)
    critic = Critic(preprocess_net=net_critic, device=device).to(device)
    optim = torch.optim.Adam(critic.parameters(), lr=1e-4)

    policy = NPGPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=env.action_space,
        observation_space=env.observation_space,
        discount_factor=0.99,
        action_bound_method="clip",
        actor_step_size=0.05,
        action_scaling=False,
        advantage_normalization=True,
    )

    if resume_path:
        policy.load_state_dict(torch.load(resume_path))
        print(f'Load {resume_path} to model.')

    return policy


def main(env, timesteps):
    # ===== Step 1: 创建环境 =====
    train_envs = DummyVectorEnv([lambda:env])
    test_envs = DummyVectorEnv([lambda:env])

    # ===== Step 2: 设置随机种子 =====
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # train_envs.seed(seed)
    # test_envs.seed(seed)

    # ===== Step 3: 获取策略 =====
    policy = get_policy(env)

    # ===== Step 4: 创建 Collector =====
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(100000, len(train_envs)), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    train_collector.collect(n_step=5000)

    # ===== Step 5: 定义训练回调 =====
    def save_best_fn(policy):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(policy.actor.state_dict(), "checkpoints/NPG_best.pth")

    log_path = "logs/tianshou_logs/NPG"
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # ===== Step 6: 启动训练器 =====
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=20,
        step_per_epoch=10000, 
        step_per_collect=1000,
        episode_per_test=10,
        batch_size=64,
        repeat_per_collect=2,
        # save_best_fn=save_best_fn,
        test_in_train=False,
        logger=logger,
    )

    print(f"\n==========Result==========\n{result}")

    torch.save(policy.state_dict(), 'checkpoints/NPGv2.pth')


def evaluate(env, path):
    test_envs = DummyVectorEnv([lambda:env])

    policy = get_policy(env, path)

    policy.eval()
    collector = Collector(policy, test_envs, exploration_noise=True)
    result = collector.collect(n_episode=1000, render=False)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
