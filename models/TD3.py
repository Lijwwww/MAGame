import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import TD3

def main(env, timesteps, save_path):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save(save_path)