import numpy as np
import os
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback  # 新增引用

def main(env, timesteps, save_freq, save_path, name_prefix):
    os.makedirs(save_path, exist_ok=True)

    # 1. 定义回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,          # 每多少步保存一次
        save_path=save_path,          # 保存文件夹路径
        name_prefix=name_prefix,      # 文件名前缀，保存后文件名如 TD3_20000_steps.zip
        verbose=2
    )
    
    model = SAC("MlpPolicy", env, verbose=1)
    
    # 2. 将 callback 传入 learn 函数
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=10)
    
    # 训练结束后保存最终模型（可选，因为callback最后也会保存附近的）
    model.save(os.path.join(save_path, "SAC"))