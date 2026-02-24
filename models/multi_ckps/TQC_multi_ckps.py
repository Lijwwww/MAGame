import os
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback

def main(env, timesteps, save_freq, save_path, name_prefix):
    
    os.makedirs(save_path, exist_ok=True)

    # 1. 定义回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,          # 每多少步保存一次
        save_path=save_path,          # 保存文件夹路径
        name_prefix=name_prefix,      # 文件名前缀
        verbose=2
    )
    
    # 2. 初始化模型
    model = TQC("MlpPolicy", env, verbose=1)
    
    # 3. 开始训练
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=10)

    model.save(os.path.join(save_path, "TQC"))