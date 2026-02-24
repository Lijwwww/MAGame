from stable_baselines3 import SAC

def main(env, timesteps, save_path):
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save(save_path)