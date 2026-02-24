from stable_baselines3 import PPO

def main(env, timesteps, save_path):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save(save_path)