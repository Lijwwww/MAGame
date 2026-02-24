from stable_baselines3 import DQN

def main(env, timesteps):
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save("checkpoints/DQN")