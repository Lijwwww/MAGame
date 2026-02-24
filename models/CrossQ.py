from sb3_contrib import CrossQ

def main(env, timesteps, save_path):
    model = CrossQ("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save(save_path)