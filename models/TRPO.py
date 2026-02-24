from sb3_contrib import TRPO

def main(env, timesteps, save_path):
    model = TRPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save(save_path)