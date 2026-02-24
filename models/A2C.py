import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import A2C

def main(env, timesteps):
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save("checkpoints/A2Cv2")