from sb3_contrib import RecurrentPPO

def main(env, timesteps):
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save("checkpoints/RecurrentPPO")