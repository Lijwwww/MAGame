from omni.isaac.gym.vec_env import VecEnvBase

import torch
import numpy as np

from datetime import datetime
import time

# VecEnv Wrapper for RL training
class VecEnvRLGames2(VecEnvBase):

    def _process_data(self):
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space


    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        self._task.pre_physics_step(actions)
        
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._render)
            self.sim_frame_count += 1

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device), reset_buf=self._task.reset_buf)

        self._states = self._task.get_states()
        self._process_data()
        
        #obs_dict = {"obs": self._obs, "states": self._states}

        terminated = self._resets.clone()  # 假设任务终止都是正常终止
        truncated = torch.zeros_like(self._resets)  # 没有显式截断，默认全 0

        obs = self._obs.cpu().numpy()
        # reward = self._rew.cpu().numpy()
        # terminated = terminated.cpu().numpy()
        # truncated = truncated.cpu().numpy()
        reward = float(self._rew)
        terminated = bool(terminated)
        truncated = bool(truncated)
        obs = obs.squeeze(0)

        # print(obs.shape, reward.shape, terminated.shape, truncated.shape, self._extras)

        return obs, reward, terminated, truncated, self._extras


    def reset(self, seed=None):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.rl_device)
        obs, _, _, _, _ = self.step(actions)

        return obs, {}
    
