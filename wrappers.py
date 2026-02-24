import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)
import numpy as np


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        self.info = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.info = None
        return obs, info

    def reward(self, current):
        if(self.info is None):
            self.info = current
            return -0.1
        x_diff = current["x_pos"] - self.info["x_pos"]
        y_diff = current["y_pos"] - self.info["y_pos"]
        y_diff = 0.1 if y_diff > 0 else 0
        if(current['status'] == 'dead' or current['status'] == 'dead'):
            return -20
        level_change = 0
        if(current['world'] > self.info['world'] or current['stage'] > self.info['stage']):
            level_change = 50
        self.info = current
        return x_diff * 0.2 + y_diff -0.5 + level_change
    

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self._skip):
            # accumulate reward for every frame skipped
            obs, reward, terminated, truncated, info = self.env.step(action)
            current_reward = self.reward(info)
            total_reward += current_reward
            if(current_reward == -25):
                return obs, -50, True, False, info
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, done, truncated, info


def apply_wrappers(env):
    """
    Applies standard Atari preprocessing.
    """
    # skip frames per decision
    env = SkipFrame(env, skip=4)

    # resize to 84x84
    env = ResizeObservation(env, shape=(84, 84))

    # grayscale conversion
    env = GrayscaleObservation(env)

    # stack last 4 frames to see movement
    env = FrameStackObservation(env, stack_size=4)

    return env