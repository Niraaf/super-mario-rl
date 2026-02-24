import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)
import numpy as np


class SkipFrame(gym.Wrapper):
    """
    Skips frames to speed up training, returning the accumulated reward.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        return obs, total_reward, done, truncated, info


def apply_wrappers(env):
    """
    Applies standard Atari/Mario preprocessing for the CNN feature extractor.
    """
    # skip frames per decision (4 frames = 1 step)
    env = SkipFrame(env, skip=4)

    # resize to 84x84 to reduce computational load
    env = ResizeObservation(env, shape=(84, 84))

    # grayscale conversion (color data isn't needed for Mario physics)
    env = GrayscaleObservation(env)

    # stack last 4 frames to capture immediate velocity and acceleration
    env = FrameStackObservation(env, stack_size=4)

    return env
