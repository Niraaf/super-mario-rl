import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)
import numpy as np
import random


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False

        for _ in range(self._skip):
            # accumulate reward for every frame skipped
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
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


class MarioSubsetRandomizer(gym.Wrapper):
    """
    Custom wrapper to execute Curriculum Learning on a specific subset of levels.
    Prevents catastrophic forgetting by randomly selecting a level
    from the 'Expanding Pool' upon every episode reset.
    """

    def __init__(self, env_ids):
        # Initialize all environments in our current curriculum pool
        self.envs = [gym.make(env_id) for env_id in env_ids]

        # Pick a random starting environment
        self.current_env = random.choice(self.envs)

        # Inherit properties from the chosen environment to satisfy Gym
        super().__init__(self.current_env)

    def reset(self, **kwargs):
        # THE CORE LOGIC: Every time Mario dies or beats a level,
        # roll the dice and completely swap the active physical environment
        self.current_env = random.choice(self.envs)

        # Overwrite the wrapped environment pointer
        self.env = self.current_env

        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
