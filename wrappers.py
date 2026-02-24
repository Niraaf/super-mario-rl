import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)
import numpy as np
import random
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv


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
    """

    def __init__(self, targets):
        self.envs = [
            SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=t)
            for t in targets
        ]

        # Pick a random starting environment
        self.current_env = random.choice(self.envs)
        super().__init__(self.current_env)

    def reset(self, **kwargs):
        self.current_env = random.choice(self.envs)
        self.env = self.current_env
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
