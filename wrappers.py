import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)
import numpy as np


def apply_wrappers(env):
    """
    Applies standard Atari preprocessing.
    """
    # resize to 84x84
    env = ResizeObservation(env, shape=(84, 84))

    # grayscale conversion
    env = GrayscaleObservation(env)

    # stack last 4 frames to see movement
    env = FrameStackObservation(env, stack_size=4)

    return env
