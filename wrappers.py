import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import numpy as np
import collections

class MarioCustomReward(gym.Wrapper):
    def __init__(self, env, skip=4, decay_target=800000):
        super().__init__(env)
        self._skip = skip
        self.info = None
        #self.current_step = 0
        self.decay_target = decay_target
        # Create a buffer to hold the last two frames for max pooling
        self._obs_buffer = collections.deque(maxlen=2)

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        info = reset_result[1] if isinstance(reset_result, tuple) else {}
        self.info = None
        self._obs_buffer.clear()
        self._obs_buffer.append(obs)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, _, done, info = step_result
                truncated = False
            else:
                obs, _, done, truncated, info = step_result

            self._obs_buffer.append(obs)

            if self.info is None:
                self.info = info

            #self.current_step += 1
            #falloff = max(0.0, 1.0 - (self.current_step / self.decay_target))

            x_diff = info["x_pos"] - self.info["x_pos"]
            is_dead = info['status'] == 'dead' or info['status'] == 'dying'
            flag_get = info['flag_get']

            step_reward = (x_diff * 0.1)
            step_reward -= 0.2

            if is_dead:
                step_reward -= 250 # Increased penalty to stop suicide sprinting
                done = True
            if flag_get:
                step_reward += 150
                done = True

            self.info = info
            total_reward += step_reward

            if done or truncated:
                break
                
        # Perform max pooling across the last two frames in the buffer
        if len(self._obs_buffer) == 2:
            max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        else:
            max_frame = self._obs_buffer[0]
            
        return max_frame, total_reward, done, truncated, info

def apply_wrappers(env):
    """
    Applies standard Atari preprocessing.
    """
    # skip frames per decision
    env = MarioCustomReward(env, skip=4, decay_target=800000)

    # resize to 84x84
    env = ResizeObservation(env, shape=(84, 84))

    # grayscale conversion
    env = GrayscaleObservation(env)

    # stack last 4 frames to see movement
    env = FrameStackObservation(env, stack_size=4)

    return env