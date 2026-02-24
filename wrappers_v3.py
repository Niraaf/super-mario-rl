import numpy as np
from collections import deque
import gymnasium as gym


class AntiStallWrapper(gym.Wrapper):
    """
    Penalizes the agent if its X-coordinate does not change significantly over a set number of frames.
    Prevents the AI from finding a safe local minimum by standing still.
    """

    def __init__(self, env, stall_threshold=120, penalty=-2.0):
        super().__init__(env)
        self.stall_threshold = stall_threshold
        self.penalty = penalty
        self.x_history = deque(maxlen=stall_threshold)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        current_x = info.get("x_pos", 0)
        self.x_history.append(current_x)

        if len(self.x_history) == self.stall_threshold:
            if max(self.x_history) - min(self.x_history) <= 2:
                reward += self.penalty

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.x_history.clear()
        return self.env.reset(**kwargs)


class ErrorDrivenCurriculumWrapper(gym.Wrapper):
    """
    SOTA Curriculum Learning: Samples from a pool of pre-built environments
    inversely proportional to their win rate.
    """

    def __init__(self, env_pool, win_window=20, promote_win_rate=0.80, epsilon=0.1):
        # env_pool is a list of tuples: [((1,1), env1), ((1,2), env2), ...]
        self.env_pool = env_pool
        self.progression_path = [target for target, _ in env_pool]

        # Initialize the wrapper with the first environment in the pool
        self.current_target, self.current_env = self.env_pool[0]
        super().__init__(self.current_env)

        self.win_window = win_window
        self.promote_win_rate = promote_win_rate
        self.epsilon = epsilon
        self.unlocked_index = 0

        # Independent win-trackers for every level
        self.level_histories = {
            target: deque(maxlen=win_window) for target in self.progression_path
        }

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if done or truncated:
            won = info.get("flag_get", False)
            self.level_histories[self.current_target].append(1 if won else 0)

            # Check for promotion on the current FRONTIER
            frontier_target = self.progression_path[self.unlocked_index]
            frontier_history = self.level_histories[frontier_target]

            if len(frontier_history) == self.win_window:
                frontier_win_rate = sum(frontier_history) / self.win_window

                if frontier_win_rate >= self.promote_win_rate:
                    if self.unlocked_index < len(self.progression_path) - 1:
                        self.unlocked_index += 1
                        new_level = self.progression_path[self.unlocked_index]
                        print(
                            f"\n[CURRICULUM] Frontier Mastered ({frontier_win_rate*100}%). Unlocking World {new_level[0]}-{new_level[1]}!"
                        )

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        unlocked_levels = self.progression_path[: self.unlocked_index + 1]

        if len(unlocked_levels) == 1:
            target_idx = 0
        else:
            weights = []
            for lvl in unlocked_levels:
                history = self.level_histories[lvl]
                win_rate = sum(history) / len(history) if len(history) > 0 else 0.0
                failure_rate = 1.0 - win_rate
                weights.append(failure_rate + self.epsilon)

            probabilities = np.array(weights) / sum(weights)
            target_idx = np.random.choice(len(unlocked_levels), p=probabilities)

        # Swap the active underlying environment!
        self.current_target, self.current_env = self.env_pool[target_idx]
        self.env = self.current_env

        return self.env.reset(**kwargs)
