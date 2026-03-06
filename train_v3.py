import sys
import gymnasium as gym
import os
import time
from typing import Callable

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from wrappers import apply_wrappers
from wrappers_v3 import AntiStallWrapper, ErrorDrivenCurriculumWrapper

# --- Setup Directories & Run Name ---
timestamp = time.strftime("%m%d_%H%M")
run_name = f"mario_lstm_phase3_{timestamp}"

LOG_DIR = "./logs/"
MODELS_DIR = "./models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# --- Dynamic Schedules & Callbacks ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Decays learning rate linearly to prevent collapse."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


class EntropyDecayCallback(BaseCallback):
    """
    Dynamically decays the entropy coefficient over the training run.
    Starts high for exploration, ends low for exploitation.
    """

    def __init__(
        self,
        initial_ent_coef=0.05,
        final_ent_coef=0.001,
        total_timesteps=10_000_000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = 1.0 - (self.num_timesteps / self.total_timesteps)
        progress = max(0.0, progress)
        current_ent_coef = self.final_ent_coef + progress * (
            self.initial_ent_coef - self.final_ent_coef
        )
        self.model.ent_coef = current_ent_coef
        self.logger.record("config/ent_coef", current_ent_coef)
        return True


class CurriculumTrackerCallback(BaseCallback):
    """
    Reaches into the ErrorDrivenCurriculumWrapper to log the current
    unlocked level and the win rate of the active frontier.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            # get_attr pulls the variables from the vectorized wrapper
            unlocked_idx = self.training_env.get_attr("unlocked_index")[0]
            progression = self.training_env.get_attr("progression_path")[0]
            histories = self.training_env.get_attr("level_histories")[0]

            # Calculate the exact win rate of whatever the current frontier is
            frontier_target = progression[unlocked_idx]
            history = histories[frontier_target]

            # Prevent division by zero if history is empty
            if len(history) > 0:
                win_rate = sum(history) / len(history)
            else:
                win_rate = 0.0

            self.logger.record("curriculum/unlocked_index", unlocked_idx)
            self.logger.record("curriculum/frontier_win_rate", win_rate)

        except Exception as e:
            pass

        return True


# --- Environment Construction ---
def make_env():
    target_pool = [(1, 1), (1, 2), (1, 3), (1, 4)]
    env_pool = []

    for target in target_pool:
        e = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=target)
        e = JoypadSpace(e, SIMPLE_MOVEMENT)
        e = GymV21CompatibilityV0(env=e, render_mode="rgb_array")
        env_pool.append((target, e))
    env = ErrorDrivenCurriculumWrapper(
        env_pool, win_window=20, promote_win_rate=0.80, epsilon=0.1
    )
    env = apply_wrappers(env)
    env = AntiStallWrapper(env, stall_threshold=120, penalty=-2.0)
    env = Monitor(env)

    return env


env = DummyVecEnv([make_env])

# --- Load or Start Fresh ---
TOTAL_TIMESTEPS = 3_000_000

if len(sys.argv) > 1:
    model_name = sys.argv[1]
    if not model_name.endswith(".zip"):
        model_name += ".zip"

    LOAD_FROM = os.path.join("models", model_name)

    print(f"------------------------------------------")
    print(f"  RESUMING PHASE 3 FROM: {LOAD_FROM}")
    print(f"------------------------------------------")
    model = RecurrentPPO.load(
        LOAD_FROM,
        env=env,
        tensorboard_log=LOG_DIR,
        device="auto",
    )
else:
    print(f"------------------------------------------")
    print(f"  STARTING FRESH PHASE 3: {run_name}")
    print(f"------------------------------------------")
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        learning_rate=linear_schedule(2.5e-4),
        ent_coef=0.05,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
    )

# --- Callbacks List ---
checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path=MODELS_DIR, name_prefix=run_name
)
entropy_callback = EntropyDecayCallback(
    initial_ent_coef=0.03,
    final_ent_coef=0.005,
    total_timesteps=TOTAL_TIMESTEPS,
)
curriculum_callback = CurriculumTrackerCallback()

callback_list = CallbackList(
    [checkpoint_callback, entropy_callback, curriculum_callback]
)

try:
    print(f"Beginning LSTM training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        tb_log_name=run_name,
        reset_num_timesteps=False,
    )

    final_path = os.path.join(MODELS_DIR, f"{run_name}_final")
    model.save(final_path)
    print(f"Phase 3 Training Finished! Saved to {final_path}.zip")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    final_path = os.path.join(MODELS_DIR, f"{run_name}_interrupted")
    model.save(final_path)
    print(f"Saved partial Phase 3 model to {final_path}.zip")

env.close()
