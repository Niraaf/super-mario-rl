import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import time
import random

from wrappers import apply_wrappers
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv


class MarioSubsetRandomizer(gym.Wrapper):
    def __init__(self, envs):
        self.envs = envs
        self.current_env = random.choice(self.envs)
        super().__init__(self.current_env)

    def reset(self, **kwargs):
        self.current_env = random.choice(self.envs)
        self.env = self.current_env
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


# generate a short timestamp
timestamp = time.strftime("%m%d_%H%M")
run_name = f"mario_cnn_phase2_{timestamp}"

models_dir = "./models/"
logs_dir = "./logs/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

target_pool = [(1, 1), (1, 2)]
env_pool = []

# Build and translate each environment individually
for target in target_pool:
    e = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=target)
    e = JoypadSpace(e, SIMPLE_MOVEMENT)
    e = GymV21CompatibilityV0(env=e)
    env_pool.append(e)

# Pass the translated environments into the randomizer
env = MarioSubsetRandomizer(env_pool)

# Apply standard observation wrappers and SB3 setup
env = apply_wrappers(env)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

LOAD_MODEL = True
# PUT THE EXACT NAME OF MODEL HERE (no .zip)
LOAD_FROM = "models/mario_cnn_resume_0223_1221_final.zip"

if LOAD_MODEL:
    print(f"------------------------------------------")
    print(f"  RESUMING TRAINING FROM: {LOAD_FROM}")
    print(f"  PHASE 2: 1-1 and 1-2 RANDOMIZER")
    print(f"------------------------------------------")

    custom_objects = {
        "learning_rate": 0.00001,
        "ent_coef": 0.005,
    }

    model = PPO.load(
        LOAD_FROM,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log=logs_dir,
        device="auto",
    )

else:
    print(f"------------------------------------------")
    print(f"  STARTING FRESH: {run_name}")
    print(f"------------------------------------------")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.00003,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=logs_dir,
        device="auto",
    )

checkpoint_callback = CheckpointCallback(
    save_freq=50000, save_path=models_dir, name_prefix=run_name
)

print("------------------------------------------")
print(f"  Run Name: {run_name}")
print(f"  Models saving to: {models_dir}")
print("------------------------------------------")

try:
    model.learn(
        total_timesteps=2000000,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        reset_num_timesteps=False,
    )

    final_path = os.path.join(models_dir, f"{run_name}_final")
    model.save(final_path)
    print(f"Phase 2 Training Finished! Saved to {final_path}.zip")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    final_path = os.path.join(models_dir, f"{run_name}_interrupted")
    model.save(final_path)
    print(f"Saved partial Phase 2 model to {final_path}.zip")
