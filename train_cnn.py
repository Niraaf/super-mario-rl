import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import faulthandler
import os
import time

faulthandler.enable()
torch.cuda.empty_cache()
# generate a short timestamp
timestamp = time.strftime("%m%d_%H%M")
run_name = f"mario_cnn_{timestamp}"

models_dir = "./models/"
logs_dir = "./logs/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

from gym_super_mario_bros.smb_random_stages_env import SuperMarioBrosRandomStagesEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from wrappers import apply_wrappers

env = SuperMarioBrosRandomStagesEnv()
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GymV21CompatibilityV0(env=env)
env = apply_wrappers(env)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# init PPO Model with CNN

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.0002,
    batch_size=256,
    n_steps=8192,
    n_epochs=15,
    ent_coef=0.001,
    clip_range = 0.1,
    target_kl=0.03,
    tensorboard_log=logs_dir,
    device="auto",
)

#model = PPO.load("./models/mario_cnn_0212_2136_final.zip", env=env, tensorboard_log=logs_dir, device="auto")
# save model every 50,000 steps so we don't lose progress
checkpoint_callback = CheckpointCallback(
    save_freq=100000, save_path=models_dir, name_prefix=run_name
)

print("------------------------------------------")
print(f"  Run Name: {run_name}")
print(f"  Models saving to: {models_dir}")
print("------------------------------------------")

try:
    model.learn(total_timesteps=20000000, callback=checkpoint_callback, reset_num_timesteps=False)

    final_path = os.path.join(models_dir, f"{run_name}_final")
    model.save(final_path)
    print(f"Training Finished! Saved to {final_path}.zip")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    final_path = os.path.join(models_dir, f"{run_name}_interrupted")
    model.save(final_path)
    print(f"Saved partial model to {final_path}.zip")
