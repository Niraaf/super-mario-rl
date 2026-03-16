import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from PIL import Image
import numpy as np
import os
import argparse

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from wrappers import apply_wrappers

parser = argparse.ArgumentParser(description="Record a GIF from a trained model.")
parser.add_argument("model_name", help="Name of the model (e.g., mario_cnn_0207_1255)")
args = parser.parse_args()

MODELS_DIR = "./models/"
REPLAYS_DIR = "./replays/"

model_name = args.model_name
if model_name.endswith(".zip"):
    model_name = model_name[:-4]

model_path = os.path.join(MODELS_DIR, model_name)

print(f"Looking for model at: {model_path}.zip")
if not os.path.exists(model_path + ".zip"):
    print(f"ERROR: Could not find model file at {model_path}.zip")
    print(f"Make sure your model is inside the '{MODELS_DIR}' folder.")
    exit()

# --- Create a dedicated folder for this specific model ---
model_replay_dir = os.path.join(REPLAYS_DIR, model_name)
os.makedirs(model_replay_dir, exist_ok=True)

# Load the brain once
print(f"Loading {model_name}...")
model = PPO.load(model_path)

# --- Define the subset of levels to evaluate ---
target_pool = [(1, 1), (1, 2)]

# Loop through each level and record a separate GIF
for world, stage in target_pool:
    level_name = f"{world}-{stage}"
    print(f"\n--- Setting up World {level_name} ---")

    # Build a fresh environment for the specific level
    env = SuperMarioBrosEnv(
        rom_mode="vanilla", lost_levels=False, target=(world, stage)
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=env, render_mode="rgb_array")
    env = apply_wrappers(env)
    env = DummyVecEnv([lambda: env])

    obs = env.reset()
    frames = []

    print(f"Recording {level_name} gameplay (until death or victory)...")
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)

        screen = env.render()
        if len(screen.shape) == 4:
            screen = screen[0]

        frames.append(Image.fromarray(screen))

        if done:
            print(f"Run finished after {i} frames!")
            break

    # --- Save the GIF with the level name ---
    save_path = os.path.join(model_replay_dir, f"{level_name}.gif")
    print(f"Saving replay to {save_path}...")
    frames[0].save(
        save_path, save_all=True, append_images=frames[1:], duration=66, loop=0
    )

    # --- Destroy the environment before the loop restarts ---
    env.close()

print(f"\nAll done! GIFs saved in: {model_replay_dir}")
