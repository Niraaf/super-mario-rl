import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from PIL import Image
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Record a GIF from a trained model.")
parser.add_argument("model_name", help="Name of the model (e.g., mario_cnn_0207_1255)")
parser.add_argument("--level", default="1-1", help="Level to test on, e.g. 1-1, 1-2, 2-1 (default: 1-1)")
parser.add_argument("--frames", type=int, default=2000, help="Number of frames to record (default: 2000)")
args = parser.parse_args()

MODELS_DIR = "./models/"
REPLAYS_DIR = "./replays/"

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from wrappers import apply_wrappers

model_name = args.model_name
if model_name.endswith(".zip"):
    model_name = model_name[:-4]

# parse level argument into world/stage tuple
try:
    world, stage = map(int, args.level.split("-"))
except ValueError:
    print(f"ERROR: Invalid level format '{args.level}'. Use format like 1-1, 1-2, 2-1.")
    exit()

print(f"Testing on World {world}-{stage}")

model_path = os.path.join(MODELS_DIR, model_name)

print(f"Looking for model at: {model_path}.zip")
if not os.path.exists(model_path + ".zip"):
    print(f"ERROR: Could not find model file at {model_path}.zip")
    print(f"Make sure your model is inside the '{MODELS_DIR}' folder.")
    exit()

env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(world, stage))
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GymV21CompatibilityV0(env=env, render_mode="rgb_array")
env = apply_wrappers(env)
env = DummyVecEnv([lambda: env])

# load VecNormalize stats if a matching .pkl file exists alongside the model
vecnorm_path = model_path + "_vecnorm.pkl"
if os.path.exists(vecnorm_path):
    print(f"Found VecNormalize stats at: {vecnorm_path}")
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False    # freeze running stats — don't update them during recording
    env.norm_reward = False  # don't normalize rewards at inference time, only observations matter
else:
    print("No VecNormalize stats found — running without reward normalization.")

print(f"Loading {model_name}...")
model = PPO.load(model_path)

frames = []
obs = env.reset()

print(f"Recording gameplay ({args.frames} frames)...")
for i in range(args.frames):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    screen = env.render()
    if len(screen.shape) == 4:
        screen = screen[0]

    frames.append(Image.fromarray(screen))

    if done:
        obs = env.reset()

os.makedirs(REPLAYS_DIR, exist_ok=True)
save_path = os.path.join(REPLAYS_DIR, f"{model_name}_world{world}-{stage}.gif")

print(f"Saving replay to {save_path}...")
frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=66, loop=0)
print("Done!")
