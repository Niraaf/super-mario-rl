import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from PIL import Image
import numpy as np

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1))
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GymV21CompatibilityV0(env=env, render_mode="rgb_array")
env = DummyVecEnv([lambda: env])

print("Loading model...")
try:
    model = PPO.load("mario_rl_model_mlp")
except:
    print("Could not find model file! Check the name.")
    exit()

# recording loop
frames = []
obs = env.reset()
done = False

print("Collecting frames... (This takes about 10 seconds)")
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    screen = env.render()

    if i == 0:
        print(f"DEBUG: Raw Image Shape is {screen.shape}")

    if len(screen.shape) == 4:
        screen = screen[0]

    img = Image.fromarray(screen)
    frames.append(img)

    if i % 50 == 0:
        print(f"Captured {i} frames...")

print("Saving GIF... (Please wait)")
frames[0].save(
    "mario_mlp_gameplay.gif",
    save_all=True,
    append_images=frames[1:],
    duration=33,
    loop=0,
)

print("------------------------------------------------")
print("SUCCESS! Saved 'mario_mlp_gameplay.gif'")
print("------------------------------------------------")
