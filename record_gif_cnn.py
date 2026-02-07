import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from PIL import Image
import numpy as np

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from wrappers import apply_wrappers

env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1))
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GymV21CompatibilityV0(env=env, render_mode="rgb_array")
env = apply_wrappers(env)
env = DummyVecEnv([lambda: env])

print("Loading CNN model...")
file_name = "mario_cnn_final"  # use filename of saved zip
try:
    model = PPO.load(file_name)
except:
    print(f"Could not find {file_name}.zip!")
    exit()

# recording loop
frames = []
obs = env.reset()

print("Recording Smart Agent...")
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    screen = env.render()

    if len(screen.shape) == 4:
        screen = screen[0]

    frames.append(Image.fromarray(screen))

    if done:
        obs = env.reset()

print("Saving 'mario_cnn_gameplay.gif'...")
frames[0].save(
    "mario_cnn_gameplay.gif",
    save_all=True,
    append_images=frames[1:],
    duration=33,
    loop=0,
)
print("Done! Open 'mario_cnn_gameplay.gif'")
