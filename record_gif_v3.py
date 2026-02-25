import os
import argparse
import numpy as np
import imageio
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from wrappers import apply_wrappers

parser = argparse.ArgumentParser(
    description="Generate GIFs from a trained Phase 3 LSTM model."
)
parser.add_argument(
    "model_name",
    type=str,
    help="Name of the model (e.g., mario_lstm_phase3_0224_1301_50000_steps)",
)
args = parser.parse_args()

raw_name = args.model_name
if not raw_name.endswith(".zip"):
    raw_name += ".zip"

clean_filename = os.path.basename(raw_name)
MODEL_PATH = os.path.join("models", clean_filename)

model_folder_name = clean_filename.replace(".zip", "")
SAVE_DIR = f"./replays/{model_folder_name}/"
os.makedirs(SAVE_DIR, exist_ok=True)

LEVELS_TO_TEST = [(1, 1)]


def make_eval_env(target):
    env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=target)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=env, render_mode="rgb_array")
    env = apply_wrappers(env)
    return env


print(f"Loading Phase 3 LSTM model from: {MODEL_PATH}")
try:
    model = RecurrentPPO.load(MODEL_PATH, device="auto")
except Exception as e:
    print(
        f"\n[ERROR] Failed to load model. Does {clean_filename} exist in the models/ folder?\nDetails: {e}"
    )
    exit()

for target in LEVELS_TO_TEST:
    world, stage = target
    print(f"\n--- Recording World {world}-{stage} ---")

    env = DummyVecEnv([lambda: make_eval_env(target)])
    obs = env.reset()

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    frames = []
    done = False
    step_count = 0

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )

        obs, reward, dones, info = env.step(action)

        episode_starts = dones
        done = dones[0]

        frame = env.envs[0].unwrapped.screen.copy()
        frames.append(frame)
        step_count += 1

        if step_count > 3000:
            print(f"Max steps (3000) reached. Mario might be stuck in {world}-{stage}!")
            break

    env.close()

    gif_path = os.path.join(SAVE_DIR, f"{world}-{stage}.gif")
    print(f"Run finished after {step_count} frames! Saving GIF to {gif_path}...")
    imageio.mimsave(gif_path, frames, fps=15, loop=0)

print(f"\nAll evaluations complete! GIFs saved in: {SAVE_DIR}")
