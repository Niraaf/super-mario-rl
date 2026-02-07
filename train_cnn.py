import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from wrappers import apply_wrappers

env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1))
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
    learning_rate=0.0001,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    tensorboard_log="./mario_tensorboard/",
    device="auto",
)

# save model every 50,000 steps so we don't lose progress
checkpoint_callback = CheckpointCallback(
    save_freq=50000, save_path="./logs/", name_prefix="mario_cnn_model"
)

# training phase
print("------------------------------------------")
print("  Vision System Online. Starting Training.")
print("  Press Ctrl+C to stop and save.          ")
print("------------------------------------------")

try:
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    model.save("mario_cnn_final")
    print("Training Finished! Model saved to 'mario_cnn_final.zip'")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    model.save("mario_cnn_final")
    print("Model saved to 'mario_cnn_final.zip'")
