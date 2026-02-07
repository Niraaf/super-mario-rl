import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

# target: (World, Level)
env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1))
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GymV21CompatibilityV0(env=env)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=2048,
    batch_size=64,
    device="auto",
)

# training phase
print("------------------------------------------")
print("  System Check Passed. Starting Training. ")
print("  Press Ctrl+C to stop and save.          ")
print("------------------------------------------")

try:
    model.learn(total_timesteps=100000)
    model.save("mario_rl_model")
    print("Training Finished! Model saved to 'mario_rl_model.zip'")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    model.save("mario_rl_model")
    print("Model saved to 'mario_rl_model.zip'")
