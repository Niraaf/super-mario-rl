import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import os
import time

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from wrappers import apply_wrappers

NUM_ENVS = 8

def make_env():
    def _init():
        env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1))
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GymV21CompatibilityV0(env=env)
        env = apply_wrappers(env)
        env = Monitor(env)
        return env
    return _init

# save VecNormalize running stats alongside each model checkpoint
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, vec_env, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.vec_env = vec_env

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps_vecnorm.pkl"
            )
            self.vec_env.save(path)
            print(f"VecNormalize stats saved to {path}")
        return True


if __name__ == '__main__':
    # generate a short timestamp
    timestamp = time.strftime("%m%d_%H%M")
    run_name = f"mario_cnn_{timestamp}"

    models_dir = "./models/"
    logs_dir = "./logs/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    # normalize rewards only (not observations — the CNN handles raw pixels fine)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # init PPO Model with CNN
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,
        n_steps=512,        # 512 steps * 8 envs = 4096 total per update (same scale as before)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,      # entropy bonus to encourage exploration and improve generalization
        tensorboard_log=logs_dir,
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, save_path=models_dir, name_prefix=run_name
    )
    vecnorm_callback = SaveVecNormalizeCallback(
        save_freq=50000, save_path=models_dir, name_prefix=run_name, vec_env=env
    )
    callbacks = CallbackList([checkpoint_callback, vecnorm_callback])

    print("------------------------------------------")
    print(f"  Run Name:       {run_name}")
    print(f"  Num Envs:       {NUM_ENVS}")
    print(f"  Steps/env:      512  (4096 total per update)")
    print(f"  Entropy coef:   0.01")
    print(f"  Reward norm:    VecNormalize (clip=10.0)")
    print(f"  Models dir:     {models_dir}")
    print("------------------------------------------")

    try:
        model.learn(total_timesteps=1000000, callback=callbacks)

        final_path = os.path.join(models_dir, f"{run_name}_final")
        model.save(final_path)
        vecnorm_final_path = os.path.join(models_dir, f"{run_name}_final_vecnorm.pkl")
        env.save(vecnorm_final_path)
        print(f"Training Finished! Saved to {final_path}.zip")
        print(f"VecNormalize stats saved to {vecnorm_final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        final_path = os.path.join(models_dir, f"{run_name}_interrupted")
        model.save(final_path)
        vecnorm_interrupted_path = os.path.join(models_dir, f"{run_name}_interrupted_vecnorm.pkl")
        env.save(vecnorm_interrupted_path)
        print(f"Saved partial model to {final_path}.zip")
        print(f"VecNormalize stats saved to {vecnorm_interrupted_path}")
