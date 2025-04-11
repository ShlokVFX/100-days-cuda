from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import gymnasium as gym

# Create log dir
log_dir = "./ppo_cartpole_tensorboard/"
env = make_vec_env("CartPole-v1", n_envs=1, monitor_dir=log_dir)

# Optional: Configure TensorBoard logging
new_logger = configure(log_dir, ["stdout", "tensorboard"])

model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

# Train the agent
model.learn(total_timesteps=100_000)
