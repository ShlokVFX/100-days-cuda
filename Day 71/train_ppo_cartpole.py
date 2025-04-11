from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Step 1: Create the environment
env = make_vec_env("CartPole-v1", n_envs=4)

# Step 2: Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Step 3: Train the agent
model.learn(total_timesteps=100_000)

# Step 4: Save the model
model.save("ppo_cartpole")
