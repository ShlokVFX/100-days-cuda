from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
# Optional plotting
# import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=20000)

rewards_list = []

# Run multiple episodes to collect rewards
num_eval_episodes = 10
for _ in range(num_eval_episodes):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    rewards_list.append(total_reward)

env.close()

# Save rewards to .npy
np.save("baseline_rewards.npy", rewards_list)
print("Saved rewards:", rewards_list)
