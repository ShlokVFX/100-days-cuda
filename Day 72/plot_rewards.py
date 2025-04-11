import numpy as np
import matplotlib.pyplot as plt

# Load reward data
baseline_rewards = np.load("baseline_rewards.npy")
parallel_rewards = np.load("parallel_rewards.npy")

# Smoothing function
def smooth(y, box_pts=10):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(smooth(baseline_rewards), label='Baseline PPO (Single Env)', color='blue')
plt.plot(smooth(parallel_rewards), label='Parallel PPO (4 Envs)', color='green')

plt.title("CartPole PPO Training Comparison")
plt.xlabel("Episodes")
plt.ylabel("Episode Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_cartpole_comparison.png")
plt.show()
