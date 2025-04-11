from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import numpy as np

def make_env():
    return gym.make("CartPole-v1")

if __name__ == '__main__':
    # Use 4 parallel environments
    env = SubprocVecEnv([make_env for _ in range(4)])

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=20000)

    # Evaluate on a single env (not parallel) to get rewards_list
    rewards_list = []
    test_env = gym.make("CartPole-v1")

    num_eval_episodes = 10
    for _ in range(num_eval_episodes):
        obs, _ = test_env.reset()
        total_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards_list.append(total_reward)

    test_env.close()
    env.close()

    # Save rewards to file
    np.save("parallel_rewards.npy", rewards_list)

    print("Saved parallel training evaluation rewards:", rewards_list)
