from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym

def make_env():
    return gym.make("CartPole-v1")
if __name__ == '__main__':
    env = SubprocVecEnv([make_env for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=5000)

    # Optional quick check — test one of the environments
    test_env = gym.make("CartPole-v1")
    obs, _ = test_env.reset()
    total_reward = 0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print("Total reward:", total_reward)
    env.close()
    test_env.close()
