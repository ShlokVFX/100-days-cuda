import torch
import torch.nn as nn
import gymnasium as gym  # Use gymnasium instead of gym
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())

# Create Environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # State size (Cart position, velocity, etc.)
action_dim = env.action_space.n  # Number of actions (Left or Right)

# Define a simple test Deep Q-Network
class TestDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TestDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Q-values

# Initialize model and move to GPU
dqn = TestDQN(state_dim, action_dim).to(device)
print("Model moved to:", device)

# Test model with a random state
state, _ = env.reset()  # Fix: Get (state, info)
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
q_values = dqn(state_tensor)  # Forward pass
print("Q-values:", q_values)

# Take a random action
action = np.random.choice(action_dim)
next_state, reward, terminated, truncated, _ = env.step(action)  # Fix: Updated return values
done = terminated or truncated  # Ensure proper termination check
print(f"Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")

# Close environment
env.close()
