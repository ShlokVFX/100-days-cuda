# **Complete List of Tabular RL Methods (Step-by-Step Learning Path)**  

This is a structured way to learn **Tabular Reinforcement Learning**, starting from simple ideas and moving to advanced ones.  

First, we learn **Multi-Armed Bandits**, where we choose actions but don’t have states. We use methods like **ε-Greedy**, **UCB**, and **Thompson Sampling** to balance exploration and exploitation.  

Next, we move to **Q-Learning**, which helps find the best action for each state. It works well for small problems but struggles with large ones. We have already implemented this in **CUDA**.  

Then comes **SARSA**, a method that learns differently from Q-learning. It considers the actual next action, making learning more stable but slower. Our next goal is to implement **SARSA in CUDA**.  

After that, we explore **Expected SARSA**, which reduces randomness by considering the average of possible actions.  

To improve stability, we learn **Double Q-Learning**, which avoids overestimating values by using two Q-tables.  

For a deeper understanding of RL math, we study **Dynamic Programming Methods**, which require full knowledge of the environment. These include **Value Iteration** and **Policy Iteration**.  

Next, we learn **Monte Carlo Methods**, which update learning based on full episodes. These methods are useful when handling larger problems.  

Then, we move to **Temporal Difference (TD) Methods**, which combine Monte Carlo and Q-learning ideas. TD(0) updates based on the next state, while TD(λ) is a more advanced version that makes learning efficient.  

Once we finish this, we go beyond tabular RL by using **Neural Networks** instead of Q-tables. This leads to **Deep Q-Learning (DQN)**, which combines **Neural Networks and RL**.  
