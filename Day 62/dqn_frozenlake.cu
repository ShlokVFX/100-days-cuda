#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

#define ACTION_SPACE_SIZE 4   // Actions in FrozenLake
#define STATE_SPACE_SIZE 16   // 4x4 grid (16 states)
#define LEARNING_RATE 0.001
#define GAMMA 0.99
#define EPSILON 0.1
#define BATCH_SIZE 32
#define EPISODES 100
#define TARGET_UPDATE 10  // Update target network every 10 episodes

// CUDA device setup
torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

// Neural Network for Q-learning (DQN)
struct DQN : torch::nn::Module {
    torch::nn::Linear fc1, fc2, fc3;

    DQN()
        : fc1(STATE_SPACE_SIZE, 128), fc2(128, 64), fc3(64, ACTION_SPACE_SIZE) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);

        // Move model to GPU if available
        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.to(device);
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }

    // Copy parameters from another DQN model
    void copy_from(const DQN& other) {
        this->to(device);
        fc1->weight.data().copy_(other.fc1->weight.data());
        fc1->bias.data().copy_(other.fc1->bias.data());

        fc2->weight.data().copy_(other.fc2->weight.data());
        fc2->bias.data().copy_(other.fc2->bias.data());

        fc3->weight.data().copy_(other.fc3->weight.data());
        fc3->bias.data().copy_(other.fc3->bias.data());
    }
};

// Replay Buffer for Experience Replay
class ReplayBuffer {
public:
    std::vector<std::tuple<torch::Tensor, int, float, torch::Tensor>> buffer;
    int max_size;

    ReplayBuffer(int max_size = 10000) : max_size(max_size) {}

    void push(torch::Tensor state, int action, float reward, torch::Tensor next_state) {
        if (buffer.size() >= max_size) {
            buffer.erase(buffer.begin());
        }
        buffer.emplace_back(state.to(device), action, reward, next_state.to(device));
    }

    std::vector<std::tuple<torch::Tensor, int, float, torch::Tensor>> sample(int batch_size) {
        std::vector<std::tuple<torch::Tensor, int, float, torch::Tensor>> batch;
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});
        return batch;
    }
};

int main() {
    torch::manual_seed(0);
    
    // Initialize networks and optimizer
    DQN policy_net, target_net;
    target_net.copy_from(policy_net);
    torch::optim::Adam optimizer(policy_net.parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    ReplayBuffer replay_buffer(10000);

    int total_episodes = EPISODES;
    float epsilon = EPSILON;

    for (int episode = 0; episode < total_episodes; ++episode) {
        // Initialize state (random tensor for mock environment)
        torch::Tensor state = torch::rand({STATE_SPACE_SIZE}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        float total_reward = 0;
        for (int t = 0; t < 1000; ++t) {
            // Epsilon-greedy action selection
            int action;
            if ((static_cast<float>(rand()) / RAND_MAX) < epsilon) {
                action = rand() % ACTION_SPACE_SIZE;
            } else {
                action = policy_net.forward(state.unsqueeze(0)).argmax(1).item<int>();
            }

            // Take action, get next state and reward (mock values)
            torch::Tensor next_state = torch::rand({STATE_SPACE_SIZE}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            float reward = (rand() % 10 > 7) ? 1.0f : 0.0f;

            replay_buffer.push(state, action, reward, next_state);
            state = next_state;
            total_reward += reward;

            // Train DQN if enough experiences collected
            if (replay_buffer.buffer.size() >= BATCH_SIZE) {
                auto batch = replay_buffer.sample(BATCH_SIZE);

                std::vector<torch::Tensor> states_batch, actions_batch, rewards_batch, next_states_batch;
                for (auto& sample : batch) {
                    states_batch.push_back(std::get<0>(sample));
                    actions_batch.push_back(torch::tensor(std::get<1>(sample), torch::TensorOptions().dtype(torch::kLong).device(device)));
                    rewards_batch.push_back(torch::tensor(std::get<2>(sample), torch::TensorOptions().dtype(torch::kFloat32).device(device)));
                    next_states_batch.push_back(std::get<3>(sample));
                }

                torch::Tensor states = torch::stack(states_batch);
                torch::Tensor actions = torch::stack(actions_batch).view({-1, 1});
                torch::Tensor rewards = torch::stack(rewards_batch);
                torch::Tensor next_states = torch::stack(next_states_batch);

                // Compute Q-values for current states
                torch::Tensor state_action_values = policy_net.forward(states).gather(1, actions).squeeze();

                // Compute max Q-value for next states from target network
              
                torch::Tensor next_state_values = std::get<0>(target_net.forward(next_states).max(1)).detach();


                // Compute expected Q-values using Bellman equation
                torch::Tensor expected_state_action_values = rewards + GAMMA * next_state_values;

                // Compute loss
                torch::Tensor loss = torch::mse_loss(state_action_values, expected_state_action_values);

                // Backpropagation
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }
        }

        // Update target network every `TARGET_UPDATE` episodes
        if (episode % TARGET_UPDATE == 0) {
            target_net.copy_from(policy_net);
        }

        std::cout << "Episode " << episode << " completed. Total Reward: " << total_reward << std::endl;
    }

    return 0;
}
