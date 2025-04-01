#include <torch/torch.h>
#include <vector>
#include <random>
#include <iostream>
#include <deque>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// ------------------ DQN Class Definition ------------------

class DQN {
public:
    int input_size;
    int hidden_size;
    int output_size;

    // Host copies of weights and biases.
    std::vector<float> h_W1, h_b1, h_W2, h_b2;
    // Device pointers.
    float *d_W1, *d_b1, *d_W2, *d_b2;

    DQN(int in_size, int hidden, int out_size)
        : input_size(in_size), hidden_size(hidden), output_size(out_size)
    {
        h_W1.resize(hidden_size * input_size);
        h_b1.resize(hidden_size, 0.0f);
        h_W2.resize(output_size * hidden_size);
        h_b2.resize(output_size, 0.0f);

        // Initialize weights with small random values.
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        for(auto &w: h_W1) { w = dist(rng); }
        for(auto &w: h_W2) { w = dist(rng); }

        cudaMalloc((void**)&d_W1, h_W1.size() * sizeof(float));
        cudaMalloc((void**)&d_b1, h_b1.size() * sizeof(float));
        cudaMalloc((void**)&d_W2, h_W2.size() * sizeof(float));
        cudaMalloc((void**)&d_b2, h_b2.size() * sizeof(float));

        cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~DQN() {
        cudaFree(d_W1);
        cudaFree(d_b1);
        cudaFree(d_W2);
        cudaFree(d_b2);
    }

    // Forward pass: compute Q-values from a one-hot encoded state.
    std::vector<float> forward(const std::vector<float>& state);

    // Copy weights/biases from another network.
    void copy_state(const DQN &other) {
        h_W1 = other.h_W1;
        h_b1 = other.h_b1;
        h_W2 = other.h_W2;
        h_b2 = other.h_b2;
        cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
};

// ------------------ CUDA Kernels ------------------

// Kernel to compute a fully connected layer: output = W * input + b
__global__ void forward_layer(const float* W, const float* b, const float* input,
                              float* output, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
        float sum = b[idx];
        for (int i = 0; i < in_size; i++) {
            sum += W[idx * in_size + i] * input[i];
        }
        output[idx] = sum;
    }
}

// Kernel to apply ReLU activation elementwise.
__global__ void relu_activation(float* vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = fmaxf(0.0f, vec[idx]);
    }
}

// ------------------ DQN Forward Implementation ------------------

std::vector<float> DQN::forward(const std::vector<float>& state) {
    float *d_input, *d_hidden, *d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(float));
    cudaMalloc((void**)&d_hidden, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, state.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (hidden_size + blockSize - 1) / blockSize;
    forward_layer<<<numBlocks, blockSize>>>(d_W1, d_b1, d_input, d_hidden, input_size, hidden_size);
    cudaDeviceSynchronize();
    relu_activation<<<numBlocks, blockSize>>>(d_hidden, hidden_size);
    cudaDeviceSynchronize();

    blockSize = 256;
    numBlocks = (output_size + blockSize - 1) / blockSize;
    forward_layer<<<numBlocks, blockSize>>>(d_W2, d_b2, d_hidden, d_output, hidden_size, output_size);
    cudaDeviceSynchronize();

    std::vector<float> output(output_size);
    cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    
    return output;
}

// ------------------ Simple Replay Memory ------------------

struct ReplayMemory {
    std::deque<std::tuple<int, int, int, float, bool>> memory;
    int capacity;
    ReplayMemory(int maxlen) : capacity(maxlen) {}
    void append(std::tuple<int, int, int, float, bool> transition) {
        if(memory.size() >= capacity) memory.pop_front();
        memory.push_back(transition);
    }
    std::vector<std::tuple<int, int, int, float, bool>> sample(int batch_size) {
        std::vector<std::tuple<int, int, int, float, bool>> batch;
        std::sample(memory.begin(), memory.end(), std::back_inserter(batch),
                    batch_size, std::mt19937{std::random_device{}()});
        return batch;
    }
};

// ------------------ FrozenLake Environment ------------------

class FrozenLakeEnv {
public:
    int agent;
    std::vector<int> holes;
    int goal;
    
    FrozenLakeEnv() {
        agent = 0;
        holes = {5, 7, 11, 12};
        goal = 15;
    }
    
    int reset() {
        agent = 0;
        return agent;
    }
    
    std::tuple<int, float, bool> step(int action) {
        int row = agent / 4;
        int col = agent % 4;
        int newRow = row, newCol = col;
        if (action == 0 && col > 0) newCol--;
        else if (action == 1 && row < 3) newRow++;
        else if (action == 2 && col < 3) newCol++;
        else if (action == 3 && row > 0) newRow--;
        int newState = newRow * 4 + newCol;
        agent = newState;
        bool done = false;
        float reward = 0.0f;
        if (newState == goal) { reward = 1.0f; done = true; }
        for (int h : holes) {
            if (newState == h) { reward = 0.0f; done = true; break; }
        }
        return std::make_tuple(newState, reward, done);
    }
};

// ------------------ Main Training Loop ------------------

int main() {
    srand(time(0));
    int num_states = 16, num_actions = 4;
    int episodes = 100;
    float gamma = 0.9f, epsilon = 1.0f;

    FrozenLakeEnv env;
    DQN policy_net(num_states, num_states, num_actions);
    DQN target_net(num_states, num_states, num_actions);
    target_net.copy_state(policy_net);

    ReplayMemory memory(500);

    for (int ep = 0; ep < episodes; ep++) {
        int state = env.reset();
        bool done = false;
        while (!done) {
            int action;
            if ((rand() / (float)RAND_MAX) < epsilon) {
                action = rand() % num_actions;
            } else {
                std::vector<float> state_vec(num_states, 0.0f);
                state_vec[state] = 1.0f;
                std::vector<float> q_vals = policy_net.forward(state_vec);
                action = std::distance(q_vals.begin(), std::max_element(q_vals.begin(), q_vals.end()));
            }
            int new_state;
            float reward;
            std::tie(new_state, reward, done) = env.step(action);
            memory.append({state, action, new_state, reward, done});
            state = new_state;
        }

        // Perform optimization if there are enough samples.
        if (memory.memory.size() > 32) {
            auto batch = memory.sample(32);
            std::vector< std::vector<float> > q_vals_batch;
            std::vector< std::vector<float> > targets_batch;
            for (auto& [s, a, ns, r, t] : batch) {
                std::vector<float> state_vec(num_states, 0.0f);
                state_vec[s] = 1.0f;
                std::vector<float> q = policy_net.forward(state_vec);

                std::vector<float> next_state_vec(num_states, 0.0f);
                next_state_vec[ns] = 1.0f;
                std::vector<float> q_next = target_net.forward(next_state_vec);
                float max_q_next = *std::max_element(q_next.begin(), q_next.end());

                std::vector<float> target_q = q; // copy current q
                target_q[a] = r + (t ? 0.0f : gamma * max_q_next);
                q_vals_batch.push_back(q);
                targets_batch.push_back(target_q);
            }
            // NOTE: For brevity, this code does not implement weight updates.
            // A full implementation would compute a loss between q_vals_batch and targets_batch,
            // then perform backpropagation and update the weights of policy_net.
        }

        if (ep % 10 == 0) {
            target_net.copy_state(policy_net);
        }
        epsilon = std::max(epsilon - 1.0f / episodes, 0.1f);
        std::cout << "Episode " << ep << " completed." << std::endl;
    }

    std::cout << "Training loop finished (no actual training updates are performed in this simplified code)." << std::endl;
    return 0;
}
