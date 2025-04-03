#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define STATE_SIZE 4 
#define ACTION_SIZE 2  
#define MEMORY_SIZE 10000
#define BATCH_SIZE 32
#define GAMMA 0.99
#define LEARNING_RATE 0.001

__global__ void forwardPassKernel(float* input, float* weights, float* output, int inputSize, int outputSize) {
    int idx = threadIdx.x;
    if (idx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[i] * weights[i * outputSize + idx];
        }
        output[idx] = sum;
    }
}

class DQN {
public:
    float *d_weights;
    float *d_input, *d_output;

    DQN() {
        cudaMalloc(&d_weights, STATE_SIZE * ACTION_SIZE * sizeof(float));
        cudaMalloc(&d_input, STATE_SIZE * sizeof(float));
        cudaMalloc(&d_output, ACTION_SIZE * sizeof(float));

        // Initialize weights on host and copy to device
        std::vector<float> h_weights(STATE_SIZE * ACTION_SIZE);
        for (auto &w : h_weights) {
            w = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
        }
        cudaMemcpy(d_weights, h_weights.data(), STATE_SIZE * ACTION_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~DQN() {
        cudaFree(d_weights);
        cudaFree(d_input);
        cudaFree(d_output);
    }

    void forward(float* state, float* q_values) {
        cudaMemcpy(d_input, state, STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        forwardPassKernel<<<1, ACTION_SIZE>>>(d_input, d_weights, d_output, STATE_SIZE, ACTION_SIZE);
        cudaMemcpy(q_values, d_output, ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

int main() {
    DQN agent;
    float state[STATE_SIZE] = {1.0, 0.5, -0.5, 0.2};
    float q_values[ACTION_SIZE];
    
    agent.forward(state, q_values);
    std::cout << "Q-values: ";
    for (int i = 0; i < ACTION_SIZE; i++) {
        std::cout << q_values[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
