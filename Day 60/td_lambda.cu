#include <iostream>
#include <cuda_runtime.h>

#define STATES 6
#define EPISODES 1000
#define ALPHA 0.1    // Learning rate
#define GAMMA 0.9    // Discount factor
#define LAMBDA 0.8   // Trace decay rate

__global__ void td_lambda_kernel(float *V, float *E, int *state_transitions, float *rewards) {
    int tid = threadIdx.x;

    if (tid < STATES) {
        float delta;
        for (int episode = 0; episode < EPISODES; episode++) {
            for (int t = 0; t < STATES - 1; t++) {
                int s = t;
                int s_next = state_transitions[t];

                // TD Error
                delta = rewards[s] + GAMMA * V[s_next] - V[s];

                // Update eligibility traces
                E[s] = E[s] * LAMBDA * GAMMA + 1.0f;

                // Update value function
                V[s] += ALPHA * delta * E[s];
            }
        }
    }
}

void td_lambda() {
    float V[STATES] = {0};  // State-value function
    float E[STATES] = {0};  // Eligibility traces
    int state_transitions[STATES] = {1, 2, 3, 4, 5, 5};  // State transitions
    float rewards[STATES] = {0, 0, 0, 0, 1, 0}; // Reward structure

    // Allocate device memory
    float *d_V, *d_E, *d_rewards;
    int *d_state_transitions;
    cudaMalloc((void**)&d_V, STATES * sizeof(float));
    cudaMalloc((void**)&d_E, STATES * sizeof(float));
    cudaMalloc((void**)&d_state_transitions, STATES * sizeof(int));
    cudaMalloc((void**)&d_rewards, STATES * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_V, V, STATES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, STATES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_state_transitions, state_transitions, STATES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, rewards, STATES * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    td_lambda_kernel<<<1, STATES>>>(d_V, d_E, d_state_transitions, d_rewards);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(V, d_V, STATES * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "TD(λ) Value Function:\n";
    for (int i = 0; i < STATES; i++) {
        std::cout << V[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_state_transitions);
    cudaFree(d_rewards);
}

int main() {
    td_lambda();
    return 0;
}
