#include <stdio.h>
#include <cuda_runtime.h>

#define STATES 6
#define EPISODES 1000
#define ALPHA 0.1f   // Learning rate
#define GAMMA 0.9f   // Discount factor

__global__ void td0_update(float *V, int *transitions, float *rewards) {
    int state = threadIdx.x;
    int next_state = transitions[state];
    float reward = rewards[state];
    
    // TD(0) Update Rule: V(s) <- V(s) + alpha * [R + gamma * V(s') - V(s)]
    V[state] += ALPHA * (reward + GAMMA * V[next_state] - V[state]);
}

void print_values(float *V) {
    for (int i = 0; i < STATES; i++) {
        printf("%.4f ", V[i]);
    }
    printf("\n");
}

int main() {
    float *d_V, *d_rewards;
    int *d_transitions;
    
    float V[STATES] = {0, 0, 0, 0, 0, 0};  // Initialize value function
    float rewards[STATES] = {0, 1, -1, 2, -2, 3}; // Example rewards
    int transitions[STATES] = {1, 2, 3, 4, 5, 5}; // Example state transitions

    cudaMalloc(&d_V, STATES * sizeof(float));
    cudaMalloc(&d_rewards, STATES * sizeof(float));
    cudaMalloc(&d_transitions, STATES * sizeof(int));
    
    cudaMemcpy(d_V, V, STATES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, rewards, STATES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, transitions, STATES * sizeof(int), cudaMemcpyHostToDevice);
    
    for (int episode = 0; episode < EPISODES; episode++) {
        td0_update<<<1, STATES>>>(d_V, d_transitions, d_rewards);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(V, d_V, STATES * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("TD(0) Value Function:\n");
    print_values(V);
    
    cudaFree(d_V);
    cudaFree(d_rewards);
    cudaFree(d_transitions);
    return 0;
}