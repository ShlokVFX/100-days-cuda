#include <stdio.h>
#include <curand_kernel.h>

#define N_STATES 9
#define N_ACTIONS 4
#define ALPHA 0.1f
#define GAMMA 0.9f
#define EPSILON 0.1f
#define EPISODES 1000

__device__ float getExpectedQ(float* Q, int state) {
    float expectedQ = 0.0f;
    for (int a = 0; a < N_ACTIONS; a++) {
        expectedQ += (1.0f / N_ACTIONS) * Q[state * N_ACTIONS + a]; // Uniform policy assumption
    }
    return expectedQ;
}

__global__ void expectedSARSA(float* Q, int* transitions, float* rewards, curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N_STATES) return;

    curandState localState = states[tid];

    for (int episode = 0; episode < EPISODES; episode++) {
        int state = tid;
        int action = curand(&localState) % N_ACTIONS;

        int next_state = transitions[state * N_ACTIONS + action];
        float reward = rewards[state * N_ACTIONS + action];

        float expectedQ = getExpectedQ(Q, next_state);

        // Update Q-value
        Q[state * N_ACTIONS + action] += ALPHA * (reward + GAMMA * expectedQ - Q[state * N_ACTIONS + action]);
    }
    states[tid] = localState;
}

int main() {
    float h_Q[N_STATES * N_ACTIONS] = {0};
    int h_transitions[N_STATES * N_ACTIONS] = { // Fake transition table
        1, 3, 0, 0, 2, 4, 1, 1, 3, 5, 2, 2,
        4, 6, 3, 3, 5, 7, 4, 4, 6, 8, 5, 5,
        7, 7, 6, 6, 8, 8, 7, 7, 8, 8, 8, 8
    };
    float h_rewards[N_STATES * N_ACTIONS] = { // Fake reward table
        0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0
    };

    float* d_Q;
    int* d_transitions;
    float* d_rewards;
    curandState* d_states;

    cudaMalloc(&d_Q, N_STATES * N_ACTIONS * sizeof(float));
    cudaMalloc(&d_transitions, N_STATES * N_ACTIONS * sizeof(int));
    cudaMalloc(&d_rewards, N_STATES * N_ACTIONS * sizeof(float));
    cudaMalloc(&d_states, N_STATES * sizeof(curandState));

    cudaMemcpy(d_Q, h_Q, N_STATES * N_ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, h_transitions, N_STATES * N_ACTIONS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, h_rewards, N_STATES * N_ACTIONS * sizeof(float), cudaMemcpyHostToDevice);

    expectedSARSA<<<1, N_STATES>>>(d_Q, d_transitions, d_rewards, d_states);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Q, d_Q, N_STATES * N_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Expected SARSA Q-values:\n");
    for (int i = 0; i < N_STATES; i++) {
        for (int j = 0; j < N_ACTIONS; j++) {
            printf("%.4f ", h_Q[i * N_ACTIONS + j]);
        }
        printf("\n");
    }

    cudaFree(d_Q);
    cudaFree(d_transitions);
    cudaFree(d_rewards);
    cudaFree(d_states);

    return 0;
}
