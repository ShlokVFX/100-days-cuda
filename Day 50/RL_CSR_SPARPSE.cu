#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <curand_kernel.h>

#define N_STATES 100     // Number of states
#define N_ACTIONS 4      // Number of actions
#define EPSILON 0.1f     // Exploration rate
#define ALPHA 0.1f       // Learning rate
#define GAMMA 0.9f       // Discount factor

// CSR Data Structures
struct CSRMatrix {
    int* rowPtr;         // Row pointer
    int* colInd;         // Column indices
    float* values;       // Non-zero values
};

__global__ void initQTable(float* Q, int nStates, int nActions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nStates * nActions) {
        curandState randState;
        curand_init(clock64(), idx, 0, &randState);
        Q[idx] = curand_uniform(&randState) * 0.1f; // Small random values
    }
}

// Random action selection for exploration
__global__ void epsilonGreedy(float* Q, int* actions, int nStates, int nActions, float epsilon) {
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state < nStates) {
        curandState randState;
        curand_init(clock64(), state, 0, &randState);

        if (curand_uniform(&randState) < epsilon) {
            actions[state] = curand(&randState) % nActions; // Random action
        } else {
            float maxQ = -1e9;
            int bestAction = 0;

            for (int a = 0; a < nActions; a++) {
                if (Q[state * nActions + a] > maxQ) {
                    maxQ = Q[state * nActions + a];
                    bestAction = a;
                }
            }

            actions[state] = bestAction;
        }
    }
}

// Q-Learning Update
__global__ void qLearningUpdate(float* Q, int* actions, float* rewards, int* nextStates, 
                                int nStates, int nActions, float alpha, float gamma) {
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state < nStates) {
        int action = actions[state];
        float reward = rewards[state];
        int nextState = nextStates[state];

        // Bellman Equation: Q(s, a) = Q(s, a) + α[R + γ max Q(s', a') - Q(s, a)]
        float maxQNext = -1e9;
        for (int a = 0; a < nActions; a++) {
            maxQNext = fmaxf(maxQNext, Q[nextState * nActions + a]);
        }

        Q[state * nActions + action] += alpha * (reward + gamma * maxQNext - Q[state * nActions + action]);
    }
}

int main() {
    // Host data
    float *h_Q, *d_Q;
    int *d_actions, *d_nextStates;
    float *d_rewards;

    // Allocate and initialize Q-table
    cudaMalloc(&d_Q, N_STATES * N_ACTIONS * sizeof(float));
    cudaMalloc(&d_actions, N_STATES * sizeof(int));
    cudaMalloc(&d_rewards, N_STATES * sizeof(float));
    cudaMalloc(&d_nextStates, N_STATES * sizeof(int));

    // Initialize Q-table
    initQTable<<<(N_STATES * N_ACTIONS + 255) / 256, 256>>>(d_Q, N_STATES, N_ACTIONS);

    // Simulate one episode (example only)
    epsilonGreedy<<<(N_STATES + 255) / 256, 256>>>(d_Q, d_actions, N_STATES, N_ACTIONS, EPSILON);

    qLearningUpdate<<<(N_STATES + 255) / 256, 256>>>(d_Q, d_actions, d_rewards, d_nextStates,
                                                     N_STATES, N_ACTIONS, ALPHA, GAMMA);

    // Copy back and print Q-table
    h_Q = new float[N_STATES * N_ACTIONS];
    cudaMemcpy(h_Q, d_Q, N_STATES * N_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final Q-table (Sample):\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < N_ACTIONS; ++j) {
            std::cout << h_Q[i * N_ACTIONS + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_nextStates);
    delete[] h_Q;

    return 0;
}
