#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>

#define N_STATES 5
#define N_ACTIONS 4
#define ALPHA 0.1f
#define GAMMA 0.9f

__global__ void initCurand(curandState* state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void boltzmannExploration(float* Q, int* actions, int nStates, int nActions, float temperature, curandState* randState) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nStates) {
        float sumExpQ = 0.0f;
        __shared__ float expQ_shared[N_ACTIONS];
        for (int a = 0; a < nActions; a++) {
            expQ_shared[a] = expf(Q[idx * nActions + a] / temperature);
            sumExpQ += expQ_shared[a];
        }
        float randVal = curand_uniform(&randState[idx]) * sumExpQ;
        float cumulativeSum = 0.0f;
        for (int a = 0; a < nActions; a++) {
            cumulativeSum += expQ_shared[a];
            if (randVal <= cumulativeSum) {
                actions[idx] = a;
                break;
            }
        }
    }
}

__global__ void epsilonGreedy(float* Q, int* actions, int nStates, int nActions, float epsilon, curandState* randState) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nStates) {
        float randVal = curand_uniform(&randState[idx]);
        if (randVal < epsilon) {
            actions[idx] = curand(&randState[idx]) % nActions;
        } else {
            float maxQ = -1e9;
            int bestAction = 0;
            for (int a = 0; a < nActions; a++) {
                float qValue = Q[idx * nActions + a];
                if (qValue > maxQ) {
                    maxQ = qValue;
                    bestAction = a;
                }
            }
            actions[idx] = bestAction;
        }
    }
}

__global__ void qLearningUpdate(float* Q, int* actions, float* rewards, int* nextStates, int nStates, int nActions, float alpha, float gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nStates) {
        int action = actions[idx];
        int nextState = nextStates[idx];
        float maxQNext = -1e9;
        for (int a = 0; a < nActions; a++) {
            maxQNext = fmaxf(maxQNext, Q[nextState * nActions + a]);
        }
        Q[idx * nActions + action] += alpha * (rewards[idx] + gamma * maxQNext - Q[idx * nActions + action]);
    }
}

int main() {
    float *d_Q, *d_rewards;
    int *d_actions, *d_nextStates;
    curandState* d_randState;
    float epsilon = 1.0f;
    float temperature = 1.0f;
    float epsilonDecay = 0.995f;
    float minEpsilon = 0.01f;
    float minTemperature = 0.1f;
    cudaMalloc(&d_Q, N_STATES * N_ACTIONS * sizeof(float));
    cudaMalloc(&d_actions, N_STATES * sizeof(int));
    cudaMalloc(&d_rewards, N_STATES * sizeof(float));
    cudaMalloc(&d_nextStates, N_STATES * sizeof(int));
    cudaMalloc(&d_randState, N_STATES * sizeof(curandState));
    float h_Q[N_STATES * N_ACTIONS];
    for (int i = 0; i < N_STATES * N_ACTIONS; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }
    cudaMemcpy(d_Q, h_Q, N_STATES * N_ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    float h_rewards[N_STATES];
    int h_nextStates[N_STATES];
    for (int i = 0; i < N_STATES; i++) {
        h_rewards[i] = ((float)rand() / RAND_MAX);
        h_nextStates[i] = rand() % N_STATES;
    }
    cudaMemcpy(d_rewards, h_rewards, N_STATES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nextStates, h_nextStates, N_STATES * sizeof(int), cudaMemcpyHostToDevice);
    initCurand<<<(N_STATES + 255) / 256, 256>>>(d_randState, time(NULL));
    for (int episode = 0; episode < 1000; ++episode) {
        if (episode % 2 == 0) {
            boltzmannExploration<<<(N_STATES + 255) / 256, 256>>>(d_Q, d_actions, N_STATES, N_ACTIONS, temperature, d_randState);
        } else {
            epsilonGreedy<<<(N_STATES + 255) / 256, 256>>>(d_Q, d_actions, N_STATES, N_ACTIONS, epsilon, d_randState);
        }
        qLearningUpdate<<<(N_STATES + 255) / 256, 256>>>(d_Q, d_actions, d_rewards, d_nextStates, N_STATES, N_ACTIONS, ALPHA, GAMMA);
        epsilon = fmaxf(epsilon * epsilonDecay, minEpsilon);
        temperature = fmaxf(temperature * epsilonDecay, minTemperature);
    }
    float Q[N_STATES * N_ACTIONS];
    cudaMemcpy(Q, d_Q, N_STATES * N_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Final Q-table (Sample):\n");
    for (int i = 0; i < N_STATES; i++) {
        for (int j = 0; j < N_ACTIONS; j++) {
            printf("%.4f ", Q[i * N_ACTIONS + j]);
        }
        printf("\n");
    }
    cudaFree(d_Q);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_nextStates);
    cudaFree(d_randState);
    return 0;
}