#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>

#define STATES 9
#define ACTIONS 4
#define EPISODES 5000
#define ALPHA 0.5
#define GAMMA 0.9
#define EPSILON_MAX 0.1
#define EPSILON_MIN 0.01
#define LAMBDA 0.001

__device__ int getNextState(int state, int action) {
    int row = state / 3, col = state % 3;
    if (action == 0 && row > 0) return state - 3;
    if (action == 1 && col < 2) return state + 1;
    if (action == 2 && row < 2) return state + 3;
    if (action == 3 && col > 0) return state - 1;
    return state;
}

__device__ int chooseAction(float *Q, int state, curandState *randState, float epsilon) {
    if (curand_uniform(randState) < epsilon) {
        return curand(randState) % ACTIONS;
    } else {
        int best_action = 0;
        float best_value = Q[state * ACTIONS];
        for (int i = 1; i < ACTIONS; i++) {
            if (Q[state * ACTIONS + i] > best_value) {
                best_value = Q[state * ACTIONS + i];
                best_action = i;
            }
        }
        return best_action;
    }
}

__global__ void sarsa_kernel(float *Q, curandState *randState) {
    int tid = threadIdx.x;
    curand_init(1234, tid, 0, &randState[tid]);
    
    for (int episode = 0; episode < EPISODES; episode++) {
        float epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * expf(-LAMBDA * episode);
        int state = curand(&randState[tid]) % STATES;
        int action = chooseAction(Q, state, &randState[tid], epsilon);
        
        for (int step = 0; step < 100; step++) {
            int next_state = getNextState(state, action);
            int next_action = chooseAction(Q, next_state, &randState[tid], epsilon);
            
            float reward = (next_state == STATES - 1) ? 10.0f : -0.1f;
            Q[state * ACTIONS + action] += ALPHA * (reward + GAMMA * Q[next_state * ACTIONS + next_action] - Q[state * ACTIONS + action]);
            
            state = next_state;
            action = next_action;
        }
    }
}

int main() {
    float *d_Q;
    curandState *d_randState;
    cudaMalloc(&d_Q, STATES * ACTIONS * sizeof(float));
    cudaMalloc(&d_randState, STATES * sizeof(curandState));
    cudaMemset(d_Q, 0, STATES * ACTIONS * sizeof(float));
    
    sarsa_kernel<<<1, STATES>>>(d_Q, d_randState);
    cudaDeviceSynchronize();
    
    float Q[STATES * ACTIONS];
    cudaMemcpy(Q, d_Q, STATES * ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("SARSA Q-values:\n");
    for (int s = 0; s < STATES; s++) {
        for (int a = 0; a < ACTIONS; a++) {
            printf("%.4f ", Q[s * ACTIONS + a]);
        }
        printf("\n");
    }
    
    cudaFree(d_Q);
    cudaFree(d_randState);
    return 0;
}
