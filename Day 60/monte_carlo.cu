#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define STATES 6
#define ACTIONS 2
#define EPISODES 1000
#define GAMMA 0.9

__device__ int generateEpisode(int *episode, float *rewards, float *transitions, curandState *state) {
    int s = curand(state) % STATES; // Start from a random state
    int length = 0;
    
    while (length < STATES) { // Limit episode length to STATES
        int a = curand(state) % ACTIONS; // Random action selection
        float rand_val = curand_uniform(state);
        float prob_sum = 0.0;
        int next_s = 0;
        
        for (int i = 0; i < STATES; i++) {
            prob_sum += transitions[s * STATES * ACTIONS + a * STATES + i];
            if (rand_val <= prob_sum) {
                next_s = i;
                break;
            }
        }
        
        episode[length * 3] = s;
        episode[length * 3 + 1] = a;
        episode[length * 3 + 2] = next_s;
        s = next_s;
        length++;
    }
    return length;
}

__global__ void monteCarloPrediction(float *V, float *rewards, float *transitions) {
    int s = threadIdx.x;
    curandState state;
    curand_init(clock64(), s, 0, &state);
    
    float returns[STATES] = {0};
    int visit_count[STATES] = {0};
    int episode[STATES * 3];
    
    for (int ep = 0; ep < EPISODES; ep++) {
        int length = generateEpisode(episode, rewards, transitions, &state);
        float G = 0.0;
        
        for (int t = length - 1; t >= 0; t--) {
            int state_t = episode[t * 3];
            int action_t = episode[t * 3 + 1];
            G = rewards[state_t * ACTIONS + action_t] + GAMMA * G;
            
            bool first_visit = true;
            for (int k = 0; k < t; k++) {
                if (episode[k * 3] == state_t) {
                    first_visit = false;
                    break;
                }
            }
            
            if (first_visit) {
                returns[state_t] += G;
                visit_count[state_t] += 1;
            }
        }
    }
    
    for (int i = 0; i < STATES; i++) {
        if (visit_count[i] > 0) {
            V[i] = returns[i] / visit_count[i];
        }
    }
}

void runMonteCarlo() {
    float V[STATES] = {0};
    float rewards[STATES * ACTIONS] = {0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 0, 6};
    float transitions[STATES * STATES * ACTIONS] = {0.6, 0.4, 0, 0, 0, 0,  0.5, 0.5, 0, 0, 0, 0,
                                                     0, 0, 0.7, 0.3, 0, 0,  0, 0, 0.6, 0.4, 0, 0,
                                                     0, 0, 0, 0, 0.8, 0.2,  0, 0, 0, 0, 0.5, 0.5};
    
    float *d_V, *d_rewards, *d_transitions;
    cudaMalloc(&d_V, STATES * sizeof(float));
    cudaMalloc(&d_rewards, STATES * ACTIONS * sizeof(float));
    cudaMalloc(&d_transitions, STATES * STATES * ACTIONS * sizeof(float));
    
    cudaMemcpy(d_rewards, rewards, STATES * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, transitions, STATES * STATES * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    
    monteCarloPrediction<<<1, STATES>>>(d_V, d_rewards, d_transitions);
    cudaMemcpy(V, d_V, STATES * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Monte Carlo Value Function:\n");
    for (int s = 0; s < STATES; s++) printf("%.4f ", V[s]);
    printf("\n");
    
    cudaFree(d_V);
    cudaFree(d_rewards);
    cudaFree(d_transitions);
}

int main() {
    runMonteCarlo();
    return 0;
}
