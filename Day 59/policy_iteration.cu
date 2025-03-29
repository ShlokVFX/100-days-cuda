#include <stdio.h>
#include <cuda_runtime.h>

#define STATES 6
#define ACTIONS 2
#define GAMMA 0.9
#define THETA 1e-6

__global__ void evaluatePolicy(float *V, int *policy, float *rewards, float *transitions, bool *stable) {
    int s = threadIdx.x;
    float v_old = V[s];
    
    int a = policy[s];
    V[s] = 0.0;
    for (int next_s = 0; next_s < STATES; next_s++) {
        V[s] += transitions[s * STATES * ACTIONS + a * STATES + next_s] * (rewards[s * ACTIONS + a] + GAMMA * V[next_s]);
    }
    
    if (fabs(v_old - V[s]) > THETA) *stable = false;
}

__global__ void improvePolicy(float *V, int *policy, float *rewards, float *transitions, bool *stable) {
    int s = threadIdx.x;
    float best_value = -1e9;
    int best_action = 0;
    
    for (int a = 0; a < ACTIONS; a++) {
        float q_sa = 0.0;
        for (int next_s = 0; next_s < STATES; next_s++) {
            q_sa += transitions[s * STATES * ACTIONS + a * STATES + next_s] * (rewards[s * ACTIONS + a] + GAMMA * V[next_s]);
        }
        if (q_sa > best_value) {
            best_value = q_sa;
            best_action = a;
        }
    }
    
    if (policy[s] != best_action) {
        policy[s] = best_action;
        *stable = false;
    }
}

void policyIteration() {
    float V[STATES] = {0};
    int policy[STATES] = {0};
    float rewards[STATES * ACTIONS] = {0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 0, 6};
    float transitions[STATES * STATES * ACTIONS] = {0.6, 0.4, 0, 0, 0, 0,  0.5, 0.5, 0, 0, 0, 0,
                                                     0, 0, 0.7, 0.3, 0, 0,  0, 0, 0.6, 0.4, 0, 0,
                                                     0, 0, 0, 0, 0.8, 0.2,  0, 0, 0, 0, 0.5, 0.5};
    
    float *d_V, *d_rewards, *d_transitions;
    int *d_policy;
    bool *d_stable, stable;
    cudaMalloc(&d_V, STATES * sizeof(float));
    cudaMalloc(&d_policy, STATES * sizeof(int));
    cudaMalloc(&d_rewards, STATES * ACTIONS * sizeof(float));
    cudaMalloc(&d_transitions, STATES * STATES * ACTIONS * sizeof(float));
    cudaMalloc(&d_stable, sizeof(bool));
    
    cudaMemcpy(d_V, V, STATES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_policy, policy, STATES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, rewards, STATES * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, transitions, STATES * STATES * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    
    int iteration = 0;
    do {
        stable = true;
        cudaMemcpy(d_stable, &stable, sizeof(bool), cudaMemcpyHostToDevice);
        evaluatePolicy<<<1, STATES>>>(d_V, d_policy, d_rewards, d_transitions, d_stable);
        improvePolicy<<<1, STATES>>>(d_V, d_policy, d_rewards, d_transitions, d_stable);
        cudaMemcpy(&stable, d_stable, sizeof(bool), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(V, d_V, STATES * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Iteration %d - V-values: ", iteration++);
        for (int s = 0; s < STATES; s++) printf("%.4f ", V[s]);
        printf("\n");
    } while (!stable);
    
    cudaMemcpy(policy, d_policy, STATES * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Optimal Policy:\n");
    for (int s = 0; s < STATES; s++) printf("%d ", policy[s]);
    printf("\n");
    
    cudaFree(d_V);
    cudaFree(d_policy);
    cudaFree(d_rewards);
    cudaFree(d_transitions);
    cudaFree(d_stable);
}

int main() {
    policyIteration();
    return 0;
}
