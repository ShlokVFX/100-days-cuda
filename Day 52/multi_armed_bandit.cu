#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define N_ARMS 10
#define N_TRIALS 1000
#define EPSILON 0.1

__global__ void bandit_kernel(float *estimates, int *counts, curandState *states, int n_trials) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState;
    
    if (tid == 0) {
        curand_init(1234, tid, 0, &localState);

        for (int t = 0; t < n_trials; t++) {
            float rand_val = curand_uniform(&localState);
            int action;
            if (rand_val < EPSILON) {
                action = curand(&localState) % N_ARMS;
            } else {
                action = 0;
                float max_estimate = estimates[0];
                for (int i = 1; i < N_ARMS; i++) {
                    if (estimates[i] > max_estimate) {
                        max_estimate = estimates[i];
                        action = i;
                    }
                }
            }

            float reward = curand_uniform(&localState);
            counts[action]++;
            estimates[action] += (reward - estimates[action]) / counts[action];
        }
    }
}

int main() {
    float *d_estimates, h_estimates[N_ARMS] = {0};
    int *d_counts, h_counts[N_ARMS] = {0};
    curandState *d_states;

    cudaMalloc(&d_estimates, N_ARMS * sizeof(float));
    cudaMalloc(&d_counts, N_ARMS * sizeof(int));
    cudaMalloc(&d_states, sizeof(curandState));

    cudaMemcpy(d_estimates, h_estimates, N_ARMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, N_ARMS * sizeof(int), cudaMemcpyHostToDevice);

    bandit_kernel<<<1, 1>>>(d_estimates, d_counts, d_states, N_TRIALS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_estimates, d_estimates, N_ARMS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_counts, N_ARMS * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Final Action Value Estimates:\n";
    for (int i = 0; i < N_ARMS; i++) {
        std::cout << "Arm " << i << ": " << h_estimates[i] << " (Chosen " << h_counts[i] << " times)\n";
    }

    cudaFree(d_estimates);
    cudaFree(d_counts);
    cudaFree(d_states);

    return 0;
}
