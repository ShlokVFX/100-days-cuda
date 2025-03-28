#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define S 10      // States
#define A 4       // Actions
#define __A__ 0.7 // Learning Rate
#define __G__ 0.99 // Discount Factor
#define EP 10000   // Training Episodes

__global__ void initR(float *R, curandState *states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < S * A) {
        curand_init(1234, i, 0, &states[i]);
        R[i] = (curand(&states[i]) % 10) / 10.0f; // Larger rewards
    }
}

__global__ void initQ(float *Q1, float *Q2) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < S * A) {
        Q1[i] = 0.0f;
        Q2[i] = 0.0f;
    }
}

__global__ void doubleQlearning(float *Q1, float *Q2, float *R, curandState *states, float eps) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < S) {
        int a = curand(&states[i]) % A;
        int next_s = curand(&states[i]) % S;
        float maxQ;
        
        if (curand_uniform(&states[i]) < 0.5f) {
            int maxA = 0;
            for (int j = 1; j < A; j++)
                if (Q1[next_s * A + j] > Q1[next_s * A + maxA])
                    maxA = j;
            maxQ = Q2[next_s * A + maxA];  
            Q1[i * A + a] += __A__ * (R[i * A + a] + __G__ * maxQ - Q1[i * A + a]);
        } else {
            int maxA = 0;
            for (int j = 1; j < A; j++)
                if (Q2[next_s * A + j] > Q2[next_s * A + maxA])
                    maxA = j;
            maxQ = Q1[next_s * A + maxA];
            Q2[i * A + a] += __A__ * (R[i * A + a] + __G__ * maxQ - Q2[i * A + a]);
        }
    }
}

int main() {
    float *d_Q1, *d_Q2, *d_R;
    curandState *d_states;
    cudaMalloc(&d_Q1, S * A * sizeof(float));
    cudaMalloc(&d_Q2, S * A * sizeof(float));
    cudaMalloc(&d_R, S * A * sizeof(float));
    cudaMalloc(&d_states, S * A * sizeof(curandState));

    initQ<<<1, S * A>>>(d_Q1, d_Q2);
    initR<<<1, S * A>>>(d_R, d_states);
    cudaDeviceSynchronize();

    for (int ep = 0; ep < EP; ep++) {
        float eps = max(0.1f, 1.0f - 0.001f * ep); // Slower Decay
        doubleQlearning<<<1, S>>>(d_Q1, d_Q2, d_R, d_states, eps);
        cudaDeviceSynchronize();
    }

    float h_Q1[S * A], h_Q2[S * A];
    cudaMemcpy(h_Q1, d_Q1, S * A * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Q2, d_Q2, S * A * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Double Q-Learning Q-values:\n";
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < A; j++)
            std::cout << (h_Q1[i * A + j] + h_Q2[i * A + j]) / 2.0 << " ";
        std::cout << "\n";
    }

    cudaFree(d_Q1);
    cudaFree(d_Q2);
    cudaFree(d_R);
    cudaFree(d_states);
    return 0;
}
