#include <iostream>
#include <curand_kernel.h>
#include <math.h>  // For fmaxf()
#include <cuda_runtime.h>

#define S 9      // States
#define A 4      // Actions
#define EPS 1000 // Episodes
#define _A_ 0.1f // Learning rate
#define _G_ 0.9f // Discount factor
#define _E_ 1.0f // Exploration rate
#define _D_ 0.99f // Decay

__global__ void init_rand(curandState *state, int seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < S) curand_init(seed, i, 0, &state[i]);
}

__device__ int rnd_action(curandState *s) {
    return curand(s) % A;
}

__global__ void q_train(float *Q, int *R, curandState *randS) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= S - 1) return;

    curandState s = randS[i];
    int st = i;
    float e = _E_;

    for (int ep = 0; ep < EPS; ep++) {
        bool done = false;

        while (!done) {
            int act;
            if (curand_uniform(&s) < e) {
                act = rnd_action(&s); // Exploration
            } else {
                // Exploitation: Choose best action
                float maxQ = -1e9;
                for (int j = 0; j < A; j++) {
                    if (Q[st * A + j] > maxQ) {
                        maxQ = Q[st * A + j];
                        act = j;
                    }
                }
            }

            // Ensure next state follows a valid transition
            int nxt = (st + act) % S;
            float r = R[nxt];

            // Find max Q-value for next state
            float maxQ = -1e9;
            for (int j = 0; j < A; j++)
                maxQ = fmaxf(maxQ, Q[nxt * A + j]);

            // Q-value update
            Q[st * A + act] += _A_ * (r + _G_ * maxQ - Q[st * A + act]);

            st = nxt;
            if (st == S - 1) done = true;
        }
        e *= _D_; // Decay exploration rate
    }
    randS[i] = s;
}

int main() {
    float h_Q[S * A];
    int h_R[S] = {-1, -1, 10, -1, -10, -1, -1, -10, 10};

    // Initialize Q-table with small random values
    for (int i = 0; i < S * A; i++) {
        h_Q[i] = ((float)(rand() % 100) / 100.0f) * 0.01f; // Small random values
    }

    float *d_Q;
    int *d_R;
    curandState *d_randS;

    cudaMalloc((void **)&d_Q, S * A * sizeof(float));
    cudaMalloc((void **)&d_R, S * sizeof(int));
    cudaMalloc((void **)&d_randS, S * sizeof(curandState));

    cudaMemcpy(d_Q, h_Q, S * A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, h_R, S * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize random states
    init_rand<<<1, S>>>(d_randS, time(NULL));

    // Train the Q-table
    q_train<<<1, S - 1>>>(d_Q, d_R, d_randS);

    // Copy results back
    cudaMemcpy(h_Q, d_Q, S * A * sizeof(float), cudaMemcpyDeviceToHost);

    // Print final Q-values
    std::cout << "Q-values:\n";
    for (int i = 0; i < S; i++) {
        std::cout << "S" << i << ": ";
        for (int j = 0; j < A; j++)
            std::cout << h_Q[i * A + j] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_Q);
    cudaFree(d_R);
    cudaFree(d_randS);

    return 0;
}
