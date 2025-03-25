#include <iostream>
#include <curand_kernel.h>
#include <math.h>
#include <cuda_runtime.h>

#define S 9       // States
#define A 4       // Actions
#define EPS 1000  // Episodes
#define _A_ 0.1f  // Base learning rate (will be adapted)
#define _G_ 0.9f  // Discount factor

// We'll use adaptive exploration with exponential decay, so no constant _E_ or _D_ macros here.

__device__ int rnd_action(curandState *s) {
    return curand(s) % A;
}

__global__ void init_rand(curandState *state, int seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < S) curand_init(seed, i, 0, &state[i]);
}

__global__ void q_train(float *Q, float *R, curandState *randS, int *visit_count) {
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id >= S) return;

    curandState s = randS[agent_id];
    int st = agent_id;
    float e = 1.0f;  // Start with full exploration

    for (int ep = 0; ep < EPS; ep++) {
        bool done = false;
        while (!done) {
            int act;
            if (curand_uniform(&s) < e) {
                act = rnd_action(&s); // Exploration: random action
            } else {
                // Exploitation: choose best action for current state
                float bestQ = -1e9;
                for (int j = 0; j < A; j++) {
                    float q_val = Q[st * A + j];
                    if (q_val > bestQ) {
                        bestQ = q_val;
                        act = j;
                    }
                }
            }

            // Transition to next state based on action (example transition)
            int nxt = (st + act) % S;
            float r = R[nxt];

            // Find max Q for the next state
            float maxQ = -1e9;
            for (int j = 0; j < A; j++)
                maxQ = fmaxf(maxQ, Q[nxt * A + j]);

            // Update visit count and compute adaptive learning rate
            int idx = st * A + act;
            visit_count[idx] += 1;
            float alpha = 1.0f / (1.0f + visit_count[idx]);

            // Compute TD error and update Q-value if significant
            float td_error = fabsf(r + _G_ * maxQ - Q[idx]);
            if (td_error > 0.01f) {
                Q[idx] += alpha * (r + _G_ * maxQ - Q[idx]);
            }

            st = nxt;
            if (st == S - 1) done = true;
        }
        e = expf(-0.001f * ep);  // Adaptive exploration decay
    }
    randS[agent_id] = s;
}

int main() {
    float h_Q[S * A] = {0};
    // Change rewards to float for fractional values
    float h_R[S] = {-0.1f, -0.1f, 10.0f, -0.5f, -10.0f, -0.5f, -0.5f, -10.0f, 10.0f};

    float *d_Q, *d_R;
    int *d_visit_count;
    curandState *d_randS;

    cudaMalloc((void **)&d_Q, S * A * sizeof(float));
    cudaMalloc((void **)&d_R, S * sizeof(float));
    cudaMalloc((void **)&d_randS, S * sizeof(curandState));
    cudaMalloc((void **)&d_visit_count, S * A * sizeof(int));

    cudaMemcpy(d_Q, h_Q, S * A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, h_R, S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_visit_count, 0, S * A * sizeof(int));

    // Initialize random states
    init_rand<<<1, S>>>(d_randS, time(NULL));
    cudaDeviceSynchronize();

    // Train the Q-table using multi-agent training (one agent per state)
    q_train<<<1, S>>>(d_Q, d_R, d_randS, d_visit_count);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Q, d_Q, S * A * sizeof(float), cudaMemcpyDeviceToHost);

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
    cudaFree(d_visit_count);

    return 0;
}
