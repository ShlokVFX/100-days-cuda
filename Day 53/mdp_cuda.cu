#include <iostream>
#include <cuda_runtime.h>

#define STATES 4
#define ACTIONS 2
#define GAMMA 0.9f
#define THRESHOLD 0.01f

__global__ void value_iteration_kernel(float *V, float *R, float *P, float *newV) {
    int state = threadIdx.x;
    if (state >= STATES) return;

    float max_value = -1e9;
    for (int action = 0; action < ACTIONS; action++) {
        float sum_value = 0.0f;
        for (int next_state = 0; next_state < STATES; next_state++) {
            float transition_prob = P[state * ACTIONS * STATES + action * STATES + next_state];
            float reward = R[state * ACTIONS + action];
            sum_value += transition_prob * (reward + GAMMA * V[next_state]);
        }
        if (sum_value > max_value)
            max_value = sum_value;
    }
    newV[state] = max_value;
}

void value_iteration(float *h_V, float *h_R, float *h_P) {
    float *d_V, *d_R, *d_P, *d_newV;
    cudaMalloc((void **)&d_V, STATES * sizeof(float));
    cudaMalloc((void **)&d_R, STATES * ACTIONS * sizeof(float));
    cudaMalloc((void **)&d_P, STATES * ACTIONS * STATES * sizeof(float));
    cudaMalloc((void **)&d_newV, STATES * sizeof(float));

    cudaMemcpy(d_V, h_V, STATES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, h_R, STATES * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, h_P, STATES * ACTIONS * STATES * sizeof(float), cudaMemcpyHostToDevice);

    bool converged = false;
    while (!converged) {
        value_iteration_kernel<<<1, STATES>>>(d_V, d_R, d_P, d_newV);
        cudaMemcpy(h_V, d_newV, STATES * sizeof(float), cudaMemcpyDeviceToHost);

        converged = true;
        for (int i = 0; i < STATES; i++) {
            if (fabs(h_V[i] - h_V[i]) > THRESHOLD) {
                converged = false;
                break;
            }
        }

        cudaMemcpy(d_V, d_newV, STATES * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_V);
    cudaFree(d_R);
    cudaFree(d_P);
    cudaFree(d_newV);
}

int main() {
    float h_V[STATES] = {0, 0, 0, 0};
    float h_R[STATES * ACTIONS] = {
        5, 10, 2, 3, 0, -1, 7, 8
    };
    float h_P[STATES * ACTIONS * STATES] = { 
        0.8, 0.2, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0,  
        0.0, 0.6, 0.4, 0.0, 0.0, 0.5, 0.5, 0.0,  
        0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.8, 0.2,  
        0.3, 0.0, 0.0, 0.7, 0.4, 0.0, 0.0, 0.6
    };

    value_iteration(h_V, h_R, h_P);

    std::cout << "Optimal Value Function:\n";
    for (int i = 0; i < STATES; i++) {
        std::cout << "V[" << i << "] = " << h_V[i] << std::endl;
    }

    return 0;
}
