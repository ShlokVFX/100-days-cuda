#include <stdio.h>
#include <math.h>

__global__ void attention(float *Q, float *K, float *V, float *output, int n) {
    __shared__ float scores[4];
    __shared__ float softmax_scores[4];

    int tid = threadIdx.x;

    // Compute dot product (Q · K^T)
    scores[tid] = Q[tid] * K[tid];
    __syncthreads();

    // Print raw attention scores
    printf("Thread %d: Raw Score = %.2f\n", tid, scores[tid]);

    // Compute softmax denominator (sum of exponentials)
    float sum_exp = 0;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(scores[i]);
    }

    // Compute softmax score
    softmax_scores[tid] = expf(scores[tid]) / sum_exp;
    __syncthreads();

    // Print softmax scores
    printf("Thread %d: Softmax Score = %.2f\n", tid, softmax_scores[tid]);

    // Compute weighted sum (Softmax × V)
    output[tid] = softmax_scores[tid] * V[tid];

    // Print final output
    printf("Thread %d: Output = %.2f\n", tid, output[tid]);
}

int main() {
    float h_Q[4] = {1, 0, 1, 0};
    float h_K[4] = {1, 1, 0, 0};
    float h_V[4] = {5, 10, 15, 20};
    float h_output[4] = {0};

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, 4 * sizeof(float));
    cudaMalloc(&d_K, 4 * sizeof(float));
    cudaMalloc(&d_V, 4 * sizeof(float));
    cudaMalloc(&d_output, 4 * sizeof(float));

    cudaMemcpy(d_Q, h_Q, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, 4 * sizeof(float), cudaMemcpyHostToDevice);

    printf("Running Attention Kernel:\n");
    attention<<<1, 4>>>(d_Q, d_K, d_V, d_output, 4);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    return 0;
}
