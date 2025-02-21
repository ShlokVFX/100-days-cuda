#include <stdio.h>
#include <math.h>

__global__ void layernorm(float *input, float *output, int n) {
    __shared__ float mean;
    __shared__ float var;
    
    int tid = threadIdx.x;

    if (tid == 0) {
        mean = 0;
        var = 0;
    }
    __syncthreads();

    // Compute mean
    atomicAdd(&mean, input[tid] / n);
    __syncthreads();

    // Compute variance
    float diff = input[tid] - mean;
    atomicAdd(&var, (diff * diff) / n);
    __syncthreads();

    // Normalize
    output[tid] = (input[tid] - mean) / sqrtf(var + 1e-5);

    // Print Debugging Info
    printf("Thread %d: Input = %.2f, Mean = %.2f, Variance = %.5f, Normalized = %.2f\n",
           tid, input[tid], mean, var, output[tid]);
}

int main() {
    float h_input[4] = {1.0, 2.0, 3.0, 4.0};
    float h_output[4] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input, 4 * sizeof(float));
    cudaMalloc(&d_output, 4 * sizeof(float));

    cudaMemcpy(d_input, h_input, 4 * sizeof(float), cudaMemcpyHostToDevice);

    printf("Running LayerNorm Kernel:\n");
    layernorm<<<1, 4>>>(d_input, d_output, 4);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
