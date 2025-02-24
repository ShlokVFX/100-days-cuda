#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>

// Define complex number type
typedef std::complex<float> Complex;
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // Macro for column-major indexing

// CUDA Kernel to print device array (Debugging)
__global__ void printDeviceArray(cufftComplex *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        printf("Device Data[%d]: (%f, %f)\n", i, data[i].x, data[i].y);
    }
}

// Host function to print host array
void printHostArray(const char* name, cufftComplex *data, int N) {
    printf("\n%s:\n", name);
    for (int i = 0; i < N; i++) {
        printf("[%d]: (%f, %f)\n", i, data[i].x, data[i].y);
    }
}

// Main function
int main() {
    int N = 8; // Must be power of 2 for Cooley-Tukey FFT
    cufftComplex *h_input, *h_output;
    cufftComplex *d_data;

    // Allocate host memory
    h_input = (cufftComplex*)malloc(sizeof(cufftComplex) * N);
    h_output = (cufftComplex*)malloc(sizeof(cufftComplex) * N);

    // Initialize input data (Real and Imaginary parts)
    for (int i = 0; i < N; i++) {
        h_input[i].x = i + 1;  // Real part
        h_input[i].y = 0.0f;   // Imaginary part (set to 0)
    }
    
    // Print input data
    printHostArray("Input Data", h_input, N);

    // Allocate device memory
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * N);
    cudaMemcpy(d_data, h_input, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    // Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    // Execute FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_output, d_data, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);

    // Print output data
    printHostArray("Output Data (FFT Result)", h_output, N);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_input);
    free(h_output);

    return 0;
}
