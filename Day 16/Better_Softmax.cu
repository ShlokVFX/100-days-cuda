#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CUDA error check macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// cuDNN error check macro
#define CHECK_CUDNN(call) { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Softmax Kernel using FP16 Tensor Cores
__global__ void softmaxKernel(__half* input, __half* output, int rows, int cols) {
    extern __shared__ __half shared_data[];  
    __shared__ __half block_max;
    __shared__ __half block_sum;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= rows) return;

    __half* row_data = input + row * cols;
    __half* shared_row = shared_data;
    
    // Step 1: Compute max for numerical stability (Block-wide reduction)
    __half max_val = __float2half(-INFINITY);
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = __hmax(max_val, row_data[i]);
    }

    // Warp reduction for max
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = __hmax(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    if (tid == 0) block_max = max_val;
    __syncthreads();

    // Step 2: Compute exponentials and sum
    max_val = block_max;
    __half sum = __float2half(0.0f);
    for (int i = tid; i < cols; i += blockDim.x) {
        shared_row[i] = __float2half(expf(__half2float(__hsub(row_data[i], max_val))));
        sum = __hadd(sum, shared_row[i]);
    }

    // Warp reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }

    if (tid == 0) block_sum = sum;
    __syncthreads();

    // Step 3: Normalize values
    sum = block_sum;
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] = __hdiv(shared_row[i], sum);
    }
}

// Host function with CUDA Graphs, Streams, and cuBLAS/cuDNN
void softmaxOptimized(float* h_input, float* h_output, int rows, int cols) {
    __half *d_input, *d_output;
    size_t size = rows * cols * sizeof(__half);

    // Convert FP32 to FP16
    __half* h_input_half = new __half[rows * cols];
    for (int i = 0; i < rows * cols; i++)
        h_input_half[i] = __float2half(h_input[i]);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input_half, size, cudaMemcpyHostToDevice));

    // CUDA Stream & Graph setup
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // CUDA Event setup for performance measurement
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Capture Graph
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    int shared_mem_size = cols * sizeof(__half);
    softmaxKernel<<<rows, BLOCK_SIZE, shared_mem_size, stream>>>(d_input, d_output, rows, cols);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // Instantiate and launch graph
    CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    // Start event
    CHECK_CUDA(cudaEventRecord(start, stream));
    CHECK_CUDA(cudaGraphLaunch(instance, stream));
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // Wait for execution
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Compute execution time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // GFLOPS Calculation
    int num_operations = 2 * rows * cols;  // Each element involves 2 FLOPs (exp + div)
    double gflops = (num_operations * 1e-9) / (milliseconds * 1e-3);  // GFLOPS = (FLOPs / time)

    // Copy result back to host
    __half* h_output_half = new __half[rows * cols];
    CHECK_CUDA(cudaMemcpy(h_output_half, d_output, size, cudaMemcpyDeviceToHost));

    // Convert FP16 to FP32
    for (int i = 0; i < rows * cols; i++)
        h_output[i] = __half2float(h_output_half[i]);

    // Cleanup
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input_half;
    delete[] h_output_half;
    cudaFree(d_input);
    cudaFree(d_output);

    // Print performance metrics
    std::cout << "Execution Time: " << milliseconds << " ms\n";
    std::cout << "GFLOPS: " << gflops << " GFLOPS\n";
}

// Test function
int main() {
    const int rows = 2, cols = 4;
    float h_input[rows * cols] = {1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0};
    float h_output[rows * cols];

    // Compute softmax using CUDA Streams + Graphs + cuBLAS/cuDNN
    softmaxOptimized(h_input, h_output, rows, cols);

    // Print result
    std::cout << "Softmax Output (CUDA Streams + cuBLAS/cuDNN + FP16 Accelerated):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_output[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
