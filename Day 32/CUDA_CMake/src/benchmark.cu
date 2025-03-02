#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "vector_add.h"

inline void gpuCheck(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "GPU Error: " << cudaGetErrorString(code)
                  << " " << file << ":" << line << std::endl;
        exit(code);
    }
}
#define gpuErrchk(ans) { gpuCheck((ans), __FILE__, __LINE__); }

int main() {
    std::vector<size_t> testSizes = {1'000'000, 2'000'000, 10'000'000, 20'000'000, 50'000'000, 100'000'000, 500'000'000};

    std::cout << "  n (elements)  |  Runtime (ms)  |  GFLOPS\n"
              << "-----------------------------------------\n";

    for (auto n : testSizes) {
        std::vector<float> h_input1(n, 1.0f), h_input2(n, 2.0f), h_output(n, 0.0f);
        float *d_input1, *d_input2, *d_output;
        size_t bytes = n * sizeof(float);

        gpuErrchk(cudaMalloc(&d_input1, bytes));
        gpuErrchk(cudaMalloc(&d_input2, bytes));
        gpuErrchk(cudaMalloc(&d_output, bytes));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        gpuErrchk(cudaMemcpyAsync(d_input1, h_input1.data(), bytes, cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_input2, h_input2.data(), bytes, cudaMemcpyHostToDevice, stream));

        solution(d_input1, d_input2, d_output, n);

        gpuErrchk(cudaMemcpyAsync(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        float milliseconds = 0.0f;
        cudaEvent_t start, stop;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));

        gpuErrchk(cudaEventRecord(start));
        solution(d_input1, d_input2, d_output, n);
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

        double gflops = (static_cast<double>(n) / (milliseconds / 1000.0)) / 1e9;
        std::cout << "  " << n << "           |  " << milliseconds << "          |  " << gflops << std::endl;

        gpuErrchk(cudaFree(d_input1));
        gpuErrchk(cudaFree(d_input2));
        gpuErrchk(cudaFree(d_output));
    }

    return 0;
}
