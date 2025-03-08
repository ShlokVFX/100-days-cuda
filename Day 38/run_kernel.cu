#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(call) \
    if((call) != CUDA_SUCCESS) { \
        std::cerr << "CUDA Error at " << __LINE__ << std::endl; \
        return EXIT_FAILURE; \
    }

int main() {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;

    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuDeviceGet(&cuDevice, 0));
    CHECK_CUDA(cuCtxCreate(&cuContext, 0, cuDevice));
    CHECK_CUDA(cuModuleLoad(&cuModule, "cuda_kernel.ptx"));
    CHECK_CUDA(cuModuleGetFunction(&cuFunction, cuModule, "cuda_kernel"));

    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    cudaMemset(d_data, 1.0, 1024 * sizeof(float));  // Initialize all to 1.0

    void *args[] = { &d_data };
    CHECK_CUDA(cuLaunchKernel(cuFunction, 1, 1, 1, 1024, 1, 1, 0, 0, args, 0));

    // Copy back results
    float h_data[1024];
    cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << h_data[0] << std::endl;  // Should print 2.0

    cudaFree(d_data);
    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);
    return 0;
}
