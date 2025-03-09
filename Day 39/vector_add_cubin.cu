#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(call) \
    if((call) != CUDA_SUCCESS) { \
        std::cerr << "CUDA Driver API error at " << __LINE__ << std::endl; \
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

    // Load the compiled CUBIN file
    CHECK_CUDA(cuModuleLoad(&cuModule, "vector_add.cubin"));
    CHECK_CUDA(cuModuleGetFunction(&cuFunction, cuModule, "vectorAdd"));

    // Allocate & initialize memory
    const int N = 1024;
    size_t size = N * sizeof(float);
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size));
    CHECK_CUDA(cuMemAlloc(&d_B, size));
    CHECK_CUDA(cuMemAlloc(&d_C, size));

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size));

    // Launch kernel
    void* args[] = { &d_A, &d_B, &d_C, (void*)&N };
    CHECK_CUDA(cuLaunchKernel(cuFunction,
                              (N + 255) / 256, 1, 1,  // Grid size
                              256, 1, 1,              // Block size
                              0, 0, args, 0));

    // Copy back result
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size));

    std::cout << "C[0] = " << h_C[0] << std::endl; // Expect 3.0

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    cuMemFree(d_A); cuMemFree(d_B); cuMemFree(d_C);
    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);

    return 0;
}
