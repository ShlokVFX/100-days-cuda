// my_fp8_test.cpp

#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>  // For __half2float conversion

// We add this flag to prevent HIP from redefining CUDA's vector types.
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
#endif

// Number of elements
#define N 16

// A simple kernel that copies an array of FP8 values
__global__ void test_fp8_kernel(__hip_fp8_e4m3_fnuz* d_in, __hip_fp8_e4m3_fnuz* d_out) {
    int tx = threadIdx.x;
    if (tx < N) {
        d_out[tx] = d_in[tx];
    }
}

int main() {
    // Host arrays to hold FP8 data
    __hip_fp8_e4m3_fnuz h_in[N], h_out[N];

    // Set conversion parameters.
    // On NVIDIA targets, HIP maps its FP8 API to CUDA constructs.
    const __hip_fp8_interpretation_t interpret = __HIP_E4M3_FNUZ;
    const __hip_saturation_t sat = __HIP_SATFINITE;

    // Convert float values to FP8 (using HIP conversion function)
    // Each input value is converted as: fp8_value = __hip_cvt_float_to_fp8(float, saturation, interpretation)
    for (int i = 0; i < N; i++) {
        h_in[i] = __hip_cvt_float_to_fp8(static_cast<float>(i + 1), sat, interpret);
    }

    // Allocate device memory
    __hip_fp8_e4m3_fnuz *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(__hip_fp8_e4m3_fnuz));
    hipMalloc(&d_out, N * sizeof(__hip_fp8_e4m3_fnuz));

    // Copy data from host to device
    hipMemcpy(d_in, h_in, N * sizeof(__hip_fp8_e4m3_fnuz), hipMemcpyHostToDevice);

    // Launch the kernel (1 block, N threads)
    hipLaunchKernelGGL(test_fp8_kernel, dim3(1), dim3(N), 0, 0, d_in, d_out);
    hipDeviceSynchronize();

    // Copy the result from device back to host
    hipMemcpy(h_out, d_out, N * sizeof(__hip_fp8_e4m3_fnuz), hipMemcpyDeviceToHost);

    // Print results.
    // Convert the FP8 result back to float.
    std::cout << "Output values:" << std::endl;
    for (int i = 0; i < N; i++) {
        // __hip_cvt_fp8_to_halfraw converts FP8 to __half.
        __half half_val = __hip_cvt_fp8_to_halfraw(h_out[i], interpret);
        float float_val = __half2float(half_val);  // Convert __half to float.
        std::cout << float_val << " ";
    }
    std::cout << std::endl;

    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
