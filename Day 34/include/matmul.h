#ifndef MATMUL_H
#define MATMUL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void solution(float* input_a, float* input_b, float* output_c,
              size_t m, size_t n, size_t k);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_H
