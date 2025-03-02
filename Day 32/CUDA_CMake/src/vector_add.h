#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <cuda_runtime.h>

// Declare your solution function
extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n);

#endif // VECTOR_ADD_H
