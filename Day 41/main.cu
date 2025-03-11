#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>

struct MemRef1D {
  void *allocated; 
  void *aligned;     
  int64_t offset;    
  int64_t sizes[1];  
  int64_t strides[1]; 
};

extern "C" void matrix_add(
    void*, void*, int64_t, int64_t, int64_t,
    void*, void*, int64_t, int64_t, int64_t,
    void*, void*, int64_t, int64_t, int64_t);

void matrix_add_wrapper(MemRef1D *A, MemRef1D *B, MemRef1D *C) {
    matrix_add(
        A->allocated,
        A->aligned,
        A->offset,
        A->sizes[0],
        A->strides[0],
        B->allocated,
        B->aligned,
        B->offset,
        B->sizes[0],
        B->strides[0],
        C->allocated,
        C->aligned,
        C->offset,
        C->sizes[0],
        C->strides[0]
    );
}

#define SIZE 1024

int main() {
    float *A_raw = new float[SIZE];
    float *B_raw = new float[SIZE];
    float *C_raw = new float[SIZE];

    for (int i = 0; i < SIZE; ++i) {
        A_raw[i] = static_cast<float>(i);
        B_raw[i] = static_cast<float>(i * 2);
    }

    MemRef1D A_desc, B_desc, C_desc;
    A_desc.allocated = reinterpret_cast<void*>(A_raw);
    A_desc.aligned   = reinterpret_cast<void*>(A_raw);
    A_desc.offset    = 0;
    A_desc.sizes[0]  = SIZE;
    A_desc.strides[0]= 1;

    B_desc.allocated = reinterpret_cast<void*>(B_raw);
    B_desc.aligned   = reinterpret_cast<void*>(B_raw);
    B_desc.offset    = 0;
    B_desc.sizes[0]  = SIZE;
    B_desc.strides[0]= 1;

    C_desc.allocated = reinterpret_cast<void*>(C_raw);
    C_desc.aligned   = reinterpret_cast<void*>(C_raw);
    C_desc.offset    = 0;
    C_desc.sizes[0]  = SIZE;
    C_desc.strides[0]= 1;

    matrix_add_wrapper(&A_desc, &B_desc, &C_desc);
    std::cout << "C[0] = " << C_raw[0] << ", C[1023] = " << C_raw[1023] << std::endl;

    delete[] A_raw;
    delete[] B_raw;
    delete[] C_raw;
    return 0;
}
