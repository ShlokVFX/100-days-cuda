# MLIR to CUDA Integration - Matrix Addition

This tutorial demonstrates how to successfully integrate MLIR-generated LLVM IR with the CUDA runtime using a simple matrix addition kernel.

## Overview
We'll create a matrix addition kernel using MLIR, convert it to LLVM IR, and integrate it with CUDA's runtime API for execution. The process involves several steps: converting MLIR to LLVM IR, generating object files, and linking them with CUDA-generated object files.

---

## Step 1: Create `cuda_add.mlir`

```mlir
module {
  func.func @matrix_add(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index

    scf.for %i = %c0 to %c1024 step %c1 {
      %a = memref.load %arg0[%i] : memref<1024xf32>
      %b = memref.load %arg1[%i] : memref<1024xf32>
      %sum = arith.addf %a, %b : f32
      memref.store %sum, %arg2[%i] : memref<1024xf32>
    }
    return
  }
}

```

---

## Step 2: Generate LLVM IR
```sh
mlir-opt cuda_add.mlir \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --convert-to-llvm | mlir-translate --mlir-to-llvmir > cuda_add.ll
```

---

## Step 3: Compile LLVM IR to Object File
```sh
llc --filetype=obj -o cuda_add.o cuda_add.ll
```

---

## Step 4: Create `main.cu`
```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>

// Define the memref descriptor structure.
struct MemRef1D {
  void *allocated;    // pointer to allocated memory
  void *aligned;      // pointer to aligned memory
  int64_t offset;     // usually 0
  int64_t sizes[1];   // size of the memref
  int64_t strides[1]; // stride (usually 1 for contiguous memory)
};

// Declaration of the MLIR-generated function (as seen in your LLVM IR).
extern "C" void matrix_add(
    void*, void*, int64_t, int64_t, int64_t,
    void*, void*, int64_t, int64_t, int64_t,
    void*, void*, int64_t, int64_t, int64_t);

// Wrapper function that takes our memref descriptors.
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
    // Allocate raw arrays.
    float *A_raw = new float[SIZE];
    float *B_raw = new float[SIZE];
    float *C_raw = new float[SIZE];

    // Initialize A and B.
    for (int i = 0; i < SIZE; ++i) {
        A_raw[i] = static_cast<float>(i);
        B_raw[i] = static_cast<float>(i * 2);
    }

    // Build memref descriptors.
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

    // Call the MLIR-generated kernel via our wrapper.
    matrix_add_wrapper(&A_desc, &B_desc, &C_desc);

    // Print some results.
    std::cout << "C[0] = " << C_raw[0] << ", C[1023] = " << C_raw[1023] << std::endl;

    // Clean up.
    delete[] A_raw;
    delete[] B_raw;
    delete[] C_raw;
    return 0;
}

```

---

## Step 5: Compile CUDA File
```sh
nvcc -c main.cu -o main.o
```

---

## Step 6: Link Object Files and Create Executable
```sh
nvcc main.o cuda_add.o -o main -lcudart
```

---

## Step 7: Run the Program
```sh
./main
```

---

## Expected Output
```
C[0] = 0, C[1023] = 3069
```

---

## Explanation
1. **MLIR Generation:** Basic matrix addition kernel written in MLIR.
2. **LLVM IR Conversion:** Using `mlir-opt` and `mlir-translate` to convert MLIR to LLVM IR.
3. **Object File Generation:** Converting LLVM IR to object file using `llc`.
4. **CUDA Compilation & Linking:** Compiling and linking the object file with the CUDA runtime API.

---

This integration approach successfully runs MLIR-generated kernels with CUDA's runtime API. 🎉

