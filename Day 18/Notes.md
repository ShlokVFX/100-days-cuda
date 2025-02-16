# Notes for cuBLAS_matmul_v2.cu

## Line-by-Line Analysis

### Includes
```c
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <assert.h>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
```
These headers include standard C libraries, CUDA runtime, cuBLAS for matrix operations, and cuRAND for random number generation.

---

### Macros & Function Definitions
```c
#define EPSILON 1.0e-2
```
Defines a small tolerance value for verifying matrix multiplication accuracy.

```c
void verify_result(float *a, float *b, float *c, int n) {
```
Function to verify if the computed matrix `c` matches expected results using a naive matrix multiplication approach.

- Uses column-major order.
- Iterates over rows and columns, computing expected results.
- Uses `assert` to compare expected vs. actual values.

---

### Main Function
```c
int main() {
```
Entry point of the program.

#### Variable Declarations
```c
int n = 4; // Keeping small for readable output
size_t bytes = n * n * sizeof(float);
```
Defines matrix size `n x n` and calculates the memory required in bytes.

```c
float *h_a, *h_b, *h_c;
float *d_a, *d_b, *d_c;
```
Declares host and device pointers for matrices A, B, and C.

#### Memory Allocation
```c
h_a = (float *)malloc(bytes);
h_b = (float *)malloc(bytes);
h_c = (float *)malloc(bytes);
cudaMalloc(&d_a, bytes);
cudaMalloc(&d_b, bytes);
cudaMalloc(&d_c, bytes);
```
Allocates memory on both host (CPU) and device (GPU).

#### Random Number Generation
```c
curandGenerator_t prng;
curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
curandGenerateUniform(prng, d_a, n * n);
curandGenerateUniform(prng, d_b, n * n);
```
- Creates a pseudo-random number generator.
- Seeds it using the system clock.
- Fills matrices A and B with random values.

#### Copying Matrices to Host for Display
```c
cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
```
Copies device matrices to host memory for printing.

#### Printing Input Matrices
```c
printf("\nMatrix A:\n");
```
Loops through matrix elements and prints them.

#### cuBLAS Handle Creation
```c
cublasHandle_t handle;
cublasCreate(&handle);
```
Initializes cuBLAS handle for GPU computations.

#### Defining Scaling Factors
```c
float alpha = 1.0f;
float beta = 0.0f;
```
Defines scalar values used in matrix multiplication: `C = alpha * A * B + beta * C`.

#### CUDA Events for Timing
```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
```
Creates CUDA events to measure execution time.

#### Recording Start Time
```c
cudaEventRecord(start);
```
Marks the start of computation.

#### Performing Matrix Multiplication
```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
```
Calls cuBLAS function to perform single-precision general matrix multiplication (SGEMM).

#### Recording Stop Time & Synchronizing
```c
cudaEventRecord(stop);
cudaEventSynchronize(stop);
```
Stops timing and ensures execution has completed before measuring time.

#### Measuring Execution Time
```c
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
double seconds = milliseconds / 1000.0;
double gflops = (2.0 * pow(n, 3)) / (seconds * 1e9);
```
Computes execution time and calculates GFLOPS performance.

#### Copying Result to Host & Printing
```c
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
```
Copies the result matrix `C` from GPU to CPU memory and prints it.

#### Printing Performance Metrics
```c
printf("\n🚀 Performance Metrics:\n");
printf(" - Execution Time: %.6f ms\n", milliseconds);
printf(" - GFLOPS: %.6f\n", gflops);
```
Displays execution time and GFLOPS.

#### Verification & Cleanup
```c
verify_result(h_a, h_b, h_c, n);
```
Validates results using the naive CPU multiplication.

```c
cublasDestroy(handle);
free(h_a);
free(h_b);
free(h_c);
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
cudaEventDestroy(start);
cudaEventDestroy(stop);
```
Frees allocated memory and destroys CUDA objects.

#### Final Output
```c
printf("\nCOMPLETED SUCCESSFULLY\n");
return 0;
```
Indicates successful execution.

