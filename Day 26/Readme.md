Below is a sample **README.md** file that explains the FFT project in a way that’s friendly for a teenager, and then walks you through the code line by line.

---

```markdown
# CUDA FFT Project: Fast Fourier Transform Explained for Teens

## Introduction
Hey there! Imagine listening to your favorite song and being able to see all the hidden musical notes that make up the tune. That’s what the **Fast Fourier Transform (FFT)** does—it takes a signal (like sound) and shows you its individual frequencies (the notes). In this project, we use CUDA (a way to make your computer's graphics card do serious math) to run an FFT quickly on the GPU.

## What Is FFT?
FFT stands for **Fast Fourier Transform**. Think of it like this:
- **Time Domain:** This is the raw song—how the sound changes over time.
- **Frequency Domain:** This is the breakdown of the song into individual notes (frequencies).
- The FFT takes the song and shows you, “Hey, here are the bass beats, the midrange, and the high treble!”

## How This Code Works
Our code uses a library called **cuFFT**, which is specially made for doing FFTs on NVIDIA GPUs. We also add extra print statements to help you see:
- The input values (the song before it’s broken down)
- What’s happening on the GPU before and after the FFT
- The final output (the hidden notes/frequencies)

## Line-by-Line Explanation

### 1. Including Libraries
```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>
```
- **`#include <stdio.h>`**  
  Lets us use functions like `printf` to print messages on the screen.
- **`#include <cuda_runtime.h>`**  
  Includes CUDA functions for managing GPU memory and launching GPU functions.
- **`#include <cufft.h>`**  
  Brings in the cuFFT library, which does the heavy lifting for our FFT.
- **`#include <complex>`**  
  Allows us to work with complex numbers (numbers with real and imaginary parts).

### 2. Defining a Complex Type and a Helper Macro
```cpp
typedef std::complex<float> Complex;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
```
- **`typedef std::complex<float> Complex;`**  
  This makes a new type called `Complex` for handling numbers that have a real part and an imaginary part.
- **`#define IDX2C(i,j,ld) (((j)*(ld))+(i))`**  
  A helper macro to convert 2D indices to a single index in an array (this one isn’t used in our main code, but it’s useful for matrix operations).

### 3. CUDA Kernel for Printing Data from the GPU
```cpp
__global__ void printDeviceArray(cufftComplex *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        printf("Device Data[%d]: (%f, %f)\n", i, data[i].x, data[i].y);
    }
}
```
- **`__global__`**  
  Declares a function (a kernel) that runs on the GPU.
- **`printDeviceArray`**  
  This kernel prints each element of an array stored on the GPU. Each thread prints one element.
- **`int i = blockIdx.x * blockDim.x + threadIdx.x;`**  
  Calculates a unique index for each thread.
- **`if (i < N)`**  
  Checks that we only try to print valid elements.
- **`printf(...)`**  
  Prints the real (`x`) and imaginary (`y`) parts of each complex number.

### 4. Host Function to Print an Array
```cpp
void printHostArray(const char* name, cufftComplex *data, int N) {
    printf("\n%s:\n", name);
    for (int i = 0; i < N; i++) {
        printf("[%d]: (%f, %f)\n", i, data[i].x, data[i].y);
    }
}
```
- **`printHostArray`**  
  This function prints an array that’s in the computer’s main memory.
- It prints a title (like "Input Data") and then each element with its real and imaginary parts.

### 5. The Main Function
```cpp
int main() {
    int N = 8; // Must be a power of 2 for the FFT to work properly
    cufftComplex *h_input, *h_output;
    cufftComplex *d_data;
```
- **`int N = 8;`**  
  Sets the number of elements in our FFT. We choose 8 because FFT algorithms work best with powers of 2.
- **`cufftComplex *h_input, *h_output;`**  
  Declares pointers for the input and output arrays in main (host) memory.
- **`cufftComplex *d_data;`**  
  Declares a pointer for the array that will live on the GPU (device memory).

#### 5.1. Allocating Memory on the Host
```cpp
    h_input = (cufftComplex*)malloc(sizeof(cufftComplex) * N);
    h_output = (cufftComplex*)malloc(sizeof(cufftComplex) * N);
```
- **`malloc`**  
  Allocates enough space for our arrays to hold `N` complex numbers.

#### 5.2. Initializing the Input Data
```cpp
    for (int i = 0; i < N; i++) {
        h_input[i].x = i + 1;  // Real part (1, 2, 3, ..., 8)
        h_input[i].y = 0.0f;   // Imaginary part is set to 0
    }
```
- This loop fills the input array with numbers 1 to 8. The imaginary part is 0 because we’re starting with real numbers.

#### 5.3. Printing the Input Data on the Host
```cpp
    printHostArray("Input Data", h_input, N);
```
- Calls our helper function to show the input data on the screen.

#### 5.4. Allocating Memory on the Device and Copying Data
```cpp
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * N);
    cudaMemcpy(d_data, h_input, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);
```
- **`cudaMalloc`**  
  Reserves space on the GPU for our data.
- **`cudaMemcpy`**  
  Copies the initialized data from the host (CPU) to the device (GPU).

#### 5.5. Debug: Print Device Data Before Running the FFT
```cpp
    printf("\n--- Checking Device Data Before FFT ---\n");
    printDeviceArray<<<1, N>>>(d_data, N);
    cudaDeviceSynchronize();
```
- Launches the `printDeviceArray` kernel with one block and `N` threads so that every element gets printed.
- **`cudaDeviceSynchronize()`**  
  Waits for the GPU to finish printing before moving on.

#### 5.6. Creating and Executing the FFT Plan
```cpp
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
```
- **`cufftHandle plan;`**  
  Declares a variable to hold our FFT “plan” (settings for the FFT).
- **`cufftPlan1d(&plan, N, CUFFT_C2C, 1);`**  
  Sets up a 1D FFT for `N` elements. `CUFFT_C2C` means we’re doing a complex-to-complex transform.

```cpp
    printf("\n--- Running FFT on GPU ---\n");
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
```
- **`cufftExecC2C`**  
  Executes the FFT. The transformation happens in-place, meaning the original data is replaced with the FFT result.
- **`CUFFT_FORWARD`** indicates a forward FFT (from time domain to frequency domain).

#### 5.7. Debug: Print Device Data After the FFT
```cpp
    printf("\n--- Checking Device Data After FFT ---\n");
    printDeviceArray<<<1, N>>>(d_data, N);
    cudaDeviceSynchronize();
```
- Prints the results stored on the GPU after the FFT has been performed.

#### 5.8. Copying the Output Data Back to the Host
```cpp
    cudaMemcpy(h_output, d_data, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);
```
- Copies the FFT result from the GPU back into host memory so you can view it.

#### 5.9. Printing the FFT Output Data on the Host
```cpp
    printHostArray("Output Data (FFT Result)", h_output, N);
```
- Displays the final FFT results on the screen.

#### 5.10. Cleanup: Freeing Memory and Resources
```cpp
    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_input);
    free(h_output);
    return 0;
}
```
- **`cufftDestroy(plan);`**  
  Cleans up the FFT plan.
- **`cudaFree(d_data);`**  
  Releases the GPU memory.
- **`free(h_input); free(h_output);`**  
  Releases the host memory.
- The program ends with `return 0;`, indicating it ran successfully.

## Conclusion
This project shows how to use CUDA and the cuFFT library to perform an FFT on a set of numbers. We use extra print statements so you can clearly see what the data looks like before the FFT, on the GPU during processing, and after the FFT is complete. It’s like taking a song, breaking it into notes, and then showing you each note step by step.

Fast Fourier Transform (FFT) Using Shared Memory + Profiling:

This task involves implementing a batched 1D FFT using CUDA and optimizing it using shared memory, coalesced memory accesses, and loop unrolling. You will then profile the performance using Nsight Systems and Nsight Compute.

    FFT is used in quantitative finance, image processing, and deep learning.
    It requires efficient global memory access patterns.
    It has high computational intensity, making it a great profiling candidate.
    It challenges you to understand warp divergence and register pressure.


Helpful Profiling commands:

//deprecated: ncu --metrics achieved_occupancy,smsp__warp_issue_stalled_long_scoreboard,sm__warps_launched.avg.per_cycle_active ./naive_kernel

ncu --metrics sm__average_warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__warps_launched.avg.per_cycle_active ./naive_kernel
ncu --metrics sm__average_warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__warps_launched.avg.per_cycle_active ./KernelOccupancy

nvcc -maxrregcount=32 Kernel_occupancy.cu -o KernelOccupancy
nvcc -maxrregcount=32 Optimized.cu -o KO
ncu --section MemoryWorkloadAnalysis ./optimized_kernel

---------------------------------------------------------

compute-sanitizer ./faultyKernel
compute-sanitizer --tool memcheck ./faultyKernel
compute-sanitizer --tool memcheck ./fixedKernel

nsys launch --trace=cuda ./faultyKernel

---------------------------------------------------------

ncu --set full ./Naive
ncu --set full ./optimized

ncu --export profile_report.ncu-rep ./FFT_cuda