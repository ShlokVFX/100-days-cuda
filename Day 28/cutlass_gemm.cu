#include <iostream>
#include <vector>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,  // A: float row-major
        float, cutlass::layout::RowMajor,  // B: float row-major
        float, cutlass::layout::RowMajor>; // C: float row-major

    Gemm gemm_operator;
    
    // Matrix dimensions (m × k) * (k × n) = (m × n)
    int m = 3, k = 3, n = 3;
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    // Allocate unified memory (accessible by both CPU & GPU)
    float *A, *B, *C;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C, m * n * sizeof(float));

    // Predefined Matrices
    float A_init[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B_init[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

    // Copy data to CUDA memory
    std::copy(A_init, A_init + (m * k), A);
    std::copy(B_init, B_init + (k * n), B);
    std::fill_n(C, m * n, 0.0f);  // Initialize C to zeros

    // Print input matrices
    printMatrix(A, m, k, "Matrix A");
    printMatrix(B, k, n, "Matrix B");

    // Setup GEMM arguments
    Gemm::Arguments args({problem_size}, {A, k}, {B, n}, {C, n}, {C, n});

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch GEMM computation
    gemm_operator(args);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();  // Ensure computation is complete

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output Result
    printMatrix(C, m, n, "Result Matrix C");

    double gflops = (2.0 * m * n * k) / (milliseconds * 1e6);
    std::cout << "Time Taken: " << milliseconds << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
