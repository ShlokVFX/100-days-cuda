#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <device_launch_parameters.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n){
	//global thread index (tid)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
if (tid < n) {
	c[tid] = a[tid] + b[tid];
	}	
}
//initialize vector of size n to int b/w 0-99

void matrix_init(int* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
	}
}

void error_check(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {
//vector of size of 2^16 (65536 elements)
	int n = 1 << 16;
	//Host vector pointers
		int* h_a, * h_b, * h_c;
		//device vector pointer
		int* d_a, * d_b, * d_c;
	//Alocation size for all vectors
	size_t bytes = sizeof(int) * n;

	//Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	//Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	//initialize vectors a and b with random values b/w 0 -99
	matrix_init(h_a, n);
	matrix_init(h_b, n);

	//copy data from 

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	//threadblock size

	int NUM_THREADS = 256;

		//Grid SIze

	int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

	//Launch Kernel on default stream w/o shmem
	vectorAdd <<<NUM_BLOCKS, NUM_THREADS>>> (d_a, d_b, d_c, n);

	//copy sum vector from device to host 
	
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	//check result for error 
	error_check(h_a, h_b, h_c, n);

	printf("Result is %d\n", d_c);

	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_c);
		
	return 0;
}
