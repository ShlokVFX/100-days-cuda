#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <device_launch_parameters.h>

__global__ void vectorAddUM(int* a, int* b, int* c, int n) {
	//global thread index (tid)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < n) {
		c[tid] = a[tid] + b[tid];
	}
}
//initialize vector of size n to int b/w 0-99

void init_vector(int* a, int* b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
		a[i] = rand() % 100;

	}
}
void check_answer(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {

	//get ID from cuda Calls 
	int id = cudaGetDevice(&id);

	//elements per array
	int n = 1 << 16;

	//declare unified memory pointers

	int* a, * b, * c;

	//Alocation size for all vectors
	size_t bytes = sizeof(int) * n;

	//Allocate host memory

	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	//initialize vectors 
	init_vector(a, b, n);

	//threadblock size

	int BLOCK_SIZE = 256;

	//Grid SIze

	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);


	//Launch Kernel on default stream w/o shmem
	vectorAddUM << <GRID_SIZE, BLOCK_SIZE >> > (a, b, c, n);

	//waitforall operations

	cudaDeviceSynchronize();

	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	check_answer(a, b, c, n);

	printf("completed successfully , result of sum is %d\n", c);

	return 0;
}
