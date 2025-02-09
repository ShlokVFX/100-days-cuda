#include <stdio.h>
#include <cuda_runtime.h>

__global__ void atomicAddKernel(int* sum) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	atomicAdd(sum, tid); //safe addition to memory
 }

int main() {
	int h_sum = 0;  //host value
	int *d_sum;    // device

	//Alocate mem on gpu

	cudaMalloc((void**)&d_sum, sizeof(int));
	cudaMemcpy(d_sum, &h_sum, sizeof(int), cudaMemcpyHostToDevice);

	atomicAddKernel<<<1, 256>>>(d_sum); //launch kernel with 256 thread and 1 block

	cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

	printf("The value after atomicAdd is %d\n", h_sum);

	cudaFree(d_sum); //free memory

	return 0;
}
