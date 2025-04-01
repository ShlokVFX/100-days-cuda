#include <iostream>
#include <cuda_runtime.h>

#define INPUT_SIZE 4
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 2
#define LEARNING_RATE 0.01

__global__ void matMulVec(float* W, float* x, float* y, int rows, int cols) {
    int i = threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

__global__ void reluActivation(float* x, int size) {
    int i = threadIdx.x;
    if (i < size) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

__global__ void reluDerivative(float* x, float* grad, int size) {
    int i = threadIdx.x;
    if (i < size) {
        grad[i] *= (x[i] > 0);
    }
}

__global__ void mseLossGradient(float* predicted, float* target, float* grad, int size) {
    int i = threadIdx.x;
    if (i < size) {
        grad[i] = 2 * (predicted[i] - target[i]) / size;
    }
}

__global__ void updateWeights(float* W, float* grad, float* input, int rows, int cols, float lr) {
    int i = threadIdx.x;
    if (i < rows) {
        for (int j = 0; j < cols; j++) {
            W[i * cols + j] -= lr * grad[i] * input[j];
        }
    }
}

void forwardPass(float* d_W1, float* d_W2, float* d_input, float* d_hidden, float* d_output) {
    matMulVec<<<1, HIDDEN_SIZE>>>(d_W1, d_input, d_hidden, HIDDEN_SIZE, INPUT_SIZE);
    reluActivation<<<1, HIDDEN_SIZE>>>(d_hidden, HIDDEN_SIZE);
    matMulVec<<<1, OUTPUT_SIZE>>>(d_W2, d_hidden, d_output, OUTPUT_SIZE, HIDDEN_SIZE);
}

void backwardPass(float* d_W1, float* d_W2, float* d_input, float* d_hidden, float* d_output,
                  float* d_target, float* d_grad_output, float* d_grad_hidden) {
    mseLossGradient<<<1, OUTPUT_SIZE>>>(d_output, d_target, d_grad_output, OUTPUT_SIZE);
    updateWeights<<<1, OUTPUT_SIZE>>>(d_W2, d_grad_output, d_hidden, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    matMulVec<<<1, HIDDEN_SIZE>>>(d_W2, d_grad_output, d_grad_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    reluDerivative<<<1, HIDDEN_SIZE>>>(d_hidden, d_grad_hidden, HIDDEN_SIZE);
    updateWeights<<<1, HIDDEN_SIZE>>>(d_W1, d_grad_hidden, d_input, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
}

int main() {
    float h_input[INPUT_SIZE] = {1.0, 0.5, -0.2, 0.1};
    float h_target[OUTPUT_SIZE] = {0.0, 1.0};
    float h_W1[INPUT_SIZE * HIDDEN_SIZE];
    float h_W2[HIDDEN_SIZE * OUTPUT_SIZE];
    float h_output[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) h_W1[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) h_W2[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_input, *d_W1, *d_W2, *d_hidden, *d_output, *d_target, *d_grad_output, *d_grad_hidden;
    cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_target, OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_grad_output, OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_grad_hidden, HIDDEN_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 1000; i++) {
        forwardPass(d_W1, d_W2, d_input, d_hidden, d_output);
        backwardPass(d_W1, d_W2, d_input, d_hidden, d_output, d_target, d_grad_output, d_grad_hidden);
    }

    cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Q-values: ";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_grad_output);
    cudaFree(d_grad_hidden);

    return 0;
}