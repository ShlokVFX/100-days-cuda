# 1D vs 2D Convolution in CUDA

## Introduction
This repository contains examples and explanations of 1D and 2D convolutions implemented using CUDA. Convolution is a fundamental operation in many signal processing and image processing applications.

## 1D Convolution
1D convolution is used primarily in signal processing. It involves sliding a kernel over a 1D input signal to produce a 1D output signal.

### Example
```cuda
__global__ void conv1D(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < inputSize - kernelSize + 1) {
        float sum = 0.0;
        for (int i = 0; i < kernelSize; i++) {
            sum += input[tid + i] * kernel[i];
        }
        output[tid] = sum;
    }
}
```

## 2D Convolution
2D convolution is used primarily in image processing. It involves sliding a 2D kernel over a 2D input image to produce a 2D output image.

### Example
```cuda
__global__ void conv2D(float *input, float *kernel, float *output, int width, int height, int kernelWidth, int kernelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfKernelWidth = kernelWidth / 2;
    int halfKernelHeight = kernelHeight / 2;

    if (x >= halfKernelWidth && x < width - halfKernelWidth && y >= halfKernelHeight && y < height - halfKernelHeight) {
        float sum = 0.0;
        for (int i = -halfKernelWidth; i <= halfKernelWidth; i++) {
            for (int j = -halfKernelHeight; j <= halfKernelHeight; j++) {
                sum += input[(y + j) * width + (x + i)] * kernel[(j + halfKernelHeight) * kernelWidth + (i + halfKernelWidth)];
            }
        }
        output[y * width + x] = sum;
    }
}
```

## Building and Running
To build and run the examples, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/100-days-cuda.git
    cd 100-days-cuda/Day 20
    ```

2. Compile the CUDA code:
    ```sh
    nvcc -o conv1D conv1D.cu
    nvcc -o conv2D conv2D.cu
    ```

3. Run the executables:
    ```sh
    ./conv1D
    ./conv2D
    ```

## Conclusion
This repository demonstrates the implementation of 1D and 2D convolutions using CUDA. These examples can be used as a starting point for more complex convolution operations in signal and image processing applications.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.