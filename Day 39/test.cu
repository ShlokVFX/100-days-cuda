#include <cuda.h>
#include <iostream>

int main() {
    CUresult res = cuInit(0);
    if (res == CUDA_SUCCESS)
        std::cout << "CUDA Driver API initialized successfully!" << std::endl;
    else
        std::cout << "CUDA Driver API failed: " << res << std::endl;

    return 0;
}
