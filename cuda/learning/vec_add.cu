#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

__global__ void helloGPU() {
    printf("Hello from GPU thread! %d\n", threadIdx.x);
}

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    
    
    return 0;
}

void gpu_hello() {
    helloGPU<<<2, 4>>>();

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "Hello from CPU" << std::endl;
}

bool fvector_add(std::vector<float> &v1; std::vector<float> &v2, std::vector<float> &result){

    if (v1.size() != v2.size()) {
        return false;
    }
    /*Number of elements*/
    const int num_elements =  v1.size();

    int num_threads = 256; /* Make a multiple of 32 for warp*/
    int num_blocks = (num_elements + num_threads - 1) / num_threads;

    /*ALlocate host memory*/
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    float* h_c = (float*)malloc(N * sizeof(float));

    /*DEVICE POINTERS*/
    float* d_a;
    float* d_b;
    float* d_c;

    /* Allocate memory on device*/
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    /* Copy memory from host to device*/
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<num_blocks, num_threads>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << "+" << h_b[i] << "=" << h_c[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return true;
}
