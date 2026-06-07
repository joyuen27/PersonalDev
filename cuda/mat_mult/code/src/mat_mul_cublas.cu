#include <vector>
#include <cublas_v2.h>

void mat_mul_cublas(std::vector<float>& m1, std::vector<float>& m2, std::vector<float>& m3, int n, int m, int k) {
    float* dev_m1;
    float* dev_m2;
    float* dev_m3;

    cudaMalloc(&dev_m1, n * k * sizeof(float));
    cudaMalloc(&dev_m2, k * m * sizeof(float));
    cudaMalloc(&dev_m3, n * m * sizeof(float));

    cudaMemcpy(dev_m1, m1.data(), n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_m2, m2.data(), k * m * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major, your matrices are row-major
    // trick: compute B^T * A^T = (A*B)^T, then the result is correct in row-major
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,        // dimensions of output (cols, rows, inner)
        &alpha,
        dev_m2, m,      // B first (column-major trick)
        dev_m1, k,      // A second
        &beta,
        dev_m3, m);     // output

    cudaMemcpy(m3.data(), dev_m3, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_m3);
}
