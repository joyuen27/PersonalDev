#include "include/mat_mul_base.h"
#include "include/mat_mul_tiled.h"
#include "include/mat_mul_cublas.h"
#include "include/mat_mul_tensor.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

void set_matrix(int n, int m, int k, float* m1, float* m2);
bool check_matrix_equality(int n, int m, std::vector<float>& m1, std::vector<float>& m2);
void nsight_base_test(int num_threads, int matrix_size);
void test_tiled();
void test_base_vs_cuBLAS();
void nsight_tiled_test(int matrix_size);

//Base Testing
int num_threads[] = {32, 64, 128, 256, 512, 1024};
// int matrix_sizes[] = {512, 2048, 4096};

//Tile Testing
int tile_sizes[] = {8, 16, 32};
int matrix_sizes[] = {2048, 4096};


int num_runs = 10;

cudaEvent_t s, e;
float test_time, cublas_time;

int main() {
    test_tiled();
    return 0;
}

void set_matrix(int n, int m, int k, float* m1, float* m2) {
    // Fill m1 with col number
    for (int r = 0; r < n; r++)
    {
        for (int c = 0; c < k; c++)
        {
            m1[r * k + c] = (float) c;
        }
    }
    // Fill m2 with row number
    for (int r = 0; r < k; r++)
    {
        for (int c = 0; c < n; c++)
        {
            m2[r * m + c] = (float) r;
        }
    }

}

bool check_matrix_equality(int n, int m, std::vector<float>& m1, std::vector<float>& m2) {
    for (int i = 0; i < n * m; i++) {
        float diff = std::fabs(m1[i] - m2[i]);
        float ref  = std::fabs(m2[i]);
        // relative tolerance for large values, absolute for small
        if (diff > 0.01f * std::max(ref, 1.0f)) {
            std::cout << "Wrong at index " << i 
                      << ": got " << m1[i] << " expected " << m2[i] << std::endl;
            return false;
        }
    }
    std::cout << "Correct!" << std::endl;
    return true;
}

void test_base_vs_cuBLAS() {
    
    // warmup cuBLAS
    for (int n : matrix_sizes) {
        std::vector<float> w1(n*n), w2(n*n), w3(n*n);
        mat_mul_cublas(w1, w2, w3, n, n, n);
    }
    cudaDeviceSynchronize();

    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int i = 0; i < sizeof(matrix_sizes) / sizeof(int); i++) {
        for (int j = 0; j < sizeof(num_threads) / sizeof(int); j++) {

            int n = matrix_sizes[i];
            int k = matrix_sizes[i];
            int m = matrix_sizes[i];

            std::vector<float> m1(n * k);
            std::vector<float> m2(k * m);
            std::vector<float> m3_test(n * m);
            std::vector<float> m3_cublas(n * m);

            set_matrix(n, m, k, m1.data(), m2.data());

            std::vector<float> test_times(num_runs);
            std::vector<float> cublas_times(num_runs);

            for (int r = 0; r < num_runs; r++) {
                cudaEventRecord(s);
                mat_mul_base_test(num_threads[j], m1, m2, m3_test, n, m, k);
                cudaEventRecord(e);
                cudaEventSynchronize(e);
                cudaEventElapsedTime(&test_times[r], s, e);

                cudaEventRecord(s);
                mat_mul_cublas(m1, m2, m3_cublas, n, m, k);
                cudaEventRecord(e);
                cudaEventSynchronize(e);
                cudaEventElapsedTime(&cublas_times[r], s, e);
            }

            std::sort(test_times.begin(), test_times.end());
            std::sort(cublas_times.begin(), cublas_times.end());
            test_time = test_times[num_runs / 2];      // median
            cublas_time = cublas_times[num_runs / 2];

            std::cout << " Tested Kernel Time: " << test_time << " for matrix size " << matrix_sizes[i] << " with " << num_threads[j] << " threads." << std::endl;
            std::cout << " Tested cuBLAS Time: " << cublas_time << " for matrix size " << matrix_sizes[i] << std::endl;

        }
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
}

void test_tiled() {

    cudaEventCreate(&s);
    cudaEventCreate(&e);

    const int num_warmup = 5;

    for (int i = 0; i < sizeof(matrix_sizes) / sizeof(int); i++) {
        int n = matrix_sizes[i];
        int k = matrix_sizes[i];
        int m = matrix_sizes[i];

        std::vector<float> m1(n * k);
        std::vector<float> m2(k * m);
        std::vector<float> m3_test(n * m);
        std::vector<float> m3_cublas(n * m);

        set_matrix(n, m, k, m1.data(), m2.data());

        std::vector<float> test_times(num_runs);
        std::vector<float> cublas_times(num_runs);

        // --- warm THIS kernel at THIS size until steady-state ---
        for (int w = 0; w < num_warmup; w++) {
            mat_mul_tiled(m1, m2, m3_test, n, m);
        }

        cudaDeviceSynchronize();

        for (int r = 0; r < num_runs; r++) {
            cudaEventRecord(s);
            mat_mul_tiled( m1, m2, m3_test, n, m);
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&test_times[r], s, e);
        }

         // --- warm cuBLAS separately, then time it ---
         for (int w = 0; w < num_warmup; w++) {
            mat_mul_cublas(m1, m2, m3_cublas, n, m, k);
        }

        cudaDeviceSynchronize();

        for (int r = 0; r < num_runs; r++) {

            cudaEventRecord(s);
            mat_mul_cublas(m1, m2, m3_cublas, n, m, k);
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&cublas_times[r], s, e);
        }

        // check_matrix_equality(n, k, m3_test, m3_cublas);

        std::sort(test_times.begin(), test_times.end());
        std::sort(cublas_times.begin(), cublas_times.end());
        test_time = test_times[num_runs / 2];      // median
        cublas_time = cublas_times[num_runs / 2];

        std::cout << " Tested Tiled Kernel Time: " << test_time << " for matrix size " << matrix_sizes[i] << " with tile size 32." << std::endl;
        std::cout << " Tested cuBLAS Time: " << cublas_time << " for matrix size " << matrix_sizes[i] << std::endl;
    
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
}

void nsight_base_test(int num_threads, int matrix_size) {
    std::vector<float> m1(matrix_size * matrix_size);
    std::vector<float> m2(matrix_size * matrix_size);
    std::vector<float> m3(matrix_size * matrix_size);

    set_matrix(matrix_size, matrix_size, matrix_size, m1.data(), m2.data());

    mat_mul_base_test(num_threads, m1, m2, m3, matrix_size, matrix_size, matrix_size);

}

void nsight_tiled_test(int matrix_size) {
    std::vector<float> m1(matrix_size * matrix_size);
    std::vector<float> m2(matrix_size * matrix_size);
    std::vector<float> m3(matrix_size * matrix_size);

    set_matrix(matrix_size, matrix_size, matrix_size, m1.data(), m2.data());

    mat_mul_tiled(m1, m2, m3, matrix_size, matrix_size);

}