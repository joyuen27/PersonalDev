#include <vector>
#include <iostream>

#define TILE_SIZE 16

__global__ void k_mat_mul_tiled(float* dev_a, float* dev_b, float* dev_c, int a_num_rows, int b_num_cols, int dim_shared) {
    __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE];

    int c_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int c_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int num_tile_steps = (dim_shared + TILE_SIZE - 1) / TILE_SIZE;

    float sum = 0;

    for (int t = 0; t < num_tile_steps; t++)
    {
        //Determine row/col for A and B
        int a_row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int a_col = TILE_SIZE * t + threadIdx.x;
        int b_row = TILE_SIZE * t + threadIdx.y;
        int b_col = blockIdx.x * TILE_SIZE + threadIdx.x;

        //Pull global memory into shared
        a_shared[threadIdx.y][threadIdx.x] = dev_a[a_row * dim_shared + a_col];
        b_shared[threadIdx.y][threadIdx.x] = dev_b[b_row * b_num_cols + b_col];

        //Synchronize all threads in block
        __syncthreads();

        //Dot product of A col and B row;
        for (int i = 0; i < TILE_SIZE; i++)
        {
            sum += a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x];
        }
        //Synchronize all threads in block
        __syncthreads();
    }

    //Set output elem
    dev_c[c_row * b_num_cols + c_col] = sum;
}

std::vector<float> mat_mul_tiled(std::vector<float> a, std::vector<float> b, int a_num_rows, int b_num_cols) {
    //Check matrix dimensions
    int a_num_cols = a.size() / a_num_rows;
    int b_num_rows = b.size() / b_num_cols;
    if (a_num_cols != b_num_rows) {
        return {};
    }

    //Set up device pointers
    float* dev_a;
    float* dev_b;
    float* dev_c;

    //Allocate on device
    cudaMalloc((void**) &dev_a, a.size() * sizeof(float));
    cudaMalloc((void**) &dev_b, b.size() * sizeof(float));
    cudaMalloc((void**) &dev_c, a_num_rows * b_num_cols * sizeof(float));

    //Copy a and b to device
    cudaMemcpy(dev_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

    //Define grid and block dimensions
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((b_num_cols + TILE_SIZE - 1)/ TILE_SIZE, (a_num_rows + TILE_SIZE - 1) / TILE_SIZE);

    //Call kernel
    k_mat_mul_tiled<<<grid_dim, block_dim>>>(dev_a,  dev_b, dev_c, a_num_rows, b_num_cols, a_num_cols);

    //Synchronize
    cudaDeviceSynchronize();

    //Copy output matrix back
    std::vector<float> c(a_num_rows * b_num_cols);
    cudaMemcpy(c.data(), dev_c, a_num_rows * b_num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    //Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return c;
}