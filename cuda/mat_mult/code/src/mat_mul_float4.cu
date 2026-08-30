#include <vector>
#include <iostream>

#define TILE_SIZE 16
#define REG_TILE_SIZE 8

__global__ void k_mat_mul_float4(float* dev_a, float* dev_b, float* dev_c, int a_num_rows, int b_num_cols, int dim_shared) {
    __shared__ float a_shared[TILE_SIZE][TILE_SIZE * REG_TILE_SIZE]; // Transposed so we can get register tile in contiguous memory
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE * REG_TILE_SIZE];

    int c_row = (blockIdx.y * TILE_SIZE + threadIdx.y) * REG_TILE_SIZE;
    int c_col = (blockIdx.x * TILE_SIZE + threadIdx.x) * REG_TILE_SIZE;

    int num_tile_steps = (dim_shared + TILE_SIZE - 1) / TILE_SIZE;

    float accum[REG_TILE_SIZE][REG_TILE_SIZE] = {};

    for (int t = 0; t < num_tile_steps; t++)
    {
        //Determine row/col for A and B
        int a_row = (blockIdx.y * TILE_SIZE + threadIdx.y) * REG_TILE_SIZE;
        int a_col = TILE_SIZE * t + threadIdx.x;
        int b_row = TILE_SIZE * t + threadIdx.y;
        int b_col = (blockIdx.x * TILE_SIZE + threadIdx.x) * REG_TILE_SIZE;

        //Pull global memory into shared
        for (int i = 0; i < REG_TILE_SIZE; i++) {
            a_shared[threadIdx.x][threadIdx.y * REG_TILE_SIZE + i] = dev_a[(a_row + i) * dim_shared + a_col];
            b_shared[threadIdx.y][threadIdx.x * REG_TILE_SIZE + i] = dev_b[b_row * b_num_cols + b_col + i];
        }

        //Synchronize all threads in block
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
        {
            float a_reg[REG_TILE_SIZE];
            float b_reg[REG_TILE_SIZE];

            for (int j = 0; j < REG_TILE_SIZE; j += 4) 
            {
                *reinterpret_cast<float4*> (&a_reg[j]) = *reinterpret_cast<float4*> (&a_shared[i][threadIdx.y * REG_TILE_SIZE + j]);
                *reinterpret_cast<float4*> (&b_reg[j]) = *reinterpret_cast<float4*> (&b_shared[i][threadIdx.x * REG_TILE_SIZE + j]);
            }

            for (int r = 0; r < REG_TILE_SIZE; r++) {
                for (int c = 0; c < REG_TILE_SIZE; c++) {
                    accum[r][c] += a_reg[r] * b_reg[c];
                }
            }
        }
        //Synchronize all threads in block
        __syncthreads();
    }

    //Set output elems
    for (int r = 0; r < REG_TILE_SIZE; r++) {
        for (int c = 0; c < REG_TILE_SIZE; c++) {
            dev_c[(c_row + r) * b_num_cols + c_col + c] = accum[r][c];
        }
    }
    
}

void mat_mul_float4(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int a_num_rows, int b_num_cols) {
    //Check matrix dimensions
    int a_num_cols = a.size() / a_num_rows;
    int b_num_rows = b.size() / b_num_cols;
    if (a_num_cols != b_num_rows) {
        return;
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
    dim3 grid_dim((b_num_cols + TILE_SIZE * REG_TILE_SIZE - 1)/ (TILE_SIZE * REG_TILE_SIZE), (a_num_rows + TILE_SIZE * REG_TILE_SIZE - 1) / (TILE_SIZE * REG_TILE_SIZE));

    //Call kernel
    k_mat_mul_float4<<<grid_dim, block_dim>>>(dev_a,  dev_b, dev_c, a_num_rows, b_num_cols, a_num_cols);

    //Synchronize
    cudaDeviceSynchronize();

    //Copy output matrix back
    cudaMemcpy(c.data(), dev_c, a_num_rows * b_num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    //Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

}
