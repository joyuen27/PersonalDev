#include <vector>
#include <iostream>

__global__ void k_mat_mul(float* dev_m1, float* dev_m2, float* dev_m3, uint32_t row_dim, uint32_t col_dim, uint32_t k)
{
    // Calculate index of the current output element to calculate for
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check idx in bounds
    if (idx >= row_dim * col_dim) 
    {
        return;
    }

    // Get pointer to current output element to calculate
    float* elem_ptr = dev_m3[idx];

    //Get m1 row
    int m1_row = idx / col_dim;

    //Get m2 col
    int m2_col = idx % row_dim;


    //Calculate dot product
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += dev_m1[m1_row * col_dim + i] * dev_m2[m2_col + i *k];
    }

    //Set pointer to sum
    *elem_ptr = sum;
}

/* Returns true if mult successful */
bool mat_mul(std::vector<float>& m1, std::vector<float>& m2, uint32_t row_1, uint32_t col_2)
{
    // Get dimensions
    uint32_t col_1 = m1.size() / row_1;
    uint32_t row_2 = m2.size() / col_2;

    // Return if matrix dimensions incorrect
    if (m1.size() % row_1 != 0 
    || m2.size() % col_2 != 0
    || col_1 != row_2)
    {
        return false;
    }

    /* Set up pointers */
    float* dev_m1;
    float* dev_m2;
    float* dev_m3;

    /* Allocate memory on device */
    cudaMalloc((void**)&dev_m1, m1.size() * sizeof(float));
    cudaMalloc((void**)&dev_m2, m2.size() * sizeof(float));
    cudaMalloc((void**)&dev_m3, (row_1 * col_2 * sizeof(float)));

    /* Copy to Device */
    cudaMemcpy(dev_m1, m1.data(), m1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_m2, m2.data(), m2.size() * sizeof(float), cudaMemcpyHostToDevice);

}