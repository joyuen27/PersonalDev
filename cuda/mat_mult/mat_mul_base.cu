#include <vector>
#include <iostream>

__global__ void k_mat_mul_base(float* dev_m1, float* dev_m2, float* dev_m3, uint32_t num_rows, uint32_t num_cols, uint32_t k)
{
    // Calculate index of the current output element to calculate for
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check idx in bounds
    if (idx >= num_rows * num_cols) 
    {
        return;
    }

    //Get m1 row
    int m1_row = idx / num_cols;

    //Get m2 col
    int m2_col = idx % num_cols;


    //Calculate dot product
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += dev_m1[m1_row * k + i] * dev_m2[m2_col + i * num_cols];
    }

    //Set pointer to sum
    dev_m3[idx] = sum;
}

/* Returns result array */
std::vector<float> mat_mul_base(std::vector<float>& m1, std::vector<float>& m2, uint32_t row_1, uint32_t col_2)
{

    // Get dimensions
    uint32_t col_1 = m1.size() / row_1;
    uint32_t row_2 = m2.size() / col_2;

    // Return if matrix dimensions incorrect
    if (m1.size() % row_1 != 0 
    || m2.size() % col_2 != 0
    || col_1 != row_2)
    {
        return {};
    }

    /* Set up return */
    std::vector<float> m3(row_1 * col_2);

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

    /* Call Kernel */
    uint16_t num_threads = 256; //Should always be some multiple of 32 for warp efficiency
    uint16_t num_blocks = (row_1 * col_2 + num_threads - 1) / num_threads;
    k_mat_mul_base<<<num_blocks, num_threads>>>(dev_m1, dev_m2, dev_m3, row_1, col_2, col_1);

    /* Wait for Sync */
    cudaDeviceSynchronize();

    /* Copy from device back to host */
    cudaMemcpy(m3.data(), dev_m3, row_1 * col_2 * sizeof(float), cudaMemcpyDeviceToHost);

    /* Free device memory */
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_m3);

    return m3;

}

int main() {
  int N = 512;
    
  std::vector<float> m1(N * N);
  std::vector<float> m2(N * N);
  
  // m1 = identity matrix
  for (int r = 0; r < N; r++)
      for (int c = 0; c < N; c++)
          m1[r * N + c] = (r == c) ? 1.0f : 0.0f;

  // m2 = fill each element with its column index
  for (int r = 0; r < N; r++)
      for (int c = 0; c < N; c++)
          m2[r * N + c] = (float)c;

  std::vector<float> result = mat_mul_base(m1, m2, N, N);

  if (result.empty()) {
      std::cout << "Matrix multiply failed" << std::endl;
      return 1;
  }

  // Identity * m2 should equal m2, so check a few elements
  bool correct = true;
  for (int r = 0; r < N; r++)
      for (int c = 0; c < N; c++)
          if (result[r * N + c] != (float)c) {
              correct = false;
              std::cout << "WRONG at " << r << "," << c 
                        << " expected " << c 
                        << " got " << result[r * N + c] << std::endl;
          }

  if (correct)
      std::cout << "All correct!" << std::endl;

    return 0;
}
