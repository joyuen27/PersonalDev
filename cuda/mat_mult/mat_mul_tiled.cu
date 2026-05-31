#include <vector>
#include <iostream>

__global__ k_mat_mul_tiled() {

}

std::vector<float> mat_mul_tiled(std::vector<float> a, std::vector<float> b, int a_num_rows, int b_num_cols) {
    //Check matrix dimensions
    int a_num_cols = a.size() / a_num_rows;
    int b_num_rows = b.size() / b_num_cols;
    if (a_num_cols != b_num_rows) {
        return {};
    } 

    //Allocate on device
    cudaMalloc()
}

int main() {
    return 0;
}
