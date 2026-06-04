#include "include/mat_mul_base.h"
#include "include/mat_mul_tiled.h"
#include "include/mat_mul_cublas.h"
#include "include/mat_mul_tensor.h"

#include <vector>
#include <iostream>
#include <cmath>

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

int main() {
    int n = 512;
    int m = 512;
    int k = 512;

    std::vector<float> m1(n * k);
    std::vector<float> m2(k * m);
    std::vector<float> m3_test(n * m);
    std::vector<float> m3_cublas(n * m);

    set_matrix(n, m, k, m1.data(), m2.data());

    mat_mul_base(m1, m2, m3_test, n, m, k);
    mat_mul_cublas(m1, m2, m3_cublas, n, m, k);

    for (int i = 0; i < n * m; i++) {
        //because cublas does different floating point addition/rounding
        if (std::fabs(m3_test[i] - m3_cublas[i]) / std::fabs(m3_cublas[i]) > 1e-2f) {
            std::cout << "Wrong!" << std::endl;
            return 0;
        }
    }
    return 0;
}
