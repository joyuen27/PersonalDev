#pragma once
#include <vector>

void mat_mul_base(std::vector<float>& m1, std::vector<float>& m2, std::vector<float>& m3, int row_1, int col_2, int k);

void mat_mul_base_test(int num_threads, std::vector<float>& m1, std::vector<float>& m2, std::vector<float>& m3, int row_1, int col_2, int k);
