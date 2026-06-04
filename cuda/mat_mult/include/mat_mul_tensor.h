#pragma once
#include <vector>

// Requires sm_70+ (Volta/Turing/Ampere). n, m, k must be multiples of 16.
void mat_mul_tensor(std::vector<float>& m1, std::vector<float>& m2, std::vector<float>& m3, int n, int m, int k);
