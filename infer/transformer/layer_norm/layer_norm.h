#pragma once

#ifndef LAYER_NORM_H
#define LAYER_NORM_H
#include <stdint.h>

#define EPS 1e-5f

void cpu_layer_norm(float* V_embd, float* ln_out,
                    float* ln_w, float* ln_b,
                    int32_t n_tokens, int32_t n_embd);

#endif
