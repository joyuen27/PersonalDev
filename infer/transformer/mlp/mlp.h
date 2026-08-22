#pragma once

#ifndef MLP_H
#define MLP_H
#include <stdint.h>

float gelu(float x);

void cpu_mlp(float* ln_embd,
             float* fc_w, float* fc_b,
             float* proj_w, float* proj_b,
             int32_t n_embd, int32_t n_tokens,
             float* mlp_delta);

#endif
