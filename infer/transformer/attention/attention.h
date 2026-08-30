#pragma once

#ifndef ATTENTION_H
#define ATTENTION_H
#include <stdint.h>

void cpu_attention(float* ln_out,
    float* attn_w, float* attn_b,
    float* proj_w, float* proj_b,
    int32_t n_tokens, int32_t n_embd, int32_t n_head,
    float* attn_output);

void cpu_attention_cached(float* ln_out,
    float* attn_w, float* attn_b,
    float* proj_w, float* proj_b,
    int32_t n_tokens, int32_t n_embd, int32_t n_head,
    float* attn_output,
    float* k_cache,
    float* v_cache,
    int n_new,
    int c_pos);

#endif
