#pragma once

#ifndef MAIN_H
#define MAIN_H
#include <stdint.h>

typedef struct {
    int32_t magic;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_embd;
    int32_t vocab_size;
    int32_t block_size;
} GPT2Config;

typedef struct {
    float* ln1_w;
    float* ln1_b;
    float* attn_w;
    float* attn_b;
    float* proj_w;
    float* proj_b;
    float* ln2_w;
    float* ln2_b;
    float* mlp_fc_w;
    float* mlp_fc_b;
    float* mlp_proj_w;
    float* mlp_proj_b;
} Block;

bool load_tokens();
bool load_model();
bool load_vocab();

#endif
