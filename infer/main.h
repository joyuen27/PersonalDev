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

bool load_tokens();
bool load_model();


#endif