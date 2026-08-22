#pragma once

#ifndef EMBEDDING_H
#define EMBEDDING_H
#include <stdint.h>

void cpu_embed(float* V_embd,
               int32_t* tokens, int32_t n_tokens,
               float* wte, float* wpe,
               int32_t n_embd);

void cpu_unembed(float* t_embd,
                 int32_t vocab_size, int32_t n_embd,
                 float* wte,
                 float* scores);

#endif
