#pragma once

#ifndef EMBEDDING_H
#define EMBEDDING_H
#include <stdint.h>

void cpu_embed(float* V_embd,
               int32_t* tokens, int32_t n_tokens,
               float* wte, float* wpe,
               int32_t n_embd);

#endif
