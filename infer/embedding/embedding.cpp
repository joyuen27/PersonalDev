#include <stdint.h>
#include <stdlib.h>

#include "embedding.h"

void cpu_embed(float* V_embd,
               int32_t* tokens, int32_t n_tokens,
               float* wte, float* wpe,
               int32_t n_embd) {

    for (int token_pos = 0; token_pos < n_tokens; token_pos++) {
        int32_t token_id = tokens[token_pos];
        for (int j = 0; j < n_embd; j++) {
            V_embd[token_pos * n_embd + j] = wte[token_id * n_embd + j] + wpe[token_pos * n_embd + j];
        }
    }
}

void cpu_unembed(float* t_embd,
                 int32_t vocab_size, int32_t n_embd,
                 float* wte,
                 float* scores) {

    for (int r = 0; r < vocab_size; r++) {
        float accum = 0;
        for (int c = 0; c < n_embd; c++) {
            accum += wte[r * n_embd + c] * t_embd[c];
        }
        scores[r] = accum;
    }
 
}