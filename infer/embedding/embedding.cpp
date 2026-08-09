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
