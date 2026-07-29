#include <stdint.h>
#include <stdlib.h>

float* cpu_embed(int32_t* tokens, int num_tokens, float* wte, float* wpe, int n_embd){
    float* ret = (float*) malloc (n_embd * num_tokens * sizeof(float));

    /* Use wte and wpe lookup to map token id/pos -> vectors*/
    for (int token_pos = 0; token_pos < num_tokens; token_pos++) {

        int32_t token_id = tokens[token_pos];

        for (int j = 0; j < n_embd; j++) {
            ret[token_pos * n_embd + j] = wte[token_id * n_embd + j] + wpe[token_pos * n_embd + j];
        }
    }

    return ret;
}