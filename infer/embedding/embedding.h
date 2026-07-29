#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

float* cpu_embed(int32_t* tokens, int num_tokens, float* wte, float* wpe, int n_embd);