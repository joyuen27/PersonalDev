#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <math.h>
#include "mlp.h"

void cpu_mlp(float* ln_embd, float* fc_w, float* fc_b, float* proj_w, float* proj_b,
            int32_t n_embd, int32_t n_tokens, float* mlp_delta)
{
    // FC Matrix Multiply
    float* fc_out = (float*) malloc (4 * n_embd * n_tokens * sizeof(float));
    for (int t = 0; t < n_tokens; t++) {
        //Get row in weight matrix
        for (int r = 0; r < 4 * n_embd; r++) {
            float accum = fc_b[r];
            //Get Token Embedding and dot product with row
            for (int i = 0; i < n_embd; i++) {
                accum += ln_embd[t * n_embd + i] * fc_w[r * n_embd + i];
            }
            fc_out[t * 4 * n_embd + r] = accum;
        }
    }

    //GeLu
    for (int i = 0; i < 4 * n_embd * n_tokens; i++) {
        fc_out[i] = gelu(fc_out[i]);
    }

    // Proj Matrix Multiply
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < n_embd; r++) {
            float accum = proj_b[r];
            for (int i = 0; i < 4 * n_embd; i++) {
                accum += fc_out[t * 4 * n_embd + i] * proj_w[r * 4 * n_embd + i];
            }
            mlp_delta[t * n_embd + r] = accum;
        }
    }
}

float gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x3)));
}
