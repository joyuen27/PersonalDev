#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "layer_norm.h"

void cpu_layer_norm(float* V_embd, float* ln_out,
                    float* ln_w, float* ln_b,
                    int32_t n_tokens, int32_t n_embd) {
    for (int t = 0; t < n_tokens; t++) {

        //Find Mean
        float accum = 0;
        for (int i = 0; i < n_embd; i++) {
            accum += V_embd[t * n_embd + i];
        }
        float mean = accum / n_embd;

        //Find Variance
        accum = 0;
        for (int i = 0; i < n_embd; i++) {
            accum += ((V_embd[t * n_embd + i] - mean) * (V_embd[t * n_embd + i] - mean));
        }
        float var = accum / n_embd;

        //Normalize, Scale, and Shift
        float denom = sqrtf(var + EPS);
        for (int i = 0; i < n_embd; i++) {
            ln_out[t * n_embd + i] = ((V_embd[t * n_embd + i] - mean) / denom)
                                      * ln_w[i] + ln_b[i];
        }
    }
}
