#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "attention.h"

void cpu_attention_cached(float* ln_out,
                     float* attn_w, float* attn_b,
                     float* proj_w, float* proj_b,
                     int32_t n_tokens, int32_t n_embd, int32_t n_head,
                     float* attn_output,
                     float* k_cache,
                     float* v_cache,
                     int n_new,
                     int c_pos) {
    
    // QKV Projections
    float* QKV = (float*)malloc(n_new * 3 * n_embd * sizeof(float));

    //Cache KV from c_pos up to n_new tokens
    for (int t = 0; t < n_new; t++) {

        for (int r = 0; r < 3 * n_embd; r++) {
            float sum = attn_b[r];
            for (int c = 0; c < n_embd; c++) {
                sum += ln_out[t * n_embd + c] * attn_w[r * n_embd + c];
            }
            QKV[t * 3 * n_embd + r] = sum;
        }

        for (int i = 0; i < n_embd; i++) {
            k_cache[(c_pos + t) * n_embd + i] = QKV[ t * 3  * n_embd + n_embd + i];
            v_cache[(c_pos + t) * n_embd + i] = QKV[ t * 3 * n_embd + 2 * n_embd + i];
        }
    }

    float* heads = (float*) calloc(n_new * n_embd, sizeof(float));

    int32_t head_dim = n_embd / n_head;
    for (int t = 0; t < n_new; t++) {
        for (int h = 0; h < n_head; h++) {
            //Find token row then get Q and K
            float* Q = QKV + t * 3 * n_embd + h * head_dim;

            float scores[n_tokens];
            //Take dot product of QK to get scores
            for (int s = 0; s <= c_pos + t; s++){
                float* K = k_cache + s * n_embd + h * head_dim;
                float dot = 0;

                for (int i = 0; i < head_dim; i++) {
                    dot += Q[i] * K[i];
                }
                scores[s] = dot / sqrtf(head_dim);
            }

            //Softmax
            //Find Max
            float max = scores[0];
            for (int i = 1; i <= c_pos + t; i++) {
                if (scores[i] > max) {
                    max = scores[i];
                }
            }

            //Get exponential sum
            float sum = 0;
            for (int i = 0; i <= c_pos + t; i++) {
                scores[i] = expf(scores[i] - max);
                sum += scores[i];
            }

            // Divide
            for (int i = 0; i <= c_pos + t; i++) {
                scores[i] = scores[i] / sum;
            }

            //Calculating change vector
            for (int i = 0; i <= c_pos + t; i++) {
                float score = scores[i];
                float* V = v_cache + i * n_embd + h * head_dim;

                for (int j = 0; j < head_dim; j++) {
                    heads[t * n_embd + h * head_dim + j] += score * V[j];
                }
            }
        }
    }

    //Matrix multiply with proj
    for (int t = 0; t < n_new; t++) {
        for (int r = 0; r < n_embd; r++) {
            float accum = proj_b[r];
            for (int c = 0; c < n_embd; c++) {
                accum += heads[t * n_embd + c] * proj_w[r * n_embd + c];
            }
            attn_output[t * n_embd + r] = accum;
        }
    }

    free(heads);
    free(QKV);

}

void cpu_attention(float* ln_out,
                     float* attn_w, float* attn_b,
                     float* proj_w, float* proj_b,
                     int32_t n_tokens, int32_t n_embd, int32_t n_head,
                     float* attn_output) {

    float* QKV = (float*)malloc(n_tokens * 3 * n_embd * sizeof(float));

    for (int i = 0; i < n_tokens; i++) {
        for (int r = 0; r < 3 * n_embd; r++) {
            float sum = attn_b[r];
            for (int c = 0; c < n_embd; c++) {
                sum += ln_out[i * n_embd + c] * attn_w[r * n_embd + c];
            }
            QKV[i * 3 * n_embd + r] = sum;
        }
    }

    float* heads = (float*) calloc(n_tokens * n_embd, sizeof(float));

    int32_t head_dim = n_embd / n_head;
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < n_head; h++) {
            //Find token row then get Q and K
            float* Q = QKV + t * 3 * n_embd + h * head_dim;

            float scores[n_tokens];
            //Take dot product of QK to get scores
            for (int s = 0; s <= t; s++){
                float* K = QKV + s * 3 * n_embd + n_embd + h * head_dim;
                float dot = 0;

                for (int i = 0; i < head_dim; i++) {
                    dot += Q[i] * K[i];
                }
                scores[s] = dot / sqrtf(head_dim);
            }

            //Softmax
            //Find Max
            float max = scores[0];
            for (int i = 1; i <= t; i++) {
                if (scores[i] > max) {
                    max = scores[i];
                }
            }

            //Get exponential sum
            float sum = 0;
            for (int i = 0; i <= t; i++) {
                scores[i] = expf(scores[i] - max);
                sum += scores[i];
            }

            // Divide
            for (int i = 0; i <= t; i++) {
                scores[i] = scores[i] / sum;
            }

            //Calculating change vector
            for (int i = 0; i <= t; i++) {
                float score = scores[i];
                float* V = QKV + i * 3 * n_embd + 2 * n_embd + h * head_dim;

                for (int j = 0; j < head_dim; j++) {
                    heads[t * n_embd + h * head_dim + j] += score * V[j];
                }
            }
        }
    }

    //Matrix multiply with proj
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < n_embd; r++) {
            float accum = proj_b[r];
            for (int c = 0; c < n_embd; c++) {
                accum += heads[t * n_embd + c] * proj_w[r * n_embd + c];
            }
            attn_output[t * n_embd + r] = accum;
        }
    }

    free(heads);
    free(QKV);

}
