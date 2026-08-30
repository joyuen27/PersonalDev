#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "main.h"
#include "embedding.h"
#include "attention.h"
#include "layer_norm.h"
#include "mlp.h"

GPT2Config cfg;
Block* blocks;

int32_t *token_ids;
int32_t n_tokens;
int32_t max_new = 20; /*Max tokens to regressively generate*/

float* wte; /* Maps token to vector*/
float* wpe; /* Maps postion to vector*/

float* lnf_w; /* Final LayerNorm Weights */
float* lnf_b; /* Final LayerNorm Biases */

char*    vocab_data;
int32_t* vocab_offsets;

float* k_cache;
float* v_cache;

// Seconds from a monotonic clock.
static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {

    load_tokens();
    load_model();
    load_vocab();

    for (int i = 0; i < n_tokens; i++) {
        printf("%s", vocab_data + vocab_offsets[token_ids[i]]);
        fflush(stdout);
    }

    // generate();
    generate_kv();

    // Dump full token sequence (prompt + generated) for PyTorch comparison
    {
        int32_t total = n_tokens + max_new;
        FILE* f = fopen("test_gen_tokens.bin", "wb");
        fwrite(&n_tokens, sizeof(int32_t), 1, f); // prompt length
        fwrite(&total,    sizeof(int32_t), 1, f); // total length
        fwrite(token_ids, sizeof(int32_t), total, f);
        fclose(f);
    }
}


void generate() {

    //Allocate all space needed for max embeddings
    int max_len = n_tokens + max_new;
    float* V_embd = (float*) malloc(cfg.n_embd * max_len * sizeof(float));
    float* ln_out       = (float*) malloc(max_len * cfg.n_embd * sizeof(float));
    float* attn_delta   = (float*) malloc(max_len * cfg.n_embd * sizeof(float));
    float* mlp_delta    = (float*) malloc(max_len * cfg.n_embd * sizeof(float));
    float* unembd_scores = (float*) malloc(cfg.vocab_size * sizeof(float));

    double t0 = now_sec();
    double ttft = 0;

    for (int step = 0; step < max_new; step++)
    {
        int32_t curr_tokens = n_tokens + step;

        // Re-embed the whole sequence every step.
        cpu_embed(V_embd, token_ids, curr_tokens, wte, wpe, cfg.n_embd);

        for (int l = 0; l < cfg.n_layer; l++) {
            Block b = blocks[l];

            //LayerNorm1
            cpu_layer_norm(V_embd, ln_out, b.ln1_w, b.ln1_b, curr_tokens, cfg.n_embd);

            //Attention
            cpu_attention(ln_out,
                        b.attn_w, b.attn_b,
                        b.proj_w, b.proj_b,
                        curr_tokens, cfg.n_embd, cfg.n_head,
                        attn_delta);

            //Add attn delta to embeddings
            for (int t = 0; t < curr_tokens; t++) {
                for (int i = 0; i < cfg.n_embd; i++){
                    V_embd[t * cfg.n_embd + i] += attn_delta[t * cfg.n_embd + i];
                }
            }

            //LayerNorm2
            cpu_layer_norm(V_embd, ln_out, b.ln2_w, b.ln2_b, curr_tokens, cfg.n_embd);

            //MLP
            cpu_mlp(ln_out, b.mlp_fc_w, b.mlp_fc_b, b.mlp_proj_w, b.mlp_proj_b, cfg.n_embd, curr_tokens, mlp_delta);

            //Add mlp delta to embeddings
            for (int t = 0; t < curr_tokens; t++) {
                for (int i = 0; i < cfg.n_embd; i++){
                    V_embd[t * cfg.n_embd + i] += mlp_delta[t * cfg.n_embd + i];
                }
            }
        }

        // Final Layer Norm
        cpu_layer_norm(V_embd, ln_out, lnf_w, lnf_b, curr_tokens, cfg.n_embd);

        // Get last token embedding from layerNorm output
        float* last = ln_out + (curr_tokens - 1) * cfg.n_embd;

        // Unembed to get vocab scores
        cpu_unembed(last, cfg.vocab_size, cfg.n_embd, wte, unembd_scores);

        // Greedy pick highest score
        float max_score = unembd_scores[0];
        int max_i = 0; // This is the token id of the next predicted token
        for (int i = 1; i < cfg.vocab_size; i++) {
            if (unembd_scores[i] > max_score) {
                max_score = unembd_scores[i];
                max_i = i;
            }
        }

        //Print next token vocab
        printf("%s", vocab_data + vocab_offsets[max_i]);
        fflush(stdout);

        //Add new token back into input to generate again
        token_ids[curr_tokens] = max_i;

        if (step == 0) ttft = now_sec() - t0;
    }

    double total = now_sec() - t0;
    printf("\nTTFT: %.1f ms\nAverage: %.1f tokens/s\n", ttft * 1000.0, max_new / total);

    free(V_embd);
    free(ln_out);
    free(attn_delta);
    free(mlp_delta);
    free(unembd_scores);
}

// KV-cached engine
void generate_kv() {

    //Allocate all space needed for max embeddings
    int max_len = n_tokens + max_new;
    float* V_embd = (float*) malloc(cfg.n_embd * max_len * sizeof(float));
    float* ln_out       = (float*) malloc(max_len * cfg.n_embd * sizeof(float));
    float* attn_delta   = (float*) malloc(max_len * cfg.n_embd * sizeof(float));
    float* mlp_delta    = (float*) malloc(max_len * cfg.n_embd * sizeof(float));
    float* unembd_scores = (float*) malloc(cfg.vocab_size * sizeof(float));

    //Allocate KV cache
    k_cache = (float*) malloc(max_len * cfg.n_embd * cfg.n_layer * sizeof(float));
    v_cache = (float*) malloc(max_len * cfg.n_embd * cfg.n_layer * sizeof(float));

    double t0 = now_sec();
    double ttft = 0;

    for (int step = 0; step < max_new; step++)
    {
        int32_t curr_tokens = n_tokens + step;

        // Prefill the whole prompt on step 0, then decode one token at a time.
        int n_new = (step == 0) ? n_tokens : 1;
        int c_pos = (step == 0) ? 0 : curr_tokens - 1;

        // Embed the new tokens
        cpu_embed(V_embd, token_ids + c_pos, n_new, wte, wpe + c_pos * cfg.n_embd, cfg.n_embd);

        for (int l = 0; l < cfg.n_layer; l++) {
            Block b = blocks[l];

            // Per-layer slice of the KV cache.
            float* k_cache_l = k_cache + l * max_len * cfg.n_embd;
            float* v_cache_l = v_cache + l * max_len * cfg.n_embd;

            //LayerNorm1
            cpu_layer_norm(V_embd, ln_out, b.ln1_w, b.ln1_b, n_new, cfg.n_embd);

            //Attention (KV cached)
            cpu_attention_cached(ln_out,
                        b.attn_w, b.attn_b,
                        b.proj_w, b.proj_b,
                        curr_tokens, cfg.n_embd, cfg.n_head,
                        attn_delta,
                        k_cache_l, v_cache_l,
                        n_new, c_pos);

            //Add attn delta to embeddings
            for (int t = 0; t < n_new; t++) {
                for (int i = 0; i < cfg.n_embd; i++){
                    V_embd[t * cfg.n_embd + i] += attn_delta[t * cfg.n_embd + i];
                }
            }

            //LayerNorm2
            cpu_layer_norm(V_embd, ln_out, b.ln2_w, b.ln2_b, n_new, cfg.n_embd);

            //MLP
            cpu_mlp(ln_out, b.mlp_fc_w, b.mlp_fc_b, b.mlp_proj_w, b.mlp_proj_b, cfg.n_embd, n_new, mlp_delta);

            //Add mlp delta to embeddings
            for (int t = 0; t < n_new; t++) {
                for (int i = 0; i < cfg.n_embd; i++){
                    V_embd[t * cfg.n_embd + i] += mlp_delta[t * cfg.n_embd + i];
                }
            }
        }

        // Final Layer Norm
        cpu_layer_norm(V_embd, ln_out, lnf_w, lnf_b, n_new, cfg.n_embd);

        // Get last (newest) token embedding from layerNorm output
        float* last = ln_out + (n_new - 1) * cfg.n_embd;

        // Unembed to get vocab scores
        cpu_unembed(last, cfg.vocab_size, cfg.n_embd, wte, unembd_scores);

        // Greedy pick highest score
        float max_score = unembd_scores[0];
        int max_i = 0; // This is the token id of the next predicted token
        for (int i = 1; i < cfg.vocab_size; i++) {
            if (unembd_scores[i] > max_score) {
                max_score = unembd_scores[i];
                max_i = i;
            }
        }

        //Print next token vocab
        printf("%s", vocab_data + vocab_offsets[max_i]);
        fflush(stdout);

        //Add new token back into input to generate again
        token_ids[curr_tokens] = max_i;

        if (step == 0) ttft = now_sec() - t0;
    }

    double total = now_sec() - t0;
    printf("\nTTFT: %.1f ms\nAverage: %.1f tokens/s\n", ttft * 1000.0, max_new / total);

    free(V_embd);
    free(ln_out);
    free(attn_delta);
    free(mlp_delta);
    free(unembd_scores);
    free(k_cache);
    free(v_cache);
}

bool load_tokens() {
    FILE *f = fopen("model/tokens.bin", "rb");
    if (f == nullptr) {
        return false;
    }

    fread(&n_tokens, sizeof(int32_t), 1, f);

    token_ids = (int32_t*)malloc((n_tokens + max_new) * sizeof(int32_t));
    fread(token_ids, sizeof(int32_t), n_tokens, f);

    fclose(f);
    return true;
}

bool load_model() {

    /* Open File */
    FILE *f = fopen("model/gpt2.bin", "rb");
    if (f == nullptr) {
        return false;
    }

    /* Load cfg */
    fread(&cfg, sizeof(GPT2Config), 1, f);

    blocks = (Block*) malloc(cfg.n_layer * sizeof(Block));

    /* Check right file */
    if (cfg.magic != 20240326) {
        return false;
    }

    /* Allocate and copy over embedding matrices */
    wte = (float*)malloc(cfg.vocab_size * cfg.n_embd * sizeof(float));
    fread(wte, sizeof(float), cfg.vocab_size * cfg.n_embd, f);

    wpe = (float*)malloc(cfg.block_size * cfg.n_embd * sizeof(float));
    fread(wpe, sizeof(float), cfg.block_size * cfg.n_embd, f);

    for (int i = 0; i < cfg.n_layer; i++) {

        blocks[i].ln1_w = (float*) malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].ln1_w, sizeof(float), cfg.n_embd, f);

        blocks[i].ln1_b = (float*)malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].ln1_b, sizeof(float), cfg.n_embd, f);

        blocks[i].attn_w = (float*)malloc(cfg.n_embd * 3 * cfg.n_embd * sizeof(float));
        fread(blocks[i].attn_w, sizeof(float), cfg.n_embd * 3 * cfg.n_embd, f);

        blocks[i].attn_b = (float*)malloc(cfg.n_embd * 3 * sizeof(float));
        fread(blocks[i].attn_b, sizeof(float), cfg.n_embd * 3, f);

        blocks[i].proj_w = (float*)malloc(cfg.n_embd * cfg.n_embd * sizeof(float));
        fread(blocks[i].proj_w, sizeof(float), cfg.n_embd * cfg.n_embd, f);

        blocks[i].proj_b = (float*)malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].proj_b, sizeof(float), cfg.n_embd, f);

        blocks[i].ln2_w = (float*) malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].ln2_w, sizeof(float), cfg.n_embd, f);

        blocks[i].ln2_b = (float*)malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].ln2_b, sizeof(float), cfg.n_embd, f);

        blocks[i].mlp_fc_w = (float*)malloc(cfg.n_embd * 4 * cfg.n_embd * sizeof(float));
        fread(blocks[i].mlp_fc_w, sizeof(float), cfg.n_embd * 4 * cfg.n_embd, f);

        blocks[i].mlp_fc_b = (float*)malloc(cfg.n_embd * 4 * sizeof(float));
        fread(blocks[i].mlp_fc_b, sizeof(float), cfg.n_embd * 4, f);

        blocks[i].mlp_proj_w = (float*)malloc(cfg.n_embd * 4 * cfg.n_embd * sizeof(float));
        fread(blocks[i].mlp_proj_w, sizeof(float), cfg.n_embd * 4 * cfg.n_embd, f);

        blocks[i].mlp_proj_b = (float*)malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].mlp_proj_b, sizeof(float), cfg.n_embd, f);
    }

    lnf_w = (float*)malloc(cfg.n_embd * sizeof(float));
    fread(lnf_w, sizeof(float), cfg.n_embd, f);

    lnf_b = (float*)malloc(cfg.n_embd * sizeof(float));
    fread(lnf_b, sizeof(float), cfg.n_embd, f);

    fclose(f);
    return true;
}

bool load_vocab() {
    FILE *f = fopen("model/vocab.bin", "rb");
    if (f == nullptr) {
        return false;
    }

    int32_t n_vocab;
    fread(&n_vocab, sizeof(int32_t), 1, f);

    vocab_offsets = (int32_t*) malloc(n_vocab * sizeof(int32_t));
    fread(vocab_offsets, sizeof(int32_t), n_vocab, f);

    long data_start = ftell(f);
    fseek(f, 0, SEEK_END);
    long data_size = ftell(f) - data_start;
    fseek(f, data_start, SEEK_SET);

    vocab_data = (char*) malloc(data_size);
    fread(vocab_data, 1, data_size, f);

    fclose(f);
    return true;
}
