#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "main.h"
#include "embedding.h"

GPT2Config cfg;

int32_t *tokens;
int32_t num_tokens;

float *wte; /* Maps token to vector*/
float *wpe; /* Maps postion to vector*/
float *ln1_w;
float *ln1_b;

int main() {

    load_tokens();

    load_model();
    
    float* embed_out = cpu_embed(tokens, num_tokens, wte, wpe, cfg.n_embd);

    FILE* f = fopen("test_embed_out.bin", "wb");
    fwrite(embed_out, sizeof(float), num_tokens * cfg.n_embd, f);
    fclose(f);
    
}

bool load_tokens() {
    FILE *f = fopen("tokens.bin", "rb");
    if (f == nullptr) {
        return false;
    }

    fread(&num_tokens, sizeof(int32_t), 1, f);

    tokens = (int32_t*)malloc(num_tokens * sizeof(int32_t));
    fread(tokens, sizeof(int32_t), num_tokens, f);
    
    fclose(f);
    return true;
}

bool load_model() {

    /* Open File */
    FILE *f = fopen("gpt2.bin", "rb");
    if (f == nullptr) {
        return false;
    }

    /* Load headers */
    fread(&cfg, sizeof(GPT2Config), 1, f);

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
        /* Allocate and copy over layernorm weights + biases*/
        ln1_w = (float*) malloc(cfg.n_embd * sizeof(float));
        fread(ln1_w, sizeof(float), cfg.n_embd, f);

        ln1_b = (float*)malloc(cfg.n_embd * sizeof(float));
        fread(ln1_b, sizeof(float), cfg.n_embd, f);

        /*TODO rest of the weights*/
    }
    fclose(f);

    return true;

}