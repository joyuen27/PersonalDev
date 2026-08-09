#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "main.h"
#include "embedding.h"
#include "attention.h"
#include "layer_norm.h"

GPT2Config cfg;
Block* blocks;

int32_t *tokens;
int32_t n_tokens;

float *wte; /* Maps token to vector*/
float *wpe; /* Maps postion to vector*/

int main() {

    load_tokens();

    load_model(blocks);

    float* V_embd = (float*) malloc (cfg.n_embd * n_tokens * sizeof(float));
    cpu_embed(V_embd, tokens, n_tokens, wte, wpe, cfg.n_embd);

    // FILE* f = fopen("test_embed_out.bin", "wb");
    // fwrite(embed_out, sizeof(float), n_tokens * cfg.n_embd, f);
    // fclose(f);

    float* ln_out = (float*)malloc(n_tokens * cfg.n_embd * sizeof(float));
    float* V_delta = (float*) malloc (cfg.n_embd * n_tokens * sizeof(float));

    for (int l = 0; l < cfg.n_layer; l++) {
        //LayerNorm1
        cpu_layer_norm(V_embd, ln_out, blocks[l].ln1_w, blocks[l].ln1_b, n_tokens, cfg.n_embd);

        //Attention
        V_delta = cpu_attention(ln_out,
                        blocks[l].attn_w, blocks[l].attn_b,
                        blocks[l].proj_w, blocks[l].proj_b,
                        n_tokens, cfg.n_embd, cfg.n_head);

        //Add delta to embeddings
        for (int t = 0; t < n_tokens; t++) {
            for (int i = 0; i < cfg.n_embd; i++){
                V_embd[t * cfg.n_embd + i] += V_delta[t * cfg.n_embd + i];
            }
        }

        //LayerNorm2
        cpu_layer_norm(V_embd, ln_out, blocks[l].ln2_w, blocks[l].ln2_b, n_tokens, cfg.n_embd);
    }

}

bool load_tokens() {
    FILE *f = fopen("tokens.bin", "rb");
    if (f == nullptr) {
        return false;
    }

    fread(&n_tokens, sizeof(int32_t), 1, f);

    tokens = (int32_t*)malloc(n_tokens * sizeof(int32_t));
    fread(tokens, sizeof(int32_t), n_tokens, f);

    fclose(f);
    return true;
}

bool load_model(Block* blocks) {

    /* Open File */
    FILE *f = fopen("gpt2.bin", "rb");
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

        blocks[i].proj_w = (float*)malloc(cfg.n_embd * cfg.n_embd *sizeof(float));
        fread(blocks[i].proj_w, sizeof(float), cfg.n_embd * cfg.n_embd, f);

        blocks[i].proj_b = (float*)malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].proj_b, sizeof(float), cfg.n_embd, f);

        blocks[i].ln2_w = (float*) malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].ln2_w, sizeof(float), cfg.n_embd, f);

        blocks[i].ln2_b = (float*)malloc(cfg.n_embd * sizeof(float));
        fread(blocks[i].ln2_b, sizeof(float), cfg.n_embd, f);
        /*TODO rest of the weights*/
    }
    fclose(f);

    return true;
}
