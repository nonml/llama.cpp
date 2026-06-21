#include "models.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen3::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,      hparams.dflash_block_size,     false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID,    hparams.dflash_mask_token_id,  false);

    std::vector<int32_t> ids;
    if (ml.get_arr(LLM_KV_DFLASH_TARGET_LAYER_IDS, ids, false)) {
        for (size_t i = 0; i < ids.size() && i < hparams.dflash_target_layer_ids.size(); ++i) {
            hparams.dflash_target_layer_ids[i] = ids[i];
        }
    }

    hparams.n_embd_inp_impl = (uint32_t)(hparams.dflash_target_layer_ids.size()) * hparams.n_embd;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_target_layer_ids = (int64_t) hparams.dflash_target_layer_ids.size();
    const int64_t n_embd_inp = n_target_layer_ids * n_embd;

    fc                = create_tensor(tn(LLM_TENSOR_FC, "weight"), {n_embd_inp, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, 0);

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,   "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<llm_build_dflash_encode>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<llm_build_dflash_decode>(*this, params);
        case LLM_GRAPH_TYPE_DFLASH_KV_PROJ:
            return std::make_unique<llm_build_dflash_kv_proj>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

ggml_tensor * llm_build_dflash_encode::build_inp_embd() const {
    const int64_t n_target_layer_ids = (int64_t) hparams.dflash_target_layer_ids.size();
    const int64_t n_embd_target_features = n_target_layer_ids * n_embd;

    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_target_features);
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_target_features, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

llm_build_dflash_encode::llm_build_dflash_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "hidden_norm_out", -1);

    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

llm_build_dflash_kv_proj::llm_build_dflash_kv_proj(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    // Input: new encoder features [n_embd, n_new] via cross_embd (from cross->v_embd)
    ggml_tensor * features = build_inp_cross_embd();

    std::vector<ggml_tensor *> k_layers, v_layers;
    k_layers.reserve(n_layer);
    v_layers.reserve(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * k = build_lora_mm(layer.wk, features);
        if (layer.wk_b) k = ggml_add(ctx0, k, layer.wk_b);
        ggml_tensor * v = build_lora_mm(layer.wv, features);
        if (layer.wv_b) v = ggml_add(ctx0, v, layer.wv_b);
        k_layers.push_back(k);
        v_layers.push_back(v);
    }

    // Stack all layer K projections along dim 1: [n_embd_k, n_layer * n_new]
    ggml_tensor * K_all = k_layers[0];
    for (int il = 1; il < n_layer; ++il) {
        K_all = ggml_concat(ctx0, K_all, k_layers[il], 1);
    }
    ggml_tensor * V_all = v_layers[0];
    for (int il = 1; il < n_layer; ++il) {
        V_all = ggml_concat(ctx0, V_all, v_layers[il], 1);
    }

    res->t_embd   = K_all;
    res->t_embd_v = V_all;

    ggml_build_forward_expand(gf, K_all);
    ggml_build_forward_expand(gf, V_all);
}

llm_build_dflash_decode::llm_build_dflash_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Noise tokens [MASK] — prefer target model's tok_embd; fall back to own for standalone use (e.g. imatrix)
    ggml_tensor * tok_embd_src = model.target_tok_embd ? model.target_tok_embd : model.tok_embd;
    ggml_tensor * noise_embd;
    if (tok_embd_src) {
        noise_embd = build_inp_embd(tok_embd_src);
    } else {
        // No tok_embd is reachable: this is a standalone graph reservation (the memory-fit probe
        // in common_get_device_memory_data, or imatrix) where the target model isn't linked yet
        // and the drafter ships without its own embeddings. The embeddings physically live on the
        // target and are not part of the draft's own memory, so reserve a placeholder of the
        // post-embedding shape instead of asserting. Never taken during real decoding, where
        // auto-setup in llama_init_from_model links the target's tok_embd before reservation.
        noise_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(noise_embd);
    }
    cb(noise_embd, "inp_noise_embd", -1);

    // Target context via llama_cross (filled from accumulated_target_ctx), graph rebuilds every step.
    // When kv_proj_valid we use the incrementally-cached K/V projections and never consume the raw
    // target context, so we must NOT register the cross-embd input (an unconsumed leaf input never
    // gets a backend buffer and would crash in set_input). Only build it for the fallback path.
    // Only the device-resident cache slots into head-split attention directly. Under
    // tensor-split (meta) the cache is absent and we recompute K/V from the accumulated
    // target context so K_tgt is head-split like K_noise (see dflash_update_kv_proj).
    // Standalone reservation (memory-fit probe / imatrix): no target context is available, so
    // reserve only the noise self-attention. This mirrors the cheap per-step cost of the real
    // kv_proj path (O(n_new)) rather than the raw-context fallback, which would otherwise reserve
    // a graph over n_ctx_train and badly overestimate the draft's memory.
    const bool standalone  = (tok_embd_src == nullptr);
    const bool use_kv_proj = cross->kv_proj_valid && cross->kv_proj_device;
    ggml_tensor * target_ctx = (use_kv_proj || standalone) ? nullptr : build_inp_cross_embd();
    const int64_t n_ctx = (use_kv_proj || standalone) ? (standalone ? 0 : (int64_t) cross->n_enc)
                                                      : target_ctx->ne[1];

    ggml_tensor * inpL = noise_embd;

    const int64_t n_tokens_kv = n_ctx + n_tokens;

    // Position tensor covering target_ctx + noise
    ggml_tensor * inp_pos_full = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens_kv);
    ggml_set_input(inp_pos_full);
    cb(inp_pos_full, "inp_pos_full", -1);

    // Q positions: last n_tokens entries (noise only)
    ggml_tensor * inp_pos_q = ggml_view_1d(ctx0, inp_pos_full, n_tokens,
            n_ctx * ggml_element_size(inp_pos_full));

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        // Q from noise only
        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        if (layer.wq_b) { Qcur = ggml_add(ctx0, Qcur, layer.wq_b); }
        cb(Qcur, "Qcur", il);

        // K_noise and V_noise: project noise through wk/wv (always fresh)
        ggml_tensor * K_noise = build_lora_mm(layer.wk, noise_norm);
        ggml_tensor * V_noise = build_lora_mm(layer.wv, noise_norm);
        if (layer.wk_b) K_noise = ggml_add(ctx0, K_noise, layer.wk_b);
        if (layer.wv_b) V_noise = ggml_add(ctx0, V_noise, layer.wv_b);

        ggml_tensor * Kcur, * Vcur;
        if (standalone) {
            // No target context: noise self-attention only (n_tokens_kv == n_tokens).
            Kcur = K_noise;
            Vcur = V_noise;
        } else {
            ggml_tensor * K_tgt, * V_tgt;
            if (use_kv_proj) {
                // Use incrementally-cached K/V projections (O(n_new) cost per step)
                auto kv = build_inp_cross_kv_proj(il);
                K_tgt = kv.k;
                V_tgt = kv.v;
            } else {
                // Fallback: compute K/V from raw target context (O(n_ctx) cost per step)
                K_tgt = build_lora_mm(layer.wk, target_ctx);
                if (layer.wk_b) K_tgt = ggml_add(ctx0, K_tgt, layer.wk_b);
                V_tgt = build_lora_mm(layer.wv, target_ctx);
                if (layer.wv_b) V_tgt = ggml_add(ctx0, V_tgt, layer.wv_b);
            }

            Kcur = ggml_concat(ctx0, K_tgt, K_noise, 1);
            Vcur = ggml_concat(ctx0, V_tgt, V_noise, 1);
        }
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens_kv);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens_kv);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        cb(Kcur, "Kcur_normed", il);

        // RoPE: K uses full positions [0..n_ctx+n_tokens-1], Q uses last n_tokens
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos_full, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Kcur, "Kcur_rope", il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos_q, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Qcur, "Qcur_rope", il);

        // Full attention (no causal mask)
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        ggml_tensor * cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "kqv_out", il);

        cur = build_lora_mm(layer.wo, cur);
        if (layer.wo_b) { cur = ggml_add(ctx0, cur, layer.wo_b); }
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "attn_res", il);

        ggml_tensor * ffn_inp = cur;
        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    ggml_tensor * output_src = model.target_output ? model.target_output : model.output;
    if (output_src) {
        cur = build_lora_mm(output_src, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}