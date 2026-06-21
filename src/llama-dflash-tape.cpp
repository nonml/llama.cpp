// DFlash tape-replay engine — free-function implementations
//
// All functions operate on dflash_capture_data and llama_memory_recurrent
// (plus auxiliary params) without accessing the llama_context class.  They
// were extracted from llama_context:: methods in bee/v0.3.2 so that the
// tape-replay subsystem can be compiled and tested independently.
//
// Key invariants:
//   * Math / logic is bit-identical to bee/v0.3.2.
//   * CUDA kernel calls use proc_address indirection via ggml_backend_reg_get_proc_address.
//   * The async state machine (replay_pending / replay_event / replay_graph_ctx)
//     works exactly as in bee: dflash_tape_replay() enqueues; dflash_tape_replay_sync()
//     waits and then runs the conv rebuild.

#include "llama-dflash-tape.h"

#include "llama-memory-recurrent.h"
#include "llama-memory.h"     // llama_memory_recurrent_copy_profile
#include "llama-kv-cache.h"
#include "llama-impl.h"       // LLAMA_LOG_INFO / WARN
#include "dflash-profile.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// Maximum CUDA device index we ever scan.  Mirrors the constant used in
// llama-context.cpp so device-loop bounds are consistent.
#ifndef GGML_CUDA_MAX_DEVICES
#  define GGML_CUDA_MAX_DEVICES 32
#endif

// ---------------------------------------------------------------------------
// Internal helpers (file-scope statics — not exposed in the header)
// ---------------------------------------------------------------------------

static ggml_backend_reg_t dtape_gpu_backend_reg() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        reg = ggml_backend_reg_by_name("ROCm");
    }
    return reg;
}

static bool dtape_is_cuda_compatible_tensor(const ggml_tensor * t) {
    if (!t || !t->data || !t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    const char * name = ggml_backend_buffer_name(t->buffer);
    return name && (std::strncmp(name, "CUDA", 4) == 0 || std::strncmp(name, "ROCm", 4) == 0);
}

static bool dtape_tensor_span_in_bounds(const ggml_tensor * t, size_t offset_bytes, size_t n_bytes) {
    if (!t) { return false; }
    const size_t total = ggml_nbytes(t);
    return offset_bytes <= total && n_bytes <= total - offset_bytes;
}

static bool dtape_backend_dev_is_gpu(ggml_backend_dev_t dev) {
    if (!dev) { return false; }
    auto type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

// Find the first GPU backend in a backends list.
static ggml_backend_t dtape_first_gpu_backend(const std::vector<ggml_backend_ptr> & backends) {
    for (const auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dtape_backend_dev_is_gpu(dev)) {
            return backend.get();
        }
    }
    return nullptr;
}

// Inline helpers imported from bee's llama-context.h (static, not in the header).
static inline bool dtape_replay_gdn_supported_s(int64_t s) {
    return s == 16 || s == 32 || s == 64 || s == 128;
}

static inline bool dtape_replay_state_shape_valid(int64_t s, int64_t h_v, uint32_t n_embd_s) {
    if (s <= 0 || h_v <= 0) { return false; }
    const uint64_t us  = (uint64_t) s;
    const uint64_t uh  = (uint64_t) h_v;
    const uint64_t max = (uint64_t) -1;
    if (us > max / us) { return false; }
    const uint64_t ss = us * us;
    if (uh > 0 && ss > max / uh) { return false; }
    return ss * uh == (uint64_t) n_embd_s;
}

// ---------------------------------------------------------------------------
// dflash_ensure_recurrent_setup
// ---------------------------------------------------------------------------

void dflash_ensure_recurrent_setup(
        dflash_capture_data * capture,
        const llama_hparams & hparams) {
    if (!capture || !capture->recurrent_layer_ids.empty()) {
        return;
    }

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (hparams.is_recr(il)) {
            int idx = (int) capture->recurrent_layer_ids.size();
            capture->recurrent_layer_ids.push_back(il);

            std::string il_str = std::to_string(il);
            capture->tape_name_map["k_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_K};
            capture->tape_name_map["v_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_V};
            capture->tape_name_map["gate-" + il_str]                   = {idx, DFLASH_TAPE_GATE};
            capture->tape_name_map["beta-" + il_str]                   = {idx, DFLASH_TAPE_BETA};
            capture->tape_name_map["qkv_mixed_pretranspose-" + il_str] = {idx, DFLASH_TAPE_QKV};
        }
    }
    capture->tape_layers.resize(capture->recurrent_layer_ids.size());
}

// ---------------------------------------------------------------------------
// dflash_set_tape_recording
// ---------------------------------------------------------------------------
//
// Toggles tape recording on/off.  Unlike the bee method, this free function
// only touches capture state.  The caller (llama_context) is responsible for
// propagating tape_gpu / tape_gpu_n_seqs into cparams and allocating GPU
// tape buffers via allocate_tape_gpu().

void dflash_set_tape_recording(
        dflash_capture_data * capture,
        bool                  enable) {
    if (!capture) {
        return;
    }

    capture->tape_enabled = enable;

    if (enable) {
        // Reset token counts on all existing tapes so a fresh decode begins
        // recording from position 0.
        for (auto & tape : capture->tapes) {
            if (tape) {
                tape->n_tokens = 0;
            }
        }
        for (auto & hidden : capture->hidden_gpu) {
            if (hidden) {
                hidden->n_tokens = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Forward declarations for internal helper functions used by tape_replay
// ---------------------------------------------------------------------------

static bool dtape_replay_gdn_direct_gpu(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_s);

static bool dtape_replay_gdn_direct_from_cpu_tape(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_s);

static bool dtape_replay_conv_gpu(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_r,
        bool                     advance_pos);

static bool dtape_replay_conv_gpu_from_cpu_tape(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_r,
        llama_seq_id             seq_id);

// Derive n_embd_s from the first valid s_l tensor (ne[0]).
// Returns 0 if no valid tensor found.
static uint32_t dtape_n_embd_s(const llama_memory_recurrent * mem,
                                const std::vector<int32_t>   & rec_ids) {
    for (int il : rec_ids) {
        ggml_tensor * s = mem->s_l[il];
        if (s && s->ne[0] > 0) { return (uint32_t) s->ne[0]; }
    }
    return 0;
}

// Derive n_embd_r from the first valid r_l tensor (ne[0]).
static uint32_t dtape_n_embd_r(const llama_memory_recurrent * mem,
                                const std::vector<int32_t>   & rec_ids) {
    for (int il : rec_ids) {
        ggml_tensor * r = mem->r_l[il];
        if (r && r->ne[0] > 0) { return (uint32_t) r->ne[0]; }
    }
    return 0;
}

// Detect multi-GPU: return true if any two recurrent state tensors live on
// different CUDA devices (by buffer pointer comparison).
static bool dtape_recurrent_is_multi_gpu(const llama_memory_recurrent * mem,
                                          const std::vector<int32_t>   & rec_ids) {
    ggml_backend_buffer_t first_buf = nullptr;
    for (int il : rec_ids) {
        ggml_tensor * s = mem->s_l[il];
        if (!s || !s->buffer) { continue; }
        if (!first_buf) { first_buf = s->buffer; continue; }
        if (s->buffer != first_buf) { return true; }
    }
    return false;
}

// ---------------------------------------------------------------------------
// dflash_tape_replay_cpu
// ---------------------------------------------------------------------------

void dflash_tape_replay_cpu(
        dflash_capture_data   * capture,
        llama_memory_recurrent * mem,
        int32_t                  cell_idx,
        int                      n_accepted) {
    if (!capture || !mem || n_accepted <= 0) {
        return;
    }

    const auto   & rec_ids  = capture->recurrent_layer_ids;
    const uint32_t n_embd_s = dtape_n_embd_s(mem, rec_ids);
    auto         & tape_layers = capture->tape_layers;

    if (capture->profile) {
        capture->profile_replay_cpu_fallback += 1;
        capture->profile_replay_layers += rec_ids.size();
    }

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers[li];

        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) { continue; }

        const int64_t S   = tape.S_k;
        const int64_t H_k = tape.H_k;
        const int64_t H_v = tape.H_v;
        if (S <= 0 || H_k <= 0 || H_v <= 0) { continue; }
        if ((size_t) S * (size_t) S * (size_t) H_v != (size_t) n_embd_s) { continue; }

        ggml_tensor * s_tensor = mem->s_l[il];
        if (!s_tensor) { continue; }

        const size_t s_offset   = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);
        const size_t state_bytes = (size_t) n_embd_s * ggml_element_size(s_tensor);
        if (!dtape_tensor_span_in_bounds(s_tensor, s_offset, state_bytes)) {
            LLAMA_LOG_WARN("%s: DFlash CPU recurrent replay state span out of bounds "
                    "(layer=%d tensor_bytes=%zu offset=%zu bytes=%zu cell=%d n_embd_s=%u)\n",
                    __func__, il, s_tensor ? ggml_nbytes(s_tensor) : (size_t) 0,
                    s_offset, state_bytes, cell_idx, n_embd_s);
            continue;
        }

        std::vector<float> state(n_embd_s);
        ggml_backend_tensor_get(s_tensor, state.data(), s_offset, n_embd_s * sizeof(float));

        for (int tok = 0; tok < n_accepted; ++tok) {
            for (int64_t hv = 0; hv < H_v; ++hv) {
                const int64_t hk = hv % H_k;
                float g_val = exp2f(tape.gate[tok * H_v + hv] * 1.442695041f);
                float b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv]));

                float       * S_h = state.data() + hv * S * S;
                const float * k_t = tape.k.data() + tok * (S * H_k) + hk * S;
                const float * v_t = tape.v.data() + tok * (S * H_v) + hv * S;

                // kv = S^T @ k, delta = (v - g*kv) * beta, S = g*S + k⊗delta (fused)
                for (int64_t col = 0; col < S; ++col) {
                    float kv = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        kv += S_h[col * S + row] * k_t[row];
                    }
                    float delta_col = (v_t[col] - g_val * kv) * b_val;
                    for (int64_t row = 0; row < S; ++row) {
                        S_h[col * S + row] = g_val * S_h[col * S + row] + k_t[row] * delta_col;
                    }
                }
            }
        }

        ggml_backend_tensor_set(s_tensor, state.data(), s_offset, n_embd_s * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// dflash_tape_replay_conv  (dispatcher)
// ---------------------------------------------------------------------------

void dflash_tape_replay_conv(
        dflash_capture_data              * capture,
        llama_memory_recurrent           * mem,
        const std::vector<ggml_backend_ptr> & backends,
        int32_t                            cell_idx,
        int                                n_accepted,
        llama_seq_id                       seq_id) {
    if (!capture || !mem || n_accepted <= 0) {
        return;
    }

    const auto   & rec_ids  = capture->recurrent_layer_ids;
    const uint32_t n_embd_r = dtape_n_embd_r(mem, rec_ids);
    auto         & tape_layers = capture->tape_layers;

    // Try GPU conv replay paths first.
    if (dtape_replay_conv_gpu(capture, mem, cell_idx, n_accepted, n_embd_r, /*advance_pos=*/true)) {
        return;
    }
    if (dtape_recurrent_is_multi_gpu(mem, rec_ids) &&
            dtape_replay_conv_gpu_from_cpu_tape(capture, mem, cell_idx, n_accepted, n_embd_r, seq_id)) {
        return;
    }

    // CPU fallback: batch async reads → single sync → CPU math → batch async writes → sync.
    struct conv_layer_data {
        size_t tape_li;
        ggml_tensor * r_tensor;
        size_t r_offset;
        int64_t conv_ch;
        int64_t conv_window;
        std::vector<float> old_window;
        std::vector<float> qkv_mixed;
        std::vector<float> new_conv;
    };
    std::vector<conv_layer_data> layers;
    layers.reserve(rec_ids.size());

    ggml_backend_t gpu_backend = dtape_first_gpu_backend(backends);
    // Use async ops only on single-GPU paths; multi-GPU state tensors on different
    // devices need synchronous reads/writes (no DeviceToDevice transfers via async).
    const bool use_async_backend = gpu_backend && !dtape_recurrent_is_multi_gpu(mem, rec_ids);

    auto get_tensor_data = [&](const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
        if (use_async_backend) {
            ggml_backend_tensor_get_async(gpu_backend, tensor, data, offset, size);
        } else {
            ggml_backend_tensor_get(tensor, data, offset, size);
        }
    };
    auto set_tensor_data = [&](ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        if (use_async_backend) {
            ggml_backend_tensor_set_async(gpu_backend, tensor, data, offset, size);
        } else {
            ggml_backend_tensor_set(tensor, data, offset, size);
        }
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers[li];
        dflash_tape_gpu       * gpu_tape  = capture->active_tape();
        dflash_tape_gpu_layer * gpu_layer = nullptr;
        if (gpu_tape && li < gpu_tape->layers.size() &&
                n_accepted <= gpu_tape->max_tokens &&
                n_accepted <= gpu_tape->n_tokens) {
            gpu_layer = &gpu_tape->layers[li];
        }

        if (!mem->r_l[il]) { continue; }
        const bool use_gpu_qkv = gpu_backend && gpu_layer && gpu_layer->qkv;
        if (!use_gpu_qkv) {
            if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) { continue; }
            if (tape.qkv_mixed.empty()) { continue; }
        }

        size_t qkv_seq_offset = 0;
        if (!use_gpu_qkv && tape.n_seqs > 1) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                if (tape.seq_ids[s] == seq_id) { found = true; break; }
                qkv_seq_offset += (size_t) tape.n_tokens * (size_t) tape.conv_channels;
            }
            GGML_ASSERT(found && "dflash_tape_replay_conv: seq_id not found in tape");
        }

        ggml_tensor * r_tensor = mem->r_l[il];
        const size_t r_offset = (size_t) cell_idx * n_embd_r * ggml_element_size(r_tensor);
        const int64_t conv_ch = use_gpu_qkv ? gpu_layer->qkv->ne[0] : tape.conv_channels;
        GGML_ASSERT(conv_ch > 0 && n_embd_r % conv_ch == 0);
        const int64_t conv_window = (int64_t)(n_embd_r / conv_ch);

        conv_layer_data & d = layers.emplace_back();
        d.tape_li    = li;
        d.r_tensor   = r_tensor;
        d.r_offset   = r_offset;
        d.conv_ch    = conv_ch;
        d.conv_window = conv_window;
        d.old_window.resize(n_embd_r);
        d.qkv_mixed.resize((size_t) n_accepted * (size_t) conv_ch);
        d.new_conv.resize(n_embd_r);

        get_tensor_data(r_tensor, d.old_window.data(), r_offset, n_embd_r * sizeof(float));
        if (use_gpu_qkv) {
            get_tensor_data(gpu_layer->qkv, d.qkv_mixed.data(), 0, d.qkv_mixed.size() * sizeof(float));
        } else {
            std::memcpy(d.qkv_mixed.data(),
                        tape.qkv_mixed.data() + qkv_seq_offset,
                        d.qkv_mixed.size() * sizeof(float));
        }
    }

    // Phase 2: single sync point for all reads
    if (use_async_backend && !layers.empty()) {
        const int64_t t_start_us = capture->profile ? ggml_time_us() : 0;
        ggml_backend_synchronize(gpu_backend);
        if (capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            capture->profile_conv_read_wait_us  += elapsed;
            capture->profile_replay_conv_wait_us += elapsed;
        }
    }

    // Phase 3: CPU conv rebuilds
    const int64_t t_cpu_start_us = capture->profile ? ggml_time_us() : 0;
    for (auto & d : layers) {
        for (int64_t w = 0; w < d.conv_window; ++w) {
            int src_pos = n_accepted + (int) w;
            for (int64_t ch = 0; ch < d.conv_ch; ++ch) {
                float val;
                if (src_pos < (int) d.conv_window) {
                    val = d.old_window[ch * d.conv_window + src_pos];
                } else {
                    val = d.qkv_mixed[(src_pos - d.conv_window) * d.conv_ch + ch];
                }
                d.new_conv[ch * d.conv_window + w] = val;
            }
        }
    }
    if (capture->profile && !layers.empty()) {
        capture->profile_conv_cpu_us += ggml_time_us() - t_cpu_start_us;
    }

    // Phase 4: issue all async writes, then sync
    for (auto & d : layers) {
        set_tensor_data(d.r_tensor, d.new_conv.data(), d.r_offset, n_embd_r * sizeof(float));
    }
    if (use_async_backend && !layers.empty()) {
        const int64_t t_start_us = capture->profile ? ggml_time_us() : 0;
        ggml_backend_synchronize(gpu_backend);
        if (capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            capture->profile_conv_write_wait_us  += elapsed;
            capture->profile_replay_conv_wait_us += elapsed;
        }
    }

    mem->cells[cell_idx].pos += n_accepted;
}

// ---------------------------------------------------------------------------
// dflash_tape_replay  (main entry point)
// ---------------------------------------------------------------------------

void dflash_tape_replay(
        dflash_capture_data              * capture,
        llama_memory_recurrent           * mem_recurrent,
        const std::vector<ggml_backend_ptr> & backends,
        llama_seq_id                       seq_id,
        int                                n_accepted) {
    if (!capture || n_accepted <= 0) {
        return;
    }

    // Ensure any previous async replay is complete before launching a new one.
    dflash_tape_replay_sync(capture, backends);

    dflash_tape_gpu * const gpu_tape = capture->active_tape();
    const bool use_gpu_tape = (gpu_tape != nullptr &&
                               n_accepted <= gpu_tape->max_tokens &&
                               n_accepted <= gpu_tape->n_tokens);

    if (!use_gpu_tape && capture->tape_layers.empty()) {
        return;
    }

    if (!mem_recurrent) {
        LLAMA_LOG_WARN("%s: tape replay requires recurrent memory\n", __func__);
        return;
    }

    const auto   & rec_ids   = capture->recurrent_layer_ids;
    const uint32_t n_embd_s  = dtape_n_embd_s(mem_recurrent, rec_ids);
    auto         & tape_layers = capture->tape_layers;

    // Find the tail cell for this seq_id.
    int32_t cell_idx = -1;
    if (seq_id >= 0 && (uint32_t) seq_id < mem_recurrent->size) {
        int32_t tail = mem_recurrent->cells[seq_id].tail;
        if (tail >= 0) {
            cell_idx = tail;
        }
    }
    if (cell_idx < 0) {
        LLAMA_LOG_WARN("%s: no active cell for seq %d\n", __func__, seq_id);
        return;
    }

    // Find first GPU backend.
    ggml_backend_t gpu_backend = dtape_first_gpu_backend(backends);

    if (!gpu_backend) {
        dflash_tape_replay_cpu(capture, mem_recurrent, cell_idx, n_accepted);
        dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
        return;
    }

    // Partial offload: if any recurrent layer's state is host memory, fall back to CPU.
    for (int li = 0; li < (int) rec_ids.size(); ++li) {
        ggml_tensor * s_tensor = mem_recurrent->s_l[rec_ids[li]];
        if (s_tensor && s_tensor->buffer && ggml_backend_buffer_is_host(s_tensor->buffer)) {
            dflash_tape_replay_cpu(capture, mem_recurrent, cell_idx, n_accepted);
            dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
            return;
        }
    }

    // Try direct GPU GDN replay (pure CUDA, zero ggml graph overhead).
    if (use_gpu_tape && dtape_replay_gdn_direct_gpu(capture, mem_recurrent, cell_idx, n_accepted, n_embd_s)) {
        capture->replay_pending         = true;
        capture->replay_gpu_backend     = nullptr; // direct path; no ggml backend to sync
        capture->replay_graph_ctx       = nullptr;
        capture->replay_direct_gpu      = true;
        capture->replay_n_accepted      = n_accepted;
        capture->replay_cell_idx        = cell_idx;
        capture->replay_seq_id          = seq_id;
        capture->replay_mem_recurrent   = mem_recurrent;
        return;
    }

    const bool multi_gpu_target = dtape_recurrent_is_multi_gpu(mem_recurrent, rec_ids);
    if (multi_gpu_target) {
        if (dtape_replay_gdn_direct_from_cpu_tape(capture, mem_recurrent, cell_idx, n_accepted, n_embd_s)) {
            dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
            return;
        }
        if (!capture->multi_gpu_replay_fallback_logged) {
            LLAMA_LOG_WARN("%s: multi-GPU target detected; exact CUDA DFlash replay unavailable,"
                    " using CPU recurrent replay fallback\n", __func__);
            capture->multi_gpu_replay_fallback_logged = true;
        }
        dflash_tape_replay_cpu(capture, mem_recurrent, cell_idx, n_accepted);
        dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
        return;
    }

    // GPU tape replay: build a ggml graph with GDN ops for all recurrent layers.
    const int n_rec = (int) rec_ids.size();
    if (n_rec == 0) {
        dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
        return;
    }

    {
        // Per layer: k_view + v_view + g_view + b_view + q + b_sigmoid + s_view + GDN
        //            + result_state + s_write + cpy = ~11 tensors
        size_t ctx_mem = ggml_tensor_overhead() * ((size_t) n_rec * 14 + 4)
                         + ggml_graph_overhead_custom((size_t) n_rec * 12, false);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * ctx = ggml_init(ctx_params);

        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, (size_t) n_rec * 12, false);

        struct replay_input {
            ggml_tensor * q;
            ggml_tensor * k;
            ggml_tensor * v;
            ggml_tensor * g;
            ggml_tensor * b;
            size_t tape_li;
        };
        std::vector<replay_input> inputs;
        inputs.reserve(n_rec);
        bool replay_graph_unsafe = false;

        for (int li = 0; li < n_rec; ++li) {
            int il = rec_ids[li];

            int64_t S, H_k, H_v;
            if (use_gpu_tape) {
                auto & tl = gpu_tape->layers[li];
                S   = tl.k->ne[0];
                H_k = tl.k->ne[1];
                H_v = tl.v->ne[1];
            } else {
                auto & tape = tape_layers[li];
                if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) { continue; }
                S   = tape.S_k;
                H_k = tape.H_k;
                H_v = tape.H_v;
            }

            if (S <= 0 || H_k <= 0 || H_v <= 0) { continue; }

            if (!dtape_replay_gdn_supported_s(S) ||
                    !dtape_replay_state_shape_valid(S, H_v, n_embd_s)) {
                LLAMA_LOG_WARN("%s: DFlash recurrent replay view out of bounds or unsupported shape "
                        "(layer=%d S=%lld H_v=%lld n_embd_s=%u use_gpu_tape=%d); %s\n",
                        __func__, il, (long long) S, (long long) H_v, n_embd_s,
                        use_gpu_tape ? 1 : 0,
                        use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }

            ggml_tensor * k_in;
            ggml_tensor * v_in;
            ggml_tensor * g_in;
            ggml_tensor * b_in;

            if (use_gpu_tape) {
                auto & tl = gpu_tape->layers[li];
                k_in = ggml_view_4d(ctx, tl.k, S, H_k, (int64_t) n_accepted, (int64_t) 1,
                    tl.k->nb[1], tl.k->nb[2], tl.k->nb[2] * n_accepted, 0);
                v_in = ggml_view_4d(ctx, tl.v, S, H_v, (int64_t) n_accepted, (int64_t) 1,
                    tl.v->nb[1], tl.v->nb[2], tl.v->nb[2] * n_accepted, 0);
                g_in = ggml_view_4d(ctx, tl.gate, (int64_t) 1, H_v, (int64_t) n_accepted, (int64_t) 1,
                    tl.gate->nb[1], tl.gate->nb[2], tl.gate->nb[2] * n_accepted, 0);
                b_in = ggml_view_4d(ctx, tl.beta, (int64_t) 1, H_v, (int64_t) n_accepted, (int64_t) 1,
                    tl.beta->nb[1], tl.beta->nb[2], tl.beta->nb[2] * n_accepted, 0);
            } else {
                k_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t) n_accepted, (int64_t) 1);
                v_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, (int64_t) n_accepted, (int64_t) 1);
                g_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) 1, H_v, (int64_t) n_accepted, (int64_t) 1);
                b_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) 1, H_v, (int64_t) n_accepted, (int64_t) 1);
                ggml_set_input(k_in); ggml_set_input(v_in);
                ggml_set_input(g_in); ggml_set_input(b_in);
            }

            // Q: zeros (attention output discarded; only state update matters)
            ggml_tensor * q_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t) n_accepted, (int64_t) 1);
            ggml_set_input(q_in);

            // State view: reads directly from recurrent memory (zero-copy).
            ggml_tensor * s_tensor = mem_recurrent->s_l[il];
            if (!s_tensor) {
                LLAMA_LOG_WARN("%s: missing recurrent state tensor for DFlash replay layer=%d; %s\n",
                        __func__, il, use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }
            const size_t state_bytes  = (size_t) n_embd_s * ggml_element_size(s_tensor);
            const size_t s_byte_offset = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);
            if (!dtape_tensor_span_in_bounds(s_tensor, s_byte_offset, state_bytes)) {
                LLAMA_LOG_WARN("%s: DFlash recurrent replay view out of bounds "
                        "(layer=%d tensor_bytes=%zu offset=%zu bytes=%zu cell=%d n_embd_s=%u); %s\n",
                        __func__, il, s_tensor ? ggml_nbytes(s_tensor) : (size_t) 0,
                        s_byte_offset, state_bytes, cell_idx, n_embd_s,
                        use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }
            ggml_tensor * s_view = ggml_view_4d(ctx, s_tensor, S, S, H_v, (int64_t) 1,
                S * ggml_element_size(s_tensor),
                S * S * ggml_element_size(s_tensor),
                S * S * H_v * ggml_element_size(s_tensor),
                s_byte_offset);

            // GDN op — same kernel as forward pass.
            // K=1: keep only the final state snapshot (slot 0).
            // beta is passed raw; the op applies sigmoid internally.
            ggml_tensor * result = ggml_gated_delta_net(ctx, q_in, k_in, v_in, g_in, b_in, s_view, (int64_t) 1);

            // Extract state from result (layout: [attn_output | K=1 state snapshot]).
            const size_t attn_bytes        = (size_t)(S * H_v * n_accepted) * ggml_element_size(result);
            const size_t result_state_bytes = (size_t) n_embd_s * ggml_element_size(result);
            if (!dtape_tensor_span_in_bounds(result, attn_bytes, result_state_bytes)) {
                LLAMA_LOG_WARN("%s: DFlash recurrent replay view out of bounds "
                        "(layer=%d result_bytes=%zu offset=%zu bytes=%zu S=%lld H_v=%lld n_accepted=%d); %s\n",
                        __func__, il, ggml_nbytes(result), attn_bytes, result_state_bytes,
                        (long long) S, (long long) H_v, n_accepted,
                        use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }
            ggml_tensor * result_state = ggml_view_1d(ctx, result, n_embd_s, attn_bytes);

            // Write-back view pointing at the same location in s_l[il].
            ggml_tensor * s_write = ggml_view_1d(ctx, s_tensor, n_embd_s, s_byte_offset);

            // GPU→GPU copy.
            ggml_tensor * cpy = ggml_cpy(ctx, result_state, s_write);
            ggml_build_forward_expand(graph, cpy);

            inputs.push_back({ q_in, k_in, v_in, g_in, b_in, (size_t) li });
        }

        if (replay_graph_unsafe) {
            ggml_free(ctx);
            if (!use_gpu_tape) {
                dflash_tape_replay_cpu(capture, mem_recurrent, cell_idx, n_accepted);
                dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
            } else {
                // GPU tape mode: can't fall back to CPU tape; just do conv.
                dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
            }
            return;
        }

        if (inputs.empty()) {
            ggml_free(ctx);
            dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
            return;
        }

        // Allocate non-view tensors on GPU, reusing a persistent buffer.
        ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
        size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, gpu_buft);

        if (needed > capture->replay_buf_size) {
            if (capture->replay_buf) {
                ggml_backend_buffer_free(capture->replay_buf);
            }
            capture->replay_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
            capture->replay_buf_size = capture->replay_buf
                ? ggml_backend_buffer_get_size(capture->replay_buf) : 0;
        }

        if (!capture->replay_buf) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU buffer for tape replay, falling back to CPU\n", __func__);
            ggml_free(ctx);
            dflash_tape_replay_cpu(capture, mem_recurrent, cell_idx, n_accepted);
            dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
            return;
        }

        // Assign tensors within the persistent buffer.
        {
            struct ggml_tallocr talloc = ggml_tallocr_new(capture->replay_buf);
            struct ggml_tensor * t = ggml_get_first_tensor(ctx);
            while (t) {
                if (t->data == nullptr && t->view_src == nullptr) {
                    ggml_tallocr_alloc(&talloc, t);
                } else if (t->view_src != nullptr && t->buffer == nullptr) {
                    ggml_backend_view_init(t);
                }
                t = ggml_get_next_tensor(ctx, t);
            }
        }

        // Upload data for inputs.
        for (auto & inp : inputs) {
            // Q: always zero (attention output discarded).
            {
                const int64_t Sq = inp.q->ne[0];
                const int64_t Hq = inp.q->ne[1];
                size_t q_size = (size_t)(Sq * Hq * n_accepted);
                if (capture->replay_zeros.size() < q_size) {
                    capture->replay_zeros.resize(q_size, 0.0f);
                }
                ggml_backend_tensor_set(inp.q, capture->replay_zeros.data(), 0, ggml_nbytes(inp.q));
            }

            // K, V, gate, beta: only upload from CPU tape if not using GPU tape.
            if (!use_gpu_tape) {
                auto & tape = tape_layers[inp.tape_li];
                const int64_t Si  = tape.S_k;
                const int64_t Hki = tape.H_k;
                const int64_t Hvi = tape.H_v;
                ggml_backend_tensor_set(inp.k, tape.k.data(), 0, Si * Hki * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.v, tape.v.data(), 0, Si * Hvi * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.g, tape.gate.data(), 0, Hvi * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.b, tape.beta.data(), 0, Hvi * n_accepted * sizeof(float));
            }
        }

        // Launch GDN ops + state copies asynchronously on the GPU.
        const int64_t t_replay_enqueue_us = capture->profile ? ggml_time_us() : 0;
        const ggml_status replay_status = ggml_backend_graph_compute_async(gpu_backend, graph);
        if (replay_status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_WARN("%s: GPU DFlash recurrent replay graph failed with status %d; %s\n",
                    __func__, (int) replay_status,
                    use_gpu_tape ? "CPU fallback unavailable for GPU tape" : "falling back to CPU");
            ggml_free(ctx);
            if (!use_gpu_tape) {
                dflash_tape_replay_cpu(capture, mem_recurrent, cell_idx, n_accepted);
                dflash_tape_replay_conv(capture, mem_recurrent, backends, cell_idx, n_accepted, seq_id);
            }
            return;
        }
        if (capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_replay_enqueue_us;
            capture->profile_replay_gdn_enqueue_us += elapsed;
            capture->profile_conv_gpu_us           += elapsed;
            capture->profile_replay_ggml_gpu       += 1;
            capture->profile_replay_layers         += inputs.size();
        }

        // Save deferred state for async completion.
        capture->replay_pending         = true;
        capture->replay_gpu_backend     = gpu_backend;
        capture->replay_graph_ctx       = ctx; // freed in dflash_tape_replay_sync
        capture->replay_direct_gpu      = false;
        capture->replay_sync_device     = -1;
        capture->replay_n_accepted      = n_accepted;
        capture->replay_cell_idx        = cell_idx;
        capture->replay_seq_id          = seq_id;
        capture->replay_mem_recurrent   = mem_recurrent;
        return; // conv rebuild deferred to dflash_tape_replay_sync()
    }
}

// ---------------------------------------------------------------------------
// dflash_tape_replay_sync
// ---------------------------------------------------------------------------

void dflash_tape_replay_sync(
        dflash_capture_data              * capture,
        const std::vector<ggml_backend_ptr> & backends) {
    if (!capture || !capture->replay_pending) {
        return;
    }

    auto * backend = capture->replay_gpu_backend;
    if (backend) {
        // Use a CUDA event for fine-grained sync instead of a full stream synchronize.
        const int64_t t_start_us = capture->profile ? ggml_time_us() : 0;
        if (!capture->replay_event) {
            auto * dev = ggml_backend_get_device(backend);
            if (dev) {
                capture->replay_event = ggml_backend_event_new(dev);
            }
        }
        if (capture->replay_event) {
            ggml_backend_event_record(capture->replay_event, backend);
            ggml_backend_event_synchronize(capture->replay_event);
        } else {
            ggml_backend_synchronize(backend);
        }
        if (capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            capture->profile_replay_wait_us     += elapsed;
            capture->profile_replay_gdn_wait_us += elapsed;
            capture->profile_replay_sync_calls  += 1;
        }
    } else if (capture->replay_direct_gpu &&
            (!capture->replay_sync_ptrs.empty() || capture->replay_sync_ptr)) {
        const int64_t t_start_us = capture->profile ? ggml_time_us() : 0;
        ggml_backend_reg_t cuda_reg = dtape_gpu_backend_reg();
        using sync_ptr_fn_t    = bool (*)(const void *);
        using sync_device_fn_t = bool (*)(int);
        auto fn_sync_ptr = cuda_reg
            ? (sync_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr")
            : nullptr;
        auto fn_sync_device = cuda_reg
            ? (sync_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_device")
            : nullptr;
        bool synced = false;
        uint64_t sync_calls = 0;
        if (fn_sync_device && capture->replay_sync_device >= 0) {
            synced = fn_sync_device(capture->replay_sync_device);
            sync_calls = 1;
        } else if (fn_sync_ptr) {
            if (capture->replay_sync_ptr) {
                synced = fn_sync_ptr(capture->replay_sync_ptr);
                sync_calls = 1;
            } else if (!capture->replay_sync_ptrs.empty()) {
                synced = true;
                for (const void * ptr : capture->replay_sync_ptrs) {
                    synced = fn_sync_ptr(ptr) && synced;
                    sync_calls++;
                }
            } else {
                synced = false;
            }
        }
        if (!synced) {
            LLAMA_LOG_WARN("%s: direct GPU tape replay sync failed\n", __func__);
        }
        if (capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            capture->profile_replay_wait_us     += elapsed;
            capture->profile_replay_gdn_wait_us += elapsed;
            capture->profile_replay_sync_calls  += sync_calls;
        }
    }

    // Free the graph context (if any).
    if (capture->replay_graph_ctx) {
        ggml_free(capture->replay_graph_ctx);
        capture->replay_graph_ctx = nullptr;
    }

    // Finish conv rebuild + position advance.
    dflash_tape_replay_conv(capture,
                            capture->replay_mem_recurrent,
                            backends,
                            capture->replay_cell_idx,
                            capture->replay_n_accepted,
                            capture->replay_seq_id);

    if (dflash_profile_has(capture->profile_flags, DFLASH_PROFILE_REPLAY)) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: replay_path=direct-gpu:%" PRIu64 " replay_path=ggml-gpu:%" PRIu64
            " replay_path=cpu-fallback:%" PRIu64 " replay_layers=%" PRIu64 " replay_sync_calls=%" PRIu64
            " gdn_enqueue=%.3f ms gdn_wait=%.3f ms conv_enqueue=%.3f ms conv_wait=%.3f ms "
            "legacy_replay_wait=%.3f ms legacy_conv_gpu_enqueue=%.3f ms "
            "legacy_conv_read_wait=%.3f ms legacy_conv_write_wait=%.3f ms conv_cpu=%.3f ms\n",
            __func__,
            capture->profile_replay_direct_gpu,
            capture->profile_replay_ggml_gpu,
            capture->profile_replay_cpu_fallback,
            capture->profile_replay_layers,
            capture->profile_replay_sync_calls,
            capture->profile_replay_gdn_enqueue_us  / 1000.0,
            capture->profile_replay_gdn_wait_us     / 1000.0,
            capture->profile_replay_conv_enqueue_us / 1000.0,
            capture->profile_replay_conv_wait_us    / 1000.0,
            capture->profile_replay_wait_us         / 1000.0,
            capture->profile_conv_gpu_us            / 1000.0,
            capture->profile_conv_read_wait_us      / 1000.0,
            capture->profile_conv_write_wait_us     / 1000.0,
            capture->profile_conv_cpu_us            / 1000.0);
    }

    capture->replay_pending     = false;
    capture->replay_direct_gpu  = false;
    capture->replay_sync_ptr    = nullptr;
    capture->replay_sync_ptrs.clear();
    capture->replay_sync_device = -1;
}

// ---------------------------------------------------------------------------
// dflash_rollback
// ---------------------------------------------------------------------------

void dflash_rollback(
        dflash_capture_data              * capture,
        llama_memory_i                   * mem_attn,
        llama_memory_recurrent           * mem_recr,
        const std::vector<ggml_backend_ptr> & backends,
        llama_seq_id                       seq_id,
        llama_seq_id                       seq_backup,
        int                                n_past_before,
        int                                n_accepted,
        bool                               tree_mode) {
    if (!mem_recr) {
        LLAMA_LOG_WARN("%s: dflash_rollback requires recurrent memory\n", __func__);
        return;
    }

    const bool profile = capture && dflash_profile_has(capture->profile_flags, DFLASH_PROFILE_COPY);
    const int64_t t_start_us = profile ? ggml_time_us() : 0;
    int64_t t_phase_us = t_start_us;
    int64_t attn_us = 0;
    int64_t recurrent_restore_us = 0;
    int64_t tape_launch_us = 0;
    llama_memory_recurrent_copy_profile recurrent_restore_profile = {};
    auto profile_lap = [&](int64_t & dst) {
        if (!profile) { return; }
        const int64_t now = ggml_time_us();
        dst += now - t_phase_us;
        t_phase_us = now;
    };

    if (mem_attn) {
        if (tree_mode) {
            // Tree mode: branch tokens may have polluted KV at accepted positions.
            // Remove ALL entries from n_past_before onwards, then restore from backup.
            mem_attn->seq_rm(seq_id, n_past_before, -1);
            mem_attn->seq_cp(seq_backup, seq_id, n_past_before, -1);
            mem_attn->seq_rm(seq_backup, -1, -1);
        } else {
            // Flat mode: no duplicate entries at the same position; keep accepted KV.
            int kv_keep_pos = n_past_before + n_accepted;
            mem_attn->seq_rm(seq_id, kv_keep_pos, -1);
        }
    }
    profile_lap(attn_us);

    // Restore recurrent state from backup, then tape replay.
    mem_recr->seq_rm(seq_id, -1, -1);
    if (profile) {
        mem_recr->recurrent_copy_profile_reset();
    }
    mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1);
    if (profile) {
        recurrent_restore_profile = mem_recr->recurrent_copy_profile();
    }
    mem_recr->seq_rm(seq_backup, -1, -1);
    profile_lap(recurrent_restore_us);

    // Replay DeltaNet state updates for accepted tokens.
    dflash_tape_replay(capture, mem_recr, backends, seq_id, n_accepted);
    profile_lap(tape_launch_us);

    if (profile) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: rollback accepted=%d attn=%.3f ms recurrent_restore=%.3f ms "
            "rollback_restore_enqueue=%.3f ms rollback_restore_sync=%.3f ms "
            "rollback_restore_layers=%" PRIu64 " rollback_restore_tensors=%" PRIu64
            " rollback_restore_cuda_d2d=%" PRIu64 " rollback_restore_fallback=%" PRIu64
            " tape_launch=%.3f ms total=%.3f ms\n",
            __func__, n_accepted,
            attn_us / 1e3,
            recurrent_restore_us / 1e3,
            recurrent_restore_profile.enqueue_us / 1e3,
            recurrent_restore_profile.sync_us / 1e3,
            recurrent_restore_profile.layers_scanned,
            recurrent_restore_profile.tensors_copied,
            recurrent_restore_profile.cuda_d2d_queued,
            recurrent_restore_profile.fallback_copies,
            tape_launch_us / 1e3,
            (ggml_time_us() - t_start_us) / 1e3);
    }
}

// ---------------------------------------------------------------------------
// dtape_replay_gdn_direct_gpu  (internal)
// ---------------------------------------------------------------------------
//
// Directly invokes the low-level CUDA GDN replay kernel via proc_address
// without building a ggml graph.  Returns true on success.

static bool dtape_replay_gdn_direct_gpu(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_s) {
    if (!capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dtape_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using ptr_device_fn_t = bool (*)(const void *, int *);
    using prepare_ptr_fn_t = bool (*)(const void *);
    using replay_fn_t = bool (*)(void *, const void *, const void *, const void *, const void *, int, int, int, int);
    auto fn_ptr_device = (ptr_device_fn_t)  ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_ptr_device");
    auto fn_prepare    = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_replay     = (replay_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_replay_gdn_state_no_check");
    if (!fn_ptr_device || !fn_prepare || !fn_replay) {
        return false;
    }

    dflash_tape_gpu * gpu_tape = capture->active_tape();
    if (!gpu_tape || n_accepted > gpu_tape->max_tokens || n_accepted > gpu_tape->n_tokens) {
        return false;
    }

    const auto & rec_ids = capture->recurrent_layer_ids;

    struct replay_launch {
        void       * state;
        const void * k;
        const void * v;
        const void * gate;
        const void * beta;
        int S;
        int H_k;
        int H_v;
        int device;
    };
    std::vector<replay_launch> launches;
    launches.reserve(rec_ids.size());

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        const int il = rec_ids[li];
        if (li >= gpu_tape->layers.size()) {
            return false;
        }

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        auto & tl = gpu_tape->layers[li];
        if (!dtape_is_cuda_compatible_tensor(s_tensor) ||
                !dtape_is_cuda_compatible_tensor(tl.k) ||
                !dtape_is_cuda_compatible_tensor(tl.v) ||
                !dtape_is_cuda_compatible_tensor(tl.gate) ||
                !dtape_is_cuda_compatible_tensor(tl.beta)) {
            return false;
        }

        const int64_t S_i64   = tl.v->ne[0];
        const int64_t H_v_i64 = tl.v->ne[1];
        const int64_t H_k_i64 = tl.k->ne[1];
        if (tl.k->ne[0] != S_i64 || tl.gate->ne[0] != 1 || tl.beta->ne[0] != 1 ||
                tl.gate->ne[1] != H_v_i64 || tl.beta->ne[1] != H_v_i64 ||
                S_i64 <= 0 || H_k_i64 <= 0 || H_v_i64 <= 0 ||
                S_i64 > std::numeric_limits<int>::max() ||
                H_k_i64 > std::numeric_limits<int>::max() ||
                H_v_i64 > std::numeric_limits<int>::max()) {
            return false;
        }
        if (!dtape_replay_gdn_supported_s(S_i64)) {
            return false;
        }
        if (!dtape_replay_state_shape_valid(S_i64, H_v_i64, n_embd_s)) {
            return false;
        }

        const size_t s_offset   = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);
        const size_t state_bytes = (size_t) n_embd_s * ggml_element_size(s_tensor);
        if (!dtape_tensor_span_in_bounds(s_tensor, s_offset, state_bytes)) {
            return false;
        }
        launches.push_back({
            (char *) s_tensor->data + s_offset,
            tl.k->data,
            tl.v->data,
            tl.gate->data,
            tl.beta->data,
            (int) S_i64,
            (int) H_k_i64,
            (int) H_v_i64,
            -1,
        });
    }

    if (launches.empty()) {
        return false;
    }

    int  replay_device       = -1;
    bool replay_mixed_devices = false;
    for (auto & launch : launches) {
        int device = -1;
        if (!fn_ptr_device(launch.state, &device)) { return false; }
        int k_device = -1, v_device = -1, gate_device = -1, beta_device = -1;
        if (!fn_ptr_device(launch.k,    &k_device)    ||
                !fn_ptr_device(launch.v,    &v_device)    ||
                !fn_ptr_device(launch.gate, &gate_device) ||
                !fn_ptr_device(launch.beta, &beta_device)) {
            return false;
        }
        if (k_device != device || v_device != device || gate_device != device || beta_device != device) {
            return false;
        }
        launch.device = device;
        if (!replay_mixed_devices && replay_device < 0) {
            replay_device = device;
        } else if (!replay_mixed_devices && device != replay_device) {
            replay_device       = -1;
            replay_mixed_devices = true;
        }
    }

    const int64_t t_start_us = capture->profile ? ggml_time_us() : 0;
    capture->replay_sync_ptrs.clear();
    capture->replay_sync_device = replay_mixed_devices ? -1 : replay_device;
    for (const auto & launch : launches) {
        if (!fn_prepare(launch.state) ||
                !fn_replay(launch.state, launch.k, launch.v, launch.gate, launch.beta,
                    n_accepted, launch.S, launch.H_k, launch.H_v)) {
            GGML_ABORT("DFlash direct GPU GDN replay launch failed after validation\n");
        }
        if (replay_mixed_devices) {
            capture->replay_sync_ptrs.push_back(launch.state);
        }
    }
    if (capture->profile) {
        const uint64_t elapsed = ggml_time_us() - t_start_us;
        capture->profile_replay_gdn_enqueue_us += elapsed;
        capture->profile_conv_gpu_us           += elapsed;
        capture->profile_replay_direct_gpu     += 1;
        capture->profile_replay_layers         += launches.size();
    }
    capture->replay_sync_ptr = replay_mixed_devices ? nullptr : launches.back().state;
    return true;
}

// ---------------------------------------------------------------------------
// dtape_replay_gdn_direct_from_cpu_tape  (internal)
// ---------------------------------------------------------------------------
//
// Multi-GPU variant: uploads CPU tape data to GPU per-layer device, runs the
// CUDA GDN kernel, then synchronises and frees temporaries.

static bool dtape_replay_gdn_direct_from_cpu_tape(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_s) {
    if (!capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dtape_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using prepare_ptr_fn_t = bool (*)(const void *);
    using replay_fn_t      = bool (*)(void *, const void *, const void *, const void *, const void *, int, int, int, int);
    using sync_ptr_fn_t    = bool (*)(const void *);
    auto fn_prepare = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_replay  = (replay_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_replay_gdn_state_no_check");
    auto fn_sync    = (sync_ptr_fn_t)    ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    if (!fn_prepare || !fn_replay || !fn_sync) {
        return false;
    }

    const auto & rec_ids    = capture->recurrent_layer_ids;
    auto       & tape_layers = capture->tape_layers;
    if (rec_ids.empty() || tape_layers.empty()) {
        return false;
    }

    struct replay_upload {
        ggml_context         * ctx  = nullptr;
        ggml_backend_buffer_t  buf  = nullptr;
        void                 * state = nullptr;
        ggml_tensor          * k    = nullptr;
        ggml_tensor          * v    = nullptr;
        ggml_tensor          * gate = nullptr;
        ggml_tensor          * beta = nullptr;
        int S   = 0;
        int H_k = 0;
        int H_v = 0;
    };
    std::vector<replay_upload> uploads;
    uploads.reserve(rec_ids.size());

    auto cleanup = [&]() {
        for (auto & u : uploads) {
            if (u.buf) { ggml_backend_buffer_free(u.buf); u.buf = nullptr; }
            if (u.ctx) { ggml_free(u.ctx);                u.ctx = nullptr; }
        }
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        if (li >= tape_layers.size()) {
            cleanup();
            return false;
        }

        const int il = rec_ids[li];
        const auto & tape = tape_layers[li];
        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) {
            cleanup();
            return false;
        }

        const int64_t S_i64   = tape.S_k;
        const int64_t H_k_i64 = tape.H_k;
        const int64_t H_v_i64 = tape.H_v;
        if (S_i64 <= 0 || H_k_i64 <= 0 || H_v_i64 <= 0 ||
                S_i64 > std::numeric_limits<int>::max() ||
                H_k_i64 > std::numeric_limits<int>::max() ||
                H_v_i64 > std::numeric_limits<int>::max()) {
            cleanup();
            return false;
        }
        if (!dtape_replay_gdn_supported_s(S_i64)) {
            cleanup();
            return false;
        }
        if ((size_t) S_i64 * (size_t) S_i64 * (size_t) H_v_i64 != (size_t) n_embd_s) {
            cleanup();
            return false;
        }

        const size_t k_elems  = (size_t) S_i64  * (size_t) H_k_i64 * (size_t) n_accepted;
        const size_t v_elems  = (size_t) S_i64  * (size_t) H_v_i64 * (size_t) n_accepted;
        const size_t gb_elems = (size_t) H_v_i64 * (size_t) n_accepted;
        if (tape.k.size() < k_elems || tape.v.size() < v_elems ||
                tape.gate.size() < gb_elems || tape.beta.size() < gb_elems) {
            cleanup();
            return false;
        }

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        if (!dtape_is_cuda_compatible_tensor(s_tensor)) {
            cleanup();
            return false;
        }

        const size_t ctx_mem = ggml_tensor_overhead() * 4;
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        ggml_context * ctx = ggml_init(ctx_params);
        if (!ctx) {
            cleanup();
            return false;
        }

        replay_upload upload;
        upload.ctx  = ctx;
        upload.k    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) k_elems);
        upload.v    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) v_elems);
        upload.gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) gb_elems);
        upload.beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) gb_elems);
        if (!upload.k || !upload.v || !upload.gate || !upload.beta) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(s_tensor->buffer);
        upload.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!upload.buf) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_tensor_set(upload.k,    tape.k.data(),    0, k_elems  * sizeof(float));
        ggml_backend_tensor_set(upload.v,    tape.v.data(),    0, v_elems  * sizeof(float));
        ggml_backend_tensor_set(upload.gate, tape.gate.data(), 0, gb_elems * sizeof(float));
        ggml_backend_tensor_set(upload.beta, tape.beta.data(), 0, gb_elems * sizeof(float));

        const size_t s_offset   = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);
        const size_t state_bytes = (size_t) n_embd_s * ggml_element_size(s_tensor);
        if (!dtape_tensor_span_in_bounds(s_tensor, s_offset, state_bytes)) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }
        upload.state = (char *) s_tensor->data + s_offset;
        upload.S     = (int) S_i64;
        upload.H_k   = (int) H_k_i64;
        upload.H_v   = (int) H_v_i64;
        uploads.push_back(upload);
    }

    if (uploads.empty()) {
        cleanup();
        return false;
    }

    for (const auto & u : uploads) {
        if (!fn_prepare(u.state)) {
            cleanup();
            return false;
        }
    }

    for (const auto & u : uploads) {
        if (!fn_replay(u.state, u.k->data, u.v->data, u.gate->data, u.beta->data,
                n_accepted, u.S, u.H_k, u.H_v)) {
            cleanup();
            GGML_ABORT("DFlash direct CPU-tape GDN replay launch failed after validation\n");
        }
    }

    bool synced = true;
    for (const auto & u : uploads) {
        synced = fn_sync(u.state) && synced;
    }
    cleanup();
    if (!synced) {
        GGML_ABORT("DFlash direct CPU-tape GDN replay sync failed after launch\n");
    }

    return true;
}

// ---------------------------------------------------------------------------
// dtape_replay_conv_gpu  (internal)
// ---------------------------------------------------------------------------
//
// GPU conv-state rebuild using the active GPU tape's QKV tensor.

static bool dtape_replay_conv_gpu(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_r,
        bool                     advance_pos) {
    if (!capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dtape_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using ptr_device_fn_t  = bool (*)(const void *, int *);
    using set_device_fn_t  = bool (*)(int);
    using sync_device_fn_t = bool (*)(int);
    using rebuild_fn_t     = bool (*)(void *, const void *, int, int, int);
    auto fn_ptr_device  = (ptr_device_fn_t)  ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_ptr_device");
    auto fn_set_device  = (set_device_fn_t)  ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_set_device");
    auto fn_sync_device = (sync_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_device");
    auto fn_rebuild     = (rebuild_fn_t)     ggml_backend_reg_get_proc_address(cuda_reg, "dflash_rebuild_conv_state");
    if (!fn_ptr_device || !fn_set_device || !fn_sync_device || !fn_rebuild) {
        return false;
    }

    dflash_tape_gpu * gpu_tape = capture->active_tape();
    if (!gpu_tape || n_accepted > gpu_tape->max_tokens || n_accepted > gpu_tape->n_tokens) {
        return false;
    }

    const auto & rec_ids = capture->recurrent_layer_ids;

    struct conv_launch {
        void       * r_state;
        const void * qkv;
        int conv_ch;
        int conv_window;
        int device;
    };
    std::vector<conv_launch> launches;
    launches.reserve(rec_ids.size());
    bool touched_devices[GGML_CUDA_MAX_DEVICES] = {};

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        const int il = rec_ids[li];
        if (li >= gpu_tape->layers.size()) {
            return false;
        }

        ggml_tensor * r_tensor   = mem_recurrent->r_l[il];
        ggml_tensor * qkv_tensor = gpu_tape->layers[li].qkv;
        if (!r_tensor || !r_tensor->data || !qkv_tensor || !qkv_tensor->data) {
            return false;
        }
        if (!dtape_is_cuda_compatible_tensor(r_tensor) ||
                !dtape_is_cuda_compatible_tensor(qkv_tensor)) {
            return false;
        }

        const int64_t conv_ch_i64 = qkv_tensor->ne[0];
        if (conv_ch_i64 <= 0 || n_embd_r % conv_ch_i64 != 0 ||
                conv_ch_i64 > std::numeric_limits<int>::max()) {
            return false;
        }

        const int64_t conv_window_i64 = n_embd_r / conv_ch_i64;
        if (conv_window_i64 <= 0 || conv_window_i64 > std::numeric_limits<int>::max()) {
            return false;
        }

        int r_dev   = -1;
        int qkv_dev = -1;
        if (!fn_ptr_device(r_tensor->data, &r_dev) ||
                !fn_ptr_device(qkv_tensor->data, &qkv_dev)) {
            return false;
        }
        if (r_dev < 0 || r_dev >= GGML_CUDA_MAX_DEVICES || qkv_dev != r_dev) {
            return false;
        }

        const size_t r_offset = (size_t) cell_idx * n_embd_r * ggml_element_size(r_tensor);
        launches.push_back({
            (char *) r_tensor->data + r_offset,
            qkv_tensor->data,
            (int) conv_ch_i64,
            (int) conv_window_i64,
            r_dev,
        });
        if (r_dev < GGML_CUDA_MAX_DEVICES) {
            touched_devices[r_dev] = true;
        }
    }

    if (launches.empty()) {
        return false;
    }

    const int64_t t_gpu_start_us = capture->profile ? ggml_time_us() : 0;
    for (const auto & launch : launches) {
        if (!fn_set_device(launch.device)) { return false; }
        if (!fn_rebuild(launch.r_state, launch.qkv, n_accepted, launch.conv_ch, launch.conv_window)) {
            return false;
        }
    }
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES; ++dev) {
        if (touched_devices[dev] && !fn_sync_device(dev)) {
            return false;
        }
    }
    if (capture->profile) {
        const uint64_t elapsed = ggml_time_us() - t_gpu_start_us;
        capture->profile_replay_conv_enqueue_us += elapsed;
        capture->profile_conv_gpu_us            += elapsed;
    }

    if (advance_pos) {
        mem_recurrent->cells[cell_idx].pos += n_accepted;
    }
    return true;
}

// ---------------------------------------------------------------------------
// dtape_replay_conv_gpu_from_cpu_tape  (internal)
// ---------------------------------------------------------------------------
//
// Multi-GPU variant of conv rebuild: uploads CPU QKV to each layer's device,
// then runs CUDA rebuild kernel and synchronises.

static bool dtape_replay_conv_gpu_from_cpu_tape(
        dflash_capture_data    * capture,
        llama_memory_recurrent * mem_recurrent,
        int32_t                  cell_idx,
        int                      n_accepted,
        uint32_t                 n_embd_r,
        llama_seq_id             seq_id) {
    if (!capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dtape_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using prepare_ptr_fn_t = bool (*)(const void *);
    using rebuild_fn_t     = bool (*)(void *, const void *, int, int, int);
    using sync_ptr_fn_t    = bool (*)(const void *);
    auto fn_prepare = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_rebuild = (rebuild_fn_t)     ggml_backend_reg_get_proc_address(cuda_reg, "dflash_rebuild_conv_state");
    auto fn_sync    = (sync_ptr_fn_t)    ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    if (!fn_prepare || !fn_rebuild || !fn_sync) {
        return false;
    }

    const auto & rec_ids    = capture->recurrent_layer_ids;
    auto       & tape_layers = capture->tape_layers;
    if (rec_ids.empty() || tape_layers.empty()) {
        return false;
    }

    struct conv_upload {
        ggml_context         * ctx      = nullptr;
        ggml_backend_buffer_t  buf      = nullptr;
        void                 * r_state  = nullptr;
        ggml_tensor          * qkv      = nullptr;
        int conv_ch     = 0;
        int conv_window = 0;
    };
    std::vector<conv_upload> uploads;
    uploads.reserve(rec_ids.size());

    auto cleanup = [&]() {
        for (auto & u : uploads) {
            if (u.buf) { ggml_backend_buffer_free(u.buf); u.buf = nullptr; }
            if (u.ctx) { ggml_free(u.ctx);                u.ctx = nullptr; }
        }
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        if (li >= tape_layers.size()) {
            cleanup();
            return false;
        }

        const int il = rec_ids[li];
        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        if (!r_tensor) { continue; }
        if (!dtape_is_cuda_compatible_tensor(r_tensor)) {
            cleanup();
            return false;
        }

        const auto & tape = tape_layers[li];
        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens || tape.qkv_mixed.empty()) {
            cleanup();
            return false;
        }

        size_t qkv_seq_offset = 0;
        if (tape.n_seqs > 1) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                if (tape.seq_ids[s] == seq_id) { found = true; break; }
                qkv_seq_offset += (size_t) tape.n_tokens * (size_t) tape.conv_channels;
            }
            if (!found) {
                cleanup();
                return false;
            }
        }

        const int64_t conv_ch_i64 = tape.conv_channels;
        if (conv_ch_i64 <= 0 || n_embd_r % conv_ch_i64 != 0 ||
                conv_ch_i64 > std::numeric_limits<int>::max()) {
            cleanup();
            return false;
        }

        const int64_t conv_window_i64 = n_embd_r / conv_ch_i64;
        if (conv_window_i64 <= 0 || conv_window_i64 > std::numeric_limits<int>::max()) {
            cleanup();
            return false;
        }

        const size_t qkv_elems = (size_t) n_accepted * (size_t) conv_ch_i64;
        if (tape.qkv_mixed.size() < qkv_seq_offset + qkv_elems) {
            cleanup();
            return false;
        }

        const size_t ctx_mem = ggml_tensor_overhead();
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        ggml_context * ctx = ggml_init(ctx_params);
        if (!ctx) {
            cleanup();
            return false;
        }

        conv_upload upload;
        upload.ctx = ctx;
        upload.qkv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) qkv_elems);
        if (!upload.qkv) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(r_tensor->buffer);
        upload.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!upload.buf) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_tensor_set(upload.qkv,
            tape.qkv_mixed.data() + qkv_seq_offset, 0, qkv_elems * sizeof(float));

        const size_t r_offset = (size_t) cell_idx * n_embd_r * ggml_element_size(r_tensor);
        upload.r_state    = (char *) r_tensor->data + r_offset;
        upload.conv_ch     = (int) conv_ch_i64;
        upload.conv_window = (int) conv_window_i64;
        uploads.push_back(upload);
    }

    if (uploads.empty()) {
        cleanup();
        return false;
    }

    for (const auto & u : uploads) {
        if (!fn_prepare(u.r_state)) {
            cleanup();
            return false;
        }
    }

    for (const auto & u : uploads) {
        if (!fn_rebuild(u.r_state, u.qkv->data, n_accepted, u.conv_ch, u.conv_window)) {
            cleanup();
            GGML_ABORT("DFlash direct CPU-tape conv replay launch failed after validation\n");
        }
    }

    bool synced = true;
    for (const auto & u : uploads) {
        synced = fn_sync(u.r_state) && synced;
    }
    cleanup();
    if (!synced) {
        GGML_ABORT("DFlash direct CPU-tape conv replay sync failed after launch\n");
    }

    mem_recurrent->cells[cell_idx].pos += n_accepted;
    return true;
}
