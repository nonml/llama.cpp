#pragma once

// DFlash tape-replay data structures for DeltaNet recurrent-state rollback.
//
// These types are extracted from llama-context.h so that the tape-replay
// implementation can live in its own translation unit without pulling in the
// full context class definition.  The free-function declarations at the bottom
// are stubs — implementations live in llama-context.cpp (or a dedicated
// llama-dflash-tape.cpp once ported).

#include "llama.h"          // llama_seq_id, llama_pos
#include "llama-hparams.h"  // llama_hparams

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations for types that are defined elsewhere
// ---------------------------------------------------------------------------

class llama_memory_recurrent;
class llama_memory_i;
struct llama_ubatch;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// These are defined in llama-ext.h as part of an anonymous enum on
// llama_dflash_context_params.  Redeclare them here so translation units that
// only include this header can use them without pulling in llama-ext.h.
#ifndef LLAMA_DFLASH_MAX_VERIFY_TOKENS
#  define LLAMA_DFLASH_MAX_VERIFY_TOKENS 25  // must be >= draft_max + 1
#endif
#ifndef LLAMA_DFLASH_MAX_SLOTS
#  define LLAMA_DFLASH_MAX_SLOTS 8
#endif

// ---------------------------------------------------------------------------
// Per-slot hidden-state buffer (CPU side, eval-callback filled)
// ---------------------------------------------------------------------------

struct dflash_layer_hidden_buf {
    std::vector<float> data;
    int64_t n_embd   = 0;
    int64_t n_tokens = 0;
};

// ---------------------------------------------------------------------------
// CPU tape — one entry per recurrent layer, filled via eval callback
// ---------------------------------------------------------------------------

struct dflash_tape_layer {
    std::vector<float> k;          // [S_k * H_k * n_tokens]  after l2_norm
    std::vector<float> v;          // [S_v * H_v * n_tokens]
    std::vector<float> gate;       // [H_v * n_tokens]         pre-exp
    std::vector<float> beta;       // [H_v * n_tokens]         pre-sigmoid
    std::vector<float> qkv_mixed;  // [conv_channels * n_tokens * n_seqs]  for conv state rebuild
    int64_t S_k = 0, H_k = 0, S_v = 0, H_v = 0;
    int64_t conv_channels = 0;
    int n_tokens = 0;
    // per-seq metadata for multi-seq verify QKV scatter
    int n_seqs = 1;
    llama_seq_id seq_ids[LLAMA_DFLASH_MAX_SLOTS] = {};
};

// ---------------------------------------------------------------------------
// GPU tape — persistent tensors the graph writes into directly
//            (no eval-callback round-trip)
// ---------------------------------------------------------------------------

struct dflash_tape_gpu_layer {
    ggml_tensor * k    = nullptr;  // [S_k, H_k, max_tokens]
    ggml_tensor * v    = nullptr;  // [S_v, H_v, max_tokens]
    ggml_tensor * gate = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * beta = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * qkv  = nullptr;  // [conv_channels, max_tokens]
    ggml_backend_buffer_t buf = nullptr;
    ggml_context *        ctx = nullptr;
    ggml_backend_dev_t    dev = nullptr;
};

// Per-slot GPU tape container.
struct dflash_tape_gpu {
    std::vector<dflash_tape_gpu_layer> layers;   // one per recurrent layer
    std::vector<int32_t>               layer_ids; // model layer index → tape index
    ggml_backend_buffer_t buf       = nullptr;
    ggml_context *        ctx       = nullptr;    // owns the tensor descriptors
    int                   max_tokens = 0;          // allocated capacity
    int                   n_tokens   = 0;          // tokens recorded this pass

    ~dflash_tape_gpu() {
        for (auto & layer : layers) {
            if (layer.buf) { ggml_backend_buffer_free(layer.buf); }
            if (layer.ctx) { ggml_free(layer.ctx); }
        }
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
    }
};

// ---------------------------------------------------------------------------
// Per-slot GPU hidden-state buffers
// ---------------------------------------------------------------------------

struct dflash_hidden_gpu {
    std::vector<ggml_tensor *>          layers;    // one [n_embd, max_tokens] tensor per captured layer
    std::vector<int32_t>                layer_ids;
    std::vector<ggml_backend_buffer_t>  bufs;
    std::vector<ggml_context *>         ctxs;
    int64_t n_embd     = 0;
    int     max_tokens = 0;
    int     n_tokens   = 0;

    ~dflash_hidden_gpu() {
        for (auto b : bufs) { if (b) { ggml_backend_buffer_free(b); } }
        for (auto c : ctxs) { if (c) { ggml_free(c); } }
    }
};

// ---------------------------------------------------------------------------
// Tape field type tag (used in tape_name_map lookups)
// ---------------------------------------------------------------------------

enum dflash_tape_type {
    DFLASH_TAPE_K    = 0,
    DFLASH_TAPE_V    = 1,
    DFLASH_TAPE_GATE = 2,
    DFLASH_TAPE_BETA = 3,
    DFLASH_TAPE_QKV  = 4,
};

// ---------------------------------------------------------------------------
// Prefill capture plan — tracks a pending suffix-prefill window per slot
// ---------------------------------------------------------------------------

struct dflash_prefill_capture_plan {
    bool active = false;

    llama_seq_id seq_id = -1;

    int32_t capture_begin = 0;
    int32_t capture_end   = 0;

    int32_t n_tokens  = 0;
    int32_t n_written = 0;
};

// ---------------------------------------------------------------------------
// Master capture state — owned by the drafter llama_context
// ---------------------------------------------------------------------------

struct dflash_capture_data {
    // Logical per-view capture gate. When false the eval callback is not
    // installed and graph builds skip hidden outputs. Layer IDs, GPU buffers,
    // tape metadata, and profile counters are preserved across toggles.
    bool capture_active = true;

    // hidden-state capture (for drafter conditioning)
    std::vector<int32_t>     layer_ids;       // layer indices to capture
    std::vector<std::string> tensor_names;    // pre-formatted "l_out-{id}" names
    std::unordered_map<std::string, size_t> hidden_name_idx; // name → index (O(1) lookup)
    // pointer to context's layer_hiddens (outer: per-slot, inner: per-captured-layer)
    std::vector<std::vector<dflash_layer_hidden_buf>> * hiddens = nullptr;

    // tape recording (for DeltaNet state rollback)
    bool tape_enabled        = false;
    bool gpu_capture_enabled = true;
    std::vector<int32_t> recurrent_layer_ids;  // model layer indices that are DeltaNet
    std::unordered_map<std::string, std::pair<int, int>> tape_name_map;  // name → (layer_idx, type)
    std::vector<dflash_tape_layer> tape_layers; // one per recurrent layer (CPU fallback)

    // GPU-resident tape: graph writes directly to these tensors (no eval-callback sync).
    // One entry per slot for multi-slot DFlash (--spec-dflash-max-slots). For single-slot
    // (default) tapes has size 1 and active_tape_idx is always 0, giving byte-identical
    // behaviour to the pre-multi-slot singleton.
    std::vector<std::unique_ptr<dflash_tape_gpu>>    tapes;
    std::vector<std::unique_ptr<dflash_hidden_gpu>>  hidden_gpu;
    std::vector<std::unique_ptr<dflash_hidden_gpu>>  prefill_gpu;   // large staging buffer for suffix prefill
    int prefill_gpu_max_tokens = 0;                                  // allocation capacity of prefill_gpu
    std::vector<dflash_prefill_capture_plan> prefill_plans;          // active capture window plans, indexed by seq/slot
    int active_tape_idx = 0;

    // Active ubatch for the in-flight process_ubatch() call. The eval callback
    // reads ubatch->n_seqs_unq / ubatch->seq_id to route hidden-state captures
    // to layer_hiddens[seq] (per-token scatter under multi-seq ubatches).
    // ggml's scheduler serialises callbacks within a graph compute, so this
    // pointer is safe to read without synchronisation.
    const llama_ubatch * ubatch = nullptr;

    // Reused scratch for the multi-seq scatter path (avoid per-ubatch alloc).
    std::vector<float> scatter_buf;

    // Opt-in DFlash profiling (GGML_DFLASH_PROFILE=summary,replay,copy,prefill,verify,trace).
    uint32_t profile_flags = 0;
    bool     profile       = false;
    bool multi_gpu_capture_fallback_logged = false;
    bool multi_gpu_replay_fallback_logged  = false;
    uint64_t profile_decode_us                   = 0;
    uint64_t profile_output_extract_us           = 0;
    uint64_t profile_raw_logits_us               = 0;
    uint64_t profile_raw_logits_bytes            = 0;
    uint64_t profile_raw_logits_skipped          = 0;
    uint64_t profile_reduced_logits_us           = 0;
    uint64_t profile_reduced_logits_ids_us       = 0;
    uint64_t profile_reduced_logits_probs_us     = 0;
    uint64_t profile_reduced_logits_bytes        = 0;
    uint64_t profile_verify_sync_split_us        = 0;
    uint64_t profile_cb_ask                      = 0;
    uint64_t profile_cb_hidden_ask               = 0;
    uint64_t profile_cb_tape_ask                 = 0;
    uint64_t profile_cb_qkv_ask                  = 0;
    uint64_t profile_cb_read                     = 0;
    uint64_t profile_cb_hidden_read              = 0;
    uint64_t profile_cb_tape_read                = 0;
    uint64_t profile_cb_qkv_read                 = 0;
    uint64_t profile_replay_wait_us              = 0;
    uint64_t profile_replay_gdn_enqueue_us       = 0;
    uint64_t profile_replay_gdn_wait_us          = 0;
    uint64_t profile_replay_conv_enqueue_us      = 0;
    uint64_t profile_replay_conv_wait_us         = 0;
    uint64_t profile_replay_layers               = 0;
    uint64_t profile_replay_sync_calls           = 0;
    uint64_t profile_replay_direct_gpu           = 0;
    uint64_t profile_replay_ggml_gpu             = 0;
    uint64_t profile_replay_cpu_fallback         = 0;
    uint64_t profile_conv_gpu_us                 = 0;
    uint64_t profile_conv_read_wait_us           = 0;
    uint64_t profile_conv_cpu_us                 = 0;
    uint64_t profile_conv_write_wait_us          = 0;
    std::unordered_map<std::string, uint64_t> profile_cb_names;

    // Function pointer for syncing a backend to its CUDA stream (resolved via
    // ggml_backend_cuda_reg_get_proc_address at runtime).
    using sync_backend_to_stream_fn_t = bool (*)(ggml_backend_t);
    sync_backend_to_stream_fn_t fn_sync_backend_to_stream           = nullptr;
    ggml_backend_t              sync_backend_to_stream_backend      = nullptr;
    struct capture_wait_backend {
        ggml_backend_t              backend = nullptr;
        sync_backend_to_stream_fn_t fn      = nullptr;
    };
    std::vector<capture_wait_backend> capture_wait_backends;

    // ---------------------------------------------------------------------------
    // Inline accessors
    // ---------------------------------------------------------------------------

    dflash_tape_gpu * active_tape() const {
        return (active_tape_idx >= 0 && active_tape_idx < (int) tapes.size())
               ? tapes[active_tape_idx].get()
               : nullptr;
    }

    dflash_hidden_gpu * active_hidden_gpu() const {
        return (active_tape_idx >= 0 && active_tape_idx < (int) hidden_gpu.size())
               ? hidden_gpu[active_tape_idx].get()
               : nullptr;
    }

    dflash_prefill_capture_plan * prefill_plan_for_seq(llama_seq_id seq_id) {
        return (seq_id >= 0 && seq_id < (llama_seq_id) prefill_plans.size())
               ? &prefill_plans[(size_t) seq_id]
               : nullptr;
    }

    const dflash_prefill_capture_plan * prefill_plan_for_seq(llama_seq_id seq_id) const {
        return (seq_id >= 0 && seq_id < (llama_seq_id) prefill_plans.size())
               ? &prefill_plans[(size_t) seq_id]
               : nullptr;
    }

    bool any_prefill_plan_active() const {
        for (const auto & plan : prefill_plans) {
            if (plan.active && plan.n_tokens > 0) { return true; }
        }
        return false;
    }

    int max_prefill_plan_tokens() const {
        int max_tokens = 0;
        for (const auto & plan : prefill_plans) {
            if (plan.active && plan.n_tokens > 0) {
                max_tokens = std::max(max_tokens, (int) plan.n_tokens);
            }
        }
        return max_tokens;
    }

    std::vector<dflash_layer_hidden_buf> * slot_hiddens(int slot) const {
        if (!hiddens || slot < 0 || slot >= (int) hiddens->size()) {
            return nullptr;
        }
        return &(*hiddens)[slot];
    }

    std::vector<dflash_layer_hidden_buf> * active_slot_hiddens() const {
        return slot_hiddens(active_tape_idx);
    }

    // Persistent GPU buffer for tape replay (avoids per-call alloc/free).
    ggml_backend_buffer_t replay_buf      = nullptr;
    size_t                replay_buf_size = 0;

    // Pre-allocated zeros buffer for Q input (avoids per-call alloc+zero).
    std::vector<float> replay_zeros;

    // Async tape-replay state (GDN launched, waiting for sync before conv rebuild).
    bool                   replay_pending         = false;
    ggml_backend_t         replay_gpu_backend     = nullptr;
    ggml_context *         replay_graph_ctx       = nullptr;
    ggml_backend_event_t   replay_event           = nullptr;  // CUDA event for fine-grained sync
    bool                   replay_direct_gpu      = false;
    const void *           replay_sync_ptr        = nullptr;
    std::vector<const void *> replay_sync_ptrs;
    int                    replay_sync_device     = -1;
    int                    replay_n_accepted      = 0;
    int32_t                replay_cell_idx        = -1;
    llama_seq_id           replay_seq_id          = 0;
    llama_memory_recurrent * replay_mem_recurrent = nullptr;

    ~dflash_capture_data() {
        if (replay_graph_ctx) { ggml_free(replay_graph_ctx); }
        if (replay_buf)       { ggml_backend_buffer_free(replay_buf); }
        if (replay_event)     { ggml_backend_event_free(replay_event); }
    }
};

// ---------------------------------------------------------------------------
// Free-function declarations
//
// These functions operate on dflash_capture_data and llama_memory_recurrent.
// Implementations live in the context / tape-replay translation unit(s).
// ---------------------------------------------------------------------------

// Kick off an async tape replay for seq_id after n_accepted tokens were accepted.
// Launches GDN kernel; conv rebuild is deferred until dflash_tape_replay_sync().
// backends must remain valid until dflash_tape_replay_sync() is called.
void dflash_tape_replay(
        dflash_capture_data              * capture,
        llama_memory_recurrent           * mem,
        const std::vector<ggml_backend_ptr> & backends,
        llama_seq_id                       seq_id,
        int                                n_accepted);

// Wait for the async GDN kernel to finish, then run conv-state rebuild.
// backends must be the same vector passed to dflash_tape_replay().
void dflash_tape_replay_sync(
        dflash_capture_data              * capture,
        const std::vector<ggml_backend_ptr> & backends);

// CPU-only tape replay path — applies n_accepted token states to cell_idx.
void dflash_tape_replay_cpu(
        dflash_capture_data   * capture,
        llama_memory_recurrent * mem,
        int32_t                  cell_idx,
        int                      n_accepted);

// Conv-state replay (CPU or GPU, dispatched internally).
void dflash_tape_replay_conv(
        dflash_capture_data              * capture,
        llama_memory_recurrent           * mem,
        const std::vector<ggml_backend_ptr> & backends,
        int32_t                            cell_idx,
        int                                n_accepted,
        llama_seq_id                       seq_id);

// Full rollback: KV trim + recurrent restore + tape replay for hybrid models.
// mem_attn is the attention KV cache (may be null if pure-recurrent model).
// tree_mode should be true when a DDTree branch batch populated the KV cache
// (forces full KV trim from n_past_before rather than the cheaper tail trim).
void dflash_rollback(
        dflash_capture_data              * capture,
        llama_memory_i                   * mem_attn,
        llama_memory_recurrent           * mem_recr,
        const std::vector<ggml_backend_ptr> & backends,
        llama_seq_id                       seq_id,
        llama_seq_id                       seq_backup,
        int                                n_past_before,
        int                                n_accepted,
        bool                               tree_mode);

// One-time setup of recurrent-layer metadata derived from hparams.
// Safe to call multiple times; subsequent calls are no-ops.
void dflash_ensure_recurrent_setup(
        dflash_capture_data * capture,
        const llama_hparams & hparams);

// Enable or disable tape recording.  When enable is false the eval-callback
// tape path is skipped; existing tape_layers data is preserved.
void dflash_set_tape_recording(
        dflash_capture_data * capture,
        bool                  enable);
