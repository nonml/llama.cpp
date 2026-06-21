### Setup & Architecture
*   **Prerequisite:** `llama-server --slot-save-path ./slots/` (Required for all slot actions).
*   **C++ Native:** Integrated directly into `llama-server` & `llama-cli`. No external Python harness/dependencies.
*   **Lightweight Checkpoints:** O(1) token position markers (`tokens.size()`), not full-state disk serialization.
*   **Diagnostic Spikes:** 3 diagnostic spikes proved functionality, then were deleted (zero server code added).

### API Reference & Under the Hood
*   **Checkpoint** (`POST /slots/:id?action=checkpoint`)
    *   *Req:* `{"name": "opt_name"}` | *Resp:* `{"id_slot": 0, "name": "opt_name", "pos": 1542, "generation_id": 3}`
    *   *Auto-Checkpoint:* Server automatically records position before every request. Responses include `checkpoint_pos` and `fill_pct` (context watermark).
*   **Rollback** (`POST /slots/:id?action=rollback`)
    *   *Req:* `{"pos": 1542, "generation_id": 3}` | *Resp:* `{"id_slot": 0, "from_pos": 1891, "to_pos": 1542, "n_removed": 349}`
    *   *Mechanics:* O(1). Calls `common_context_seq_rm` + truncates tokens.
    *   *Safety:* `generation_id` increments on reset. Mismatches reject rollbacks to prevent stale state corruption.
*   **Fork** (`POST /slots/fork`)
    *   *Req/Resp:* `{"src_slot": 0, "dst_slot": 1, "p0": 0, "p1": 1542}` (`p1=-1`/omitted = end of seq).
    *   *Mechanics:* Metadata-only (`common_context_seq_cp`). Zero memory cost. Shared KV cells persist until all references are freed via `seq_rm`/erase.

### Recurrent/Hybrid Models (e.g., qwen35)
*   **Whole-Sequence Prompt Cache:** Works automatically in prefix-only mode (`prompt_cache: true` in `GET /props`). 
*   **Behavior:** Cannot trim by position. Reuses cache *only* if incoming prompt is a full prefix match. Swapped-out side-sessions restore without re-prefilling if snapshot remains cached.
*   **RAM Sizing (`--cache-ram`):** Scales heavily with `--cache-type-k/-v`. (e.g., `qwen35` 27B q8_0 KV = ~40 KB/token. 33,229 tokens = 1.31 GB; ~80k tokens = ~3 GB). Cache fits very few entries.

### Usage Patterns
*   **Tool Failure Recovery:** Use auto-returned `checkpoint_pos` & `generation_id` to rollback failed calls instantly.
*   **Proactive Compaction:** Monitor `fill_pct` in response. Crosses ~80% → rollback, compact history, re-prefill.
*   **Side-Sessions:** Fork `src` to `dst`, generate, erase `dst` (shared prefix survives in `src`).
*   **CLI Commands:** `/checkpoint` (prints pos), `/rollback <pos>` (clears history).

### Spike Results (2026-05-31)
| Spike | Status | Findings |
| :--- | :--- | :--- |
| **#1 Prefix-share** | **Success / Bug** | `seq_cp` works in unified mode (`kv_unified=true`). Forking is free. Eviction frees unique cells. **BUG:** Naive restore (`set_data`) duplicates shared trunk (2500 vs 1500 cells). `PARTIAL_ONLY` serializes full range (ratio 1.00), no dedup. |
| **#2 PCIe Cross** | **Timeout** | CPU benchmark too slow; needs GPU. |
| **#3 pSEQ Rebind** | **Crashed** | Same `kv_unified` test bug as original #1. |

### Deferred Work & Validation
*   **Deferred Delta-Loader Patch:** Spike #1 "world 3" fix requires Layer 1 C++ patch (delta save/load keyed on branch divergence) to prevent trunk duplication on restore. 
*   *Decision:* Defers gating features (artifact cache / pi keeper). Only build if runtime measures prove re-reads are frequent and re-prefill is computationally expensive.
*   **Validation Checklist:**
    1. Verify `cache_prompt` reuses prefix (monitor `fill_pct` / prefill times).
    2. Run end-to-end >80% collapse test to ensure agent coherence.