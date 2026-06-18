#include "allreduce.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// CUDA AllReduce for tensor-parallel inference across two GPUs.
//
// Provides an in-place sum reduction over matching tensors on two CUDA
// devices in the same process.  Used by the tensor-split path alongside
// NCCL; targets setups without NVLink, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.  On Windows the default
//     copy threshold is 1 byte, so this path is opt-in
//     (GGML_CUDA_AR_COPY_THRESHOLD); the in-kernel wait can miss peer stores
//     on WDDM consumer GPUs.  A spin watchdog fails the graph if it does.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// These accesses must carry *system* scope.  CUDA's `volatile` only stops the
// compiler caching the value in a register (it emits ld./st.volatile, which
// bypass L1); it says nothing about coherence with a store issued by the other
// GPU, so a poller could keep observing a stale token indefinitely.  That is
// not theoretical: with a plain volatile load, one block in eight would
// periodically miss its peer's arrival token and spin forever, wedging the
// whole server (see the watchdog below, which is what caught it).
//
// atomicAdd_system() is indeed unavailable here -- it needs
// hostNativeAtomicSupported, which PCIe-attached consumer GPUs without NVLink
// do not have -- but that restriction applies to atomic read-modify-write
// ops.  Plain scoped load/store (ld/st.relaxed.sys) are not RMW, are available
// on sm_70+, and work fine over PCIe.  There is exactly one writer and one
// reader per arrival int, so relaxed ordering is sufficient; the surrounding
// __threadfence_system() calls still provide the release/acquire edges that
// order the D2H payload writes against the token.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    asm volatile("st.volatile.global.b32 [%0], %1;" :: "l"(p), "r"(token) : "memory");
#else
    *(volatile int *)p = token;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    int v;
    asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
#else
    return *(const volatile int *)p;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// Rendezvous watchdog.  Phase 2 of the kernel publishes an arrival token with a
// plain volatile store + system threadfence: these are PCIe-attached consumer
// GPUs with no NVLink, so atomicAdd_system() is unavailable (see the note above
// ggml_cuda_ar_signal_set) and nothing guarantees the peer ever observes that
// store.  A single dropped signal used to strand the waiter in an unbounded
// spin -- one GPU pinned at 100% forever, its peer idle, the calling host
// thread stuck in cudaEventSynchronize, and the server wedged with no error
// logged anywhere.  Two mitigations:
//   * the waiter re-publishes its own token every RESIGNAL iterations, so a
//     dropped store repairs itself instead of deadlocking both sides;
//   * the spin is capped, so a rendezvous that is genuinely unrecoverable
//     bails out and reports instead of hanging.
// RESIGNAL must stay a power of two (it is used as an AND mask).
static constexpr uint64_t GGML_CUDA_AR_SPIN_RESIGNAL_ITERS = 4096;

// Index of the per-block stall marker within the arrival stride.  ARRIVAL_STRIDE
// is 64 B but only int[0] carries the token, so int[1] is free.  The kernel
// writes the stalled token there; the host reads it back via arrival.host.
static constexpr int GGML_CUDA_AR_STALL_INT = 1;

// int[2] records the peer token value the stalled block last observed, so the
// host can tell "never saw the peer's store" from "peer never got here".
static constexpr int GGML_CUDA_AR_STALL_OBSERVED_INT = 2;

// ~0.9 s: the loop measured at ~4.5 us/iteration (__nanosleep(100) costs far
// more than the 100 ns it asks for), so this is ~200k iterations.  Long enough
// that healthy rendezvous rarely trip it, short enough that a stall does not
// dominate the run.  Override with GGML_CUDA_AR_SPIN_TIMEOUT_ITERS.
static constexpr uint64_t GGML_CUDA_AR_SPIN_TIMEOUT_ITERS_DEFAULT = 200ull * 1000;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token,
        uint64_t                    spin_timeout_iters) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    __shared__ int s_stalled;

    if (tid == 0) {
        s_stalled = 0;

        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        uint64_t spins = 0;
        while (ggml_cuda_ar_signal_get(other_slot) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            ++spins;

            // Re-publish our own arrival token periodically: without NVLink or
            // system-scope atomics the original store above is not guaranteed
            // to be observed by the peer, and one dropped store would otherwise
            // strand both sides here permanently.
            if ((spins & (GGML_CUDA_AR_SPIN_RESIGNAL_ITERS - 1)) == 0) {
                ggml_cuda_ar_signal_set(my_slot, token);
                __threadfence_system();
            }

            // Watchdog.  Bail rather than spin forever, and leave a marker the
            // host can read back to report which call stalled.  Phase 3 is the
            // only phase that writes recvbuf, so giving up before it leaves the
            // output tensor unreduced.  The host fails the graph; do not keep
            // using those values.
            if (spins >= spin_timeout_iters) {
                // Record what we actually saw at the peer's slot before giving
                // up: comparing it against what the peer really published tells
                // a visibility failure apart from the peer never arriving.
                // Write the observed value first so the marker is only set once
                // the payload beside it is valid.
                ggml_cuda_ar_signal_set(my_slot + GGML_CUDA_AR_STALL_OBSERVED_INT,
                                        ggml_cuda_ar_signal_get(other_slot));
                __threadfence_system();
                ggml_cuda_ar_signal_set(my_slot + GGML_CUDA_AR_STALL_INT, token);
                __threadfence_system();
                s_stalled = 1;
                break;
            }
        }
    }

    __syncthreads();

    if (s_stalled) {
        return;
    }

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (bid == 0 && tid < count - tail) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) +
                ggml_cuda_cast<float>(host_other[tail + tid]));
        }
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    uint64_t call_count;
    uint64_t spin_timeout_iters; // per-block cap on the phase-2 rendezvous spin
    bool     stall_reported;     // gate the one-time explanatory block
    uint64_t stall_count;        // total rendezvous stalls seen this session
    bool     repair_enabled;     // GGML_CUDA_AR_REPAIR: sync after each AR so stalls can be repaired
    bool     unrepaired_stall;   // host must fail the graph; do not continue with unreduced tensors

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;
};

// Process-wide: any pipeline that leaves an unrepaired stall poisons further
// AllReduce so the meta backend can fail the graph (false from allreduce would
// run the butterfly copy path on the same GPUs).
static bool ggml_cuda_ar_fatal_stall = false;

static void ggml_cuda_ar_note_fatal_stall(void) {
    ggml_cuda_ar_fatal_stall = true;
}

bool ggml_cuda_ar_unrepaired_stall(void) {
    return ggml_cuda_ar_fatal_stall;
}

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

// Host-side view of the same per-block token block.  The arrival ring is mapped
// pinned memory, so the CPU can read back the stall markers the kernel wrote.
static int * ggml_cuda_ar_arrival_host_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.host + offset);
}

// Report any block whose phase-2 rendezvous tripped the watchdog.  Such a block
// skips its output phase, so without this the only symptom would be silently
// wrong numbers -- previously it was an unbounded hang instead.  Called after
// the slot's kernels are known complete, so the markers are stable.
static bool ggml_cuda_ar_scan_stalls(ggml_cuda_ar_pipeline * p, bool * stalled) {
    if (p->arrival.host == nullptr) {
        return false;
    }

    constexpr int ARRIVAL_INTS = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    bool any = false;

    for (int i = 0; i < p->n_devices; ++i) {
      for (int slot = 0; slot < GGML_CUDA_AR_POOL_SIZE; ++slot) {
        int * base = ggml_cuda_ar_arrival_host_ptr(p, slot, i);
        for (int b = 0; b < GGML_CUDA_AR_KERNEL_BLOCKS; ++b) {
            int * cell = base + b * ARRIVAL_INTS;
            const int stalled_token = cell[GGML_CUDA_AR_STALL_INT];
            if (stalled_token == 0) {
                continue;
            }
            stalled[i] = true;
            any        = true;
            const int observed = cell[GGML_CUDA_AR_STALL_OBSERVED_INT];
            cell[GGML_CUDA_AR_STALL_INT]          = 0;
            cell[GGML_CUDA_AR_STALL_OBSERVED_INT] = 0;

            p->stall_count++;

            // What the peer really published, straight out of mapped host
            // memory.  If these carry the awaited token then the store simply
            // never became visible to the waiter; if they lag, the peer never
            // reached this rendezvous.
            const int   peer = (i == 0) ? 1 : 0;
            const int * peer_base = ggml_cuda_ar_arrival_host_ptr(p, slot, peer);
            char peer_tokens[256];
            int  off = 0;
            for (int pb = 0; pb < GGML_CUDA_AR_KERNEL_BLOCKS && off < (int) sizeof(peer_tokens) - 16; ++pb) {
                off += snprintf(peer_tokens + off, sizeof(peer_tokens) - off, "%s%d",
                                pb ? "," : "", peer_base[pb * ARRIVAL_INTS]);
            }

            GGML_LOG_ERROR(
                "%s: AllReduce rendezvous stalled #%llu (device %d, slot %d, block %d): awaited token "
                "%d, last observed %d after %llu spins.\n"
                "  peer(device %d) published tokens per block: [%s]\n",
                __func__, (unsigned long long) p->stall_count,
                p->devices[i], slot, b, stalled_token, observed,
                (unsigned long long) p->spin_timeout_iters,
                p->devices[peer], peer_tokens);

            if (!p->stall_reported) {
                p->stall_reported = true;
                if (p->repair_enabled) {
                    GGML_LOG_ERROR(
                        "  Cross-GPU handshake in allreduce.cu stalled. Repair copies the peer result.\n"
                        "  Raise GGML_CUDA_AR_SPIN_TIMEOUT_ITERS if this is a false positive under contention.\n");
                } else {
                    GGML_LOG_ERROR(
                        "  Cross-GPU handshake in allreduce.cu stalled. Output is unreduced; the graph will fail.\n"
                        "  GGML_CUDA_AR_REPAIR=1 copies the peer result and continues. Raise\n"
                        "  GGML_CUDA_AR_SPIN_TIMEOUT_ITERS if this is a false positive under contention.\n");
                }
            }
        }
      }
    }

    return any;
}

// Drain this AR call, then repair any device whose rendezvous stalled.
//
// A block that trips the watchdog returns before phase 3, so its device's
// tensor never received the peer's contribution.  The peer, having completed,
// already holds the full sum -- so copying the peer's tensor over the stalled
// device's restores exactly what the AllReduce was supposed to produce.
//
// This has to run here rather than at the next acquire_slot: by then the next
// subgraph has already consumed the unreduced tensor.  That costs the host's
// run-ahead over the AR, which is why it only syncs the compute streams the AR
// kernels were launched on.
static bool ggml_cuda_ar_sync_and_repair(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t *        backends,
        ggml_tensor **          tensors,
        int64_t                 ne,
        ggml_type               input_type) {
    if (p->arrival.host == nullptr) {
        return true;
    }

    for (int i = 0; i < p->n_devices; ++i) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    }

    bool stalled[GGML_CUDA_MAX_DEVICES] = {};
    if (!ggml_cuda_ar_scan_stalls(p, stalled)) {
        return true;
    }

    const size_t nbytes = (size_t) ne * ggml_type_size(input_type);

    for (int i = 0; i < p->n_devices; ++i) {
        if (!stalled[i]) {
            continue;
        }
        const int peer = 1 - i; // valid for n_devices == 2 only

        if (stalled[peer]) {
            // Both sides bailed, so neither tensor holds a complete sum and
            // there is nothing left to copy from.  Never observed in practice
            // (stalls have only ever been one-sided), but say so plainly
            // rather than pretend the step was fine.
            GGML_LOG_ERROR("%s: both devices stalled on the same AllReduce -- cannot repair\n", __func__);
            p->unrepaired_stall = true;
            ggml_cuda_ar_note_fatal_stall();
            return true;
        }

        // Copy the whole tensor, not just the stalled chunk: the peer completed
        // every chunk of this call, so its buffer is correct throughout.
        CUDA_CHECK(cudaMemcpyPeer(tensors[i]->data, p->devices[i],
                                  tensors[peer]->data, p->devices[peer], nbytes));
    }

    return true;
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    if (p->unrepaired_stall) {
        return { 0, 0 };
    }

    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }

        // When the repair path is off nothing else drains the AR kernels, so
        // this is where watchdog markers get noticed.  Detection is a couple of
        // calls late, which is too late to repair.  Fail the graph rather than
        // keep generating on unreduced tensors.
        if (!p->repair_enabled) {
            bool stalled[GGML_CUDA_MAX_DEVICES] = {};
            if (ggml_cuda_ar_scan_stalls(p, stalled)) {
                p->unrepaired_stall = true;
                ggml_cuda_ar_note_fatal_stall();
            }
        }
    }

    return { slot, (int) p->call_count };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    // Windows WDDM + consumer GPUs: in-kernel spin on mapped host flags misses
    // peer stores. Default to copy-engine (CUDA events) for every size.
    // Linux keeps the 1 MB cutoff. Override with GGML_CUDA_AR_COPY_THRESHOLD.
#if defined(_WIN32)
    const uint64_t copy_threshold_default = 1;
#else
    const uint64_t copy_threshold_default = GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT;
#endif
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", copy_threshold_default);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    p->spin_timeout_iters =
        ggml_cuda_ar_env_u64("GGML_CUDA_AR_SPIN_TIMEOUT_ITERS", GGML_CUDA_AR_SPIN_TIMEOUT_ITERS_DEFAULT);
    p->stall_reported   = false;
    p->stall_count      = 0;
    p->repair_enabled   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_REPAIR", 0) != 0;
    p->unrepaired_stall = false;
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU, "
                  "copy-engine threshold %zu B\n",
                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20,
                  (size_t) p->copy_threshold);

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight work before tearing down resources.  Syncing only
    // p->streams[i] is not enough: the chunked kernel and the copy path's
    // add_kernel both run on the caller's *compute* stream, not the AR stream.
    // The chunked kernel reads and writes host_buf and arrival -- mapped pinned
    // host memory -- for the whole length of its rendezvous, so cudaFreeHost'ing
    // those while it is still resident lets the GPU keep DMA-ing into host pages
    // the OS has already handed back.  cudaDeviceSynchronize drains every stream
    // on the device, compute streams included.
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        const cudaError_t rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) {
            // The device is already wedged (sticky error, or the display driver
            // reset it out from under us).  The drain did not happen, so every
            // buffer below is still a potential DMA target.  Leak the pipeline
            // rather than unpin pages the GPU may still write to: this only runs
            // on teardown, and a leak at exit is cheaper than corrupting the
            // kernel's page tables.
            GGML_LOG_ERROR("%s: device %d did not drain (%s); leaking AllReduce buffers\n",
                           __func__, p->devices[i], cudaGetErrorString(rc));
            return;
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { cudaEventDestroy(p->ev_pool[i][s].cpy[c]); }
            }
            if (p->ev_pool[i][s].h2d) { cudaEventDestroy(p->ev_pool[i][s].h2d); }
            if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->host_large_read_done[i]);
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->dev_tmp_kernel_done[i]);
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamDestroy(p->streams[i]);
        }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    if (p->unrepaired_stall) {
        return true;
    }
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    if (p->unrepaired_stall) {
        return true;
    }

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Decide between copy-engine and chunked kernel paths based on the working
    // type's actual byte count.  No upper bound: copy_outer slices reductions
    // larger than copy_bytes into copy_bytes-sized pieces.
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            if (p->unrepaired_stall) {
                return true;
            }
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                    token, \
                    p->spin_timeout_iters)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }

        // Only the chunked-kernel path runs the in-kernel rendezvous, so it is
        // the only one that can strand a device mid-reduction.  Repairing a
        // stall requires draining both compute streams here, which measured at
        // roughly 4x slower end to end (23.4 -> 5.7 t/s on 2x RTX 3060, and it
        // provoked ~10x more stalls), so it is opt-in rather than the default.
        // With it off, a stall is logged from acquire_slot and the graph fails.
        if (p->repair_enabled) {
            ok = ggml_cuda_ar_sync_and_repair(p, backends, tensors, ne, input_type) && ok;
        }
    }

    return ok;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// HIP and MUSA lack the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and __nanosleep that this
// implementation relies on, so the internal AllReduce is a CUDA-only feature.
// The dispatcher in ggml-cuda.cu treats a nullptr pipeline as "init failed"
// and silently falls back to the meta backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

bool ggml_cuda_ar_unrepaired_stall(void) {
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
