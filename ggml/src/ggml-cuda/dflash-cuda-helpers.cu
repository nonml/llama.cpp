#include <cuda_runtime.h>
#include <cstddef>

__global__ static void k_dflash_rebuild_conv_state(
        float * __restrict__ r_state,
        const float * __restrict__ qkv,
        const int n_accepted,
        const int conv_ch,
        const int conv_window) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_ch) return;

    float * dst = r_state + (size_t) ch * conv_window;
    for (int w = 0; w < conv_window; ++w) {
        const int src_pos = n_accepted + w;
        dst[w] = src_pos < conv_window
            ? dst[src_pos]
            : qkv[(size_t)(src_pos - conv_window) * conv_ch + ch];
    }
}

extern "C" bool dflash_rebuild_conv_state(
        void * r_state, const void * qkv,
        int n_accepted, int conv_ch, int conv_window) {
    if (!r_state || !qkv) return false;
    if (n_accepted <= 0 || conv_ch <= 0 || conv_window <= 0) return false;

    cudaPointerAttributes r_attr;
    cudaError_t r_err = cudaPointerGetAttributes(&r_attr, r_state);
    if (r_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaPointerAttributes qkv_attr;
    cudaError_t qkv_err = cudaPointerGetAttributes(&qkv_attr, qkv);
    if (qkv_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000 || defined(GGML_USE_HIP)
    if (r_attr.type != cudaMemoryTypeDevice || qkv_attr.type != cudaMemoryTypeDevice ||
            r_attr.device != qkv_attr.device) {
        return false;
    }
#else
    if (r_attr.memoryType != cudaMemoryTypeDevice || qkv_attr.memoryType != cudaMemoryTypeDevice ||
            r_attr.device != qkv_attr.device) {
        return false;
    }
#endif

    (void)cudaSetDevice(r_attr.device);
    const int block = 256;
    const int grid = (conv_ch + block - 1) / block;
    k_dflash_rebuild_conv_state<<<grid, block, 0, cudaStreamPerThread>>>(
        (float *) r_state, (const float *) qkv, n_accepted, conv_ch, conv_window);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool dflash_cuda_prepare_ptr(const void * ptr) {
    if (!ptr) return false;

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000 || defined(GGML_USE_HIP)
    if (attr.type != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#else
    if (attr.memoryType != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#endif

    return true;
}

extern "C" bool dflash_cuda_set_device(int device) {
    return device >= 0 && cudaSetDevice(device) == cudaSuccess;
}

extern "C" bool dflash_cuda_synchronize_ptr(const void * ptr) {
    if (!ptr) return false;

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000 || defined(GGML_USE_HIP)
    if (attr.type != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#else
    if (attr.memoryType != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#endif

    return cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
}

extern "C" bool dflash_cuda_ptr_device(const void * ptr, int * device) {
    if (!ptr || !device) return false;

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000 || defined(GGML_USE_HIP)
    if (attr.type != cudaMemoryTypeDevice) {
        return false;
    }
    *device = attr.device;
#else
    if (attr.memoryType != cudaMemoryTypeDevice) {
        return false;
    }
    *device = attr.device;
#endif

    return true;
}

extern "C" bool dflash_cuda_synchronize_device(int device) {
    if (device < 0) return false;

    if (cudaSetDevice(device) != cudaSuccess) {
        return false;
    }
    return cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
}
