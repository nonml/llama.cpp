#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

static void fail(const char * msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static ggml_tensor * make_fattn(
        ggml_context * ctx,
        ggml_type      type_k,
        ggml_type      type_v) {
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 1, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type_k,        128, 1, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type_v,        128, 1, 1, 1);

    return ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
}

static ggml_tensor * make_scale(
        ggml_context * ctx,
        ggml_type      type) {
    ggml_tensor * x = ggml_new_tensor_1d(ctx, type, 128);

    return ggml_scale(ctx, x, 2.0f);
}

static void test_cpu_rejects_turbo_fattn(ggml_backend_t backend) {
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * fattn = make_fattn(ctx, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0);
    if (ggml_backend_supports_op(backend, fattn)) {
        ggml_free(ctx);
        fail("CPU backend must reject flash attention with turbo K because it has no CPU vec_dot kernel");
    }

    ggml_free(ctx);
}

static void test_cpu_accepts_f16_fattn(ggml_backend_t backend) {
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * fattn = make_fattn(ctx, GGML_TYPE_F16, GGML_TYPE_F16);
    if (!ggml_backend_supports_op(backend, fattn)) {
        ggml_free(ctx);
        fail("CPU backend should support flash attention with f16 K/V");
    }

    ggml_free(ctx);
}

static void test_cpu_rejects_bf16_scale(ggml_backend_t backend) {
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * scale = make_scale(ctx, GGML_TYPE_BF16);
    if (ggml_backend_supports_op(backend, scale)) {
        ggml_free(ctx);
        fail("CPU backend must reject BF16 scale because the CPU scale kernel only supports F32");
    }

    ggml_free(ctx);
}

static void test_cpu_accepts_f32_scale(ggml_backend_t backend) {
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * scale = make_scale(ctx, GGML_TYPE_F32);
    if (!ggml_backend_supports_op(backend, scale)) {
        ggml_free(ctx);
        fail("CPU backend should support F32 scale");
    }

    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fail("failed to initialize CPU backend");
    }

    test_cpu_accepts_f16_fattn(backend);
    test_cpu_rejects_turbo_fattn(backend);
    test_cpu_accepts_f32_scale(backend);
    test_cpu_rejects_bf16_scale(backend);

    ggml_backend_free(backend);
    return 0;
}
