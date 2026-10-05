// SPDX-License-Identifier: MIT
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-pass flash attention, CK backend. BSHD layout: (batch, seqlen,
// num_heads, head_size). Strides are in elements, not bytes. Returns 0 on
// success, negative on failure (no matching kernel instantiation or
// invalid arguments).
int flash_attn_rocm_fwd(const void* q_ptr,
                         const void* k_ptr,
                         const void* v_ptr,
                         void* out_ptr,
                         void* softmax_lse_ptr,
                         int32_t q_batch_stride,
                         int32_t k_batch_stride,
                         int32_t v_batch_stride,
                         int32_t o_batch_stride,
                         int32_t q_row_stride,
                         int32_t k_row_stride,
                         int32_t v_row_stride,
                         int32_t o_row_stride,
                         int32_t q_head_stride,
                         int32_t k_head_stride,
                         int32_t v_head_stride,
                         int32_t o_head_stride,
                         int32_t batch,
                         int32_t seqlen_q,
                         int32_t seqlen_k,
                         int32_t num_heads,
                         int32_t num_heads_k,
                         int32_t head_size,
                         float softmax_scale,
                         int is_causal,
                         int window_size_left,
                         int window_size_right,
                         int is_bf16,
                         void* stream);

#ifdef __cplusplus
}
#endif
