// SPDX-License-Identifier: MIT
//
// Raw FFI declarations matching `kernels/flash_attn_shim.h` field-for-field.

use core::ffi::{c_int, c_void};

unsafe extern "C" {
    pub(crate) fn flash_attn_rocm_fwd(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        out_ptr: *mut c_void,
        softmax_lse_ptr: *mut c_void,

        q_batch_stride: i32,
        k_batch_stride: i32,
        v_batch_stride: i32,
        o_batch_stride: i32,

        q_row_stride: i32,
        k_row_stride: i32,
        v_row_stride: i32,
        o_row_stride: i32,

        q_head_stride: i32,
        k_head_stride: i32,
        v_head_stride: i32,
        o_head_stride: i32,

        batch: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_size: i32,

        softmax_scale: f32,

        is_causal: c_int,
        window_size_left: c_int,
        window_size_right: c_int,
        is_bf16: c_int,

        stream: *mut c_void,
    ) -> c_int;
}
