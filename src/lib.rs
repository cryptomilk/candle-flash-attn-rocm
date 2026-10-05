// SPDX-License-Identifier: MIT

mod ffi;

use candle_core::{Result, Tensor};

/// Flash-attention v2 forward pass, CK backend.
///
/// Computes `softmax(Q @ K^T * softmax_scale) @ V` without materializing the
/// full attention score matrix. Multi-query and grouped-query attention are
/// supported by using `k`/`v` tensors with fewer heads than `q` — the number
/// of heads in `q` must be divisible by the number of heads in `k`/`v`.
///
/// # Arguments
///
/// * `q` - Query tensor, shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor, shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor, shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `softmax_scale` - Scale applied to `Q @ K^T` before softmax.
/// * `causal` - Apply a causal mask (token `i` only attends to tokens `<= i`).
///
/// Returns a tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
///
/// # Errors
///
/// Returns an error if `q`, `k`, `v` are not F16/BF16 BSHD tensors, or if the
/// CK forward kernel has no matching instantiation for the given shapes.
pub fn flash_attn(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _softmax_scale: f32,
    _causal: bool,
) -> Result<Tensor> {
    candle_core::bail!("candle-flash-attn-rocm: forward pass not yet implemented")
}

/// Flash-attention v2 forward pass with a sliding attention window, CK backend.
///
/// Same as [`flash_attn`], but instead of a plain causal mask, each query
/// token only attends to key/value tokens within `window_size_left` tokens to
/// its left and `window_size_right` tokens to its right.
///
/// # Arguments
///
/// * `q` - Query tensor, shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor, shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor, shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `softmax_scale` - Scale applied to `Q @ K^T` before softmax.
/// * `window_size_left` - Limit attention to this many tokens to the left, or
///   `None` for unlimited.
/// * `window_size_right` - Limit attention to this many tokens to the right,
///   or `None` for unlimited.
///
/// `window_size_left = None` with `window_size_right = Some(0)` is a causal
/// mask, equivalent to `flash_attn(q, k, v, softmax_scale, true)`.
///
/// Returns a tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
///
/// # Errors
///
/// Returns an error if `q`, `k`, `v` are not F16/BF16 BSHD tensors, or if the
/// CK forward kernel has no matching instantiation for the given shapes.
pub fn flash_attn_windowed(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _softmax_scale: f32,
    _window_size_left: Option<usize>,
    _window_size_right: Option<usize>,
) -> Result<Tensor> {
    candle_core::bail!("candle-flash-attn-rocm: forward pass not yet implemented")
}
