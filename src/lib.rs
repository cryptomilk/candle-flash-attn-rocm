// SPDX-License-Identifier: MIT

mod ffi;

use candle_core::rocm_backend::{RocmStorageSlice, SendSyncDeviceMemory};
use candle_core::{CpuStorage, CustomOp3, Layout, Result, RocmDevice, RocmStorage, Shape, Tensor};

struct FlashAttn {
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
}

/// Converts a shape/stride value to the `i32` the FFI boundary takes, instead
/// of silently truncating a value candle tracks as `usize`.
fn to_i32(value: usize, what: &str) -> Result<i32> {
    match i32::try_from(value) {
        Ok(v) => Ok(v),
        Err(_) => candle_core::bail!("flash-attn-rocm: {what} ({value}) overflows i32"),
    }
}

/// `None`, or a window at least as wide as `seqlen_k`, both mean "unlimited"
/// to the CK shim, which it spells `-1`. The `< seqlen_k` cutoff mirrors the
/// native shim's own `window_size_left/right >= seqlen_k` clamp.
fn window_to_i32(window: Option<usize>, seqlen_k: usize) -> Result<i32> {
    match window.filter(|w| *w < seqlen_k) {
        Some(w) => to_i32(w, "window size"),
        None => Ok(-1),
    }
}

impl FlashAttn {
    /// Runs the CK forward kernel for one `(Q, K, V)` dtype, allocating the
    /// output and softmax-LSE scratch buffer and wrapping the result back
    /// into a [`RocmStorage`]. `wrap` is the `RocmStorageSlice` variant
    /// constructor matching `T` (`RocmStorageSlice::F16` or `::BF16`).
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn rocm_fwd_t<T>(
        &self,
        device: &RocmDevice,
        q: &SendSyncDeviceMemory<T>,
        q_l: &Layout,
        k: &SendSyncDeviceMemory<T>,
        k_l: &Layout,
        v: &SendSyncDeviceMemory<T>,
        v_l: &Layout,
        is_bf16: bool,
        wrap: fn(SendSyncDeviceMemory<T>) -> RocmStorageSlice,
    ) -> Result<(RocmStorage, Shape)> {
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        if q_stride.len() != 4 || k_stride.len() != 4 || v_stride.len() != 4 {
            candle_core::bail!(
                "flash-attn-rocm expects input tensors of rank 4 (q: {}, k: {}, v: {})",
                q_stride.len(),
                k_stride.len(),
                v_stride.len()
            )
        }
        if q_stride[3] != 1 {
            candle_core::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[3] != 1 {
            candle_core::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[3] != 1 {
            candle_core::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (b_sz, seqlen_q, num_heads, head_size) = q_l.shape().dims4()?;
        let (_, seqlen_k, num_heads_k, _) = k_l.shape().dims4()?;
        let expected_kv = (b_sz, seqlen_k, num_heads_k, head_size);
        if expected_kv != k_l.shape().dims4()? {
            candle_core::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims4()? {
            candle_core::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size > 256 {
            candle_core::bail!(
                "flash-attn-rocm only supports head dimensions up to 256 (got {head_size})"
            )
        }
        if head_size % 8 != 0 {
            candle_core::bail!(
                "flash-attn-rocm only supports head sizes that are a multiple of 8 (got {head_size})"
            )
        }
        if num_heads % num_heads_k != 0 {
            candle_core::bail!(
                "number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}"
            )
        }

        let window_size_left = window_to_i32(self.window_size_left, seqlen_k)?;
        let window_size_right = window_to_i32(self.window_size_right, seqlen_k)?;
        // Causal is the special case where window_size_right == 0 and
        // window_size_left is unlimited (-1); see flash_attn_shim.cpp.
        let is_causal = window_size_left < 0 && window_size_right == 0;

        let q_batch_stride = to_i32(q_stride[0], "q batch stride")?;
        let k_batch_stride = to_i32(k_stride[0], "k batch stride")?;
        let v_batch_stride = to_i32(v_stride[0], "v batch stride")?;
        let o_batch_stride = to_i32(o_stride[0], "output batch stride")?;
        let q_row_stride = to_i32(q_stride[1], "q row stride")?;
        let k_row_stride = to_i32(k_stride[1], "k row stride")?;
        let v_row_stride = to_i32(v_stride[1], "v row stride")?;
        let o_row_stride = to_i32(o_stride[1], "output row stride")?;
        let q_head_stride = to_i32(q_stride[2], "q head stride")?;
        let k_head_stride = to_i32(k_stride[2], "k head stride")?;
        let v_head_stride = to_i32(v_stride[2], "v head stride")?;
        let o_head_stride = to_i32(o_stride[2], "output head stride")?;
        let b_sz_i32 = to_i32(b_sz, "batch size")?;
        let seqlen_q_i32 = to_i32(seqlen_q, "seqlen_q")?;
        let seqlen_kv_i32 = to_i32(seqlen_k, "seqlen_k")?;
        let num_heads_i32 = to_i32(num_heads, "num_heads")?;
        let num_heads_k_i32 = to_i32(num_heads_k, "num_heads_k")?;
        let head_size_i32 = to_i32(head_size, "head_size")?;

        let elem_count = out_shape.elem_count();
        let out_mem = device.alloc::<T>(elem_count)?;
        let softmax_lse = device.alloc_zeros::<f32>(b_sz * num_heads * seqlen_q)?;

        // SAFETY: `start_offset()` is an element index within each tensor's
        // own allocation, and candle's `Layout`/`Storage` invariants
        // guarantee the whole strided `(batch, seqlen, heads, head_size)`
        // extent reachable from that offset via the strides passed below
        // stays within the backing allocation; `ptr_at` scales the offset by
        // `size_of::<T>()`, so each pointer lands inside its buffer. `out_mem`
        // and `softmax_lse` are freshly allocated with `elem_count` and
        // `b_sz * num_heads * seqlen_q` elements respectively, matching what
        // the CK kernel writes.
        let status = unsafe {
            ffi::flash_attn_rocm_fwd(
                q.ptr_at(q_l.start_offset()).cast_const(),
                k.ptr_at(k_l.start_offset()).cast_const(),
                v.ptr_at(v_l.start_offset()).cast_const(),
                out_mem.as_ptr(),
                softmax_lse.as_ptr(),
                q_batch_stride,
                k_batch_stride,
                v_batch_stride,
                o_batch_stride,
                q_row_stride,
                k_row_stride,
                v_row_stride,
                o_row_stride,
                q_head_stride,
                k_head_stride,
                v_head_stride,
                o_head_stride,
                b_sz_i32,
                seqlen_q_i32,
                seqlen_kv_i32,
                num_heads_i32,
                num_heads_k_i32,
                head_size_i32,
                self.softmax_scale,
                i32::from(is_causal),
                window_size_left,
                window_size_right,
                i32::from(is_bf16),
                device.stream().as_raw().cast(),
            )
        };
        if status != 0 {
            candle_core::bail!("flash-attn-rocm: CK forward kernel failed with status {status}")
        }

        Ok((
            RocmStorage {
                slice: wrap(out_mem),
                device: device.clone(),
            },
            out_shape,
        ))
    }
}

impl CustomOp3 for FlashAttn {
    fn name(&self) -> &'static str {
        "flash-attn-rocm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for flash-attn-rocm")
    }

    fn rocm_fwd(
        &self,
        q: &RocmStorage,
        q_l: &Layout,
        k: &RocmStorage,
        k_l: &Layout,
        v: &RocmStorage,
        v_l: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        match (&q.slice, &k.slice, &v.slice) {
            (RocmStorageSlice::F16(qm), RocmStorageSlice::F16(km), RocmStorageSlice::F16(vm)) => {
                self.rocm_fwd_t(
                    &q.device,
                    qm,
                    q_l,
                    km,
                    k_l,
                    vm,
                    v_l,
                    false,
                    RocmStorageSlice::F16,
                )
            }
            (
                RocmStorageSlice::BF16(qm),
                RocmStorageSlice::BF16(km),
                RocmStorageSlice::BF16(vm),
            ) => self.rocm_fwd_t(
                &q.device,
                qm,
                q_l,
                km,
                k_l,
                vm,
                v_l,
                true,
                RocmStorageSlice::BF16,
            ),
            _ => candle_core::bail!(
                "flash-attn-rocm requires q, k, v to all be f16 or all be bf16 (got q={:?}, k={:?}, v={:?})",
                q.slice.dtype(),
                k.slice.dtype(),
                v.slice.dtype()
            ),
        }
    }
}

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
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };
    let op = FlashAttn {
        softmax_scale,
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
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
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}
