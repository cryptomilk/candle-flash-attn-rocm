// SPDX-License-Identifier: MIT
//
// Correctness tests: compare the CK `flash_attn` kernel against a naive F32
// SDPA reference. Requires a ROCm GPU; there is no CPU fallback.

use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};

/// Deterministic, bounded-magnitude input so F16 can represent large tensors
/// without overflowing to infinity (unlike an unscaled `arange`).
fn bounded_tensor(
    shape: (usize, usize, usize, usize),
    modulus: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let elem_count = shape.0 * shape.1 * shape.2 * shape.3;
    let half = modulus as f32 / 2.0;
    let data: Vec<f32> = (0..elem_count)
        .map(|i| ((i % modulus) as f32 - half) / half)
        .collect();
    Ok(Tensor::from_vec(data, shape, device)?.to_dtype(dtype)?)
}

/// `(seqlen_q, seqlen_k)` additive mask: 0 where `k_idx <= q_idx`, `-inf`
/// otherwise.
fn causal_mask(seqlen_q: usize, seqlen_k: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seqlen_q * seqlen_k];
    for q_idx in 0..seqlen_q {
        for k_idx in (q_idx + 1)..seqlen_k {
            data[q_idx * seqlen_k + k_idx] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(data, (seqlen_q, seqlen_k), device)?)
}

/// Naive `softmax(Q @ K^T * scale [+ causal_mask]) @ V`, computed per-head in
/// F32. GQA is handled by mapping each query head to `head / groups` on the
/// K/V side, avoiding the need for a broadcast/expand of K/V.
fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let (b_sz, seqlen_q, num_heads, _head_dim) = q.dims4()?;
    let (_, seqlen_k, num_heads_k, _) = k.dims4()?;
    let groups = num_heads / num_heads_k;
    let device = q.device();
    let mask = causal
        .then(|| causal_mask(seqlen_q, seqlen_k, device))
        .transpose()?;

    let mut batches = Vec::with_capacity(b_sz);
    for b in 0..b_sz {
        let mut heads = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let kv_h = h / groups;
            let q_h = q.i((b, .., h, ..))?.contiguous()?;
            let k_h = k.i((b, .., kv_h, ..))?.contiguous()?;
            let v_h = v.i((b, .., kv_h, ..))?.contiguous()?;
            let att = (q_h.matmul(&k_h.t()?)? * softmax_scale as f64)?;
            let att = match &mask {
                Some(m) => att.broadcast_add(m)?,
                None => att,
            };
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            heads.push(att.matmul(&v_h)?);
        }
        batches.push(Tensor::stack(&heads, 1)?);
    }
    let out = Tensor::stack(&batches, 0)?;
    Ok(out.to_dtype(in_dtype)?)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let diff = (a - b)?.abs()?.flatten_all()?.max(0)?;
    Ok(diff.to_vec0::<f32>()?)
}

#[allow(clippy::too_many_arguments)]
fn run_test(
    batch: usize,
    seqlen_q: usize,
    seqlen_k: usize,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    dtype: DType,
    causal: bool,
    tol: f32,
) -> Result<()> {
    let device = Device::new_rocm(0)?;
    let q = bounded_tensor((batch, seqlen_q, heads_q, head_dim), 251, &device, dtype)?;
    let k = bounded_tensor((batch, seqlen_k, heads_kv, head_dim), 193, &device, dtype)?;
    let v = bounded_tensor((batch, seqlen_k, heads_kv, head_dim), 157, &device, dtype)?;

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let expected = naive_sdpa(&q, &k, &v, softmax_scale, causal)?;
    let actual = candle_flash_attn_rocm::flash_attn(&q, &k, &v, softmax_scale, causal)?;

    assert_eq!(actual.dims(), &[batch, seqlen_q, heads_q, head_dim]);

    let diff = max_abs_diff(&expected, &actual)?;
    assert!(diff < tol, "max abs diff {diff} exceeds tolerance {tol}");
    Ok(())
}

/// MHA, non-causal, F16.
#[test]
fn test_non_causal_mha_f16() -> Result<()> {
    run_test(2, 32, 32, 4, 4, 128, DType::F16, false, 1e-2)
}

/// MHA, causal, F16.
#[test]
fn test_causal_mha_f16() -> Result<()> {
    run_test(2, 32, 32, 4, 4, 128, DType::F16, true, 1e-2)
}

/// MHA, non-causal, BF16 (wider tolerance: fewer mantissa bits than F16).
#[test]
fn test_non_causal_mha_bf16() -> Result<()> {
    run_test(2, 32, 32, 4, 4, 128, DType::BF16, false, 5e-2)
}

/// MHA, causal, BF16.
#[test]
fn test_causal_mha_bf16() -> Result<()> {
    run_test(2, 32, 32, 4, 4, 128, DType::BF16, true, 5e-2)
}

/// GQA (8 query heads : 2 kv heads), causal, F16.
#[test]
fn test_gqa_causal_f16() -> Result<()> {
    run_test(2, 32, 32, 8, 2, 128, DType::F16, true, 1e-2)
}

/// GQA (8 query heads : 2 kv heads), causal, BF16.
#[test]
fn test_gqa_causal_bf16() -> Result<()> {
    run_test(2, 32, 32, 8, 2, 128, DType::BF16, true, 5e-2)
}

/// Medium sequence length (256), causal, F16.
#[test]
fn test_medium_seqlen() -> Result<()> {
    run_test(1, 256, 256, 4, 4, 128, DType::F16, true, 1e-2)
}
