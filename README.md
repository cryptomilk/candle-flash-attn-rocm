# candle-flash-attn-rocm

FlashAttention-2 forward pass for [candle](https://github.com/huggingface/candle)
on AMD ROCm GPUs, backed by AMD's
[Composable Kernel](https://github.com/ROCm/composable_kernel) (CK) fused-MHA
kernels. This is a ROCm counterpart to upstream candle's CUDA
`candle-flash-attn` crate: forward-pass only, built for inference.

## What it provides

- `flash_attn` runs the causal or non-causal forward pass. It uses the
  batched BSHD layout.
- `flash_attn_windowed` adds sliding-window attention. It takes left and
  right window sizes.
- `flash_attn_varlen` and `flash_attn_varlen_windowed` handle packed
  variable-length sequences (`cu_seqlens`-style). They work with or without
  a sliding window.

All entry points accept F16 or BF16 tensors and natively support
multi-query/grouped-query attention (`num_heads_q` a multiple of
`num_heads_kv`). See the doc comments on each function in `src/lib.rs` for
exact shapes and arguments (`cargo doc --open`).

## Example

```rust
use candle_core::{DType, Device, Tensor};
use candle_flash_attn_rocm::flash_attn;

let device = Device::new_rocm(0)?;
let (batch, seqlen, num_heads, head_dim) = (2, 1024, 16, 128);
let q = Tensor::randn(0f32, 1f32, (batch, seqlen, num_heads, head_dim), &device)?
    .to_dtype(DType::F16)?;
let k = q.clone();
let v = q.clone();

let softmax_scale = 1f32 / (head_dim as f32).sqrt();
let out = flash_attn(&q, &k, &v, softmax_scale, /* causal */ true)?;
```

## Build requirements

Compiling this crate requires a ROCm toolchain with `hipcc`. There is no
pure-Rust fallback, and compilation always runs even if a ROCm GPU isn't
present on the build machine.

- ROCm with `hipcc` installed, found via (in order): the `HIPCC` env var, a
  `ROCM_PATH` env var, or `/opt/rocm` on `PATH`.
- The `composable_kernel` git submodule checked out:
  `git submodule update --init composable_kernel`.
- A C++20-capable Clang (ships with ROCm's `hipcc`).

Target architecture selection, in order of priority:

1. `ROCM_TARGET_ARCH`, a comma-separated list, e.g. `ROCM_TARGET_ARCH=gfx90a,gfx1100`.
2. Auto-detected via `rocminfo`, if available.
3. Falls back to `gfx90a,gfx1100`.

Only the `gfx9` family (CDNA, e.g. `gfx90a`) and the `gfx11` family (RDNA3,
e.g. `gfx1100`) have vendored kernel instantiations in `kernels/generated/`.
Other architectures need the `composable_kernel` `generate.py` run manually,
with the output added to `kernels/generated/` (see `kernels/README.md`).

`CANDLE_FLASH_ATTN_ROCM_BUILD_DIR` can point `build.rs` at a persistent
directory to cache compiled objects across builds, instead of `OUT_DIR`.

## Limitations

- Forward pass only. No backward or training support.
- F16 and BF16 only; F32 tensors must be cast by the caller.
- Head dimensions up to 256, a multiple of 8, but only `head_dim = 128` has
  vendored kernel instantiations right now. Other head dims need
  regenerating `kernels/generated/` (see `kernels/README.md`).
- No ALiBi or KV-cache Rust entry points yet, though the vendored CK kernel
  instantiations already include ALiBi variants for a future `flash_attn_alibi`.
