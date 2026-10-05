# csrc

`flash_attn_ck/` is vendored, forward-pass-only, from
[ROCm/flash-attention](https://github.com/ROCm/flash-attention)
`csrc/flash_attn_ck/` at commit `a369df707e1980fb328abcc1733e3457ec10155f`.
Files are copied unmodified.

Vendored:
- `mha_fwd.cpp` — forward pass, constructs CK's `fmha_fwd_traits`/`fmha_fwd_args`
- `flash_common.cpp` / `flash_common.hpp` — shared utilities
- `mha_fwd_head_grouping_utils.hpp` — GQA head-grouping dispatch

Intentionally skipped (not needed for inference, or PyTorch-specific):
- `flash_api.cpp` (pybind11 module)
- `mha_bwd.cpp`, `mha_varlen_bwd.cpp` (backward pass)

`mha_varlen_fwd.cpp` and `mha_fwd_kvcache.cpp` will be added in a later
commit when the varlen/KV-cache API lands.

These files still depend on PyTorch's `at::Tensor` and pybind11; a later
commit replaces that glue with a C-ABI shim while keeping the CK dispatch
logic intact.
