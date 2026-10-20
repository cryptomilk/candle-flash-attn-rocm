# kernels

`generated/` holds CK template instantiations produced by the
`composable_kernel` submodule's `generate.py`. These files are generated,
not hand-edited.

Regenerate with (from the repo root, submodule pinned as in `.gitmodules`):

```
python3 composable_kernel/example/ck_tile/01_fmha/generate.py \
    -d fwd --receipt 2 --optdim 128 \
    --filter "*ndropout*" --targets gfx90a,gfx1100,gfx1200,gfx1201 \
    --output_dir kernels/generated/
```

This produces inference-only instantiations (no dropout) for head_dim=128,
covering gfx90a (gfx9 family, CDNA), gfx1100 (gfx11 family, RDNA3), and
gfx1200/gfx1201 (gfx12 family, RDNA4), both causal and non-causal masks,
and both ALiBi and no-bias variants. It also produces `fmha_fwd_api.cpp`,
the dispatch function that selects the right instantiation at runtime.
gfx1200 and gfx1201 are separate RDNA4 chips but share one `gfx12` family
in CK's codegen, so listing both targets doesn't add any files beyond
listing just one.
