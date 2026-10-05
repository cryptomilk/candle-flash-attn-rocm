// SPDX-License-Identifier: MIT
//
// C-ABI shim around CK's fused-MHA forward pass. Reimplements the dispatch
// logic from csrc/flash_attn_ck/mha_fwd.cpp and
// csrc/flash_attn_ck/mha_fwd_head_grouping_utils.hpp without PyTorch types.

#include "flash_attn_shim.h"

#include "fmha_fwd.hpp"
#include "fmha_fwd_head_grouping.hpp"

#include <hip/hip_runtime.h>

#include <string>
#include <utility>

namespace {

fmha_fwd_traits get_fwd_traits(const mask_info& mask, bool is_bf16, int32_t head_size)
{
    return fmha_fwd_traits{head_size,
                            head_size,
                            is_bf16 ? std::string("bf16") : std::string("fp16"),
                            false, // is_group_mode
                            true,  // is_v_rowmajor
                            false, // has_logits_soft_cap
                            mask.type,
                            bias_enum::no_bias,
                            true,  // has_lse
                            false, // has_dropout
                            quant_scale_enum::no_scale};
}

fmha_fwd_args get_fwd_args(const mask_info& mask,
                            const void* q_ptr,
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
                            float softmax_scale)
{
    // softmax_lse is (batch, num_heads, seqlen_q), row-major contiguous.
    const ck_tile::index_t nhead_stride_lse = seqlen_q;
    const ck_tile::index_t batch_stride_lse =
        static_cast<ck_tile::index_t>(num_heads) * seqlen_q;

    return fmha_fwd_args{q_ptr,
                          k_ptr,
                          v_ptr,
                          nullptr, // bias_ptr
                          nullptr, // q_descale_ptr
                          nullptr, // k_descale_ptr
                          nullptr, // v_descale_ptr
                          nullptr, // rand_val_ptr
                          softmax_lse_ptr,
                          out_ptr,
                          nullptr, // seqstart_q_ptr
                          nullptr, // seqstart_k_ptr
                          nullptr, // seqlen_q_ptr
                          nullptr, // seqlen_k_ptr
                          nullptr, // cu_seqlen_q_ptr
                          nullptr, // cu_seqlen_k_ptr
                          nullptr, // block_scale_seqstart_q_ptr
                          nullptr, // block_scale_seqstart_k_ptr
                          nullptr, // seqstart_v_scale_ptr
                          nullptr, // sink_ptr
                          seqlen_q,
                          seqlen_k,
                          batch,
                          seqlen_q,  // max_seqlen_q
                          head_size, // hdim_q
                          head_size, // hdim_v
                          num_heads,
                          num_heads_k,
                          0, // num_head_q_total
                          0, // head_start
                          softmax_scale,
                          0.0f, // logits_soft_cap
                          q_row_stride,
                          k_row_stride,
                          v_row_stride,
                          0, // stride_bias
                          0, // stride_randval
                          o_row_stride,
                          0, // stride_q_descale
                          0, // stride_k_descale
                          0, // stride_v_descale
                          q_head_stride,
                          k_head_stride,
                          v_head_stride,
                          0, // nhead_stride_bias
                          0, // nhead_stride_randval
                          nhead_stride_lse,
                          o_head_stride,
                          0, // nhead_stride_q_descale
                          0, // nhead_stride_k_descale
                          0, // nhead_stride_v_descale
                          q_batch_stride,
                          k_batch_stride,
                          v_batch_stride,
                          0, // batch_stride_bias
                          0, // batch_stride_randval
                          batch_stride_lse,
                          o_batch_stride,
                          0, // batch_stride_q_descale
                          0, // batch_stride_k_descale
                          0, // batch_stride_v_descale
                          mask.left,
                          mask.right,
                          0, // sink_size
                          static_cast<ck_tile::index_t>(mask.type),
                          0,    // min_seqlen_q
                          0.0f, // p_drop
                          false, // s_randval
                          std::pair<uint64_t, uint64_t>{0, 0}, // drop_seed_offset
                          0,  // block_scale_size_q
                          0}; // block_scale_size_kv
}

fmha_fwd_traits get_varlen_fwd_traits(const mask_info& mask, bool is_bf16, int32_t head_size)
{
    return fmha_fwd_traits{head_size,
                            head_size,
                            is_bf16 ? std::string("bf16") : std::string("fp16"),
                            true,  // is_group_mode
                            true,  // is_v_rowmajor
                            false, // has_logits_soft_cap
                            mask.type,
                            bias_enum::no_bias,
                            true,  // has_lse
                            false, // has_dropout
                            quant_scale_enum::no_scale};
}

// Builds `fmha_fwd_args` for group mode (packed/varlen sequences). Unlike
// `get_fwd_args`, `seqstart_q_ptr`/`seqstart_k_ptr` carry the cu_seqlens
// device pointers CK uses to locate each sequence within the packed q/k/v
// buffers, batch strides are always 0, and the struct's `seqlen_q`/`seqlen_k`
// fields are repurposed by CK's group-mode path to hold `total_q`/`total_k`.
fmha_fwd_args get_varlen_fwd_args(const mask_info& mask,
                                   const void* q_ptr,
                                   const void* k_ptr,
                                   const void* v_ptr,
                                   void* out_ptr,
                                   void* softmax_lse_ptr,
                                   const int32_t* cu_seqlens_q,
                                   const int32_t* cu_seqlens_k,
                                   int32_t q_row_stride,
                                   int32_t k_row_stride,
                                   int32_t v_row_stride,
                                   int32_t o_row_stride,
                                   int32_t q_head_stride,
                                   int32_t k_head_stride,
                                   int32_t v_head_stride,
                                   int32_t o_head_stride,
                                   int32_t batch,
                                   int32_t total_q,
                                   int32_t total_k,
                                   int32_t max_seqlen_q,
                                   int32_t num_heads,
                                   int32_t num_heads_k,
                                   int32_t head_size,
                                   float softmax_scale)
{
    // softmax_lse is (num_heads, total_q), row-major contiguous.
    const ck_tile::index_t nhead_stride_lse = total_q;

    return fmha_fwd_args{q_ptr,
                          k_ptr,
                          v_ptr,
                          nullptr, // bias_ptr
                          nullptr, // q_descale_ptr
                          nullptr, // k_descale_ptr
                          nullptr, // v_descale_ptr
                          nullptr, // rand_val_ptr
                          softmax_lse_ptr,
                          out_ptr,
                          cu_seqlens_q, // seqstart_q_ptr
                          cu_seqlens_k, // seqstart_k_ptr
                          nullptr,      // seqlen_q_ptr
                          nullptr,      // seqlen_k_ptr
                          nullptr,      // cu_seqlen_q_ptr
                          nullptr,      // cu_seqlen_k_ptr
                          nullptr,      // block_scale_seqstart_q_ptr
                          nullptr,      // block_scale_seqstart_k_ptr
                          nullptr,      // seqstart_v_scale_ptr
                          nullptr,      // sink_ptr
                          total_q,
                          total_k,
                          batch,
                          max_seqlen_q,
                          head_size, // hdim_q
                          head_size, // hdim_v
                          num_heads,
                          num_heads_k,
                          0, // num_head_q_total
                          0, // head_start
                          softmax_scale,
                          0.0f, // logits_soft_cap
                          q_row_stride,
                          k_row_stride,
                          v_row_stride,
                          0, // stride_bias
                          0, // stride_randval
                          o_row_stride,
                          0, // stride_q_descale
                          0, // stride_k_descale
                          0, // stride_v_descale
                          q_head_stride,
                          k_head_stride,
                          v_head_stride,
                          0, // nhead_stride_bias
                          0, // nhead_stride_randval
                          nhead_stride_lse,
                          o_head_stride,
                          0, // nhead_stride_q_descale
                          0, // nhead_stride_k_descale
                          0, // nhead_stride_v_descale
                          0, // batch_stride_q
                          0, // batch_stride_k
                          0, // batch_stride_v
                          0, // batch_stride_bias
                          0, // batch_stride_randval
                          0, // batch_stride_lse
                          0, // batch_stride_o
                          0, // batch_stride_q_descale
                          0, // batch_stride_k_descale
                          0, // batch_stride_v_descale
                          mask.left,
                          mask.right,
                          0, // sink_size
                          static_cast<ck_tile::index_t>(mask.type),
                          0,    // min_seqlen_q
                          0.0f, // p_drop
                          false, // s_randval
                          std::pair<uint64_t, uint64_t>{0, 0}, // drop_seed_offset
                          0,  // block_scale_size_q
                          0}; // block_scale_size_kv
}

// Mirrors flash::maybe_dispatch_head_grouped_fwd from
// csrc/flash_attn_ck/mha_fwd_head_grouping_utils.hpp, with `is_bf16`
// replacing the PyTorch `at::ScalarType` dtype tag.
template <typename FmhaFwdTypeTag>
float dispatch_head_grouped(const ck_tile::stream_config& sc,
                             const fmha_fwd_traits& traits,
                             const fmha_fwd_args& args,
                             ck_tile::index_t num_heads,
                             ck_tile::index_t num_heads_k,
                             ck_tile::index_t group_size)
{
    using TypeConfig = FmhaFwdTypeConfig<FmhaFwdTypeTag>;
    return fmha_fwd_head_grouping::run_fwd_head_grouped<typename TypeConfig::QDataType,
                                                        typename TypeConfig::KDataType,
                                                        typename TypeConfig::VDataType,
                                                        typename TypeConfig::ODataType,
                                                        float,
                                                        typename TypeConfig::LSEDataType,
                                                        typename TypeConfig::RandValOutputDataType>(
        sc,
        traits,
        args,
        num_heads,
        num_heads_k,
        group_size,
        false, // use_blockscale_qscale
        [](const auto& grouped_traits, auto& grouped_args, const auto& grouped_sc) {
            return fmha_fwd(grouped_traits, grouped_args, grouped_sc);
        });
}

float maybe_dispatch_head_grouped(const ck_tile::stream_config& sc,
                                   const fmha_fwd_traits& traits,
                                   const fmha_fwd_args& args,
                                   ck_tile::index_t num_heads,
                                   ck_tile::index_t num_heads_k,
                                   ck_tile::index_t batch,
                                   ck_tile::index_t seqlen_k,
                                   ck_tile::index_t head_size,
                                   bool is_bf16)
{
    namespace head_grouping = fmha_fwd_head_grouping;

    if (head_grouping::disabled_by_env()) {
        return -1.0f;
    }

    // fp16 and bf16 are both 2 bytes per element.
    constexpr size_t kElemBytes = 2;
    const auto group_size_opt = head_grouping::get_head_group_size(
        num_heads, num_heads_k, batch, seqlen_k, head_size, head_size, kElemBytes, kElemBytes);
    if (!group_size_opt.has_value() || group_size_opt.value() >= num_heads) {
        return -1.0f;
    }

    if (is_bf16) {
        return dispatch_head_grouped<FmhaFwdBf16>(
            sc, traits, args, num_heads, num_heads_k, group_size_opt.value());
    }
    return dispatch_head_grouped<FmhaFwdFp16>(
        sc, traits, args, num_heads, num_heads_k, group_size_opt.value());
}

} // namespace

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
                         void* stream)
{
    if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        head_size <= 0) {
        return -2;
    }
    if (num_heads % num_heads_k != 0) {
        return -2;
    }
    if (head_size > 256 || head_size % 8 != 0) {
        return -2;
    }

    if (window_size_left >= seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k) {
        window_size_right = -1;
    }
    // causal=true is equivalent to causal=false when seqlen_q == 1.
    if (seqlen_q == 1) {
        is_causal = 0;
    }

    mask_info mask;
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and
        // window_size_left < 0.
        window_size_right = 0;
        const std::string ident = "b:" + std::to_string(window_size_left) + ",0";
        mask = mask_info::decode(ident, seqlen_q, seqlen_k);
    } else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", seqlen_q, seqlen_k);
    } else {
        const std::string ident =
            "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(ident, seqlen_q, seqlen_k);
    }

    const fmha_fwd_traits traits = get_fwd_traits(mask, is_bf16 != 0, head_size);
    const fmha_fwd_args args = get_fwd_args(mask,
                                             q_ptr,
                                             k_ptr,
                                             v_ptr,
                                             out_ptr,
                                             softmax_lse_ptr,
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
                                             batch,
                                             seqlen_q,
                                             seqlen_k,
                                             num_heads,
                                             num_heads_k,
                                             head_size,
                                             softmax_scale);

    const ck_tile::stream_config stream_config{static_cast<hipStream_t>(stream)};

    float t = maybe_dispatch_head_grouped(
        stream_config, traits, args, num_heads, num_heads_k, batch, seqlen_k, head_size, is_bf16 != 0);
    if (t < 0.0f) {
        t = fmha_fwd(traits, args, stream_config);
    }

    return t >= 0.0f ? 0 : -1;
}

int flash_attn_rocm_varlen_fwd(const void* q_ptr,
                                const void* k_ptr,
                                const void* v_ptr,
                                void* out_ptr,
                                void* softmax_lse_ptr,
                                const int32_t* cu_seqlens_q,
                                const int32_t* cu_seqlens_k,
                                int32_t q_row_stride,
                                int32_t k_row_stride,
                                int32_t v_row_stride,
                                int32_t o_row_stride,
                                int32_t q_head_stride,
                                int32_t k_head_stride,
                                int32_t v_head_stride,
                                int32_t o_head_stride,
                                int32_t batch,
                                int32_t total_q,
                                int32_t total_k,
                                int32_t max_seqlen_q,
                                int32_t max_seqlen_k,
                                int32_t num_heads,
                                int32_t num_heads_k,
                                int32_t head_size,
                                float softmax_scale,
                                int is_causal,
                                int window_size_left,
                                int window_size_right,
                                int is_bf16,
                                void* stream)
{
    if (batch <= 0 || total_q <= 0 || total_k <= 0 || max_seqlen_q <= 0 || max_seqlen_k <= 0 ||
        num_heads <= 0 || num_heads_k <= 0 || head_size <= 0) {
        return -2;
    }
    if (num_heads % num_heads_k != 0) {
        return -2;
    }
    if (head_size > 256 || head_size % 8 != 0) {
        return -2;
    }
    if (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr) {
        return -2;
    }

    if (window_size_left >= max_seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
        window_size_right = -1;
    }
    // causal=true is equivalent to causal=false when max_seqlen_q == 1.
    if (max_seqlen_q == 1) {
        is_causal = 0;
    }

    mask_info mask;
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and
        // window_size_left < 0.
        window_size_right = 0;
        const std::string ident = "b:" + std::to_string(window_size_left) + ",0";
        mask = mask_info::decode(ident, max_seqlen_q, max_seqlen_k);
    } else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", max_seqlen_q, max_seqlen_k);
    } else {
        const std::string ident =
            "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(ident, max_seqlen_q, max_seqlen_k);
    }

    const fmha_fwd_traits traits = get_varlen_fwd_traits(mask, is_bf16 != 0, head_size);
    const fmha_fwd_args args = get_varlen_fwd_args(mask,
                                                    q_ptr,
                                                    k_ptr,
                                                    v_ptr,
                                                    out_ptr,
                                                    softmax_lse_ptr,
                                                    cu_seqlens_q,
                                                    cu_seqlens_k,
                                                    q_row_stride,
                                                    k_row_stride,
                                                    v_row_stride,
                                                    o_row_stride,
                                                    q_head_stride,
                                                    k_head_stride,
                                                    v_head_stride,
                                                    o_head_stride,
                                                    batch,
                                                    total_q,
                                                    total_k,
                                                    max_seqlen_q,
                                                    num_heads,
                                                    num_heads_k,
                                                    head_size,
                                                    softmax_scale);

    const ck_tile::stream_config stream_config{static_cast<hipStream_t>(stream)};

    float t = maybe_dispatch_head_grouped(stream_config,
                                           traits,
                                           args,
                                           num_heads,
                                           num_heads_k,
                                           batch,
                                           max_seqlen_k,
                                           head_size,
                                           is_bf16 != 0);
    if (t < 0.0f) {
        t = fmha_fwd(traits, args, stream_config);
    }

    return t >= 0.0f ? 0 : -1;
}
