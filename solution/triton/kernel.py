import torch
import math


def _dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """
    Dequantize deep_gemm packed KV cache.
    Input shape: [num_pages, page_size, 1, 132] int8 (viewed as uint8 bytes)
    Output shape: [num_pages, page_size, 128] float32
    """
    k_u8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_with_scale = k_u8.shape
    head_dim = head_dim_with_scale - 4  # 128

    kv_flat = k_u8.view(num_pages, page_size * head_dim_with_scale)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
    return fp8_float * scale


def topk_indexer_kernel(
    q_index_fp8,      # [batch_size, 64, 128], float8_e4m3fn
    k_index_cache_fp8,  # [num_pages, 64, 1, 132], int8 packed fp8+scale
    weights,          # [batch_size, 64], float32
    seq_lens,         # [batch_size], int32
    block_table,      # [batch_size, max_num_pages], int32
    topk_indices,     # [batch_size, 2048], int32 (destination tensor)
):
    """
    DSA TopK indexer baseline implementation.
    Destination-passing style implementation.
    Fills topk_indices in-place with shape [batch_size, 2048], int32.
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device
    q = q_index_fp8.to(torch.float32)
    k_all = _dequant_fp8_kv_cache(k_index_cache_fp8)

    topk_indices.fill_(-1)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len <= 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        k_paged = k_all[page_indices]
        k = k_paged.reshape(-1, index_head_dim)[:seq_len]
        q_b = q[b]

        scores = q_b @ k.T
        scores_relu = torch.relu(scores)
        final_scores = (scores_relu * weights[b][:, None]).sum(dim=0)

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token
        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return topk_indices


def sparse_attention_kernel(
    q_c,               # q_nope: [num_tokens, 16, 512]
    q_r,               # q_pe: [num_tokens, 16, 64]
    kv_c_pool,         # ckv_cache: [num_pages, 64, 512]
    kv_r_pool,         # kpe_cache: [num_pages, 64, 64]
    kv_indices,        # sparse_indices: [num_tokens, 2048]
    sm_scale,          # float
    out_tensor,        # output: [num_tokens, 16, 512]
    lse_tensor         # lse: [num_tokens, 16]
):
    """
    Correct implementation matching the reference from definition file.
    Handles variable num_tokens, -1 padding in indices, and uses base-2 log for LSE.
    """
    device = q_c.device
    num_tokens, num_qo_heads, head_dim_ckv = q_c.shape
    head_dim_kpe = q_r.shape[-1]
    num_pages, page_size, _ = kv_c_pool.shape
    topk = kv_indices.shape[-1]

    # Verify constants (should match definition)
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = kv_c_pool.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kv_r_pool.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

    # Process each token
    for t in range(num_tokens):
        indices = kv_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask].to(torch.long)

        if valid_indices.numel() == 0:
            out_tensor[t].zero_()
            lse_tensor[t].fill_(-float('inf'))
            continue

        # Get KV entries for valid indices
        Kc = Kc_all[valid_indices]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[valid_indices]  # [num_valid, head_dim_kpe]
        qn = q_c[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_r[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits: [num_qo_heads, num_valid]
        logits = torch.matmul(qn, Kc.T) + torch.matmul(qp, Kp.T)
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE: log2(exp(logits_scaled).sum())
        lse = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)  # [num_qo_heads]
        lse_tensor[t] = lse.to(lse_tensor.dtype)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = torch.matmul(attn, Kc)  # [num_qo_heads, head_dim_ckv]
        out_tensor[t] = out.to(out_tensor.dtype)

    return out_tensor
