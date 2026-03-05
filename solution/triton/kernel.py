import torch
import math

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
