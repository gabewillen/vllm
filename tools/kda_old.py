# SPDX-License-Identifier: Apache-2.0
"""Gaudi (HPU) KDA kernels for GLM-5.3-Flash written as plain torch ops.

Everything here is shape-static and free of host syncs so it can be captured
by ``torch.compile(backend="hpu_backend")``.  Semantics follow the HF
reference (``recurrent_kimi_delta_attention`` / ``chunk_kimi_delta_attention``):

* per-key-dim log-decay ``g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))``
* ``beta = sigmoid(b)``; q/k are L2-normalised (eps inside the sqrt), q scaled
  by ``head_dim ** -0.5``
* recurrent state per head is ``h[V, K]`` (vLLM layout ``[slots, H, V, K]``),
  kept in fp32.

Padding tokens are neutralised by forcing ``g = 0`` (decay 1) and
``beta = 0`` so they never change the recurrent state, which lets prefill run
over the padded bucket without knowing the true lengths on the host.
"""

import torch
import torch.nn.functional as F


def kda_gate(f: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float) -> torch.Tensor:
    """f: [..., H, K] raw forget-gate projection -> fp32 log-decay."""
    H, K = f.shape[-2], f.shape[-1]
    return lower_bound * torch.sigmoid(
        torch.exp(a_log.float().reshape(H, 1)) * (f.float() + dt_bias.float().reshape(H, K)))


def _l2n(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def conv_decode(x: torch.Tensor, conv_state: torch.Tensor, conv_w_t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Single-token causal depthwise conv + silu.

    x: [B, D]; conv_state: [slots, W-1, D]; conv_w_t: [W, D] fp32; idx: [B] long.
    Updates conv_state rows ``idx`` in place and returns fp32 [B, D].
    """
    st = conv_state.index_select(0, idx)
    win = torch.cat([st, x.unsqueeze(1).to(st.dtype)], dim=1)  # [B, W, D]
    out = F.silu((win.float() * conv_w_t.unsqueeze(0)).sum(dim=1))
    conv_state.index_copy_(0, idx, win[:, 1:])
    return out


def conv_prefill(x: torch.Tensor, lens: torch.Tensor, conv_state: torch.Tensor, conv_w_t: torch.Tensor,
                 load_idx: torch.Tensor, store_idx: torch.Tensor, has_init: torch.Tensor) -> torch.Tensor:
    """Causal depthwise conv over padded sequences.

    x: [B, S, D] (padding already zeroed); lens: [B] valid lengths (long);
    has_init: [B] 0/1.  Writes the last W-1 valid inputs back to conv_state.
    """
    B, S, D = x.shape
    W = conv_w_t.shape[0]
    st = conv_state.index_select(0, load_idx) * has_init.view(B, 1, 1).to(conv_state.dtype)
    xpad = torch.cat([st, x.to(st.dtype)], dim=1)  # [B, S + W - 1, D]
    xf = xpad.float()
    out = xf[:, 0:S] * conv_w_t[0]
    for j in range(1, W):
        out = out + xf[:, j:j + S] * conv_w_t[j]
    # last W-1 valid inputs live at xpad[lens + 0 .. lens + W - 2]
    gidx = (lens.view(B, 1) + torch.arange(W - 1, device=x.device).view(1, W - 1))
    new_st = torch.gather(xpad, 1, gidx.unsqueeze(-1).expand(B, W - 1, D))
    conv_state.index_copy_(0, store_idx, new_st)
    return F.silu(out)


def kda_decode_step(q, k, v, g, beta, rec_state, idx, scale):
    """One recurrent step for B independent sequences.

    q/k/g: [B, H, K]; v: [B, H, V]; beta: [B, H]; rec_state: [slots, H, V, K].
    Returns fp32 [B, H, V]; updates rec_state rows ``idx`` in place.
    """
    h = rec_state.index_select(0, idx).float()
    q = _l2n(q) * scale
    k = _l2n(k)
    h = h * torch.exp(g).unsqueeze(2)
    proj = torch.matmul(h, k.unsqueeze(-1)).squeeze(-1)
    v_new = (v.float() - proj) * beta.float().unsqueeze(-1)
    h = h + v_new.unsqueeze(-1) * k.unsqueeze(2)
    o = torch.matmul(h, q.unsqueeze(-1)).squeeze(-1)
    rec_state.index_copy_(0, idx, h.to(rec_state.dtype))
    return o


def _unit_lower_inverse(A: torch.Tensor, base: int = 16) -> torch.Tensor:
    """(I - A)^-1 for strictly lower-triangular A [..., C, C].

    Diagonal ``base``-blocks use the exact row recurrence (as in HF), batched
    over all blocks; blocks are then merged pairwise with
    [[P, 0], [Q, R]]^-1 = [[P^-1, 0], [R^-1 Q' P^-1, R^-1]] (Q' = A21).
    Keeps the traced graph small (base - 1 + log2(C / base) steps).
    """
    C = A.shape[-1]
    lead = A.shape[:-2]
    nb = C // base
    blk = torch.stack([A[..., i * base:(i + 1) * base, i * base:(i + 1) * base] for i in range(nb)], dim=-3)
    t = blk.clone()
    for i in range(1, base):
        row = t[..., i, :i].clone()
        sub = t[..., :i, :i].clone()
        t[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    inv = t + torch.eye(base, dtype=A.dtype, device=A.device)  # [..., nb, base, base]
    size = base
    while size < C:
        n2 = inv.shape[-3] // 2
        P = inv[..., 0::2, :, :]
        R = inv[..., 1::2, :, :]
        Q = torch.stack([A[..., (2 * j + 1) * size:(2 * j + 2) * size, 2 * j * size:(2 * j + 1) * size]
                         for j in range(n2)], dim=-3)
        off = R @ Q @ P
        z = torch.zeros_like(P)
        top = torch.cat([P, z], dim=-1)
        bot = torch.cat([off, R], dim=-1)
        inv = torch.cat([top, bot], dim=-2)
        size *= 2
    return inv.reshape(*lead, C, C)


def _decayed_scores(xq: torch.Tensor, kk: torch.Tensor, g: torch.Tensor, blk: int = 16) -> torch.Tensor:
    """M[i, j] = sum_k xq[i,k] kk[j,k] exp(g[i,k] - g[j,k]) for j <= i (block-lower).

    xq/kk/g: [N, C, K] with g the within-chunk cumulative log-decay (non-
    increasing).  Instead of materialising [C, C, K] decay tensors the chunk
    is split into ``blk``-token blocks and every exponential is factored into
    bounded terms: off-diagonal blocks use factors <= 1 plus a per-block-pair
    decay vector; diagonal blocks use exp(g0 - g_j) <= exp(5 * blk) (the KDA
    log-decay is >= lower_bound = -5 per token), which fits in fp32 for
    blk = 16.  Entries above the block diagonal are zero; callers mask the
    diagonal blocks' upper triangle.
    """
    N, C, K = xq.shape
    nb = C // blk
    xb, kb, gb = (t.reshape(N, nb, blk, K) for t in (xq, kk, g))
    g0 = gb[:, :, :1]      # block start
    gl = gb[:, :, -1:]     # block end
    xs = xb * torch.exp(gb - g0)        # <= 1
    kd = kb * torch.exp(g0 - gb)        # diagonal-block keys
    kl = kb * torch.exp(gl - gb)        # off-diagonal keys, <= 1
    diag = torch.matmul(xs, kd.transpose(-1, -2))  # [N, nb, blk, blk]
    rows = []
    zero = torch.zeros(N, blk, blk, dtype=xq.dtype, device=xq.device)
    for a in range(nb):
        blocks = []
        for b in range(nb):
            if b < a:
                dec = torch.exp(g0[:, a] - gl[:, b])  # [N, 1, K], <= 1
                blocks.append(torch.matmul(xs[:, a] * dec, kl[:, b].transpose(-1, -2)))
            elif b == a:
                blocks.append(diag[:, a])
            else:
                blocks.append(zero)
        rows.append(torch.cat(blocks, dim=-1))
    return torch.cat(rows, dim=-2)  # [N, C, C]


def kda_chunk_prefill(q, k, v, g, beta, init_state, scale, chunk_size: int = 64):
    """Chunked KDA over padded sequences (port of HF chunk_kimi_delta_attention).

    q/k/g: [B, S, H, K]; v: [B, S, H, V]; beta: [B, S, H]; init_state: [B, H, V, K]
    (fp32).  Returns (out fp32 [B, S, H, V], final_state [B, H, V, K]).
    """
    B, S, H, K = k.shape
    V = v.shape[-1]
    q = _l2n(q) * scale
    k = _l2n(k)
    q, k, v, g = (x.float().transpose(1, 2) for x in (q, k, v, g))  # [B, H, S, *]
    beta = beta.float().transpose(1, 2)  # [B, H, S]
    C = chunk_size
    pad = (C - S % C) % C
    if pad:
        q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, pad))
    N = (S + pad) // C
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    # NOTE: flatten to <= 5-D contiguous tensors. On Gaudi SW 1.24.1 a 6-D
    # broadcast mul+sum fed by a transpose-derived (strided) view returned
    # wrong values (see tools/repro_6d_b.py); contiguous inputs are fine.
    BHN = B * H * N
    q, k, v, g, k_beta, v_beta = (x.reshape(BHN, C, x.shape[-1]) for x in (q, k, v, g, k_beta, v_beta))

    g = g.cumsum(dim=-2)  # [BHN, C, K]
    tri_incl = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)
    tri_strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    attn = -_decayed_scores(k_beta, k, g).masked_fill(tri_incl, 0.0)
    attn = _unit_lower_inverse(attn)  # == HF row recurrence + I
    u = attn @ v_beta
    w = attn @ (k_beta * g.exp())

    # intra-chunk q.k scores with decay (i >= j)
    qk = _decayed_scores(q, k, g).masked_fill(tri_strict, 0.0)

    q, k, g, u, w, qk = (x.reshape(B * H, N, C, x.shape[-1]) for x in (q, k, g, u, w, qk))
    state = init_state.float().transpose(-1, -2).reshape(B * H, K, V)
    outs = []
    for n in range(N):
        g_n = g[:, n]
        v_new = u[:, n] - w[:, n] @ state
        o = (q[:, n] * g_n.exp()) @ state + qk[:, n] @ v_new
        outs.append(o)
        g_last = g_n[:, -1:]  # [BH, 1, K]
        state = state * g_last.transpose(-1, -2).exp() + (k[:, n] * (g_last - g_n).exp()).transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=1).reshape(B, H, N * C, V)[:, :, :S].transpose(1, 2)
    return out, state.reshape(B, H, K, V).transpose(-1, -2)
