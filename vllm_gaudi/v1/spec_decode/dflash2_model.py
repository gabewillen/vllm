# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block-diffusion drafter for HPU (tensor parallel, static shapes).

Reference: z-lab DFlash2 (`DFlash2DraftModel`): a Qwen3 backbone whose
non-causal attention reads K/V projected from target hidden-state taps
("context") plus the query block [anchor, mask x (B-1)], grouped dynamic
causal convolutions around attention and MLP, and a low-rank candidate
selector that walks the top-k draft logits left to right.

HPU layout choices:
  * attention heads / MLP columns are sharded over TP ranks; fc, the convs and
    the selector are small and replicated;
  * the context K/V cache is dense per request: rows ``slot * L + position``
    of a [n_slots * L, head_dim] tensor per layer (L = max_model_len + block),
    so every step reads a fixed [B, L] window and compiles to one shape per
    batch bucket;
  * the target's vocab-parallel embedding and LM head are reused (the drafter
    checkpoint ships identical copies).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open


@dataclass
class DFlash2Config:
    hidden: int
    inter: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    eps: float
    rope_theta: float
    block: int
    mask_id: int
    taps: list[int]
    conv_k: int
    conv_group: int
    sel_rank: int
    sel_topk: int

    @classmethod
    def load(cls, path: str) -> "DFlash2Config":
        c = json.load(open(os.path.join(path, "config.json")))
        d = c.get("dflash_config", {})
        return cls(hidden=c["hidden_size"], inter=c["intermediate_size"], layers=c["num_hidden_layers"],
                   heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
                   head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                   eps=c["rms_norm_eps"], rope_theta=c.get("rope_parameters", {}).get("rope_theta", 10000.0),
                   block=d.get("block_size", c.get("block_size")), mask_id=d.get("mask_token_id", c.get("mask_token_id")),
                   taps=list(d.get("target_layer_ids", c.get("target_layer_ids"))), conv_k=d["conv_kernel_size"],
                   conv_group=d["conv_group_size"], sel_rank=d["selector_rank"], sel_topk=d["selector_top_k"])


class Comm:
    """TP collectives; world=1 makes every op the identity (unit tests)."""

    def __init__(self, rank: int = 0, world: int = 1, all_reduce: Callable | None = None,
                 all_gather: Callable | None = None):
        self.rank, self.world = rank, world
        self._ar, self._ag = all_reduce, all_gather

    def all_reduce(self, x):
        return x if self.world == 1 else self._ar(x)

    def all_gather(self, x, dim=-1):
        return x if self.world == 1 else self._ag(x, dim)


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _rope(x, cos, sin):
    # x [..., T, D] with cos/sin broadcastable [..., T, D]; neox rotate-half.
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _grouped_conv(h, dyn, base, group):
    """Causal grouped dynamic conv over the block axis.

    h [B, T, H]; dyn [B, T, K, G] (per-group dynamic taps); base [K, H].
    out[t] = sum_o (base[o] + dyn[t, o, group]) * h[t - o]
    """
    B, T, H = h.shape
    K = base.shape[0]
    G = H // group
    out = None
    for o in range(K):
        v = h if o == 0 else F.pad(h[:, :T - o], (0, 0, o, 0))
        w = dyn[:, :, o].unsqueeze(-1).expand(B, T, G, group).reshape(B, T, H) + base[o].to(h.dtype)
        term = w * v
        out = term if out is None else out + term
    return out


class DFlash2Drafter(torch.nn.Module):
    """Weights + compute.  All tensors are registered buffers (no autograd)."""

    def __init__(self, cfg: DFlash2Config, comm: Comm, max_len: int, n_slots: int, device, dtype=torch.bfloat16):
        super().__init__()
        self.cfg, self.comm = cfg, comm
        self.dtype = dtype
        tp = comm.world
        assert cfg.heads % tp == 0 and cfg.kv_heads % tp == 0 and cfg.inter % tp == 0
        self.hq, self.hk = cfg.heads // tp, cfg.kv_heads // tp
        self.it = cfg.inter // tp
        self.L = max_len
        self.n_slots = n_slots
        self.device = device
        D = cfg.head_dim
        # rope tables for every position a block can reach
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        t = torch.arange(max_len, dtype=torch.float32)
        fr = torch.outer(t, inv)
        emb = torch.cat([fr, fr], dim=-1)
        self.register_buffer("cos", emb.cos().to(dtype).to(device), persistent=False)
        self.register_buffer("sin", emb.sin().to(dtype).to(device), persistent=False)
        self.k_cache = [torch.zeros(n_slots * max_len, self.hk, D, dtype=dtype, device=device)
                        for _ in range(cfg.layers)]
        self.v_cache = [torch.zeros(n_slots * max_len, self.hk, D, dtype=dtype, device=device)
                        for _ in range(cfg.layers)]

    # ------------------------------------------------------------------ load
    def load(self, path: str):
        cfg, r = self.cfg, self.comm.rank
        D = cfg.head_dim
        f = safe_open(os.path.join(path, "model.safetensors"), "pt")

        def get(name):
            return f.get_tensor(name).to(self.dtype)

        def buf(name, t):
            self.register_buffer(name, t.contiguous().to(self.device), persistent=False)

        buf("fc_w", get("fc.weight"))
        buf("hidden_norm_w", get("hidden_norm.weight"))
        buf("norm_w", get("norm.weight"))
        qs, ks = slice(r * self.hq * D, (r + 1) * self.hq * D), slice(r * self.hk * D, (r + 1) * self.hk * D)
        isl = slice(r * self.it, (r + 1) * self.it)
        for i in range(cfg.layers):
            p = f"layers.{i}."
            buf(f"l{i}_in_norm", get(p + "input_layernorm.weight"))
            buf(f"l{i}_post_norm", get(p + "post_attention_layernorm.weight"))
            buf(f"l{i}_q", get(p + "self_attn.q_proj.weight")[qs])
            buf(f"l{i}_k", get(p + "self_attn.k_proj.weight")[ks])
            buf(f"l{i}_v", get(p + "self_attn.v_proj.weight")[ks])
            buf(f"l{i}_o", get(p + "self_attn.o_proj.weight")[:, qs])
            buf(f"l{i}_qn", get(p + "self_attn.q_norm.weight"))
            buf(f"l{i}_kn", get(p + "self_attn.k_norm.weight"))
            gate = get(p + "mlp.gate_proj.weight")[isl]
            up = get(p + "mlp.up_proj.weight")[isl]
            buf(f"l{i}_gu", torch.cat([gate, up], 0))
            buf(f"l{i}_down", get(p + "mlp.down_proj.weight")[:, isl])
            for c in ("attention_conv", "mlp_conv"):
                buf(f"l{i}_{c}_base", get(p + f"{c}.base_kernel"))
                buf(f"l{i}_{c}_proj", get(p + f"{c}.kernel_projection.weight"))
        buf("sel_pred", get("candidate_selector.predecessor_codebook"))
        buf("sel_succ", get("candidate_selector.successor_codebook"))
        buf("sel_hproj", get("candidate_selector.hidden_projection.weight"))
        m = torch.load(os.path.join(path, "mask_embedding.pt"), map_location="cpu")
        assert int(m["mask_token_id"]) == cfg.mask_id
        buf("mask_emb", m["embedding"].to(self.dtype))
        return self

    # ------------------------------------------------------------- compute
    def _heads_norm(self, x, w, nh):
        # x [..., nh*D] -> [..., nh, D] RMS-normed per head
        return _rms(x.unflatten(-1, (nh, self.cfg.head_dim)), w, self.cfg.eps)

    def write_context(self, aux, positions, rows):
        """Project target taps to context K/V and store them.

        aux [N, taps*H] target taps (layer-major concat), positions [N],
        rows [N] flat cache rows (slot * L + position).
        """
        cfg = self.cfg
        ctx = _rms(F.linear(aux, self.fc_w), self.hidden_norm_w, cfg.eps)
        cos, sin = self.cos[positions].unsqueeze(-2), self.sin[positions].unsqueeze(-2)
        for i in range(cfg.layers):
            k = self._heads_norm(F.linear(ctx, getattr(self, f"l{i}_k")), getattr(self, f"l{i}_kn"), self.hk)
            k = _rope(k, cos, sin)
            v = F.linear(ctx, getattr(self, f"l{i}_v")).unflatten(-1, (self.hk, cfg.head_dim))
            self.k_cache[i].index_copy_(0, rows, k)
            self.v_cache[i].index_copy_(0, rows, v)

    def _conv_prepare(self, i, name, h):
        cfg = self.cfg
        G = cfg.hidden // cfg.conv_group
        dyn = F.linear(h, getattr(self, f"l{i}_{name}_proj")).unflatten(-1, (2, cfg.conv_k, G))
        base = getattr(self, f"l{i}_{name}_base")
        out = _grouped_conv(h, dyn[:, :, 0].contiguous(), base[0], cfg.conv_group)
        return out, dyn[:, :, 1].contiguous()

    def _conv_finish(self, i, name, h, dyn):
        return _grouped_conv(h, dyn, getattr(self, f"l{i}_{name}_base")[1], self.cfg.conv_group)

    def query(self, anchor_emb, slots, ctx_len):
        """Run the query block and return post-norm hidden states [B, T-1, H].

        anchor_emb [B, H] embedding of the anchor (bonus) token, slots [B] cache
        slot per request, ctx_len [B] number of valid context positions (the
        block sits at positions ctx_len .. ctx_len + T - 1).
        """
        cfg = self.cfg
        B, T, D, L = anchor_emb.shape[0], cfg.block, cfg.head_dim, self.L
        h = torch.cat([anchor_emb.unsqueeze(1), self.mask_emb.view(1, 1, -1).expand(B, T - 1, -1)], dim=1)
        qpos = ctx_len.unsqueeze(1) + torch.arange(T, device=h.device).unsqueeze(0)  # [B, T]
        cos, sin = self.cos[qpos].unsqueeze(-2), self.sin[qpos].unsqueeze(-2)  # [B, T, 1, D]
        rows = (slots * L).unsqueeze(1) + torch.arange(L, device=h.device).unsqueeze(0)  # [B, L]
        rows = rows.reshape(-1)
        valid = torch.arange(L, device=h.device).unsqueeze(0) < ctx_len.unsqueeze(1)  # [B, L]
        bias = torch.cat([torch.where(valid, 0.0, float("-inf")),
                          torch.zeros(B, T, device=h.device)], dim=1).view(B, 1, 1, L + T)
        scale = D ** -0.5
        grp = self.hq // self.hk
        for i in range(cfg.layers):
            res = h
            x = _rms(h, getattr(self, f"l{i}_in_norm"), cfg.eps)
            x, dyn = self._conv_prepare(i, "attention_conv", x)
            q = self._heads_norm(F.linear(x, getattr(self, f"l{i}_q")), getattr(self, f"l{i}_qn"), self.hq)
            k = self._heads_norm(F.linear(x, getattr(self, f"l{i}_k")), getattr(self, f"l{i}_kn"), self.hk)
            v = F.linear(x, getattr(self, f"l{i}_v")).unflatten(-1, (self.hk, D))
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)  # [B, T, h, D]
            kc = self.k_cache[i].index_select(0, rows).view(B, L, self.hk, D)
            vc = self.v_cache[i].index_select(0, rows).view(B, L, self.hk, D)
            kk = torch.cat([kc, k], dim=1).transpose(1, 2)  # [B, hk, L+T, D]
            vv = torch.cat([vc, v], dim=1).transpose(1, 2)
            qq = q.transpose(1, 2).reshape(B, self.hk, grp * T, D)  # heads grouped per kv head
            s = torch.matmul(qq, kk.transpose(-1, -2)).float() * scale  # [B, hk, grp*T, L+T]
            s = s + bias.view(B, 1, 1, L + T)
            p = torch.softmax(s, dim=-1).to(vv.dtype)
            o = torch.matmul(p, vv).view(B, self.hq, T, D).transpose(1, 2).reshape(B, T, self.hq * D)
            o = self.comm.all_reduce(F.linear(o, getattr(self, f"l{i}_o")))
            o = self._conv_finish(i, "attention_conv", o, dyn)
            h = res + o
            res = h
            x = _rms(h, getattr(self, f"l{i}_post_norm"), cfg.eps)
            x, dyn = self._conv_prepare(i, "mlp_conv", x)
            gu = F.linear(x, getattr(self, f"l{i}_gu"))
            x = F.silu(gu[..., :self.it]) * gu[..., self.it:]
            x = self.comm.all_reduce(F.linear(x, getattr(self, f"l{i}_down")))
            x = self._conv_finish(i, "mlp_conv", x, dyn)
            h = res + x
        return _rms(h, self.norm_w, cfg.eps)[:, 1:]

    def topk_logits(self, hidden, head_w, vocab_start):
        """Vocab-parallel top-k: local head shard -> local top-k -> gather -> top-k.

        hidden [N, H], head_w [V_local, H] this rank's LM-head shard.
        Returns (values [N, k], global ids [N, k]).
        """
        k = self.cfg.sel_topk
        # Candidate ids stay int32: an int64-source gather fails to compile on
        # Gaudi (synStatus 26), int32 is fine and vocab ids fit.
        lg = F.linear(hidden, head_w).float()
        val, idx = torch.topk(lg, k, dim=-1)
        idx = idx.to(torch.int32) + vocab_start
        if self.comm.world > 1:
            val = self.comm.all_gather(val, -1)
            idx = self.comm.all_gather(idx.float(), -1).to(torch.int32)  # ids < 2^24 are exact
            val, j = torch.topk(val, k, dim=-1)
            idx = idx.gather(-1, j)
        return val, idx

    def select(self, hidden, unary, cand, anchor_ids):
        """Greedy candidate-selector walk.  hidden [B, T-1, H]; unary/cand [B, T-1, k]."""
        hp = F.linear(hidden, self.sel_hproj).float()  # [B, T-1, r]
        pred = anchor_ids.to(torch.int32)
        cand = cand.to(torch.int32)
        out = []
        for t in range(hidden.shape[1]):
            a = self.sel_pred.index_select(0, pred).float() * hp[:, t]  # [B, r]
            succ = self.sel_succ.index_select(0, cand[:, t].reshape(-1)).float().view(cand.shape[0], -1, a.shape[-1])
            sc = unary[:, t] + torch.einsum("br,bkr->bk", a, succ)
            j = torch.argmax(sc, dim=-1, keepdim=True)
            pred = cand[:, t].contiguous().gather(-1, j).squeeze(-1)
            out.append(pred)
        return torch.stack(out, dim=1).to(torch.int64)
