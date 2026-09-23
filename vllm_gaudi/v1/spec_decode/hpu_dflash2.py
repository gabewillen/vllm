# SPDX-License-Identifier: Apache-2.0
"""DFlash2 proposer for the HPU runner (see dflash2_model.py for the drafter).

Per engine step, for every request:
  * the verified tokens of the step (decode: the [anchor, drafts] block;
    prefill: the prompt chunk) get their target taps projected into the
    drafter's context K/V cache;
  * the query block [bonus, mask x 7] at positions ctx_len .. ctx_len + 7
    attends over context positions < ctx_len and yields 7 greedy drafts.
Rows of tokens that are padding or were rejected land either in the dummy
slot 0 or past ctx_len, where the next step overwrites them and the mask
hides them.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
from vllm.logger import init_logger

from vllm_gaudi.v1.spec_decode.dflash2_model import Comm, DFlash2Config, DFlash2Drafter

logger = init_logger(__name__)
_DEBUG = os.environ.get("GLM53_DFLASH_DEBUG") == "1"
_RANDOM_DRAFTS = os.environ.get("GLM53_DFLASH_RANDOM_DRAFTS") == "1"  # verify-path losslessness test


class HpuDFlash2Proposer:

    def __init__(self, vllm_config, device, runner):
        spec = vllm_config.speculative_config
        self.path = spec.draft_model_config.model
        self.cfg = DFlash2Config.load(self.path)
        self.k = spec.num_speculative_tokens
        assert self.k == self.cfg.block - 1, (f"DFlash2 drafter needs num_speculative_tokens={self.cfg.block - 1}, "
                                              f"got {self.k}")
        self.device = device
        self.runner = runner
        self.max_len = vllm_config.model_config.max_model_len + 2 * self.cfg.block
        self.n_slots = vllm_config.scheduler_config.max_num_seqs + 1
        self._step_fn = self._step

    # ------------------------------------------------------------ setup
    def load_model(self, target_model):
        comm = Comm(get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size(),
                    tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)
        self.m = DFlash2Drafter(self.cfg, comm, self.max_len, self.n_slots, self.device).load(self.path)
        lm = getattr(target_model, "language_model", target_model)
        lm.model.set_aux_hidden_state_layers(tuple(self.cfg.taps))
        self.embed = lm.model.embed_tokens
        head = lm.lm_head
        si = head.shard_indices
        n_valid = si.org_vocab_end_index - si.org_vocab_start_index
        # drop vocab padding rows so they never enter the top-k
        self.head_w = head.weight[:n_valid]
        self.vocab_start = si.org_vocab_start_index
        logger.info("DFlash2 drafter loaded from %s: %d layers, taps %s, block %d, %d slots x %d positions",
                    self.path, self.cfg.layers, self.cfg.taps, self.cfg.block, self.n_slots, self.max_len)

    def compile(self, compile_fn):
        self._step_fn = compile_fn(self._step)

    # ------------------------------------------------------------ compute
    def _step(self, aux, pos, rows, anchor_ids, slots, ctx_len):
        m = self.m
        m.write_context(aux, pos, rows)
        hid = m.query(self.embed(anchor_ids), slots, ctx_len)  # [B, T-1, H]
        B, T1, H = hid.shape
        val, cand = m.topk_logits(hid.reshape(-1, H), self.head_w, self.vocab_start)
        kk = cand.shape[-1]
        return m.select(hid, val.view(B, T1, kk), cand.view(B, T1, kk), anchor_ids)

    def _slot(self, req_id):
        base = self.runner._gdn_req_to_base_slot.get(req_id)
        return 0 if base is None else base + 1

    def _run(self, aux, pos, rows, anchor, slots, ctx_len):
        dev = self.device
        t = lambda a, dt=torch.int64: torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).to(dt).to(
            dev, non_blocking=True)
        return self._step_fn(aux, t(pos), t(rows), t(anchor), t(slots), t(ctx_len))

    def propose_decode(self, aux, sampled_token_ids, num_decodes, padded_bs, num_tokens, req_ids,
                       num_scheduled):
        """aux [padded_bs * num_tokens, taps*H] from the verify forward."""
        L = self.max_len
        n_rows = padded_bs * num_tokens
        pos = np.zeros(n_rows, np.int64)
        rows = np.arange(n_rows, dtype=np.int64) % L  # dummy slot 0
        anchor = np.zeros(padded_bs, np.int64)
        slots = np.zeros(padded_bs, np.int64)
        ctx_len = np.zeros(padded_bs, np.int64)
        nts = self.runner.input_batch.num_tokens_no_spec
        for i in range(num_decodes):
            acc = sampled_token_ids[i]
            if not acc:
                continue
            s = self._slot(req_ids[i])
            n_after = int(nts[i])
            n_in = num_scheduled[i]
            start = n_after - len(acc) - 1
            p = start + np.arange(n_in)
            pos[i * num_tokens:i * num_tokens + n_in] = p
            if s:
                rows[i * num_tokens:i * num_tokens + n_in] = s * L + p
            anchor[i] = acc[-1]
            slots[i] = s
            ctx_len[i] = n_after - 1
        out = self._run(aux, pos, rows, anchor, slots, ctx_len)[:num_decodes]
        if _RANDOM_DRAFTS:
            out = torch.randint(1000, 100000, out.shape, device=out.device, dtype=out.dtype)
        if _DEBUG:
            print("DFlash2 step: req0 start=%d accepted=%s ctx_len=%d drafts=%s" % (int(pos[0]),
                  list(sampled_token_ids[0]), int(ctx_len[0]), out[0].tolist()), flush=True)
        return out

    def propose_prefill(self, aux, req_ids, sampled, bs, seq_len, num_computed, num_new):
        """aux [bs * seq_len, taps*H] from one prefill batch (req-major, right padded)."""
        L = self.max_len
        n_rows = bs * seq_len
        pos = np.zeros(n_rows, np.int64)
        rows = np.arange(n_rows, dtype=np.int64) % L
        anchor = np.zeros(bs, np.int64)
        slots = np.zeros(bs, np.int64)
        ctx_len = np.zeros(bs, np.int64)
        for i, rid in enumerate(req_ids):
            s = self._slot(rid)
            p = num_computed[i] + np.arange(num_new[i])
            pos[i * seq_len:i * seq_len + num_new[i]] = p
            if s:
                rows[i * seq_len:i * seq_len + num_new[i]] = s * L + p
            anchor[i] = sampled[i]
            slots[i] = s
            ctx_len[i] = num_computed[i] + num_new[i]
        out = self._run(aux, pos, rows, anchor, slots, ctx_len)[:len(req_ids)]
        if _DEBUG:
            print("DFlash2 prefill: computed=%s new=%s anchor=%s drafts=%s" % (num_computed, num_new,
                  list(sampled), out[0].tolist()), flush=True)
        return out
