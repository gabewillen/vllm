# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility package for GLM5.3 sequence-parallel helpers."""

import torch
import torch.nn.functional as F


def fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
    """Portable fallback for the vision tower's fused Q/KV RMSNorm."""
    q = F.rms_norm(qr, (qr.shape[-1],), q_weight, eps)
    k = F.rms_norm(kv, (kv.shape[-1],), kv_weight, eps)
    return q, k
