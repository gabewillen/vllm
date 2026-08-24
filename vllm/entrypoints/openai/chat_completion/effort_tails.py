# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenize dynamic-effort level variants without re-tokenizing the body.

Every level variant of a dynamic-effort request (§13.3) is the same
conversation with a different tail: the level sentence, the think-off block or
the hidden off-vote question. Tokenizing the whole prompt once per level costs
~0.1 s each at agent-sized prompts, so only the default level goes through the
full renderer; the other levels reuse its token ids up to a cut and tokenize the
few hundred characters after it.

The cut is the start of the last special token (`<|im_start|>` and friends)
inside the variants' common text prefix. HF tokenizers split added special
tokens out before BPE, so a text cut at one is a hard tokenization boundary:
`encode(a + b) == encode(a) + encode(b)` whenever `b` starts with a special
token. The default variant proves the boundary on the way (its full-render ids
must end with the re-encoded tail); if anything does not line up the caller
falls back to a full render.
"""

from collections.abc import Iterable, Sequence


def common_prefix_len(texts: Sequence[str]) -> int:
    """Length of the longest character prefix shared by every text."""
    if not texts:
        return 0
    common = min(len(t) for t in texts)
    first = texts[0]
    for text in texts[1:]:
        # Binary search on slice equality: each probe is one C-level compare,
        # so a 300k-character prompt costs ~20 compares, not a Python loop.
        lo, hi = 0, min(common, len(text))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if text[:mid] == first[:mid]:
                lo = mid
            else:
                hi = mid - 1
        common = lo
    return common


def special_token_cut(
    texts: Sequence[str], special_tokens: Iterable[str]
) -> int | None:
    """Start of the last special token wholly inside the texts' common prefix."""
    common = common_prefix_len(texts)
    if common == 0:
        return None
    first = texts[0]
    cut = max(
        (first.rfind(tok, 0, common) for tok in special_tokens if tok),
        default=-1,
    )
    return cut if cut > 0 else None


def tokenize_variant_tails(
    encode,
    default_text: str,
    default_ids: Sequence[int],
    variant_texts: Sequence[str],
    special_tokens: Iterable[str],
) -> list[list[int]] | None:
    """Token ids of each variant, tokenizing only the text after the cut.

    Args:
        encode: `text -> list[int]` without special-token insertion.
        default_text: the rendered text of the variant already tokenized.
        default_ids: the full-render token ids of `default_text`.
        variant_texts: rendered text of every variant, `default_text` included.
        special_tokens: added tokens the tokenizer splits before BPE.

    Returns:
        One id list per entry of `variant_texts`, or `None` when no cut proves
        exact - the caller then tokenizes each variant in full.
    """
    cut = special_token_cut([default_text, *variant_texts], special_tokens)
    if cut is None:
        return None
    default_tail = encode(default_text[cut:])
    body_len = len(default_ids) - len(default_tail)
    if body_len <= 0 or list(default_ids[body_len:]) != list(default_tail):
        return None
    body = list(default_ids[:body_len])
    return [
        body + (default_tail if text == default_text else encode(text[cut:]))
        for text in variant_texts
    ]
