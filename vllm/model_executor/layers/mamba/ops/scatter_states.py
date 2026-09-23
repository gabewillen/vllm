import torch


def scatter_states(
    state: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """HPU-compatible reference scatter for the KDA recurrent state cache."""
    state[indices.to(torch.long)] = src


def _register_warmup(**kwargs) -> None:
    """Keep the vLLM Triton-op initialization API on the HPU fallback."""
    return None


# The GLM KDA layer calls this during construction.  The portable HPU
# implementation has no compiled kernel to warm up, but exposing the method
# lets the same model code run unchanged.
scatter_states.register_warmup = _register_warmup
