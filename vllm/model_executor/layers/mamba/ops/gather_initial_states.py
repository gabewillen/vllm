import torch


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """HPU-compatible reference gather for the KDA recurrent state cache."""
    output = torch.zeros(
        (indices.numel(), *state.shape[1:]), dtype=state.dtype, device=state.device
    )
    valid = has_initial_state.to(torch.bool)
    if valid.any():
        output[valid] = state[indices[valid].to(torch.long)]
    return output
