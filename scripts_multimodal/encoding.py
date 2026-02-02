import torch


def rate_encode(x: torch.Tensor, time_steps: int):
    probs = torch.sigmoid(x)
    spikes = torch.rand(time_steps, *probs.shape, device=probs.device) < probs
    return spikes.float()
