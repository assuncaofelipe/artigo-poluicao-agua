import torch


def rate_encode(x: torch.Tensor, time_steps: int):
    """
    Simple rate coding: at each time step, emit spikes with prob = normalized x.
    Assumes x is normalized (mean 0, std 1). We squash to [0,1] with sigmoid.
    """
    probs = torch.sigmoid(x)
    spikes = torch.rand(time_steps, *probs.shape, device=probs.device) < probs
    return spikes.float()
