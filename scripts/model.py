from typing import Tuple

import torch
from torch import nn


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input > thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        thresh = ctx.thresh
        # Fast sigmoid surrogate gradient
        grad = 1.0 / (1.0 + torch.abs(input - thresh)) ** 2
        return grad_output * grad, None


class LIFLayer(nn.Module):
    def __init__(self, beta: float = 0.95, thresh: float = 1.0):
        super().__init__()
        self.beta = beta
        self.thresh = thresh

    def forward(self, input_cur: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + input_cur
        spk = SpikeFn.apply(mem, self.thresh)
        mem = mem - spk * self.thresh
        return spk, mem


class SNNRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        beta: float = 0.95,
        thresh: float = 1.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = LIFLayer(beta=beta, thresh=thresh)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # spikes: [T, B, F]
        time_steps, batch_size, _ = spikes.shape
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=spikes.device)
        out_sum = torch.zeros(batch_size, self.fc2.out_features, device=spikes.device)

        for t in range(time_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            out_sum += self.fc2(spk1)

        out = out_sum / time_steps
        return out, out_sum
