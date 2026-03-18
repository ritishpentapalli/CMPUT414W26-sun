from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    spatial: torch.Tensor
    frequency: torch.Tensor


class DualDomainLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, phase_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.phase_weight = phase_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> LossBreakdown:
        spatial = F.l1_loss(pred, target)

        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        mag_loss = F.l1_loss(pred_mag, target_mag)

        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_delta = pred_phase - target_phase
        phase_loss = torch.mean(1.0 - torch.cos(phase_delta))

        frequency = mag_loss + self.phase_weight * phase_loss
        total = self.alpha * spatial + self.beta * frequency
        return LossBreakdown(total=total, spatial=spatial, frequency=frequency)
