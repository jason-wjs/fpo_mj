from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ExponentialMovingAverage:
  def __init__(
    self,
    model: nn.Module,
    decay: float = 0.95,
    device: Optional[torch.device] = None,
  ):
    self.decay = decay
    self.device = device if device is not None else next(model.parameters()).device
    self.shadow_params = {}
    self.model_params = {}
    self.backup_params = {}

    for name, param in model.named_parameters():
      if param.requires_grad:
        self.model_params[name] = param
        self.shadow_params[name] = param.data.clone().to(self.device)

  @torch.no_grad()
  def update(self):
    for name, param in self.model_params.items():
      if param.requires_grad:
        self.shadow_params[name].mul_(self.decay).add_(
          param.data.to(self.device), alpha=1.0 - self.decay
        )

  @torch.no_grad()
  def reset_to_current(self):
    for name, param in self.model_params.items():
      if param.requires_grad:
        self.shadow_params[name].copy_(param.data.to(self.device))

  def state_dict(self):
    return {"decay": self.decay, "shadow_params": self.shadow_params}

  def load_state_dict(self, state_dict):
    self.decay = state_dict["decay"]
    self.shadow_params = state_dict["shadow_params"]

