from __future__ import annotations

import torch


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
  if act_name == "elu":
    return torch.nn.ELU()
  if act_name == "selu":
    return torch.nn.SELU()
  if act_name == "relu":
    return torch.nn.ReLU()
  if act_name == "crelu":
    return torch.nn.CELU()
  if act_name == "lrelu":
    return torch.nn.LeakyReLU()
  if act_name == "tanh":
    return torch.nn.Tanh()
  if act_name == "sigmoid":
    return torch.nn.Sigmoid()
  if act_name == "identity":
    return torch.nn.Identity()
  raise ValueError(f"Invalid activation function '{act_name}'.")
