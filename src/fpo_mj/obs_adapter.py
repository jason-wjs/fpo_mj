"""Utilities for extracting runner-ready observations from mjlab TensorDicts."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def extract_group_observation(
  observations: Mapping[str, torch.Tensor],
  group_name: str,
  obs_groups: Mapping[str, tuple[str, ...]],
) -> torch.Tensor:
  if group_name not in obs_groups:
    raise KeyError(f"missing observation group definition: {group_name}")

  tensors: list[torch.Tensor] = []
  for key in obs_groups[group_name]:
    if key not in observations:
      raise KeyError(f"missing observation key '{key}' for group '{group_name}'")
    tensor = observations[key]
    if tensor.ndim != 2:
      raise ValueError(
        f"expected 2D observation tensor for key '{key}', got shape {tuple(tensor.shape)}"
      )
    tensors.append(tensor)

  if not tensors:
    raise ValueError(f"observation group '{group_name}' is empty")
  if len(tensors) == 1:
    return tensors[0]
  return torch.cat(tensors, dim=-1)


def extract_actor_critic_observations(
  observations: Mapping[str, torch.Tensor],
  obs_groups: Mapping[str, tuple[str, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
  actor_obs = extract_group_observation(observations, "actor", obs_groups)
  critic_obs = extract_group_observation(observations, "critic", obs_groups)
  return actor_obs, critic_obs
