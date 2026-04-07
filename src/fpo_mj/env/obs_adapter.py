from __future__ import annotations

import torch
from tensordict import TensorDictBase


class ObservationAdapter:
  def __init__(self, obs_groups: dict[str, tuple[str, ...]]):
    self.obs_groups = obs_groups

  def adapt(self, observations: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
    actor_obs = self._concat_group(observations, "actor")
    critic_obs = self._concat_group(observations, "critic")
    return actor_obs, critic_obs

  def _concat_group(
    self, observations: TensorDictBase, group_name: str
  ) -> torch.Tensor:
    if group_name not in self.obs_groups:
      raise KeyError(f"Missing obs group definition: {group_name}")

    parts: list[torch.Tensor] = []
    for key in self.obs_groups[group_name]:
      if key not in observations.keys():
        raise KeyError(f"Missing observation key: {key}")
      value = observations[key]
      if not isinstance(value, torch.Tensor):
        raise TypeError(f"Observation '{key}' must be a tensor")
      if value.ndim != 2:
        raise ValueError(
          f"expected 2D observation tensor for key '{key}', got shape {tuple(value.shape)}"
        )
      parts.append(value)

    if not parts:
      raise ValueError(f"Observation group '{group_name}' is empty")

    if len(parts) == 1:
      return parts[0]
    return torch.cat(parts, dim=-1)
