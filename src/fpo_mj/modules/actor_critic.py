from __future__ import annotations

import torch
import torch.nn as nn

from fpo_mj.utils import resolve_nn_activation


class ActorCritic(nn.Module):
  is_recurrent = False

  def __init__(self, num_actor_obs: int, num_critic_obs: int, num_actions: int, cfg):
    super().__init__()
    activation = resolve_nn_activation(cfg.activation)

    self.num_actions = num_actions
    self.timestep_embed_dim = cfg.timestep_embed_dim
    self.mlp_output_scale = cfg.actor_mlp_output_scale
    self.cfm_loss_t_inverse_cdf_beta = cfg.cfm_loss_t_inverse_cdf_beta
    self.sampling_steps = cfg.sampling_steps
    self.cfm_loss_reduction = cfg.cfm_loss_reduction
    self.actor_scale = cfg.actor_scale
    self.action_perturb_std = cfg.action_perturb_std

    actor_hidden_dims = cfg.actor_hidden_dims
    critic_hidden_dims = cfg.critic_hidden_dims
    actor_layers: list[nn.Module] = [
      nn.Linear(num_actor_obs + self.timestep_embed_dim + num_actions, actor_hidden_dims[0]),
      activation,
    ]
    for layer_index in range(len(actor_hidden_dims)):
      if layer_index == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
      else:
        actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
        actor_layers.append(activation)
    self.actor = nn.Sequential(*actor_layers)

    critic_layers: list[nn.Module] = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation]
    for layer_index in range(len(critic_hidden_dims)):
      if layer_index == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
      else:
        critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
        critic_layers.append(activation)
    self.critic = nn.Sequential(*critic_layers)

    if getattr(cfg, "compile_flow", False):
      self._integrate_flow_impl = torch.compile(self._integrate_flow, mode="reduce-overhead")
    else:
      self._integrate_flow_impl = self._integrate_flow

  def reset(self, dones=None):
    return None

  def act(self, observations: torch.Tensor, **kwargs):
    device = observations.device
    batch_size = observations.shape[0]
    x_t = torch.randn(size=(batch_size, self.num_actions), device=device)
    full_t_path = torch.linspace(1.0, 0.0, self.sampling_steps + 1, device=device)
    t_current = full_t_path[:-1]
    dt = full_t_path[1:] - t_current
    x_t = self._integrate_flow_impl(observations, x_t, t_current, dt, self.sampling_steps)
    actions = self.actor_scale * x_t
    if self.training and self.action_perturb_std > 0:
      actions = actions + self.action_perturb_std * torch.randn_like(actions)
    return actions

  def act_inference(self, observations, eval_mode="zero", eval_fixed_seed=12345):
    device = observations.device
    batch_size = observations.shape[0]
    if eval_mode == "zero":
      x_t = torch.zeros(size=(batch_size, self.num_actions), device=device)
    elif eval_mode == "fixed_seed":
      generator = torch.Generator(device=device)
      generator.manual_seed(eval_fixed_seed)
      x_t = torch.randn(size=(batch_size, self.num_actions), device=device, generator=generator)
    elif eval_mode == "random":
      x_t = torch.randn(size=(batch_size, self.num_actions), device=device)
    else:
      raise ValueError(f"Unknown eval_mode: {eval_mode}")
    full_t_path = torch.linspace(1.0, 0.0, self.sampling_steps + 1, device=device)
    t_current = full_t_path[:-1]
    dt = full_t_path[1:] - t_current
    x_t = self._integrate_flow_impl(observations, x_t, t_current, dt, self.sampling_steps)
    return self.actor_scale * x_t

  def get_cfm_loss(self, observations, actions, eps, t, actor: torch.nn.Module | None = None):
    actor = actor or self.actor
    scaled_actions = actions / self.actor_scale
    embedded_t = self._embed_timestep(t)
    x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
    actor_obs_expanded = observations[:, None, :].expand(observations.shape[0], eps.shape[1], -1)
    mlp_output = actor(torch.cat([actor_obs_expanded, embedded_t, x_t], dim=-1))
    mlp_output = self.mlp_output_scale * mlp_output
    velocity_pred = mlp_output
    x0_pred = x_t - t * velocity_pred
    x1_pred = x0_pred + velocity_pred
    target_velocity = eps - scaled_actions[:, None, :]
    loss = self._compute_squared_error(velocity_pred, target_velocity)
    return loss, x1_pred, x0_pred

  def evaluate(self, critic_observations, **kwargs):
    return self.critic(critic_observations)

  def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
    freqs = 2 ** torch.arange(self.timestep_embed_dim // 2, device=t.device)
    scaled_t = t * freqs
    return torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)

  def _integrate_flow(self, observations, x_t, t_current, dt, flow_steps):
    batch_size = observations.shape[0]
    half_dim = self.timestep_embed_dim // 2
    freqs = 2 ** torch.arange(half_dim, device=observations.device, dtype=observations.dtype)
    for i in range(flow_steps):
      t_val = t_current[i].reshape(1, 1)
      scaled_t = t_val * freqs
      embedded_t = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1).expand(batch_size, -1)
      mlp_output = self.actor(torch.cat([observations, embedded_t, x_t], dim=-1))
      u = self.mlp_output_scale * mlp_output
      x_t = x_t + u * dt[i]
    return x_t

  def _compute_squared_error(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if self.cfm_loss_reduction == "mean":
      return torch.mean((predictions - targets) ** 2, dim=-1)
    if self.cfm_loss_reduction == "sum":
      return torch.sum((predictions - targets) ** 2, dim=-1)
    squared_errors = (predictions - targets) ** 2
    return torch.sum(squared_errors, dim=-1) / (squared_errors.shape[-1] ** 0.5)
