from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fpo_mj.modules import ActorCritic
from fpo_mj.modules.ema import ExponentialMovingAverage
from fpo_mj.storage import RolloutStorage


def clamp_ste(x, min=None, max=None):
  clamped = x.clamp(min=min, max=max)
  return x + (clamped - x).detach()


class FPO:
  def __init__(self, policy: ActorCritic, cfg, device="cpu"):
    self.device = device
    self.policy = policy.to(device)
    if cfg.weight_decay > 0:
      self.optimizer = optim.AdamW(
        self.policy.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas, weight_decay=cfg.weight_decay
      )
    else:
      self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)
    self.ema_decay = cfg.ema_decay
    self.ema_warmup_steps = cfg.ema_warmup_steps
    self.tot_timesteps = 0
    self.ema = ExponentialMovingAverage(self.policy.actor, decay=cfg.ema_decay, device=device) if cfg.ema_decay > 0.0 else None
    self.storage = None
    self.transition = RolloutStorage.Transition()
    self.clip_param = cfg.clip_param
    self.num_learning_epochs = cfg.num_learning_epochs
    self.num_mini_batches = cfg.num_mini_batches
    self.value_loss_coef = cfg.value_loss_coef
    self.knn_entropy_coef = cfg.knn_entropy_coef
    self.knn_entropy_k = cfg.knn_entropy_k
    self.gamma = cfg.gamma
    self.lam = cfg.lam
    self.max_grad_norm = cfg.max_grad_norm
    self.use_clipped_value_loss = cfg.use_clipped_value_loss
    self.desired_kl = cfg.desired_kl
    self.schedule = cfg.schedule
    self.learning_rate = cfg.learning_rate
    self.normalize_advantage_per_mini_batch = cfg.normalize_advantage_per_mini_batch
    self.normalize_advantage = cfg.normalize_advantage
    self.n_samples_per_action = cfg.n_samples_per_action
    self.cfm_loss_clamp = cfg.cfm_loss_clamp
    self.cfm_loss_clamp_negative_advantages = cfg.cfm_loss_clamp_negative_advantages
    self.cfm_loss_clamp_negative_advantages_max = cfg.cfm_loss_clamp_negative_advantages_max
    self.cfm_diff_clamp_max = cfg.cfm_diff_clamp_max
    self.advantage_clamp = cfg.advantage_clamp
    self.storage_action_noise_std = cfg.storage_action_noise_std
    self.trust_region_mode = cfg.trust_region_mode

  def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
    self.storage = RolloutStorage(
      num_envs,
      num_transitions_per_env,
      actor_obs_shape,
      critic_obs_shape,
      actions_shape,
      self.device,
      self.n_samples_per_action,
    )

  def act(self, obs, critic_obs):
    self.transition.actions = self.policy.act(obs).detach()
    self.transition.values = self.policy.evaluate(critic_obs).detach()
    if self.storage_action_noise_std > 0:
      self.transition.actions = self.transition.actions + self.storage_action_noise_std * torch.randn_like(self.transition.actions)
    cfm_loss_eps = torch.randn((self.storage.num_envs, self.n_samples_per_action, self.policy.num_actions), device=self.device)
    uniform_t = torch.rand((self.storage.num_envs, self.n_samples_per_action, 1), device=self.device)
    beta = self.policy.cfm_loss_t_inverse_cdf_beta
    cfm_loss_t = 0.005 + 0.99 * (1.0 - (1.0 - uniform_t) ** (1.0 / beta))
    self.transition.initial_cfm_loss, self.transition.x1_pred, _ = self.policy.get_cfm_loss(
      obs, self.transition.actions, cfm_loss_eps, cfm_loss_t
    )
    self.transition.initial_cfm_loss = self.transition.initial_cfm_loss.detach()
    self.transition.x1_pred = self.transition.x1_pred.detach()
    self.transition.cfm_loss_eps = cfm_loss_eps
    self.transition.cfm_loss_t = cfm_loss_t
    self.transition.observations = obs
    self.transition.privileged_observations = critic_obs
    return self.transition.actions

  def process_env_step(self, rewards, dones, infos):
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in infos:
      self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1)
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.policy.reset(dones)

  def compute_returns(self, last_critic_obs):
    last_values = self.policy.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(
      last_values,
      self.gamma,
      self.lam,
      normalize_advantage=self.normalize_advantage and not self.normalize_advantage_per_mini_batch,
    )

  def update(self, obs_normalizer=None, privileged_obs_normalizer=None):
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_kl = 0.0
    all_grad_norms_before = []
    all_grad_norms_after = []
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    mini_batch_step = 0
    for (
      obs_batch,
      critic_obs_batch,
      actions_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_x1_pred_batch,
      old_cfm_loss_batch,
      old_cfm_loss_eps_batch,
      old_cfm_loss_t_batch,
      _hid_states_batch,
      _masks_batch,
    ) in generator:
      if self.normalize_advantage_per_mini_batch:
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
      positive_clamp, negative_clamp = self.advantage_clamp
      advantages_batch = advantages_batch.clamp(-negative_clamp, positive_clamp)
      cfm_loss_batch, x1_pred_batch, _x0_pred_batch = self.policy.get_cfm_loss(
        obs_batch, actions_batch, old_cfm_loss_eps_batch, old_cfm_loss_t_batch
      )
      value_batch = self.policy.evaluate(critic_obs_batch)
      entropy_bonus = self._compute_knn_entropy(x1_pred_batch, self.knn_entropy_k) if self.knn_entropy_coef > 0 else None
      if self.schedule == "adaptive":
        kl_mean = ((x1_pred_batch.detach() - old_x1_pred_batch) ** 2).mean()
        if kl_mean > self.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
          param_group["lr"] = self.learning_rate
        mean_kl += kl_mean.item()
      if self.cfm_loss_clamp > 0:
        old_cfm_loss_batch = torch.clamp(old_cfm_loss_batch, max=self.cfm_loss_clamp)
        cfm_loss_batch = torch.clamp(cfm_loss_batch, max=self.cfm_loss_clamp)
      if self.cfm_loss_clamp_negative_advantages:
        cfm_loss_batch = torch.where(
          advantages_batch < 0,
          cfm_loss_batch.clamp(max=self.cfm_loss_clamp_negative_advantages_max),
          cfm_loss_batch,
        )
      log_ratio = clamp_ste(old_cfm_loss_batch - cfm_loss_batch, max=self.cfm_diff_clamp_max)
      ratio = torch.exp(log_ratio)
      if self.trust_region_mode == "ppo":
        surrogate = -advantages_batch * ratio
        surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
      elif self.trust_region_mode == "spo":
        surrogate_loss = -torch.mean(
          ratio * advantages_batch - torch.abs(advantages_batch) / (2.0 * self.clip_param) * (ratio - 1.0) ** 2
        )
      else:
        surrogate = -advantages_batch * ratio
        surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        ppo_loss = torch.max(surrogate, surrogate_clipped)
        spo_loss = -(ratio * advantages_batch - torch.abs(advantages_batch) / (2.0 * self.clip_param) * (ratio - 1.0) ** 2)
        surrogate_loss = torch.where(advantages_batch > 0, ppo_loss, spo_loss).mean()
      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()
      loss = surrogate_loss + self.value_loss_coef * value_loss
      if entropy_bonus is not None:
        loss -= self.knn_entropy_coef * entropy_bonus
      self.optimizer.zero_grad()
      loss.backward()
      total_grad_norm_before = 0.0
      for p in self.policy.parameters():
        if p.grad is not None:
          total_grad_norm_before += p.grad.data.norm(2).item() ** 2
      total_grad_norm_before = total_grad_norm_before**0.5
      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      total_grad_norm_after = 0.0
      for p in self.policy.parameters():
        if p.grad is not None:
          total_grad_norm_after += p.grad.data.norm(2).item() ** 2
      total_grad_norm_after = total_grad_norm_after**0.5
      self.optimizer.step()
      all_grad_norms_before.append(total_grad_norm_before)
      all_grad_norms_after.append(total_grad_norm_after)
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_bonus.item() if entropy_bonus is not None else 0.0
      mini_batch_step += 1
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if self.schedule == "adaptive":
      mean_kl /= num_updates
    self.storage.clear()
    self.tot_timesteps += 1
    loss_dict = {"surrogate_loss": mean_surrogate_loss, "value_loss": mean_value_loss}
    if self.knn_entropy_coef > 0:
      loss_dict["entropy_loss"] = mean_entropy
    metrics_dict = {"clip_param": self.clip_param}
    if self.schedule == "adaptive":
      metrics_dict["kl"] = mean_kl
    if all_grad_norms_before:
      metrics_dict["mean_grad_norm_before_clip"] = np.mean(all_grad_norms_before)
      metrics_dict["mean_grad_norm_after_clip"] = np.mean(all_grad_norms_after)
    if obs_normalizer is not None:
      obs_std = obs_normalizer.std.cpu()
      metrics_dict["obs_norm_min_std"] = obs_std.min().item()
      metrics_dict["obs_norm_max_std"] = obs_std.max().item()
      metrics_dict["obs_norm_mean_std"] = obs_std.mean().item()
    if privileged_obs_normalizer is not None:
      priv_obs_std = privileged_obs_normalizer.std.cpu()
      metrics_dict["privileged_obs_norm_min_std"] = priv_obs_std.min().item()
      metrics_dict["privileged_obs_norm_max_std"] = priv_obs_std.max().item()
      metrics_dict["privileged_obs_norm_mean_std"] = priv_obs_std.mean().item()
    loss_dict["metrics"] = metrics_dict
    return loss_dict

  def _compute_knn_entropy(self, x0_pred: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(x0_pred, x0_pred, p=2)
    dists = dists + torch.eye(dists.shape[-1], device=dists.device).unsqueeze(0) * 1e9
    knn_dists = torch.topk(dists, k=k, dim=-1, largest=False).values[..., -1]
    return torch.log(knn_dists + 1e-8).mean()
