from __future__ import annotations

import torch


class RolloutStorage:
  class Transition:
    def __init__(self):
      self.observations = None
      self.privileged_observations = None
      self.actions = None
      self.rewards = None
      self.dones = None
      self.values = None
      self.initial_cfm_loss = None
      self.x1_pred = None
      self.cfm_loss_eps = None
      self.cfm_loss_t = None

    def clear(self):
      self.__init__()

  def __init__(
    self,
    num_envs,
    num_transitions_per_env,
    obs_shape,
    privileged_obs_shape,
    actions_shape,
    device="cpu",
    n_samples_per_action=32,
  ):
    self.device = device
    self.num_transitions_per_env = num_transitions_per_env
    self.num_envs = num_envs
    self.n_samples_per_action = n_samples_per_action
    self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
    self.privileged_observations = torch.zeros(
      num_transitions_per_env, num_envs, *privileged_obs_shape, device=device
    )
    self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
    self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
    self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.initial_cfm_loss = torch.zeros(num_transitions_per_env, num_envs, n_samples_per_action, device=device)
    self.cfm_loss_eps = torch.zeros(
      num_transitions_per_env, num_envs, n_samples_per_action, *actions_shape, device=device
    )
    self.cfm_loss_t = torch.zeros(num_transitions_per_env, num_envs, n_samples_per_action, 1, device=device)
    self.x1_pred = torch.zeros(
      num_transitions_per_env, num_envs, n_samples_per_action, *actions_shape, device=device
    )
    self.step = 0

  def add_transitions(self, transition: Transition):
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
    self.observations[self.step].copy_(transition.observations)
    self.privileged_observations[self.step].copy_(transition.privileged_observations)
    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.initial_cfm_loss[self.step].copy_(transition.initial_cfm_loss)
    self.cfm_loss_eps[self.step].copy_(transition.cfm_loss_eps)
    self.cfm_loss_t[self.step].copy_(transition.cfm_loss_t)
    self.x1_pred[self.step].copy_(transition.x1_pred)
    self.step += 1

  def clear(self):
    self.step = 0

  def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
      next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
      next_is_not_terminal = 1.0 - self.dones[step].float()
      delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
      advantage = delta + next_is_not_terminal * gamma * lam * advantage
      self.returns[step] = advantage + self.values[step]
    self.advantages = self.returns - self.values
    if normalize_advantage:
      self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

  def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
    observations = self.observations.flatten(0, 1)
    privileged_observations = self.privileged_observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_cfm_loss = self.initial_cfm_loss.flatten(0, 1)
    old_cfm_eps = self.cfm_loss_eps.flatten(0, 1)
    old_cfm_t = self.cfm_loss_t.flatten(0, 1)
    old_x1_pred = self.x1_pred.flatten(0, 1)
    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size
        batch_idx = indices[start:end]
        yield (
          observations[batch_idx],
          privileged_observations[batch_idx],
          actions[batch_idx],
          values[batch_idx],
          advantages[batch_idx],
          returns[batch_idx],
          old_x1_pred[batch_idx],
          old_cfm_loss[batch_idx],
          old_cfm_eps[batch_idx],
          old_cfm_t[batch_idx],
          (None, None),
          None,
        )
