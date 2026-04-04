from __future__ import annotations

import os
import statistics
from collections import deque
from time import perf_counter

import torch

from fpo_mj.algorithms import FPO
from fpo_mj.env import ObservationAdapter
from fpo_mj.modules import ActorCritic, EmpiricalNormalization


class _TensorboardLogger:
  def __init__(self, log_dir: str):
    from torch.utils.tensorboard import SummaryWriter

    self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    self.logger_type = "tensorboard"

  def add_scalar(self, tag: str, value: float, step: int) -> None:
    self.writer.add_scalar(tag, value, step)

  def save_model(self, path: str, _step: int) -> None:
    return None

  def close(self) -> None:
    self.writer.close()


class _WandbLogger:
  def __init__(self, log_dir: str, project: str, run_name: str):
    import wandb

    self._wandb = wandb
    self._wandb.init(project=project, name=run_name or None, dir=log_dir)
    self.logger_type = "wandb"

  def add_scalar(self, tag: str, value: float, step: int) -> None:
    self._wandb.log({tag: value}, step=step)

  def save_model(self, path: str, _step: int) -> None:
    self._wandb.save(path, base_path=os.path.dirname(path))

  def close(self) -> None:
    self._wandb.finish()


class FpoOnPolicyRunner:
  def __init__(self, env, train_cfg, log_dir: str | None = None, device: str = "cpu"):
    self.cfg = train_cfg
    self.device = device
    self.env = env
    self.adapter = ObservationAdapter(train_cfg.obs_groups)

    obs = self.env.get_observations()
    actor_obs, critic_obs = self.adapter.adapt(obs)
    num_obs = actor_obs.shape[1]
    num_privileged_obs = critic_obs.shape[1]

    policy = ActorCritic(num_obs, num_privileged_obs, self.env.num_actions, cfg=train_cfg.policy).to(self.device)
    self.alg = FPO(policy, cfg=train_cfg.algorithm, device=self.device)
    self.alg.init_storage(
      self.env.num_envs,
      train_cfg.num_steps_per_env,
      [num_obs],
      [num_privileged_obs],
      [self.env.num_actions],
    )

    self.num_steps_per_env = train_cfg.num_steps_per_env
    self.save_interval = train_cfg.save_interval
    self.empirical_normalization = train_cfg.empirical_normalization
    self.log_dir = log_dir
    self.writer = None
    self.tot_timesteps = 0
    self.current_learning_iteration = 0
    self.git_status_repos: list[str] = []

    if self.empirical_normalization:
      self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
      self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
    else:
      self.obs_normalizer = torch.nn.Identity().to(self.device)
      self.critic_obs_normalizer = torch.nn.Identity().to(self.device)

  def add_git_repo_to_log(self, repo_file_path: str):
    self.git_status_repos.append(repo_file_path)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    self._ensure_logger()
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

    obs = self.env.get_observations()
    actor_obs, critic_obs = self.adapter.adapt(obs)
    actor_obs = actor_obs.to(self.device)
    critic_obs = critic_obs.to(self.device)
    self.train_mode()

    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    for it in range(self.current_learning_iteration, self.current_learning_iteration + num_learning_iterations):
      iteration_start = perf_counter()
      print(
        f"[FPO] Starting iteration {it}: collecting {self.num_steps_per_env} rollout steps across {self.env.num_envs} envs.",
        flush=True,
      )
      for _ in range(self.num_steps_per_env):
        actions = self.alg.act(actor_obs, critic_obs)
        next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
        actor_obs, critic_obs = self.adapter.adapt(next_obs)
        actor_obs = self.obs_normalizer(actor_obs.to(self.device))
        critic_obs = self.critic_obs_normalizer(critic_obs.to(self.device))
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        self.alg.process_env_step(rewards, dones, infos)
        cur_reward_sum += rewards
        cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
          rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
          lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
          cur_reward_sum[new_ids] = 0
          cur_episode_length[new_ids] = 0

      self.alg.compute_returns(critic_obs)
      loss_dict = self.alg.update(
        obs_normalizer=self.obs_normalizer if self.empirical_normalization else None,
        privileged_obs_normalizer=self.critic_obs_normalizer if self.empirical_normalization else None,
      )
      if self.alg.ema is not None:
        self.alg.ema.update()
      self.current_learning_iteration = it
      self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
      self._log_iteration(it, loss_dict, rewbuffer, lenbuffer)
      iteration_duration = perf_counter() - iteration_start
      print(
        f"[FPO] Finished iteration {it}: total_timesteps={self.tot_timesteps}, duration_s={iteration_duration:.2f}",
        flush=True,
      )
      if self.log_dir is not None and it % self.save_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    if self.log_dir is not None:
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
    if self.writer is not None:
      self.writer.close()

  def save(self, path: str, infos=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state_dict = self.alg.policy.state_dict()
    if self.alg.ema is not None and self.alg.tot_timesteps >= self.alg.ema_warmup_steps:
      for name, ema_param in self.alg.ema.shadow_params.items():
        full_name = f"actor.{name}"
        if full_name in model_state_dict:
          model_state_dict[full_name] = ema_param.clone()
    saved_dict = {
      "model_state_dict": model_state_dict,
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": {**(infos or {}), "env_state": {"common_step_counter": getattr(self.env.unwrapped, "common_step_counter", 0)}},
    }
    if self.alg.ema is not None:
      saved_dict["ema_state_dict"] = self.alg.ema.state_dict()
    if self.empirical_normalization:
      saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
      saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
    torch.save(saved_dict, path)
    if self.writer is not None:
      self.writer.save_model(path, self.current_learning_iteration)

  def load(self, path: str, load_optimizer: bool = True):
    loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
    self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
    if load_optimizer:
      self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    if self.alg.ema is not None and "ema_state_dict" in loaded_dict:
      self.alg.ema.load_state_dict(loaded_dict["ema_state_dict"])
    if self.empirical_normalization:
      self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
      self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
    self.current_learning_iteration = loaded_dict["iter"]
    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def train_mode(self):
    self.alg.policy.train()
    if self.empirical_normalization:
      self.obs_normalizer.train()
      self.critic_obs_normalizer.train()

  def eval_mode(self):
    self.alg.policy.eval()
    if self.empirical_normalization:
      self.obs_normalizer.eval()
      self.critic_obs_normalizer.eval()

  def evaluate(
    self,
    num_episodes: int,
    eval_modes: tuple[str, ...] | None = None,
    eval_fixed_seed: int | None = None,
  ) -> dict[str, dict[str, float]]:
    eval_modes = eval_modes or tuple(self.cfg.flow_eval_modes)
    eval_fixed_seed = eval_fixed_seed if eval_fixed_seed is not None else self.cfg.flow_eval_fixed_seed
    self.eval_mode()
    results: dict[str, dict[str, float]] = {}
    for mode in eval_modes:
      obs, _extras = self.env.reset()
      episode_rewards: list[float] = []
      episode_lengths: list[float] = []
      reward_acc = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
      length_acc = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
      while len(episode_rewards) < num_episodes:
        actor_obs, _critic_obs = self.adapter.adapt(obs)
        actor_obs = actor_obs.to(self.device)
        if self.empirical_normalization:
          actor_obs = self.obs_normalizer(actor_obs)
        with torch.no_grad():
          actions = self.alg.policy.act_inference(
            actor_obs,
            eval_mode=mode,
            eval_fixed_seed=eval_fixed_seed,
          )
        obs, rewards, dones, _infos = self.env.step(actions.to(self.env.device))
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        reward_acc += rewards
        length_acc += 1
        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        for idx in done_ids:
          episode_rewards.append(float(reward_acc[idx].item()))
          episode_lengths.append(float(length_acc[idx].item()))
          reward_acc[idx] = 0
          length_acc[idx] = 0
          if len(episode_rewards) >= num_episodes:
            break
      results[mode] = {
        "mean_reward": statistics.mean(episode_rewards),
        "mean_episode_length": statistics.mean(episode_lengths),
        "num_episodes": float(len(episode_rewards)),
      }
    return results

  def _ensure_logger(self):
    if self.log_dir is None or self.writer is not None:
      return
    os.makedirs(self.log_dir, exist_ok=True)
    if self.cfg.logger == "wandb":
      self.writer = _WandbLogger(self.log_dir, self.cfg.wandb_project, self.cfg.run_name)
    else:
      self.writer = _TensorboardLogger(self.log_dir)
    self.writer.add_scalar("System/initialized", 1.0, 0)

  def _log_iteration(self, iteration: int, loss_dict: dict, rewbuffer: deque, lenbuffer: deque):
    if self.writer is None:
      return
    self.writer.add_scalar("Loss/surrogate_loss", loss_dict["surrogate_loss"], iteration)
    self.writer.add_scalar("Loss/value_loss", loss_dict["value_loss"], iteration)
    for key, value in loss_dict["metrics"].items():
      self.writer.add_scalar(f"Metrics/{key}", value, iteration)
    if len(rewbuffer) > 0:
      self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), iteration)
      self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), iteration)
