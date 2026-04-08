from __future__ import annotations

import os
import statistics
import time
from collections import deque
from dataclasses import asdict

import torch

from fpo_mj.algorithms import FPO
from fpo_mj.env import ObservationAdapter
from fpo_mj.modules import ActorCritic, EmpiricalNormalization


class _TensorboardLogger:
  def __init__(self, log_dir: str, cfg_dict: dict | None = None):
    from torch.utils.tensorboard import SummaryWriter

    self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    self.logger_type = "tensorboard"

  def add_scalar(self, tag: str, value: float, step: int) -> None:
    self.writer.add_scalar(tag, value, step)

  def save_model(self, path: str, _step: int) -> None:
    return None

  def store_config(self, _env_cfg, _train_cfg: dict) -> None:
    return None

  def save_file(self, _path: str) -> None:
    return None

  def close(self) -> None:
    self.writer.close()


class _WandbLogger:
  def __init__(self, log_dir: str, cfg_dict: dict):
    from rsl_rl.utils.wandb_utils import WandbSummaryWriter

    self.writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=cfg_dict)
    self.logger_type = "wandb"

  def add_scalar(self, tag: str, value: float, step: int) -> None:
    self.writer.add_scalar(tag, value, step)

  def save_model(self, path: str, _step: int) -> None:
    self.writer.save_model(path, _step)

  def store_config(self, env_cfg, train_cfg: dict) -> None:
    self.writer.store_config(env_cfg, train_cfg)

  def save_file(self, path: str) -> None:
    self.writer.save_file(path)

  def close(self) -> None:
    self.writer.stop()
    self.writer.close()


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
    self.tot_time = 0.0
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
    ep_extras: list[dict] = []
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations

    for it in range(start_it, total_it):
      start = time.time()
      for _ in range(self.num_steps_per_env):
        actions = self.alg.act(actor_obs, critic_obs)
        next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
        actor_obs, critic_obs = self.adapter.adapt(next_obs)
        actor_obs = self.obs_normalizer(actor_obs.to(self.device))
        critic_obs = self.critic_obs_normalizer(critic_obs.to(self.device))
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        self.alg.process_env_step(rewards, dones, infos)
        if "episode" in infos:
          ep_extras.append(infos["episode"])
        elif "log" in infos:
          ep_extras.append(infos["log"])
        cur_reward_sum += rewards
        cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
          rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
          lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
          cur_reward_sum[new_ids] = 0
          cur_episode_length[new_ids] = 0

      stop = time.time()
      collect_time = stop - start
      start = stop
      self.alg.compute_returns(critic_obs)
      loss_dict = self.alg.update(
        obs_normalizer=self.obs_normalizer if self.empirical_normalization else None,
        privileged_obs_normalizer=self.critic_obs_normalizer if self.empirical_normalization else None,
      )
      self._update_ema()
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      self._log_iteration(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        rewbuffer=rewbuffer,
        lenbuffer=lenbuffer,
        ep_extras=ep_extras,
      )
      if self.log_dir is not None and it % self.save_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
      ep_extras.clear()

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

  def get_inference_policy(
    self,
    device: str | None = None,
    eval_mode: str = "zero",
    eval_fixed_seed: int = 12345,
  ):
    inference_device = device or self.device
    self.eval_mode()
    self.alg.policy.to(inference_device)
    if self.empirical_normalization:
      self.obs_normalizer.to(inference_device)

    def policy(observations) -> torch.Tensor:
      actor_obs, _critic_obs = self.adapter.adapt(observations)
      actor_obs = actor_obs.to(inference_device)
      if self.empirical_normalization:
        actor_obs = self.obs_normalizer(actor_obs)
      with torch.no_grad():
        return self.alg.policy.act_inference(
          actor_obs,
          eval_mode=eval_mode,
          eval_fixed_seed=eval_fixed_seed,
        )

    return policy

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
    cfg_dict = self._build_logger_cfg_dict()
    env_cfg = getattr(self.env, "cfg", None) or getattr(self.env.unwrapped, "cfg", None)
    if self.cfg.logger == "wandb":
      self.writer = _WandbLogger(self.log_dir, cfg_dict)
    else:
      self.writer = _TensorboardLogger(self.log_dir, cfg_dict)
    self.writer.store_config(env_cfg, cfg_dict)
    self.writer.add_scalar("System/initialized", 1.0, 0)

  def _log_iteration(
    self,
    it: int,
    start_it: int,
    total_it: int,
    collect_time: float,
    learn_time: float,
    loss_dict: dict,
    rewbuffer: deque,
    lenbuffer: deque,
    ep_extras: list[dict],
  ):
    if self.writer is None:
      return
    collection_size = self.num_steps_per_env * self.env.num_envs
    iteration_time = collect_time + learn_time
    self.tot_timesteps += collection_size
    self.tot_time += iteration_time

    extras_string = ""
    if ep_extras:
      for key in ep_extras[0]:
        values: list[float] = []
        for ep_info in ep_extras:
          if key not in ep_info:
            continue
          value = ep_info[key]
          if isinstance(value, torch.Tensor):
            if value.numel() == 0:
              continue
            values.extend(value.detach().reshape(-1).cpu().tolist())
          else:
            values.append(float(value))
        if not values:
          continue
        mean_value = statistics.mean(values)
        tag = key if "/" in key else f"Episode/{key}"
        label = f"{key}:" if "/" in key else f"Mean episode {key}:"
        self.writer.add_scalar(tag, mean_value, it)
        extras_string += f"{label:>40} {mean_value:.4f}\n"

    scalar_losses = {key: value for key, value in loss_dict.items() if key != "metrics"}
    for key, value in scalar_losses.items():
      self.writer.add_scalar(f"Loss/{key}", value, it)
    self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)

    action_noise_std = float(self.cfg.policy.action_perturb_std)
    self.writer.add_scalar("Policy/action_perturb_std", action_noise_std, it)

    fps = int(collection_size / iteration_time) if iteration_time > 0 else 0
    self.writer.add_scalar("Perf/total_fps", fps, it)
    self.writer.add_scalar("Perf/collection_time", collect_time, it)
    self.writer.add_scalar("Perf/learning_time", learn_time, it)

    for key, value in loss_dict["metrics"].items():
      self.writer.add_scalar(f"Metrics/{key}", value, it)
    if len(rewbuffer) > 0:
      mean_reward = statistics.mean(rewbuffer)
      mean_episode_length = statistics.mean(lenbuffer)
      self.writer.add_scalar("Train/mean_reward", mean_reward, it)
      self.writer.add_scalar("Train/mean_episode_length", mean_episode_length, it)
      if self.writer.logger_type != "wandb":
        self.writer.add_scalar("Train/mean_reward/time", mean_reward, int(self.tot_time))
        self.writer.add_scalar("Train/mean_episode_length/time", mean_episode_length, int(self.tot_time))

    width = 80
    pad = 40
    log_string = f'{"#" * width}\n'
    log_string += f'\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n'
    if self.cfg.run_name:
      log_string += f'{"Run name:":>{pad}} {self.cfg.run_name}\n'
    log_string += (
      f'{"Total steps:":>{pad}} {self.tot_timesteps} \n'
      f'{"Steps per second:":>{pad}} {fps:.0f} \n'
      f'{"Collection time:":>{pad}} {collect_time:.3f}s \n'
      f'{"Learning time:":>{pad}} {learn_time:.3f}s \n'
    )
    for key, value in scalar_losses.items():
      log_string += f'{f"Mean {key} loss:":>{pad}} {float(value):.4f}\n'
    if len(rewbuffer) > 0:
      log_string += f'{"Mean reward:":>{pad}} {statistics.mean(rewbuffer):.2f}\n'
      log_string += f'{"Mean episode length:":>{pad}} {statistics.mean(lenbuffer):.2f}\n'
    log_string += f'{"Action perturb std:":>{pad}} {action_noise_std:.4f}\n'
    log_string += extras_string
    done_it = it + 1 - start_it
    remaining_it = total_it - start_it - done_it
    eta = self.tot_time / done_it * remaining_it if done_it > 0 else 0.0
    log_string += (
      f'{"-" * width}\n'
      f'{"Iteration time:":>{pad}} {iteration_time:.2f}s\n'
      f'{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n'
      f'{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n'
    )
    print(log_string)

  def _build_logger_cfg_dict(self) -> dict:
    cfg_dict = asdict(self.cfg)
    cfg_dict.setdefault("algorithm", {})
    cfg_dict["algorithm"].setdefault("rnd_cfg", None)
    cfg_dict.setdefault("num_steps_per_env", self.num_steps_per_env)
    return cfg_dict

  def _update_ema(self) -> None:
    if self.alg.ema is None:
      return
    if self.alg.tot_timesteps == self.alg.ema_warmup_steps:
      self.alg.ema.reset_to_current()
    elif self.alg.tot_timesteps > self.alg.ema_warmup_steps:
      self.alg.ema.update()
