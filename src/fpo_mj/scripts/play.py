from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from fpo_mj.config import build_default_fpo_runner_cfg
from fpo_mj.runners import FpoOnPolicyRunner
from fpo_mj.supported_tasks import SUPPORTED_TASKS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent_type: Literal["fpo", "ppo"]
  checkpoint_path: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  num_envs: int | None = 1
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  eval_mode: Literal["zero", "random", "fixed_seed"] = "zero"
  eval_fixed_seed: int = 12345


def _resolve_device(device_override: str | None) -> str:
  if device_override is not None:
    return device_override
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  if torch.cuda.is_available():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return f"cuda:{local_rank}"
  return "cpu"


def _resolve_checkpoint(
  cfg: PlayConfig,
  log_root_path: Path,
) -> Path:
  if cfg.checkpoint_path is not None:
    return Path(cfg.checkpoint_path)
  if cfg.wandb_run_path is None:
    raise ValueError("Either checkpoint_path or wandb_run_path must be provided.")
  checkpoint_path, _was_cached = get_wandb_checkpoint_path(
    log_root_path,
    Path(cfg.wandb_run_path),
    cfg.wandb_checkpoint_name,
  )
  return checkpoint_path


def _resolve_viewer(viewer: str) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _build_fpo_policy(task_id: str, env, checkpoint_path: Path, device: str, cfg: PlayConfig):
  agent_cfg = build_default_fpo_runner_cfg(task_id)
  runner = FpoOnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
  runner.load(str(checkpoint_path), load_optimizer=False)
  return runner.get_inference_policy(
    device=device,
    eval_mode=cfg.eval_mode,
    eval_fixed_seed=cfg.eval_fixed_seed,
  )


def _build_ppo_policy(task_id: str, env, checkpoint_path: Path, device: str, cfg: PlayConfig):
  if cfg.eval_mode != "zero" or cfg.eval_fixed_seed != 12345:
    print("[WARN] Ignoring FPO-specific eval settings for PPO playback.")

  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  if is_dataclass(agent_cfg):
    runner_train_cfg = asdict(agent_cfg)
  elif isinstance(agent_cfg, dict):
    runner_train_cfg = agent_cfg
  else:
    runner_train_cfg = vars(agent_cfg)
  runner = runner_cls(env, runner_train_cfg, log_dir=None, device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def run_play(task_id: str, cfg: PlayConfig) -> None:
  if task_id not in SUPPORTED_TASKS:
    raise ValueError(f"Unsupported task_id: {task_id}")

  configure_torch_backends()
  os.environ["MUJOCO_GL"] = "egl"
  device = _resolve_device(cfg.device)

  env_cfg = load_env_cfg(task_id, play=True)
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")
  if cfg.camera is not None:
    print("[WARN] camera selection is not yet implemented and will be ignored.")
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  if cfg.agent_type == "fpo":
    agent_cfg = build_default_fpo_runner_cfg(task_id)
    log_root_path = Path("logs") / "fpo_mj" / agent_cfg.experiment_name
  else:
    agent_cfg = load_rl_cfg(task_id)
    log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name

  checkpoint_path = _resolve_checkpoint(cfg, log_root_path)
  log_dir = checkpoint_path.parent
  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if cfg.agent_type == "fpo":
    policy = _build_fpo_policy(task_id, env, checkpoint_path, device, cfg)
  else:
    policy = _build_ppo_policy(task_id, env, checkpoint_path, device, cfg)

  resolved_viewer = _resolve_viewer(cfg.viewer)
  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  import mjlab.tasks  # noqa: F401

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(SUPPORTED_TASKS),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
  )
  run_play(task_id=chosen_task, cfg=args)


if __name__ == "__main__":
  main()
