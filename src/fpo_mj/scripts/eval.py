from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from fpo_mj.config import FpoRunnerCfg, build_default_fpo_runner_cfg
from fpo_mj.runners import FpoOnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

SUPPORTED_TASKS = ("Mjlab-Velocity-Flat-Unitree-G1",)


@dataclass
class EvalConfig:
  env: ManagerBasedRlEnvCfg
  agent: FpoRunnerCfg
  checkpoint_path: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  num_episodes: int = 10
  eval_modes: tuple[str, ...] = ("zero", "random")
  eval_fixed_seed: int = 12345
  gpu_ids: list[int] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "EvalConfig":
    if task_id not in SUPPORTED_TASKS:
      raise ValueError(f"Unsupported task_id: {task_id}")
    return EvalConfig(
      env=load_env_cfg(task_id),
      agent=build_default_fpo_runner_cfg(task_id),
    )


def _resolve_device() -> str:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    return "cpu"
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  return f"cuda:{local_rank}"


def _resolve_checkpoint(cfg: EvalConfig, log_root_path: Path) -> Path:
  if cfg.checkpoint_path is not None:
    return Path(cfg.checkpoint_path)
  if cfg.wandb_run_path is not None:
    checkpoint_path, _was_cached = get_wandb_checkpoint_path(
      log_root_path,
      Path(cfg.wandb_run_path),
      cfg.wandb_checkpoint_name,
    )
    return checkpoint_path
  return get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)


def run_eval(task_id: str, cfg: EvalConfig) -> dict[str, dict[str, float]]:
  device = _resolve_device()
  os.environ["MUJOCO_GL"] = "egl"
  configure_torch_backends()
  log_root_path = Path("logs") / "fpo_mj" / cfg.agent.experiment_name
  checkpoint_path = _resolve_checkpoint(cfg, log_root_path)
  env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  runner = FpoOnPolicyRunner(env, cfg.agent, log_dir=None, device=device)
  runner.load(str(checkpoint_path), load_optimizer=False)
  results = runner.evaluate(
    num_episodes=cfg.num_episodes,
    eval_modes=cfg.eval_modes,
    eval_fixed_seed=cfg.eval_fixed_seed,
  )
  env.close()
  return results


def main():
  import mjlab.tasks  # noqa: F401

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(SUPPORTED_TASKS),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    EvalConfig,
    args=remaining_args,
    default=EvalConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
  )
  run_eval(task_id=chosen_task, cfg=args)


if __name__ == "__main__":
  main()
