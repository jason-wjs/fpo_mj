from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import tyro

from fpo_mj.config import FpoRunnerCfg, build_default_fpo_runner_cfg
from fpo_mj.runners import FpoOnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder

SUPPORTED_TASKS = ("Mjlab-Velocity-Flat-Unitree-G1",)


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: FpoRunnerCfg
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  gpu_ids: list[int] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    if task_id not in SUPPORTED_TASKS:
      raise ValueError(f"Unsupported task_id: {task_id}")
    return TrainConfig(
      env=load_env_cfg(task_id),
      agent=build_default_fpo_runner_cfg(task_id),
    )


def _resolve_device() -> tuple[str, int]:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    return "cpu", 0
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  return f"cuda:{local_rank}", local_rank


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  device, local_rank = _resolve_device()
  os.environ["MUJOCO_GL"] = "egl"
  configure_torch_backends()

  seed = cfg.agent.seed + local_rank if device != "cpu" else cfg.agent.seed
  cfg.agent.seed = seed
  cfg.env.seed = seed

  env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None)
  resume_path: Path | None = None
  log_root_path = log_dir.parent

  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      resume_path, _was_cached = get_wandb_checkpoint_path(
        log_root_path,
        Path(cfg.wandb_run_path),
        cfg.wandb_checkpoint_name,
      )
    else:
      resume_path = get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  runner = FpoOnPolicyRunner(env, cfg.agent, str(log_dir), device)
  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)

  if resume_path is not None:
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

  runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> Path:
  args = args or TrainConfig.from_task(task_id)
  log_root_path = Path("logs") / "fpo_mj" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name
  run_train(task_id, args, log_dir)
  return log_dir


def main():
  import mjlab.tasks  # noqa: F401

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(SUPPORTED_TASKS),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
  )
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
