"""Task-specific FPO defaults for mjlab tasks."""

from __future__ import annotations

from dataclasses import fields

from mjlab.tasks.registry import load_rl_cfg

from fpo_mj.config import FpoAlgorithmCfg, FpoPolicyCfg, FpoRunnerCfg


SUPPORTED_TASKS = {
  "Mjlab-Velocity-Flat-Unitree-G1",
}


def _copy_base_runner_fields(base_cfg) -> dict:
  allowed = {field.name for field in fields(FpoRunnerCfg)}
  return {
    key: getattr(base_cfg, key)
    for key in allowed
    if hasattr(base_cfg, key) and key not in {"policy", "algorithm", "device"}
  }


def build_fpo_runner_cfg(task_id: str) -> FpoRunnerCfg:
  if task_id not in SUPPORTED_TASKS:
    raise KeyError(f"unsupported task_id: {task_id}")

  base_cfg = load_rl_cfg(task_id)
  copied = _copy_base_runner_fields(base_cfg)
  copied["experiment_name"] = "g1_velocity_fpo"

  if task_id == "Mjlab-Velocity-Flat-Unitree-G1":
    return FpoRunnerCfg(
      **copied,
      policy=FpoPolicyCfg(
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(768, 768, 768),
        activation="elu",
      ),
      algorithm=FpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
      ),
    )

  raise AssertionError(f"unreachable task branch: {task_id}")
