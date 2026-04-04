from mjlab.tasks.registry import load_rl_cfg

from fpo_mj.config import FpoRunnerCfg
from fpo_mj.task_registry import SUPPORTED_TASKS, build_fpo_runner_cfg


def test_supported_tasks_contains_g1_velocity() -> None:
    assert "Mjlab-Velocity-Flat-Unitree-G1" in SUPPORTED_TASKS


def test_build_fpo_runner_cfg_preserves_official_common_fields() -> None:
    task_id = "Mjlab-Velocity-Flat-Unitree-G1"
    base_cfg = load_rl_cfg(task_id)

    cfg = build_fpo_runner_cfg(task_id)

    assert isinstance(cfg, FpoRunnerCfg)
    assert cfg.obs_groups == base_cfg.obs_groups
    assert cfg.clip_actions == base_cfg.clip_actions
    assert cfg.num_steps_per_env == base_cfg.num_steps_per_env
    assert cfg.max_iterations == base_cfg.max_iterations
    assert cfg.logger == base_cfg.logger
    assert cfg.wandb_project == base_cfg.wandb_project
    assert cfg.resume == base_cfg.resume
    assert cfg.upload_model == base_cfg.upload_model
    assert cfg.experiment_name == "g1_velocity_fpo"
    assert cfg.policy.actor_hidden_dims == (256, 256, 256)
    assert cfg.policy.critic_hidden_dims == (768, 768, 768)
    assert cfg.algorithm.n_samples_per_action == 32
    assert cfg.algorithm.num_learning_epochs == 32


def test_build_fpo_runner_cfg_rejects_unknown_task() -> None:
    try:
        build_fpo_runner_cfg("Mjlab-Velocity-Flat-Does-Not-Exist")
    except KeyError as exc:
        assert "Mjlab-Velocity-Flat-Does-Not-Exist" in str(exc)
    else:
        raise AssertionError("expected KeyError for unsupported task")
