import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_rl_cfg

from fpo_mj.config import FpoRunnerCfg, build_default_fpo_runner_cfg


def test_build_default_fpo_runner_cfg_copies_official_common_fields():
    official = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1")

    cfg = build_default_fpo_runner_cfg("Mjlab-Velocity-Flat-Unitree-G1")

    assert isinstance(cfg, FpoRunnerCfg)
    assert cfg.seed == official.seed
    assert cfg.obs_groups == official.obs_groups
    assert cfg.clip_actions == official.clip_actions
    assert cfg.num_steps_per_env == official.num_steps_per_env
    assert cfg.max_iterations == official.max_iterations
    assert cfg.logger == official.logger
    assert cfg.wandb_project == official.wandb_project


def test_build_default_fpo_runner_cfg_uses_g1_fpo_defaults():
    cfg = build_default_fpo_runner_cfg("Mjlab-Velocity-Flat-Unitree-G1")

    assert cfg.experiment_name == "g1_flat_fpo"
    assert cfg.policy.actor_hidden_dims == (256, 256, 256)
    assert cfg.policy.critic_hidden_dims == (768, 768, 768)
    assert cfg.algorithm.n_samples_per_action == 32
    assert cfg.algorithm.num_learning_epochs == 32
