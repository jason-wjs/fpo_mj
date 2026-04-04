from dataclasses import replace
from pathlib import Path

from fpo_mj.config import build_default_fpo_runner_cfg
from fpo_mj.runners import FpoOnPolicyRunner
from fpo_mj.scripts.eval import EvalConfig, run_eval
from fpo_mj.scripts.train import TrainConfig, run_train
from tests.test_runner import FakeVecEnv, _make_cfg


def test_train_config_from_task_uses_fpo_runner_cfg():
    cfg = TrainConfig.from_task("Mjlab-Velocity-Flat-Unitree-G1")

    assert cfg.agent.class_name == "FpoOnPolicyRunner"
    assert cfg.agent.obs_groups == {"actor": ("actor",), "critic": ("critic",)}


def test_run_train_writes_checkpoint_with_fake_env(monkeypatch, tmp_path):
    monkeypatch.setattr("fpo_mj.scripts.train.ManagerBasedRlEnv", lambda cfg, device, render_mode=None: FakeVecEnv())
    monkeypatch.setattr("fpo_mj.scripts.train.RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
    monkeypatch.setattr("fpo_mj.scripts.train.dump_yaml", lambda path, payload: Path(path).parent.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr("fpo_mj.scripts.train.add_wandb_tags", lambda tags: None)
    monkeypatch.setattr("fpo_mj.scripts.train.configure_torch_backends", lambda: None)

    cfg = TrainConfig.from_task("Mjlab-Velocity-Flat-Unitree-G1")
    cfg.agent = _make_cfg()
    cfg.agent.logger = "tensorboard"

    run_train("Mjlab-Velocity-Flat-Unitree-G1", cfg, tmp_path / "train")

    assert list((tmp_path / "train").glob("model_*.pt"))


def test_run_eval_loads_checkpoint_and_returns_metrics(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(checkpoint_dir), device="cpu")
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    checkpoint_path = sorted(checkpoint_dir.glob("model_*.pt"))[-1]

    monkeypatch.setattr("fpo_mj.scripts.eval.ManagerBasedRlEnv", lambda cfg, device, render_mode=None: FakeVecEnv())
    monkeypatch.setattr("fpo_mj.scripts.eval.RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
    monkeypatch.setattr("fpo_mj.scripts.eval.get_checkpoint_path", lambda log_path, run_dir, checkpoint: checkpoint_path)
    monkeypatch.setattr("fpo_mj.scripts.eval.get_wandb_checkpoint_path", lambda log_path, run_path, checkpoint: (checkpoint_path, True))
    monkeypatch.setattr("fpo_mj.scripts.eval.configure_torch_backends", lambda: None)

    cfg = EvalConfig.from_task("Mjlab-Velocity-Flat-Unitree-G1")
    cfg.agent = _make_cfg()
    cfg.checkpoint_path = str(checkpoint_path)
    results = run_eval("Mjlab-Velocity-Flat-Unitree-G1", cfg)

    assert set(results) == {"zero", "random"}
