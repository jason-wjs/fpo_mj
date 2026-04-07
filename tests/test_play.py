from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_runner import FakeVecEnv, _make_cfg


def _make_env_cfg():
    return SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        viewer=SimpleNamespace(height=240, width=320),
        commands={},
        terminations={"time_out": object()},
    )


def test_run_play_uses_fpo_runner_and_native_viewer(monkeypatch, tmp_path):
    from fpo_mj.scripts.play import PlayConfig, run_play

    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    load_calls = []
    policy_calls = []
    viewer_runs = []

    class FakeFpoRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            self.env = env
            self.train_cfg = train_cfg
            self.device = device

        def load(self, path, load_optimizer=False):
            load_calls.append((path, load_optimizer))
            return {}

        def get_inference_policy(self, device=None, eval_mode="zero", eval_fixed_seed=12345):
            policy_calls.append((device, eval_mode, eval_fixed_seed))
            return lambda obs: torch.zeros(self.env.num_envs, self.env.num_actions)

    class FakeViewer:
        def __init__(self, env, policy):
            self.env = env
            self.policy = policy

        def run(self):
            viewer_runs.append(self.policy(self.env.get_observations()).shape)

    monkeypatch.setattr("fpo_mj.scripts.play.configure_torch_backends", lambda: None)
    monkeypatch.setattr("fpo_mj.scripts.play.load_env_cfg", lambda task_id, play=False: _make_env_cfg())
    monkeypatch.setattr("fpo_mj.scripts.play.ManagerBasedRlEnv", lambda cfg, device, render_mode=None: FakeVecEnv())
    monkeypatch.setattr("fpo_mj.scripts.play.RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
    monkeypatch.setattr("fpo_mj.scripts.play.FpoOnPolicyRunner", FakeFpoRunner)
    monkeypatch.setattr("fpo_mj.scripts.play.NativeMujocoViewer", FakeViewer)
    monkeypatch.setattr("fpo_mj.scripts.play.ViserPlayViewer", FakeViewer)

    cfg = PlayConfig(
        agent_type="fpo",
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        viewer="native",
        eval_mode="fixed_seed",
        eval_fixed_seed=17,
    )
    run_play("Mjlab-Velocity-Flat-Unitree-G1", cfg)

    assert load_calls == [(str(checkpoint_path), False)]
    assert policy_calls == [("cpu", "fixed_seed", 17)]
    assert viewer_runs == [(2, 3)]


def test_run_play_uses_ppo_runner_and_warns_about_eval_mode(monkeypatch, tmp_path, capsys):
    from fpo_mj.scripts.play import PlayConfig, run_play

    checkpoint_path = tmp_path / "policy.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    load_calls = []
    viewer_runs = []
    init_cfg_types = []

    class FakePpoRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            self.env = env
            self.train_cfg = train_cfg
            self.device = device
            init_cfg_types.append(type(train_cfg))

        def load(self, path, load_cfg=None, strict=True, map_location=None):
            load_calls.append((path, load_cfg, strict, map_location))
            return {}

        def get_inference_policy(self, device=None):
            return lambda obs: torch.ones(self.env.num_envs, self.env.num_actions)

    class FakeViewer:
        def __init__(self, env, policy):
            self.env = env
            self.policy = policy

        def run(self):
            viewer_runs.append(self.policy(self.env.get_observations()).shape)

    monkeypatch.setattr("fpo_mj.scripts.play.configure_torch_backends", lambda: None)
    monkeypatch.setattr("fpo_mj.scripts.play.load_env_cfg", lambda task_id, play=False: _make_env_cfg())
    monkeypatch.setattr("fpo_mj.scripts.play.load_rl_cfg", lambda task_id: SimpleNamespace(experiment_name="g1_velocity", clip_actions=1.0))
    monkeypatch.setattr("fpo_mj.scripts.play.load_runner_cls", lambda task_id: FakePpoRunner)
    monkeypatch.setattr("fpo_mj.scripts.play.ManagerBasedRlEnv", lambda cfg, device, render_mode=None: FakeVecEnv())
    monkeypatch.setattr("fpo_mj.scripts.play.RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
    monkeypatch.setattr("fpo_mj.scripts.play.NativeMujocoViewer", FakeViewer)
    monkeypatch.setattr("fpo_mj.scripts.play.ViserPlayViewer", FakeViewer)

    cfg = PlayConfig(
        agent_type="ppo",
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        viewer="native",
        eval_mode="random",
        eval_fixed_seed=99,
    )
    run_play("Mjlab-Velocity-Flat-Unitree-G1", cfg)

    stdout = capsys.readouterr().out
    assert "Ignoring FPO-specific eval settings" in stdout
    assert init_cfg_types == [dict]
    assert load_calls == [(str(checkpoint_path), {"actor": True}, True, "cpu")]
    assert viewer_runs == [(2, 3)]


def test_run_play_prefers_checkpoint_path_over_wandb(monkeypatch, tmp_path):
    from fpo_mj.scripts.play import PlayConfig, run_play

    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    load_calls = []

    class FakeFpoRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            self.env = env

        def load(self, path, load_optimizer=False):
            load_calls.append(path)
            return {}

        def get_inference_policy(self, device=None, eval_mode="zero", eval_fixed_seed=12345):
            return lambda obs: torch.zeros(self.env.num_envs, self.env.num_actions)

    class FakeViewer:
        def __init__(self, env, policy):
            self.env = env
            self.policy = policy

        def run(self):
            return None

    monkeypatch.setattr("fpo_mj.scripts.play.configure_torch_backends", lambda: None)
    monkeypatch.setattr("fpo_mj.scripts.play.load_env_cfg", lambda task_id, play=False: _make_env_cfg())
    monkeypatch.setattr("fpo_mj.scripts.play.ManagerBasedRlEnv", lambda cfg, device, render_mode=None: FakeVecEnv())
    monkeypatch.setattr("fpo_mj.scripts.play.RslRlVecEnvWrapper", lambda env, clip_actions=None: env)
    monkeypatch.setattr("fpo_mj.scripts.play.FpoOnPolicyRunner", FakeFpoRunner)
    monkeypatch.setattr("fpo_mj.scripts.play.NativeMujocoViewer", FakeViewer)
    monkeypatch.setattr("fpo_mj.scripts.play.ViserPlayViewer", FakeViewer)
    monkeypatch.setattr(
        "fpo_mj.scripts.play.get_wandb_checkpoint_path",
        lambda *args, **kwargs: pytest.fail("wandb checkpoint resolution should not be used when checkpoint_path is provided"),
    )

    cfg = PlayConfig(
        agent_type="fpo",
        checkpoint_path=str(checkpoint_path),
        wandb_run_path="entity/project/run",
        device="cpu",
        viewer="native",
    )
    run_play("Mjlab-Velocity-Flat-Unitree-G1", cfg)

    assert load_calls == [str(checkpoint_path)]
