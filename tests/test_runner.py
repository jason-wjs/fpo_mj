from pathlib import Path

import torch
from tensordict import TensorDict

from fpo_mj.config import build_default_fpo_runner_cfg
from fpo_mj.runners import FpoOnPolicyRunner


class FakeVecEnv:
    def __init__(self):
        self.num_envs = 2
        self.num_actions = 3
        self.device = torch.device("cpu")
        self.max_episode_length = 4
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.common_step_counter = 0
        self.unwrapped = self
        self._step_count = 0

    def get_observations(self):
        return TensorDict(
            {
                "actor": torch.randn(self.num_envs, 5),
                "critic": torch.randn(self.num_envs, 7),
            },
            batch_size=[self.num_envs],
        )

    def reset(self):
        self._step_count = 0
        self.episode_length_buf.zero_()
        return self.get_observations(), {"log": {}}

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        self._step_count += 1
        self.common_step_counter += self.num_envs
        self.episode_length_buf += 1
        dones = torch.ones(self.num_envs, dtype=torch.long) if self._step_count >= 2 else torch.zeros(self.num_envs, dtype=torch.long)
        if dones.any():
            self.episode_length_buf[dones > 0] = 0
        return self.get_observations(), torch.ones(self.num_envs), dones, {"log": {}}

    def close(self):
        return None


def _make_cfg():
    cfg = build_default_fpo_runner_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    cfg.logger = "tensorboard"
    cfg.num_steps_per_env = 2
    cfg.max_iterations = 1
    cfg.save_interval = 1
    cfg.policy.actor_hidden_dims = (8, 8)
    cfg.policy.critic_hidden_dims = (16, 16)
    cfg.policy.timestep_embed_dim = 4
    cfg.policy.sampling_steps = 4
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    cfg.algorithm.n_samples_per_action = 4
    return cfg


def test_runner_learn_writes_checkpoint(tmp_path):
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(tmp_path), device="cpu")

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    checkpoints = list(Path(tmp_path).glob("model_*.pt"))
    assert checkpoints


def test_runner_load_restores_iteration_and_env_state(tmp_path):
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(tmp_path), device="cpu")
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    checkpoint_path = sorted(Path(tmp_path).glob("model_*.pt"))[-1]

    new_env = FakeVecEnv()
    new_runner = FpoOnPolicyRunner(new_env, _make_cfg(), log_dir=str(tmp_path / "reload"), device="cpu")
    infos = new_runner.load(str(checkpoint_path))

    assert new_runner.current_learning_iteration == 0
    assert infos["env_state"]["common_step_counter"] == new_env.common_step_counter


def test_runner_evaluate_returns_metrics_for_each_mode(tmp_path):
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(tmp_path), device="cpu")

    results = runner.evaluate(num_episodes=2, eval_modes=("zero", "random"))

    assert set(results) == {"zero", "random"}
    for mode in results.values():
        assert mode["num_episodes"] >= 2
        assert "mean_reward" in mode
        assert "mean_episode_length" in mode


def test_runner_get_inference_policy_supports_viewer_eval_modes(tmp_path):
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(tmp_path), device="cpu")
    obs = runner.env.get_observations()

    zero_policy = runner.get_inference_policy(device="cpu", eval_mode="zero")
    random_policy = runner.get_inference_policy(device="cpu", eval_mode="random")
    fixed_policy = runner.get_inference_policy(device="cpu", eval_mode="fixed_seed", eval_fixed_seed=7)

    zero_actions = zero_policy(obs)
    random_actions = random_policy(obs)
    fixed_actions_a = fixed_policy(obs)
    fixed_actions_b = fixed_policy(obs)

    assert zero_actions.shape == (runner.env.num_envs, runner.env.num_actions)
    assert random_actions.shape == (runner.env.num_envs, runner.env.num_actions)
    assert fixed_actions_a.shape == (runner.env.num_envs, runner.env.num_actions)
    assert torch.equal(fixed_actions_a, fixed_actions_b)


def test_runner_emits_startup_heartbeat_and_console_progress(monkeypatch, tmp_path, capsys):
    recorded_scalars = []

    class FakeLogger:
        def __init__(self, log_dir, cfg_dict=None):
            self.log_dir = log_dir
            self.logger_type = "tensorboard"

        def add_scalar(self, tag, value, step):
            recorded_scalars.append((tag, value, step))

        def save_model(self, path, step):
            return None

        def store_config(self, env_cfg, train_cfg):
            return None

        def save_file(self, path):
            return None

        def close(self):
            return None

    monkeypatch.setattr("fpo_mj.runners.fpo_on_policy_runner._TensorboardLogger", FakeLogger)
    cfg = _make_cfg()
    runner = FpoOnPolicyRunner(FakeVecEnv(), cfg, log_dir=str(tmp_path), device="cpu")

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    stdout = capsys.readouterr().out
    assert "Learning iteration 0/1" in stdout
    assert "Total steps:" in stdout
    assert ("System/initialized", 1.0, 0) in recorded_scalars


def test_runner_uses_official_wandb_writer(monkeypatch, tmp_path):
    init_calls = []
    stored_configs = []
    recorded_scalars = []

    class FakeWandbWriter:
        def __init__(self, log_dir, flush_secs, cfg):
            init_calls.append((log_dir, flush_secs, cfg))

        def store_config(self, env_cfg, train_cfg):
            stored_configs.append((env_cfg, train_cfg))

        def add_scalar(self, tag, value, step):
            recorded_scalars.append((tag, value, step))

        def save_model(self, path, step):
            return None

        def save_file(self, path):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("rsl_rl.utils.wandb_utils.WandbSummaryWriter", FakeWandbWriter)
    cfg = _make_cfg()
    cfg.logger = "wandb"
    runner = FpoOnPolicyRunner(FakeVecEnv(), cfg, log_dir=str(tmp_path), device="cpu")

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert init_calls
    assert stored_configs
    assert ("System/initialized", 1.0, 0) in recorded_scalars


def test_runner_defers_ema_updates_until_after_warmup():
    class SpyEma:
        def __init__(self):
            self.reset_calls = 0
            self.update_calls = 0
            self.shadow_params = {}

        def reset_to_current(self):
            self.reset_calls += 1

        def update(self):
            self.update_calls += 1

        def state_dict(self):
            return {}

    cfg = _make_cfg()
    cfg.algorithm.ema_warmup_steps = 2
    runner = FpoOnPolicyRunner(FakeVecEnv(), cfg, log_dir=None, device="cpu")
    runner.alg.ema = SpyEma()

    runner.learn(num_learning_iterations=3, init_at_random_ep_len=False)

    assert runner.alg.tot_timesteps == 3
    assert runner.alg.ema.reset_calls == 1
    assert runner.alg.ema.update_calls == 1
